"""Local HyperQwen patch: build DFlash drafter placeholder embed/lm_head on the meta device.

Why: the DFlash2 drafter checkpoint ships no embed_tokens / lm_head (it shares the
target's). vLLM still allocates both placeholders on the GPU first (2 x ~2.37 GiB
for vocab 248320 x 5120 bf16), then replaces them with the target's modules.
With --enable-sleep-mode those allocations go through the cumem pool, which on
WSL2 cannot spill to system RAM, so the transient peak (~23.5 GiB with the
vision tower resident) exceeds the ~23.25 GiB the pool can map and the engine
OOMs during draft load. Constructing the two placeholders on the meta device removes the
peak; they are discarded when vLLM swaps in the target's modules.

Only active while load_dflash_model runs, and only when the drafter declares no
own embed/lm_head. After load it verifies both ended up on the GPU (shared).
Mounted via PYTHONPATH from docker-compose.override.yml. Chain-loads the
system sitecustomize it shadows.
"""
import importlib.abc
import importlib.util
import os
import sys

_ACTIVE = False


def _patch_vocab_embedding():
    import torch
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe

    orig_init = vpe.VocabParallelEmbedding.__init__
    if getattr(orig_init, "_hq_cpu_placeholder", False):
        return

    def __init__(self, *args, **kwargs):
        if _ACTIVE:
            with torch.device("meta"):
                return orig_init(self, *args, **kwargs)
        return orig_init(self, *args, **kwargs)

    __init__._hq_cpu_placeholder = True
    vpe.VocabParallelEmbedding.__init__ = __init__  # ParallelLMHead inherits it


def _wrap_loader(mod):
    orig = mod.load_dflash_model

    def load_dflash_model(target_model, vllm_config):
        global _ACTIVE
        hf = vllm_config.speculative_config.draft_model_config.hf_config
        own = getattr(hf, "has_own_lm_head", False) or getattr(hf, "has_own_embed_tokens", False)
        if own or os.environ.get("HQ_DFLASH_CPU_PLACEHOLDERS", "1") == "0":
            return orig(target_model, vllm_config)
        _patch_vocab_embedding()
        _ACTIVE = True
        try:
            model = orig(target_model, vllm_config)
        finally:
            _ACTIVE = False
        left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if t.device.type == "meta"]
        if left:
            raise RuntimeError(f"hq-patch: drafter tensors not shared with the target "
                               f"(still meta placeholders): {left[:4]}; set HQ_DFLASH_CPU_PLACEHOLDERS=0")
        import gc
        gc.collect()
        print("[hq-patch] DFlash drafter embed/lm_head built on meta, shared from target", flush=True)
        return model

    mod.load_dflash_model = load_dflash_model


class _Finder(importlib.abc.MetaPathFinder):
    TARGETS = ("vllm.v1.worker.gpu.spec_decode.dflash.utils",)

    def find_spec(self, name, path, target=None):
        if name not in self.TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _wrap_loader(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())

# Chain-load the system sitecustomize this file shadows.
_sys_sc = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_sys_sc):
    with open(_sys_sc) as _f:
        exec(compile(_f.read(), _sys_sc, "exec"), {"__name__": "_system_sitecustomize"})
