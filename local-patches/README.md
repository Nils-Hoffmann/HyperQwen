# Local deployment: RTX 3090, Docker Desktop / WSL2

Branch `local/rtx3090-wsl2` on top of upstream `syv-ai/HyperQwen` main.

- `docker-compose.override.yml`: pinned image digest, gateway on 0.0.0.0:18020 (nginx config from gpu-middleware),
  engine on 127.0.0.1:18022, sleep-mode cumem patch (`comfy-headless/patches/cumem.py`), and this dir's patch.
- `env.rtx3090-wsl2.example`: the `.env` in use (API key removed). Copy to `.env` and set `VLLM_API_KEY`.
- `dflash-meta-placeholders/sitecustomize.py`: builds the DFlash drafter's shared embed/lm_head
  placeholders on the meta device. Without it, `--enable-sleep-mode` + vision OOMs during draft load on WSL2.

Vision on WSL2: only ~1.1 GiB of host memory can be pinned, so the tower is offloaded to pageable RAM
(`VISION_OFFLOAD=1` + `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`). That keeps 64k context (KV pool 67,126 tokens).
