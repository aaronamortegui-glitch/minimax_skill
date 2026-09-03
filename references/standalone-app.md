# Building your own app, without ComfyUI

## Executive summary

**It is possible, it is officially documented, and it will not run on a single consumer GPU.**

| | Requires |
|---|---|
| Official MiniMax baseline | 8x NVIDIA B200 in one node |
| Official "consumer" recipe | **2x RTX 5090 + >=200 GiB RAM** |
| Latency reference | 4x H200 → 75 s per clip (1344x768, 124 frames, 50 steps) |

What lets H3 run on **one 24 GB GPU** is **ComfyUI's dynamic offload**: it moves weights
between VRAM and RAM on the fly (in the log it shows as `Model MiniMaxH3 prepared for
dynamic VRAM loading. 19995MB Staged`). Your own app would have to reimplement that
mechanism or demand the hardware in the table.

**Conclusion: on consumer hardware ComfyUI is not convenience, it is what makes the model
viable.**

## If the hardware is there: two options

### SGLang — the official route

The one MiniMax documents.

- Python 3.12, pinned commit `2511743bd784e69e5a81ca3d926a000711dae4ab`
- `uv pip install -e "python[diffusion]" --prerelease=allow`
- **One service per mode**: FL2VA and Ref2VA need separate instances
- Online FP8 quantization: **-40% peak VRAM**, no speed gain on datacenter GPUs

**API it exposes** — asynchronous, OpenAI-style:

```
POST /v1/videos            -> {prompt, task, duration, resolution, steps} -> job id
GET  /v1/videos/{id}       -> status
GET  /v1/videos/{id}/content -> download the mp4
```

Consume it with `curl` or any HTTP client, polling until it finishes.

### vLLM-Omni — the community route

Community maintained, **experimental**. Its real advantage: it supports **FL2VA and Ref2VA
in a single service**, without standing up two instances.

## What the app would have to solve

Everything ComfyUI handles today and you would have to replace:

1. **Dynamic weight offload** across VRAM, RAM and disk. The critical piece with one GPU.
2. **Segmentation and tracking** if you want person replacement: SAM 3.1 plus per-object
   tracking across the clip.
3. **Latent-space masking** — a pixel mask is not enough: it has to be reduced to the
   latent geometry or the original bleeds through.
4. **Pose preprocessing** (DWPose) if you want motion transfer.
5. **The `17k+5` frame grid** and aspect cropping.
6. **Queueing and polling**, which SGLang's API already gives you.

## Migrating from the scripts in this repo

The scripts in `scripts/` speak **HTTP to ComfyUI** (`POST /prompt`, `GET /history/{id}`) —
they import nothing from ComfyUI as a library. The structure is already that of an API
client with polling.

Changing backend means **rewriting only the transport layer**: where a JSON node graph is
assembled today, send the `/v1/videos` payload instead. Frame-grid logic, aspect cropping,
reference preparation, waiting and export all carry over unchanged.

`h3_swap.py` is the one that migrates least: its value is the segmentation and
latent-masking node chain, which does not exist on the SGLang route and would have to be
built separately.

## Middle path: leave the interface, keep the engine

ComfyUI **already is an HTTP server**. You can run it without opening the browser and treat
it as a backend — which is exactly what these scripts do. Your own app with its own UI can
talk to it over HTTP and the end user never sees ComfyUI.

That gets you 90% of the goal — not depending on the interface — without losing the dynamic
offload that makes 24 GB work.

## Sources

- Official local deployment: `platform.minimax.io/docs/guides/local-deploy-h3`
- Alternative local service with a web console and REST, tuned for a single GPU:
  `github.com/PullMyBoots/X-MinimaxH3` — supports Ref2VA with images, videos and audios,
  with INT8 and W4A8 quantization. **Validated for SM89 (RTX 4090)**; on Blackwell (sm_120)
  the quantized kernels are not portable as-is. Requires Linux or WSL2.
