# Hardware: what card you need, how long it takes, and renting one

## The one thing that explains everything else

The weights do not fit on any consumer card. In the smallest variants they total about
**42 GB** — 21 GB DiT + 15.7 GB text encoder + VAEs. (Full precision is 123.6 GB.)

What makes the model run on a 24 GB card anyway is ComfyUI's **dynamic offload**: it streams
weights between VRAM and system RAM as it samples. In the log it looks like
`Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged`.

So:

> **VRAM decides how fast. System RAM decides whether it runs at all.**

A 12 GB card with 32 GB of RAM works. A 24 GB card with 16 GB of RAM will thrash. Whatever
does not fit in VRAM has to live in RAM, and whatever does not fit in RAM goes to disk,
which is where it stops being usable.

## Tiers

Rows marked **measured** are from this repo's own runs. The rest are community reports and
should be treated as ballpark — quality descriptions especially vary by who is judging.

| VRAM | RAM | What you get | 5 s clip |
|---|---|---|---|
| 6-8 GB | 16 GB+ | 4-bit quantized variants only. Mushy detail, rough audio. Bottom of the range | 10-15 min |
| 12 GB | **32 GB+** | Works with heavy offload. 480p territory. Needs fast NVMe | ~6 min at 480p |
| 16 GB | 32 GB+ | Noticeably cleaner and faster, still short of full quality | ~6 min at 480p |
| **24 GB** | **64 GB** | **The practical target for the pruned int8 workflow. Full 768p** | **10 min at 768p, video ref — measured** |
| 32 GB+ | 64 GB+ | Headroom for longer clips and LoRA training alongside | a few min at 720p |

Reference points reported by others on 24 GB cards: an RTX 4090 with 128 GB RAM doing 12 s
HD in 10-13 min; the same card doing 20.75 s at 608x352 in ~5.9 min. A 16 GB AMD card took
22 min for a 5 s square clip — AMD is a rougher path.

## Measured times — 1x RTX 5090 Laptop 24 GB, 64 GB RAM

All at `--steps 8-10`, 1344x768 native, with the optimizations in
`scripts/START_COMFY_H3_FAST.bat`.

**Generation with a video reference** (reference fixed at ~5 s):

| Output | Time |
|---|---|
| 5.2 s | 10 min |
| 8.0 s | 14.5 min |
| 10.1 s | 17.8 min |
| 16.5 s | **hangs** — VRAM saturates, it never errors |

**Generation with image references only:**

| Output | Refs | Time |
|---|---|---|
| 8.0 s | 2 images | ~15 min |
| 8.0 s | 5 images | ~17 min |
| 15.1 s (362 frames) | images | 16.4 min |

**Inpainting** (face replacement in existing footage):

| Frames | Duration | Time |
|---|---|---|
| 22 | 0.92 s | ~2 min |
| 39 | 1.63 s | ~3 min |
| 107 | 4.46 s | ~7.7 min |
| 124 | 5.17 s | ~6 min |

**Tracking probe** (`probe_tracks.py`, no sampling): 16-24 s for 22 frames, 80 s for 124.
Always cheap, always run it first.

**Cost per step by reference type** — this is the number that predicts everything:

| References | s/step |
|---|---|
| 2 images | 80 |
| 5 images | 98 |
| 1 video + its soundtrack | 198 |

A video reference costs roughly **2.5x** what image references cost. Keep it short and
fixed at ~5 s; then output length scales close to linear. If the reference grows along with
the output, cost explodes.

Images only: duration scales with exponent **~1.5**, resolution with **~1.3**.

## Getting more out of a smaller card

In rough order of payoff:

1. **`--use-sage-attention`.** Measured **1.62x faster with no quality loss**, on hardware
   you already have. If you change one thing, change this.
2. **Drop the short side to 480.** Resolution scales with exponent ~1.3, so going 768 → 480
   is worth roughly 1.8x. This is the main lever on 12-16 GB cards.
3. **`--steps 10` with the turbo LoRA** instead of 20 clean. Measured as equivalent quality
   at half the time. (Only for fl2va; on ref2va the v0.1 LoRA is weak — see
   [comfyui.md](comfyui.md).)
4. **Shorter clips, then join.** Duration scales at ~1.5, so two 5 s clips cost less than
   one 10 s clip and give you cut points for free.
5. **Tune `--cache-ram` to what you actually have.** The value in the launcher (40) assumes
   64 GB. On 32 GB use 16-20. Set it too high and you swap to disk, which is worse than
   not caching.
6. **Raise `--vram-headroom` on small cards.** It exists to stop the allocator running out
   of slack and hanging. 2 GB suits 24 GB; try 1-1.5 on 12 GB.

**Do not** add `fp8_matrix_mult` (the weights are already int8 and it degrades them), and
`--high-ram` is incompatible with `--cache-ram`.

## Renting a GPU instead

Worth it when you need a card you do not own, want to run several clips in parallel, or are
doing a one-off batch. Prices from early-to-mid 2026 and they move.

| Provider | RTX 4090 24 GB | A100 80 GB | H100 |
|---|---|---|---|
| RunPod Community | $0.34/hr | $1.19/hr | $1.99/hr |
| RunPod Secure | $0.74/hr | $1.39/hr | $2.89/hr |
| Vast.ai (unverified hosts) | from $0.34/hr | from $0.50/hr | from $0.90/hr |
| Vast.ai (verified datacenter) | higher | — | $1.50-1.87/hr |

**What that means per clip.** A 4090 at ~13 min for an 8 s clip:

- RunPod Community at $0.34/hr → **about $0.07 per clip**
- RunPod Secure at $0.74/hr → **about $0.16 per clip**

Even at the higher rate this is cheap. What actually costs money is idle time: the model
weights are ~42 GB to pull down, so **the download and setup can outlast the generation**.
Budget 15-30 minutes of paid time before your first frame, and use a persistent volume so
you pay that once rather than per session.

**Choosing between them.** Vast.ai is a marketplace and reaches the lowest headline prices,
but unverified hosts run an effective 20-40% above list once restarts and downtime are
counted. For a batch that has to finish, cost-per-finished-run beats cost-per-hour: fewer
restarts and no storage-while-paused fees can win at a higher sticker rate. RunPod Secure
is the conservative pick; RunPod Community sits in between.

**What to rent.** A single 24 GB card (4090 or better) is the sweet spot — it is exactly the
configuration everything in this repo was measured against, so the timings transfer. Going
bigger buys headroom rather than proportional speed, because past 24 GB the offload
pressure is already gone. Two 5090s plus 200 GB of RAM is what the official recipe wants,
and that only matters if you are leaving ComfyUI for a standalone SGLang deployment — see
[standalone-app.md](standalone-app.md).

**Check before you commit:** that the instance gives you enough **system RAM**, not just
VRAM. A 24 GB card with 16 GB of RAM will be slower than a 12 GB card with 48 GB, because
the offload has nowhere to go. Ask for 2-3x the VRAM in RAM.

## Sources

- [MiniMax H3 VRAM guide, 6 GB cards to the 5090](https://www.mindstudio.ai/blog/minimax-h3-run-locally-guide)
- [ComfyUI VRAM offloading guide, 8-16 GB cards](https://www.instasd.com/post/comfyui-vram-offloading-guide)
- [MiniMax H3 day-0 support in ComfyUI](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui)
- [How much GPU memory is required for BF16 (Comfy-Org/MiniMax-H3 discussion)](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/6)
- [RunPod pricing](https://www.runpod.io/pricing)
- [RunPod vs Vast.ai pricing comparison 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-runpod-vs-vastai-2026/)
- [RTX 4090 cloud pricing across providers](https://getdeploying.com/gpus/nvidia-rtx-4090)
- [Deploying MiniMax H3 on GPU cloud](https://www.spheron.network/blog/deploy-minimax-h3-gpu-cloud/)
