# minimax-h3 skill

A [Claude skill](https://docs.claude.com/en/docs/claude-code/skills) plus the scripts it
drives, for generating video with synchronized audio locally using **MiniMax H3** — an
open-weights 33B video model that produces picture and sound together, including spoken
dialogue with lip sync.

Everything here was measured on **one RTX 5090 Laptop (24 GB) with 64 GB RAM**, on Windows,
against ComfyUI. Timings, limits and failure modes are what actually happened, not what the
docs promise.

## What it covers

- text-to-video and image-to-video with generated audio
- a consistent character across shots, from reference images
- speech in a specific voice, from a reference audio
- **replacing a person's face inside existing footage**, preserving the original frames
- putting your character through another video's motion
- pose transfer with ControlNet
- the real limits of each, and what it costs in wall-clock time
- what hardware you need, from 8 GB cards up, and what renting one costs

## Install

Drop the whole folder into your skills directory:

```
~/.claude/skills/minimax-h3/
```

Claude reads `SKILL.md` and pulls in `references/` as needed.

The scripts assume a **ComfyUI portable install** and resolve paths relative to their own
location, so put them at:

```
<ComfyUI_portable>/h3/
```

and launch ComfyUI with `scripts/START_COMFY_H3_FAST.bat`, which carries the measured
optimizations (SageAttention gives **1.62x** for free, and `--vram-headroom 2` prevents the
98%-VRAM hang).

They talk plain HTTP to ComfyUI on `127.0.0.1:8188` and import nothing from it, so they are
straightforward to point at a different backend later — see
[references/standalone-app.md](references/standalone-app.md).

## Scripts

| Script | What it does |
|---|---|
| `h3.py` | main CLI. text/image/audio/video references → video. Picks fl2va or ref2va automatically |
| `h3_swap.py` | face and person replacement inside existing footage, via segmentation + latent masking. Also takes a fixed `--mask-box` when text segmentation cannot isolate the subject |
| `probe_tracks.py` | **run this before any swap.** Tracking only, ~1 min, prints per-frame coverage per object and writes a visual overlay |
| `sheet.py` | contact strip of a video, for reviewing without playing it |
| `h3_pose.py` | pose transfer through the Fun ControlNet |
| `h3_swap_long.py` | chunked swap runner with a JSON config and resume |
| `batch_shots.py` | example: run a whole scene shot by shot, handling the frame grid by pad-and-trim. Edit the `SHOTS` table and paths |

## The three things worth knowing up front

**Duration is quantized.** Frame counts must land on `17k+5` at 24 fps — 5, 22, 39, 56, 73,
90, 107, 124... The scripts round for you, but when you are cutting a scene to length this
is the constraint that shapes everything. `batch_shots.py` shows the pad-and-trim technique
for shots that fall between grid values.

**1080p is not available.** The open model runs at 768 px short side, native 1344x768 —
which is 1.75, not 16:9. Generate native and crop. The 2K upscale is a paid-platform
component not in the open release.

**System RAM matters as much as VRAM.** The weights total ~42 GB and never fit on a
consumer card; ComfyUI streams them between VRAM and RAM. A 24 GB card with 16 GB of RAM is
slower than a 12 GB card with 48 GB. Rule of thumb: 2-3x the VRAM in RAM. See
[references/hardware.md](references/hardware.md) for the tiers, measured times, and GPU
rental costs (~$0.07-0.16 per 8 s clip on a rented 4090).

**It hangs instead of erroring.** Past roughly 10 s of output with a video reference, VRAM
saturates and ComfyUI swaps weights forever without failing. Watch for `0/8` after five
minutes with >23,500 MiB used, and kill it.

## License

The scripts are MIT. Model weights and ComfyUI node packs carry their own licenses.
