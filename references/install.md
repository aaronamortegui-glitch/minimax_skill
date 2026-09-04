# Installing everything the model needs

A checklist to work through in order. Each step ends with a check you can run, so a failure
is caught where it happens rather than three steps later as a confusing error.

The target is a **ComfyUI portable install on Windows**. On Linux the same pieces apply;
only the launcher differs.

## 0. Is the hardware enough?

24 GB of VRAM and 64 GB of system RAM is the comfortable floor for the reference flow.
Less is possible and slower; see [hardware.md](hardware.md) for measured times per tier and
for renting a GPU instead.

The reason a 24 GB card can run a 21 GB model plus a 15.7 GB text encoder at all is
ComfyUI's dynamic weight offload. That is why this route, and not a standalone app.

## 1. ComfyUI portable

Download the Windows portable build and unpack it. Everything below lives under that
folder, referred to here as `<portable>`.

The bundled interpreter is `<portable>\python_embeded\python.exe` — **use it for everything**,
never a system python. The scripts in this skill assume it.

```
<portable>\python_embeded\python.exe -c "import torch,sys;print(torch.__version__, torch.version.cuda, sys.version.split()[0])"
```

Known-good: `2.12.0+cu130`, CUDA `13.0`, Python `3.13`. The CUDA build matters — see the
model table: `int8_convrot` weights are the correct variant for cu130, and the `fp8_scaled`
ones are for cu12x. Mixing them wastes a 21 GB download.

## 2. ComfyUI version

MiniMax H3 and SAM3 ship **inside ComfyUI**, not as custom nodes. Confirmed present in
0.34.0 as `comfy_extras.nodes_minimax_h3` and `comfy_extras.nodes_sam3`. If those nodes are
missing, the fix is updating ComfyUI, not hunting for a node pack.

## 3. The models

All from [huggingface.co/Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3),
whose folder layout mirrors ComfyUI's own: drop each file into `<portable>\ComfyUI\models\<folder>`.
The full table of files and folders is in [comfyui.md](comfyui.md) → *Models*.

Minimum to generate anything through the reference flow:

| | Folder |
|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot` (21 GB) | `diffusion_models` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` (15.7 GB) | `text_encoders` |
| `minimax_h3_video_vae_fp16`, `minimax_h3_audio_vae_fp32` | `vae` |
| `minimax_h3_ref2v_turbo_4step_v0.1` | `loras` |

Add `sam3.1_multiplex_fp16` in `checkpoints` before doing any inpainting, and the pose
ControlNet only if you need pose transfer.

Budget around 60 GB and an hour of downloading.

## 4. Custom nodes — only five, and only two for the basic route

Install into `<portable>\ComfyUI\custom_nodes`, via ComfyUI Manager or `git clone`.

| Pack | Gives | Needed for |
|---|---|---|
| `comfyui-videohelpersuite` | `VHS_LoadVideo` | **any video reference** |
| `ComfyUI-NKD-Basic-Tools` | `NKDMaskOps`, `NKDMaskOpsLean`, `NKDAVLatent` | inpainting |
| `MaskVidExperiments` | `MVEx_SubjectCrop`, `MVEx_SubjectUncrop` | inpainting |
| `ComfyUI-H3-FunControl` | `H3FunControlApply`, `H3FunControlLoader` | pose transfer |
| `comfyui_controlnet_aux` | `DWPreprocessor` | pose transfer |

Everything else the scripts use is core ComfyUI.

**Check what is actually loaded**, rather than trusting that a folder exists — a node pack
that fails to import leaves its directory in place and simply registers nothing:

```
<portable>\python_embeded\python.exe -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8188/object_info')); [print('%-28s %s' % (c, d[c]['python_module'] if c in d else 'MISSING')) for c in ['MiniMaxH3ReferenceToVideo','SAM3_VideoTrack','VHS_LoadVideo','NKDMaskOps','MVEx_SubjectCrop','H3FunControlApply','DWPreprocessor']]"
```

Anything printing `MISSING` is a pack that did not load; its traceback is in the ComfyUI
startup log.

## 5. SageAttention (optional, worth it)

`--use-sage-attention` measured **1.62x faster at identical quality** — the single best
return of any flag here. It needs SageAttention 2.2 installed into the embedded python.
Skip it on the first pass and add it once generation works; if the launcher reports it
missing, ComfyUI still runs without it.

## 6. Launch

Use `scripts/START_COMFY_H3_FAST.bat`, which carries the flag set explained in
[comfyui.md](comfyui.md) → *Launch and optimizations*. Confirm in the log:

- `Using sage attention` (if you installed it)
- `Enabled fp16 accumulation`
- the server listening on `127.0.0.1:8188`

## 7. The scripts

Copy this skill's `scripts/` into `<portable>\h3\`. They talk HTTP to the running ComfyUI,
so it must be up first. Run them with the embedded interpreter:

```
cd <portable>\h3
..\python_embeded\python.exe h3.py -p prompt.txt --seconds 5.17
```

## 8. Smoke test

Generate the shortest thing possible — 5 frames — before spending real time:

```
..\python_embeded\python.exe h3.py -p prompt.txt --seconds 0.21 --steps 4
```

If that produces a file, the install is sound. First run is slow: the 15.7 GB text encoder
has to load. `--cache-ram 40` keeps it resident for subsequent runs, which is most of why
the second generation feels so much faster than the first.
