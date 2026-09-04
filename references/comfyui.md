# Technical reference — the ComfyUI route

Detail for when something breaks or the wiring needs changing.

## Launch and optimizations

```
START_COMFY_H3_FAST.bat
```

| Flag | What it does |
|---|---|
| `--use-sage-attention` | SageAttention 2.2. **Measured: 1.62x faster, identical quality.** |
| `--fast fp16_accumulation cublas_ops` | fp16 accumulation + cuBLAS kernels |
| `--cache-ram 40` | caches nodes in RAM, avoids reloading the 15.7 GB text encoder |
| `--vram-headroom 2` | keeps 2 GB free. Without it VRAM reaches 98% and it sticks |

Check the log for `Using sage attention` and `Enabled fp16 accumulation`.

**Do not use:** `fp8_matrix_mult` (weights are already int8, it degrades) ·
`--high-ram` (incompatible with `--cache-ram`) · a low `--det-thr` on SAM3 (fragments
detection).

**Untested:** `--fast-disk`, which prefers NVMe offload over unpinned RAM. Could help
exactly in the saturation case.

## Models

| | File | Folder |
|---|---|---|
| DiT, references | `minimax_h3_ref2va_pruned_int8_convrot` (21 GB) | `diffusion_models` |
| DiT, text/image | `minimax_h3_fl2va_pruned_int8_convrot` (21 GB) | `diffusion_models` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq` (15.7 GB) | `text_encoders` |
| VAE | `minimax_h3_video_vae_fp16` + `minimax_h3_audio_vae_fp32` | `vae` |
| Turbo LoRA ref2v | `minimax_h3_ref2v_turbo_4step_v0.1` | `loras` |
| Turbo LoRA fl2v | v1.0, **v1.1**, 8step v1.0, turbo v4 | `loras` |
| Pose ControlNet | `minimax_h3_fun_controlnet_union_pruned_bf16` (4.2 GB) | `controlnet` |
| Latent upscaler | `minimax_h3_latent_upscaler_3d_fp16` | `latent_upscale_models` |
| Segmentation | `sam3.1_multiplex_fp16` | `checkpoints` |
| Live preview | `taeh3` | `vae_approx` |

`int8_convrot` is the correct variant with torch cu130. The `fp8_scaled` ones only if you
drop to cu12x.

**For ref2va the only turbo LoRA that exists is v0.1**, early and weak. That is why the
reference flow is pinned at 20 steps for quality, or 10 as a compromise. fl2va does have a
mature v1.1.

## Sampling

Guidance-free: **there is no CFG to tune**. The only strength lever is
`MiniMaxH3SigmaShift` (12.0 video / 3.0 audio by default).

| Config | Steps | LoRA | When |
|---|---|---|---|
| Quality | 20 | none | final delivery |
| **Balanced** | **10** | turbo | **measured: equal to 20 steps, half the time** |
| Draft | 4 | turbo | fl2va only. On ref2va the v0.1 ruins quality |

Sampler `res_multistep`, scheduler `simple`, `BasicGuider`.
For inpainting: `KSampler`, cfg 1.

## Reference labels

Numbered **by wiring order**, not by what you imagine:
images → videos (each video's soundtrack takes its `<Audio j>` **before** its `<Video k>`)
→ standalone audios.

If the number of references changes, **renumber the labels in the prompt**.

In the API the autogrow inputs use nested names, zero-indexed:
`"ref_images.ref_image_0"`, `"ref_audios.ref_audio_0"`, `"ref_videos.ref_video_0"`,
`"ref_video_audios.ref_video_audio_0"`.
Dynamic combos likewise: `"mode": "tracked"` plus `"mode.crop_scale": 1.75`.

## The two methods, side by side

![Segmentation vs recreation](images/method-comparison.png)

Both rows are the same job — put a different man into existing footage — done the two ways.
The third column is the amplified difference against the source plate, and it is the whole
argument: with segmentation the frame is black except one bright patch where the face is;
with recreation it is white everywhere, because every pixel was generated anew.

Read the bottom band as the decision. In short: **segmentation when the plate must survive,
recreation when the mask cannot hold.**

One consequence of recreation that the figure shows and that is easy to miss until you are
in the edit: **the output does not keep the source aspect ratio.** A 2.5:1 plate comes back
at the model's native 1.78:1, with picture invented above and below and the horizontal
composition shifted. Plan on cropping back, and do not expect a frame-for-frame match with
the neighbouring shots.

## Recasting a live-action shot by recreation (`ref2va`)

Feeding a shot as `<Video 1>` and a character as `<Picture 1>` recreates the shot with a new
person in it. It is much cheaper than inpainting and needs no mask, but it comes with one
property you cannot argue away: **every pixel is regenerated**. The shot is a *reference*,
not a locked plate. You get a very close plano — camera move, blocking and action survive —
but the background, the extras and the grain are reinterpreted. If the original must stay
intact pixel for pixel, that is inpainting, not this.

Five things decide whether the recast lands, in order of how much they cost to get wrong.

### 1. The seed is a parameter, not a detail

On the hard beats, **re-roll before you re-engineer**. Measured on one shot where the
character had to appear behind the glass of a phone booth, small and dim: five different
prompt configurations all kept the original face — simple instruction, added negation,
full authority framing, one multi-view sheet, three separate crops. The *same* configuration
at two other seeds put the right man in the booth on the first try.

The corollary is a trap worth naming: three failures in a row look exactly like a broken
method. They are not proof of one. Change the seed twice before you conclude anything about
the approach, and if you do change the approach, change **one** thing.

### 2. A positive-only instruction can be satisfied by changing nothing

"Replace the man in `<Video 1>` with the man in `<Picture 1>`" is an assertion the model can
honour by leaving the frame alone — and on a low-detail face, that is what it does. It needs
the original ruled **out**:

```
<Video 1> defines ONLY the scene: the camera move, the framing, the street, the other
pedestrians, the grade and the action. It does NOT define the face.
The clean-shaven young man in it is REPLACED and must not appear.
...
Never keep the original clean-shaven face.
```

That framing transfers identity reliably. The price is looser shot fidelity — it will
re-stage a beat rather than match it frame for frame. A "locked, frame-for-frame" prompt
inverts the trade: faithful staging, original face with your features painted on top.

### 3. Stills carry identity — a video reference of the character does not

When the scene already arrives as `<Video 1>`, adding a turnaround clip of the character as a
second video reference makes things **worse**, not better: the two video signals compete, the
identity one loses, and shot fidelity degrades as well. Measured: two video references split
the shot in half, at ~4 s in one test and ~5.5 s in another, with the character changing at
the seam.

Reserve the character turnaround for **inpainting**, where the scene is a plate rather than a
reference and nothing competes with it. For recreation, use stills.

### 4. Individual close-up crops beat a sheet, and a sheet can get drawn into the frame

Give the model **separate images**, each a head-and-shoulders crop of one view — front,
profile, three-quarter — as `<Picture 1>`, `<Picture 2>`, `<Picture 3>`. A single multi-view
sheet is one image in which each face occupies maybe 15% of the area, so the face the model
receives is small, and a small reference face cannot replace a small shot face.

A collage assembled from *video frames*, with their varied lighting and backgrounds, is worse
still: in one run the model rendered the six-panel sheet **inside the shot**, as a picture
within the picture. A clean studio sheet does not do this, but cut it into panels anyway.

### 5. Name every feature you need, including the hair

The identity clause has to cover whatever the original has strongly. "Short dark hair" does
not contradict a protagonist whose hair is dark and swept back, so the model keeps his — and
the shot reads as the wrong person. Say the shape, and say what it is not:

```
short black hair cropped close at the sides and receding at the temples
...
This is his face AND his hair in every frame. His hair is short and cropped, never long
and never swept back over the ears.
```

## Where you cut the shot decides what the model can do

Cutting at the point where the character becomes recognisable is the intuitive choice and it
is usually wrong. **Cut on the original shot boundaries.**

A shot that begins mid-move hands the model an entrance it cannot explain. One phone-booth
shot was cut from the moment the face was legible; the real shot starts 2.9 s earlier as an
almost-black macro of the telephone earpiece and pulls back through the hand, the chin and
finally the profile. Cut short, the recreation had no move to inherit and opened on a face.
Cut whole, it reproduced the entire pull-back and the character was right at every stage of
it — including his reflection in the glass.

The same rule at the other end: **cut after the gesture completes**, not on the frame where
it reads. A hang-up, a head lifting, a hand arriving at the face — if the last beat is
clipped, the shot has no ending and no seed will invent one.

Three practical consequences:

- **Find the real boundary.** Sample the source every 0.2 s across the suspected cut and look;
  a scene-change filter misses cuts into darkness, and this kind of shot often opens on black.
- **Choose the length backwards from the out-point.** The out-point is fixed by the next
  shot, and the duration must land on the frame grid, so the in-point is
  `out − (grid frames / 24)`. For a 6.58 s shot (158 frames) ending at 54.00, that is 47.42 —
  which then has to fall *after* the real cut. If it does not, step down a grid rung.
- **Describe the whole move in the prompt**, beat by beat, including the part with no face in
  it. The frames before the subject appears are not filler; they are what the model needs in
  order to arrive at the right framing.

Note that a shot opening on black or on a macro defeats a first-and-last-frame check — the
first frame has no subject to verify. Check inside the shot instead.

## Splicing a swapped piece back into a longer plate

When only a second of a long shot needs replacing, generate that second and splice it. Two
things will bite:

- **Do not assume two cuts of the same moment share an origin.** Sub-cuts made in separate
  sessions start at different frames. Measure it: `psnr` between the candidate offset of the
  parent and the piece gives ~46 dB when aligned and ~26 dB one frame off. Scan a few
  offsets and take the peak.
- **Better still, cut the piece from the parent you are splicing back into**, so alignment is
  guaranteed by construction rather than by measurement. Re-derive any mask coordinates on
  that clip — the framing may match while the starting frame does not.

Verify the finished splice against the plate the same way: a correct assembly of a shot with
two small replaced regions reads around 39 dB over the whole shot. A low number means a
misalignment, not a strong swap.

## Check the original before you call it a defect

Two mistakes that cost real GPU hours, both from reviewing the output without the plate next
to it:

- **An extra read as a failed replacement.** A man in sunglasses standing by the phone booth
  looked like the swap had missed. He is in the original footage. Nothing had failed.
- **A protagonist who is not in frame yet.** The first 4.5 s of one 8 s shot contain no
  protagonist at all in the original; he walks in afterwards. There was nothing there to
  replace.

Also check what resolution the plate actually is. All the cuts in one project turned out to
be **640×266** — a face behind glass at that size is a smudge, and the model will invent
whoever is in it. And when a source file looks promising because it is 1080p, confirm it is
the original and not an earlier render of your own work.

## Inpainting — the bug that cost two days

`NKDAVLatent.latent_mask` must receive the **`latent_mask` output of `NKDMaskOps`, which is
SLOT 2**, not slot 0.

Slot 0 is the pixel mask. ComfyUI resizes it trilinearly, it blurs across frames and **the
original bleeds through**: the model reconstructs what was already there (hair does not go
away, held objects are lost). With slot 2 the behaviour changes completely.

`NKDMaskOps` → 3 outputs: `mask` (0), `mask_inverted` (1), `latent_mask` (2).

The `MaskedContext` nodes are **not** spatial inpainting: they are temporal continuation
(video extend).

**Full chain:**
```
VHS_LoadVideo -> SAM3_VideoTrack -> SAM3_TrackToMask -> MVEx_SubjectCrop
  -> NKDMaskOps -> [slot 2] -> NKDAVLatent -> KSampler -> VAEDecode
  -> MVEx_SubjectUncrop
```

## Tracking with SAM3

**Always** run `probe_tracks.py` first. Read the **coverage percentages**, not just the
pictures: an object at 0% on several frames **is unusable**, because with a broken mask the
MVEx crop wanders and the repaint lands in the wrong place.

- **The detection text matters more than the threshold.** `"the whole head of a person
  including all of the hair"` gave two stable tracks where `"the face"` gave one. Dropping
  `--det-thr` to 0.25 made it **worse**: it fragments into more fragile objects.
- **On long clips SAM3 loses the subject and re-registers it under another index.** If the
  track splits, pass the list: `--object-indices "3,4"`. They are the same person.
- **Indices change between videos.** Re-probe every time. Better: do not chain passes.
- **Text does not always move the mask.** Mentioning knives gave 32.9% vs 32.8% without.
- **"Intermittent" is not always bad.** The rule is: a mask that *flickers* on and off ruins
  the crop geometry. One that **grows or shrinks monotonically** — the subject entering or
  leaving frame — works fine. Measured: 0% → 3% → 14% → 55% produced clean output.
- **Asking for the face can return the whole body, and sometimes that is better.** In a
  distant, motion-blurred shot the face is not a detectable object and SAM3 returns the
  full silhouette. Repainting the whole figure lets the model supply the wardrobe too,
  which beats pasting a face onto the original body.
- **Overhead over a crowd, text segmentation does NOT work.** Three attempts with different
  texts and threshold 0.25: nothing, or 0.2% blobs on random pedestrians. From directly
  above the subject is one dark coat among dozens and SAM3 cannot reason about who is
  standing still. Use `--mask-box` for that.

## Manual mask: `--mask-box "x,y,w,h"`

When SAM3 cannot isolate the subject, `h3_swap.py` accepts a **fixed rectangle** in source
pixels and skips segmentation entirely:

```
h3_swap.py --video seg.mp4 --mask-box "308,128,60,74" --img face.png -p p.txt --seconds 0.92
```

It renders a white-on-black video of the same size and frame count
(`VHS_LoadVideo` → `ImageToMask`) and feeds it where `SAM3_TrackToMask` would go. `--detect`
becomes optional: you need one of the two.

Only for **near-static** subjects; the box follows nothing. Use small `--expand` and
`--feather` (6) if the box is a few tens of pixels — the defaults (30/20) are enormous at
that scale.

## Getting onto the frame grid without stretching

If a shot has fewer frames than `17k+5` asks for, do **not** stretch it and do **not** pull
frames from the neighbouring shot (that one has a different face and it wrecks the mask).

**Pad the tail by cloning the last frame** up to the grid duration, generate, then **trim
the output** back to the shot's real length:

```
ffmpeg -i shot.mp4 -vf "tpad=stop_mode=clone:stop_duration=0.175,fps=24" \
       -af "apad=pad_dur=0.175" out_padded.mp4     # 36 frames -> 39
# ... generate ...
ffmpeg -i generated.mp4 -frames:v 36 final.mp4     # back to 36
```

Pad the **audio too** (`apad`): `VHS_LoadVideo` fails if the file has no audio track, and
the graph needs `["10", 2]` for `NKDAVLatent`.

The cloned frames never reach the final file. Same idea works with frames borrowed from the
next shot when padding is not an option — generate long, trim back.

## The 124-frame ceiling — the most expensive failure of all

**Above roughly 124 frames the replacement silently does not apply.** The shot comes out
byte-for-byte looking like the original: no error, no warning, no clue in the log, and the
full generation time spent. Measured on one 5.2 s shot — at 141 frames nothing changed, at
124 frames the same shot came out perfect with every other parameter identical.

Cap it. `batch_shots.py` refuses to go past 124 and trims the shot instead, printing a
warning. If a shot genuinely needs to be longer, cut it in two.

This one cost twice, because the first time it was misdiagnosed: a manual retry that
changed `--upscale-mp` *and* the frame count fixed it, and the fix was credited to
`--upscale-mp`. It was the frame count. **When a retry changes two things, it has proved
nothing.**

Beware of trusting a difference metric here. A mean-absolute-difference against the source
read *higher* on the failed shot (20.0) than on shots where the swap worked (8.9, 10.6) —
global re-encode shift, not a changed face. The metric detects change, not the right
change. Only look at the frames.

## Inpainting trade-offs you cannot dodge

- **`--denoise` is the identity/performance dial, and it belongs per character.**
  - `1.0` destroys lip sync: the masked region is repainted from scratch every frame, so
    the mouth is invented rather than inherited.
  - `0.85` keeps per-frame structure — mouth, eyes, blinks — while identity still changes.
    The default for anyone who talks on screen.
  - `0.95` is needed when the original has a **strong feature competing with the
    reference**. A character with a big grey beard came out grey at 0.85 (only the
    moustache went dark) and correctly black at 0.95. Neither a bigger mask (`expand` 25 →
    50) nor three different detection texts moved it; only denoise did.
- **Beard density comes from the mask, not from `--upscale-mp`.** *(This corrects an
  earlier claim in these notes.)* The two were changed together in one test and the credit
  went to the wrong one. Re-measured separately: at `--upscale-mp` 0.8 and 1.4 the beard is
  identical. What produces a full beard is a detection text whose mask actually covers the
  jaw and chin. Growing `--expand` alone does nothing (18 vs 45 identical, 80 vs 140
  identical).
- **Include the neck in the detection text, not just the face.** A face-only mask stops at
  the jaw: the model paints a beard down to the mask edge and the original pale neck shows
  below it, so the head reads as pasted on. Adding the neck moves the seam somewhere nobody
  looks. Measured: `"the face, the jaw and the beard of a person"` gave 4.0% coverage,
  `"the head and the neck of a person, including the chin, the jaw and the throat"` gave
  11.1% on the same shot.
- **A head mask replaces the subject's hair**, which is sometimes the point and sometimes
  a loss — it costs the original silhouette. At large `--expand` it also reaches the collar
  and **imports the reference image's wardrobe**. At `denoise 0.85` the hair's shape and
  length survive even inside the mask; only its colour shifts toward the reference.
- **A head mask swallows a hat.** For a character in a hat, keep the detection tight
  (`"the face and the beard of a person"`) or the model repaints the hat too.
- **Keep every parameter identical across the shots of one scene.** A value tuned per shot
  gives each shot a different finish, and mismatched finish is more visible in an edit than
  any single shot's imperfection.
- **Two people: one pass each.** With one mask over two faces the model blends features.
  Spatial anchoring by prompt does not fix it.

## Replacing someone across a whole dialogue scene

Two problems appear that never show up in a single shot.

### 1. The model lip-syncs whoever is on screen, even the listener

H3's lip sync is native and it is driven by **the audio baked into the latent**, not by who
is speaking. Point it at a reaction shot and it will put the other character's dialogue in
your character's mouth.

`audio_mode` does not save you: `keep`, `regenerate` and `follow mask` all leave the audio
conditioning the picture. There is no "ignore audio" mode.

**Fix it at the source: feed the swap a silent audio track, and re-attach the real audio
afterwards.** `VHS_LoadVideo` requires an audio stream, so it must be silence rather than
absence:

```
ffmpeg -i shot.mp4 -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
       -map 0:v -map 1:a -c:v copy -c:a aac -shortest shot_silent.mp4
# ... swap on shot_silent.mp4 ...
ffmpeg -i generated.mp4 -i shot.mp4 -map 0:v -map 1:a -c:v copy -c:a aac final.mp4
```

Fighting this with prompt wording is a trap. A strong instruction ("his mouth stays CLOSED
and still") does work — but it also closes a mouth that should hang open in surprise, and
flattens the expression. Softening it to preserve the performance then loses the lip-sync
fight again. Silent audio removes the conflict instead of balancing it.

Classify each shot from the **original footage**, not from the script: does this character's
mouth articulate, or not? Three modes are enough — *speaking*, *silent and listening*,
*reacting without speech* — and only the first keeps its real audio going in.

### 2. Identity drifts from shot to shot

Segmentation decides **where** the model paints, not **who** it paints. With frontal stills
as the only reference, every non-frontal angle gets invented, differently each time.

A **video reference** fixes it: generate one turnaround of your character once — a slow
360° of the head against a plain background, even lighting, ~5 s — and pass that same clip
as `--ref-video` on every shot. It carries identity through angles the way stills cannot.

The catch, measured: a turnaround shows the face under changing light, so as a **colour**
signal it is weaker than one well-lit still. On a character whose original had a strong
competing feature (that grey beard again), the video reference brought the grey back while
stills at `denoise 0.95` did not. **Video reference for consistency where the original does
not compete; stills where it does.** It is a per-character choice, and it costs about 60%
more time per shot.

## Pose transfer (ControlNet)

`h3_pose.py`. DWPose extracts the skeleton → `H3FunControlApply` drives the motion →
`MiniMaxH3ReferenceToVideo` supplies identity.

- **The skeleton MUST be at the exact generation size**: `DWPreprocessor` →
  `ImageScale(w,h)` → `H3FunControlApply`. Without it, a shape error.
- `--strength` goes **up to 2.0**. At 1.0 the pose is followed loosely; **1.8 does follow
  it**, at the cost of detail and identity.
- **The background is reinvented**: it generates from scratch, it does not preserve the scene.
- DWPose extracts body, hands and face. **Not objects** — props do not survive.

## Scaling and memory

| References | s/step | 8 s of output |
|---|---|---|
| 2 images | 80 | ~15 min |
| 5 images | 98 | ~17 min |
| 1 video + its track | 198 | ~35 min |

Images only: duration scales with exponent **~1.5**, resolution with **~1.3**.
With a video reference **fixed at ~5 s**, output scales **almost linearly**.

**362 frames in a single pass fit** in 24 GB with image references (measured, 16.4 min).
Chunking introduces seams; only do it if it genuinely will not fit.

**With a video reference the ceiling is between 10.1 s (works) and 16.5 s (hangs).**

## Node traps already paid for

- **`tile=` silently drops rows when the inputs differ in size.** Every comparison sheet in
  this workflow needs `scale=W:H:force_original_aspect_ratio=decrease,pad=W:H:...` on each
  frame first, or half the grid comes back black and you review the wrong thing.
- **A difference metric detects change, not the *right* change.** A mean-absolute-difference
  against the source read *higher* on a shot where the swap silently did nothing (20.0) than
  on shots where it worked (8.9, 10.6) — global re-encode shift. Look at the frames.

- `CreateVideo` without audio + `SaveVideo` **blows up** with libx264.
- The embedded python's `hf` CLI **does not work** (missing `venv`). Use
  `huggingface_hub.snapshot_download` from a script.
- Always validate with `--dry-run` before spending GPU: `/prompt` validates the whole graph
  and returns the exact error without sampling.
- Third-party workflows may hardcode the int8 text encoder. Patch the JSON to `nvfp4_awq`
  instead of downloading 27 redundant GB.

## Pending / unverified

- **Latent upscale**: `MinimaxH3LatentUpscaler3D` + `LTXVSeparateAVLatent` /
  `LTXVConcatAVLatent` to upscale only the video and rejoin. This is **the route to real
  1080p** and the highest-value open item.
- **Video extend**: `MiniMaxH3StreamLiveExtensionAVToVHS` and the `MaskedContext` nodes.
- **Audio inpainting**: `MVEx_AudioMaskToLatent` and `NKDAVLatent`'s `follow mask` mode, to
  regenerate only a stretch of sound.
- **Wan 2.2 Animate** (`WanAnimate2ToVideo`) would be the right stack for full-person
  replacement **with wardrobe**, because it separates `pose_strength` from
  `reference_image_strength`. One attempt came out black. Suspects: `cfg 1.0` with `uni_pc`
  (Wan is **not** guidance-free, it needs 3-6), `pose_video` probably expects a
  pre-extracted skeleton, and the `trim_latent`/`trim_image` outputs were ignored.
  **Find the official workflow before guessing further.**
