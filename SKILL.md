---
name: minimax-h3
description: Generate video with synchronized audio locally using MiniMax H3. Covers both routes - through ComfyUI (works today on a single 24 GB GPU) and as a standalone app with SGLang or vLLM-Omni. Includes text-to-video, image-to-video, consistent characters, voice cloning, replacing a person inside existing footage, pose transfer, and the real limits of each. Use when asked to generate, extend, replace people, clone voices or transfer motion in video without paid services.
---

# MiniMax H3

Open-weights video model with **native audio**: it generates picture and sound together,
including spoken dialogue with lip sync. 33B parameters.

**If you are new to this, read sections 1, 2 and 3 and you can start asking for things.**

---

## 1. What it can do

| I want | Possible | Called |
|---|---|---|
| Write text, get video with sound | Yes | T2V (fl2va) |
| Animate a still image | Yes | I2V (fl2va) |
| My own character, consistent across shots | Yes | ref2va with images |
| Speech in *a specific voice* | Yes | ref2va with a reference audio |
| Two characters with distinct voices | Yes | ref2va, describe each voice |
| Replace someone's face in **my** footage | Yes | inpainting with a mask |
| My character performing another video's motion | Yes | video reference, or pose ControlNet |
| Music and sound effects | Yes, generated | in the prompt, `Audio:` section |
| Video longer than 15 seconds | Not in one pass | cut into pieces |
| True 1080p | **No** | the open model is 768p |
| Keep objects held in hands while replacing | Only one route | see recipe 4 |

---

## 2. The hard limits

Not negotiable, they come from the model.

**Duration: the frame count must land on the `17k+5` grid at 24 fps.** Valid values: 5, 22,
39, 56, 73, 90, 107, 124, 141, 158, 175, 192... MiniMax documents 4-15 s as the supported
range, but **the short end of the grid does work** — 22 frames (0.92 s) produces clean
output for inpainting, where the model only repaints a region of real footage rather than
inventing motion. The scripts round for you.

**Resolution: 768 px short side.** Native 1344x768. That is **not 16:9** (it is 1.75), and
1080 is not divisible by 16, so **1080p cannot be generated**. Generate native and **crop**
to exact 16:9 or 9:16. Upscaling afterwards adds size, not detail.

**References: 9 images, 3 videos, 3 video soundtracks, 3 standalone audios.**

**Not in the open release:** `H3-Context-IR` and `H3-Regenerate-2K`, the paid platform's 2K
upscale. That is why the ceiling is 768p.

**Voice cloning is undocumented** for open H3-Base, but works through the audio reference
route in ComfyUI. Verified.

**Hardware: 24 GB VRAM + 64 GB RAM is the practical target.** Smaller cards work with
heavier offload and lower resolution; see section 5. Renting a 4090 costs roughly
$0.07-0.16 per 8 s clip.

---

## 3. The two routes

### Route A — ComfyUI (the one that works on consumer hardware)

Start here, always. Launch with `scripts/START_COMFY_H3_FAST.bat`; see
[references/comfyui.md](references/comfyui.md) for detail.

### Route B — standalone app, no ComfyUI

**Possible and officially documented, but it needs real hardware:**

| | Requires |
|---|---|
| Official "consumer" recipe | **2x RTX 5090 + 200 GiB RAM** |
| Official baseline | 8x B200 |

What makes a single 24 GB GPU viable is ComfyUI's **dynamic weight offload**, shuttling
weights between VRAM and RAM as it samples. A standalone app has to reimplement that or
demand the hardware above. Here ComfyUI is not convenience, it is what makes the model run.

See [references/standalone-app.md](references/standalone-app.md) for framework, API and
what such an app has to solve.

---

## 4. Recipes

The scripts talk HTTP to a running ComfyUI. Put them in `<portable>/h3/` and run them with
the portable install's python: `cd <portable>\h3` then `..\python_embeded\python.exe <script> ...`

### 1. Text to video with sound

```
h3.py -p prompt.txt --seconds 8
```

### 2. My character, speaking in a given voice

```
h3.py -p prompt.txt --img face.png body.png --audio voice.mp3 --seconds 8 --vertical
```

The reference audio is trimmed to 15 s and normalized. **It does not become the
soundtrack** — it tells the model what that voice sounds like. Put the dialogue in **double
quotes** inside the prompt.

### 3. Replace a face in my own footage

```
probe_tracks.py --video source.mp4              # see which index is which person
h3_swap.py --video source.mp4 --object-indices "1" --img face.png -p prompt.txt --seconds 5
```

Real footage is preserved; only the masked region is repainted. **One person per pass** —
with two faces under one mask the model blends them.

This is the route that keeps an existing look intact: costume, hair, lighting, grade and
depth of field all survive because they are never regenerated.

Three things decide whether it works, all covered in
[references/comfyui.md](references/comfyui.md):

- **Stay under 124 frames.** Past that the swap silently does not apply — the shot comes
  out identical to the original, with no error and the full generation time spent.
- **`--denoise 0.85`, not 1.0**, or lip sync is destroyed. Raise to 0.95 only when the
  original has a strong feature fighting your reference.
- **Feed silent audio for anyone who is not the one speaking.** The model lip-syncs whoever
  is on screen to whatever is on the soundtrack, so a listener gets the other character's
  dialogue in their mouth. Re-attach the real audio after generating.

### 4. My character performing another video's motion

```
h3.py -p simple.txt --video motion.mp4 --video-seconds 5.17 --img character.png \
      --seconds 10 --vertical --ref-size match
```

with a **three-sentence** prompt:

> Recreate the video in `<Video 1>` exactly as it is, with the same movement, the same
> room, the same camera and the same framing, but change the person for the man in
> `<Picture 1>`. Everything else stays the same.

**Counterintuitive: over-specifying hurts.** Twenty-line prompts with authority sections,
timelines and locks lost to three sentences. With a video reference the model already has
the information; excess competes with it.

It is also the only route that preserved **objects held in the hands** — inpainting lost
them and pose ControlNet does not extract objects.

Note it **recreates, it does not preserve pixels.** If the background must be literally the
original, use recipe 3.

**The exception to "keep it short":** when the face is small in frame, three sentences are
not enough and the model keeps the original face. Then you need a reference-authority
block — see section 7.

**Before re-engineering the prompt, re-roll the seed.** On a hard beat — a face that is
small, dim or behind glass — five different prompt configurations kept the original face
and the *same* configuration at another seed fixed it on the first try. Three failures in a
row look like a broken method and are not proof of one.

**Pass separate close-up crops, not one multi-view sheet**, and do not add a video reference
of the character on top of the scene video: the two video signals compete and identity
loses. Full write-up in
[references/comfyui.md](references/comfyui.md) → *Recasting a live-action shot by
recreation*.

**Cut on the original shot boundaries, not where the face becomes legible.** A shot that
begins mid-move gives the model an entrance it cannot explain, and the move is lost; cut
whole, it is inherited. Cut after the closing gesture completes, too. See
[references/comfyui.md](references/comfyui.md) → *Where you cut the shot decides what the
model can do*.

---

## 5. Cost and hardware

**What card you need.** The weights total ~42 GB, so they never fit. ComfyUI streams them
between VRAM and system RAM as it samples, which means **VRAM decides how fast and system
RAM decides whether it runs at all**. Ask for 2-3x the VRAM in RAM.

| VRAM | RAM | What you get | 5 s clip |
|---|---|---|---|
| 6-8 GB | 16 GB+ | 4-bit variants only, mushy detail | 10-15 min |
| 12-16 GB | **32 GB+** | Works with heavy offload, 480p territory | ~6 min at 480p |
| **24 GB** | **64 GB** | **The practical target. Full 768p** | **10 min at 768p** |
| 32 GB+ | 64 GB+ | Headroom for longer clips and LoRA training | a few min |

**Renting is cheap.** An RTX 4090 runs $0.34-0.74/hr, which is **about $0.07-0.16 per 8 s
clip**. What costs money is the ~42 GB of weights you have to pull down first — use a
persistent volume so you pay that once.

Full tiers, per-step costs, tuning for smaller cards and provider comparison:
[references/hardware.md](references/hardware.md).

**Times measured on 1x RTX 5090 Laptop 24 GB, 64 GB RAM**, with `--steps 8-10`:

| Output | Route | Time |
|---|---|---|
| 5.2 s | video reference | 10 min |
| 8.0 s | video reference | 14.5 min |
| 10.1 s | video reference | 17.8 min |
| **16.5 s** | video reference | **hangs** |
| 8.0 s | 2 image references | ~15 min |
| 5.2 s | inpainting | ~6 min |
| 0.9 s | inpainting | ~2 min |

**Rule: keep the video reference short and fixed (~5 s).** Then output scales close to
linear. If the reference grows with the output, cost explodes.

---

## 6. The failures that cost hours

**It hangs without erroring.** Above ~10 s of output with a video reference, VRAM hits 98%
and ComfyUI swaps weights forever. **It does not fail, it sticks.** If after 5 minutes it
still reads `0/8` and `nvidia-smi` shows >23,500 MiB, kill it and shorten. This cost 46
minutes once.

**References beat the prompt on palette.** If the source footage is black and white and the
character reference is in colour, you get a colour patch. **Convert the reference to
greyscale first.** The prompt alone does not win.

**Character sheets with several views.** Feed a 4-view sheet and you get one person per
panel. **Crop individual panels** and say to generate exactly ONE. Crop the captions too
(`FRONT VIEW`, titles): the model reads text and reproduces it.

**Never tell the model what expression to wear when replacing a face.** It obeys, and
overrides the performance in the footage. A prompt saying "a soft, slightly amused
expression" turned a startled open-mouthed reaction into a calm smile, and broke lip sync
along with it. Describe identity only, then explicitly instruct it to keep the original
expression, mouth movement, gaze and blink timing.

---

## 7. Writing the prompt

**To recreate something that already exists** (recipe 4): three sentences. Less is more.

**To replace a face in existing footage** (recipe 3): identity only, plus an explicit
preservation clause:

> Change ONLY his identity. His face is the face of the man in `<Picture 1>`: [description].
>
> Keep EVERYTHING else exactly as it is in the video: his hair, his expression, the movement
> of his mouth as he speaks, where his eyes are looking, the timing of his blinks, his
> clothes, the background, the light and the depth of field.
>
> His lips must move exactly as they move in the video, frame by frame, in sync with the
> dialogue. Do not give him a different expression.

**When the face is small in frame** and the model keeps the original, use a reference
authority block naming what each reference governs and what it must not:

> REFERENCE AUTHORITY
> `<Picture 1>` and `<Picture 2>` define THE MAN, and they override the face of the person in
> the video completely: [description]. His beard is his defining feature and must be
> unmistakable in every single frame.
> `<Video 1>` defines ONLY the scene: the camera move, the framing, the location, the other
> people, the colour grading and the action. It does NOT define the face. The clean-shaven
> man in it is REPLACED and must not appear.
>
> What happens, in order:
> 1. [beat] 2. [beat] 3. [beat]
>
> Never keep the original face. The man from `<Picture 1>` is the only person in [wardrobe],
> in every frame, near or far.

Numbering the beats matters. A shot where the subject starts behind glass, in shadow and
half hidden, was treated as background until the prompt made "he is already the bearded man
from the first frame, even behind the glass" its own numbered beat.

**To build a new scene**, in this order — the model weighs what comes first:

1. **Reference authority** — as above. The most valuable section. Without it, studio
   backdrops, burnt-in subtitles and the reference's clothing leak in.
2. **Characters**, with `Generate exactly ONE` each, and scale if it matters.
3. **Place**, plus an anchor ("the amps stay behind him").
4. **Timeline in seconds**, a beat every 1.5-2.5 s. Do not pack in more than fits.
5. **Camera with a speed curve**, not just a direction. With a frontal reference **never
   exceed a quarter orbit** — the model invents the back.
6. **Audio**: exact dialogue in double quotes. Several voices, describe **each separately**
   and insist they stay distinct.
7. **Locks** listing what must not change.

**To change a voice's character you need force and negatives.** "low, dry, film-noir voice"
moved nothing. "Extremely deep, hoarse, sandpaper rasp on every word, destroyed smoker
voice, **absolutely not clean or youthful**" did.

---

## 8. Reviewing without playing the video

```
sheet.py video.mp4 --n 8
```

Writes a PNG contact strip and reports whether audio exists.

To judge identity you must **crop the face and enlarge it** — in thumbnails of a dark video
you see nothing and reach the wrong conclusion. To learn **what** changed, diff against the
source instead of comparing by eye.

Sample at a fine interval when the subject moves fast. Sampling every 0.1 s missed a
character who was only on screen for 7 frames, and led to the wrong conclusion that there
was nothing there to replace.

---

## 9. Reference files

- [references/comfyui.md](references/comfyui.md) — models, nodes, parameters, wiring traps,
  and everything measured and broken along the way.
- [references/hardware.md](references/hardware.md) — VRAM tiers, measured times, how to get
  more out of a smaller card, and renting a GPU.
- [references/standalone-app.md](references/standalone-app.md) — building the app without
  ComfyUI: frameworks, API, hardware.
