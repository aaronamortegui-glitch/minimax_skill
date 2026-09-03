#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_shots.py - replace a character's face shot by shot across a whole scene,
preserving the original footage (segmentation + inpainting with MiniMax H3).

    python batch_shots.py            # every shot in the table
    python batch_shots.py 09 11      # only those

Why this exists: shots in a real scene run anywhere from 0.8 to 6 seconds, and the
model only accepts frame counts on the 17k+5 grid. Rather than stretching a shot or
borrowing frames from its neighbour (which holds a different face and wrecks the
mask), this pads the tail by CLONING the last frame up to the grid duration,
generates, and then TRIMS the output back to the shot's real length. The cloned
frames never reach the final file.

Edit SOURCE, the CHARACTERS recipes and the SHOTS table for your own scene.
"""
import os, subprocess, sys

HERE     = os.path.dirname(os.path.abspath(__file__))
PORTABLE = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable"
PY       = os.path.join(PORTABLE, "python_embeded", "python.exe")
SWAP     = os.path.join(PORTABLE, "h3", "h3_swap.py")

SOURCE   = os.path.join(HERE, "scene.mp4")     # the full scene to cut from
CUTS     = os.path.join(HERE, "cuts")          # per-shot source clips
OUTPUTS  = os.path.join(HERE, "outputs")
PROMPTS  = os.path.join(HERE, "prompts")
REFS     = os.path.join(HERE, "refs")

GRID = [17 * k + 5 for k in range(0, 12)]      # 5, 22, 39, 56, ... 192

# denoise: 1.0 repaints the masked region from scratch and DESTROYS lip sync.
# Below 1.0 the per-frame structure survives (mouth, eyes, blinks) and the
# performance is preserved. 0.85 is the measured compromise.
DENOISE = float(os.environ.get("SWAP_DENOISE", "0.85"))

# One recipe per character.
#   Mask the FACE, not the whole head, to keep the subject's own hair and collar.
#   A whole-head mask replaces the hair and, at large expand, imports the
#   reference image's wardrobe.
#   If the subject wears a hat, keep the detection text tight ("face and beard"):
#   asking for the whole head swallows the hat and the model repaints it.
#   Beard density comes from --upscale-mp, not from a bigger mask. --expand on a
#   face mask controls how far the beard spreads: 25 gives a goatee, 40 fills the
#   cheeks and jaw without reaching the neck.
CHARACTERS = {
    "hero":  {"detect": "the face, the jaw and the beard of a person",
              "refs":   ["hero_face1.png", "hero_face2.png"],
              "prompt": "hero.txt", "expand": 40, "feather": 20},
    "elder": {"detect": "the face and the beard of a person",
              "refs":   ["elder_face.png"],
              "prompt": "elder.txt", "expand": 25, "feather": 18},
}

# shot id: (start seconds, end seconds, character)
SHOTS = {
    "01": (13.90, 17.80, "hero"),
    "02": (38.05, 43.28, "elder"),
}


def sh(*a):
    subprocess.run(list(a), check=True)


def grid_above(n):
    """Smallest grid value >= n frames."""
    for g in GRID:
        if g >= n:
            return g
    return GRID[-1]


def run(key):
    t0, t1, who = SHOTS[key]
    c = CHARACTERS[who]
    real = int(round((t1 - t0) * 24))          # the shot's true frame count
    n = grid_above(real)                        # what the grid demands
    src = os.path.join(CUTS, "%s_%s.mp4" % (key, who))
    pad = os.path.join(CUTS, "%s_%s_pad.mp4" % (key, who))
    gen = os.path.join(OUTPUTS, "gen_%s_%s.mp4" % (key, who))
    out = os.path.join(OUTPUTS, "shot_%s_%s.mp4" % (key, who))

    print("\n=== shot %s (%s) %.2f-%.2f | %d real frames -> %d on the grid"
          % (key, who, t0, t1, real, n))

    for d in (CUTS, OUTPUTS):
        if not os.path.isdir(d):
            os.makedirs(d)

    sh("ffmpeg", "-y", "-v", "error", "-ss", str(t0), "-to", str(t1), "-i", SOURCE,
       "-vf", "fps=24", "-c:v", "libx264", "-crf", "14", "-preset", "medium",
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", src)

    if n > real:
        extra = (n - real) / 24.0 + 0.05
        print("    padding %d frames with the last one" % (n - real))
        # audio gets padded too: VHS_LoadVideo fails without an audio track
        sh("ffmpeg", "-y", "-v", "error", "-i", src,
           "-vf", "tpad=stop_mode=clone:stop_duration=%.3f,fps=24" % extra,
           "-af", "apad=pad_dur=%.3f" % extra,
           "-c:v", "libx264", "-crf", "14", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", pad)
        source_clip = pad
    else:
        source_clip = src

    sh(PY, SWAP, "--video", source_clip, "--detect", c["detect"], "--object-indices", "0",
       "-p", os.path.join(PROMPTS, c["prompt"]),
       "--img", *[os.path.join(REFS, r) for r in c["refs"]],
       "--seconds", "%.4f" % (n / 24.0), "--steps", "10",
       "--denoise", "%.2f" % DENOISE,
       "--expand", str(c["expand"]), "--feather", str(c["feather"]),
       "--upscale-mp", "1.4", "-o", gen)

    # trim back to the real length and restore the shot's own audio
    sh("ffmpeg", "-y", "-v", "error", "-i", gen, "-i", src,
       "-map", "0:v", "-map", "1:a", "-frames:v", str(real),
       "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-shortest", out)
    print("    -> %s" % out)


if __name__ == "__main__":
    keys = sys.argv[1:] or sorted(SHOTS)
    for k in keys:
        if k not in SHOTS:
            print("unknown shot:", k); sys.exit(1)
    for k in keys:
        run(k)
    print("\ndone.")
