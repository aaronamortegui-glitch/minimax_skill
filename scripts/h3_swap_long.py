#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
h3_swap_long.py - replace one or more people in a LONG video.

Splits the video into chunks that fit in VRAM, runs h3_swap.py over each chunk
(one pass per subject, chained), and joins them at the end. Resumable: chunks
already done are skipped.

Chunking introduces seams, so prefer a single pass when the clip fits. Config
lives in a JSON file:

{
  "video": "C:/.../video_to_test.mp4",
  "crop": "960:720:160:0",
  "seconds_per_chunk": 5.17,
  "seed": 2,
  "steps": 8,
  "subjects": [
    {"name": "andres", "detect": "the face and head of the man singing",
     "refs": ["and_face_bw.png"], "prompt_file": "swap_andres.txt", "expand": 40},
    {"name": "victor", "detect": "the face and head of the man playing guitar",
     "refs": ["vic_face_bw.png"], "prompt_file": "swap_victor.txt", "expand": 40}
  ]
}

Usage:
  python h3_swap_long.py config.json
  python h3_swap_long.py config.json --only-chunk 3     # redo only chunk 3
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), "python_embeded", "python.exe")
SWAP = os.path.join(HERE, "h3_swap.py")


def sh(*a):
    r = subprocess.run(list(a))
    if r.returncode != 0:
        raise SystemExit("failed: %s" % " ".join(str(x) for x in a[:4]))


def probe_duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", p], capture_output=True, text=True)
    return float(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only-chunk", type=int, help="redo a single chunk (1-based)")
    ap.add_argument("--no-concat", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    video = cfg["video"]
    base = os.path.dirname(os.path.abspath(video))
    work = os.path.join(base, "swap_chunks")
    os.makedirs(work, exist_ok=True)

    spc = float(cfg.get("seconds_per_chunk", 5.17))
    dur = probe_duration(video)
    nchunks = int(dur // spc) + (1 if dur % spc > 0.4 else 0)
    crop = cfg.get("crop")
    seed = int(cfg.get("seed", 2))
    steps = int(cfg.get("steps", 8))
    subs = cfg["subjects"]

    print("video: %s" % video)
    print("duration: %.2f s -> %d chunks of %.2f s" % (dur, nchunks, spc))
    print("subjects: %s" % ", ".join(s["name"] for s in subs))
    print("estimate: ~%d min (%d chunks x %d subjects x ~8 min)\n"
          % (nchunks * len(subs) * 8, nchunks, len(subs)))

    finals = []
    for c in range(nchunks):
        idx = c + 1
        final = os.path.join(work, "chunk_%02d_final.mp4" % idx)
        finals.append(final)
        if a.only_chunk and idx != a.only_chunk:
            continue
        if os.path.exists(final) and not a.only_chunk:
            print("[%d/%d] already done, skipping" % (idx, nchunks)); continue

        # 1) cut the chunk out of the original
        src = os.path.join(work, "chunk_%02d_src.mp4" % idx)
        vf = ["-vf", crop] if crop else []
        print("[%d/%d] cutting from %.2f s" % (idx, nchunks, c * spc))
        sh("ffmpeg", "-y", "-v", "error", "-ss", str(c * spc), "-t", str(spc + 0.2),
           "-i", video, *vf, "-c:v", "libx264", "-crf", "12", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", src)

        # 2) one pass per subject, chained
        cur = src
        for s in subs:
            out = os.path.join(work, "chunk_%02d_%s.mp4" % (idx, s["name"]))
            if os.path.exists(out):
                print("   %s already done, skipping" % s["name"]); cur = out; continue
            print("   pass: %s" % s["name"])
            cmd = [PY, SWAP, "--video", cur, "--detect", s["detect"],
                   "-p", s["prompt_file"], "--img"] + list(s["refs"]) + \
                  ["--seconds", str(spc), "--steps", str(steps), "--seed", str(seed),
                   "--expand", str(s.get("expand", 40)),
                   "--crop-scale", str(s.get("crop_scale", 1.75)),
                   "--feather", str(s.get("feather", 20)),
                   "--denoise", str(s.get("denoise", 1.0)),
                   "--object-indices", str(s.get("object_indices", "")),
                   "--det-thr", str(s.get("det_thr", 0.5)),
                   "--temporal-expand", str(s.get("temporal_expand", 1)),
                   "--temporal-smooth", str(s.get("temporal_smooth", 0)),
                   "-o", out]
            sh(*cmd)
            cur = out
        os.replace(cur, final) if cur != final else None
        print("   -> %s\n" % final)

    if a.no_concat:
        return
    missing = [f for f in finals if not os.path.exists(f)]
    if missing:
        print("\n%d chunks still missing, not joining yet." % len(missing)); return

    listfile = os.path.join(work, "listfile.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in finals:
            f.write("file '%s'\n" % p.replace("\\", "/"))
    out = os.path.join(base, "swap_FULL.mp4")
    print("joining %d chunks..." % len(finals))
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listfile,
       "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out)
    print("DONE -> %s" % out)


if __name__ == "__main__":
    main()
