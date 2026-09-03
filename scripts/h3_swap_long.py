#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
h3_swap_long.py - reemplaza una o varias personas en un video LARGO.

Trocea el video en segmentos que caben en VRAM, corre h3_swap.py sobre cada trozo
(una pasada por sujeto, encadenadas), y pega todo al final. Se puede reanudar: los
trozos ya hechos se saltan.

Config en un JSON:

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

Uso:
  python h3_swap_long.py config.json
  python h3_swap_long.py config.json --only-chunk 3     # rehacer solo el trozo 3
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), "python_embeded", "python.exe")
SWAP = os.path.join(HERE, "h3_swap.py")


def sh(*a):
    r = subprocess.run(list(a))
    if r.returncode != 0:
        raise SystemExit("fallo: %s" % " ".join(str(x) for x in a[:4]))


def probe_duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", p], capture_output=True, text=True)
    return float(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only-chunk", type=int, help="rehacer un solo trozo (1-based)")
    ap.add_argument("--no-concat", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    video = cfg["video"]
    base = os.path.dirname(os.path.abspath(video))
    work = os.path.join(base, "swap_trozos")
    os.makedirs(work, exist_ok=True)

    spc = float(cfg.get("seconds_per_chunk", 5.17))
    dur = probe_duration(video)
    nchunks = int(dur // spc) + (1 if dur % spc > 0.4 else 0)
    crop = cfg.get("crop")
    seed = int(cfg.get("seed", 2))
    steps = int(cfg.get("steps", 8))
    subs = cfg["subjects"]

    print("video: %s" % video)
    print("duracion: %.2f s -> %d trozos de %.2f s" % (dur, nchunks, spc))
    print("sujetos: %s" % ", ".join(s["name"] for s in subs))
    print("estimado: ~%d min (%d trozos x %d sujetos x ~8 min)\n"
          % (nchunks * len(subs) * 8, nchunks, len(subs)))

    finales = []
    for c in range(nchunks):
        idx = c + 1
        final = os.path.join(work, "trozo_%02d_final.mp4" % idx)
        finales.append(final)
        if a.only_chunk and idx != a.only_chunk:
            continue
        if os.path.exists(final) and not a.only_chunk:
            print("[%d/%d] ya hecho, salto" % (idx, nchunks)); continue

        # 1) cortar el trozo del original
        src = os.path.join(work, "trozo_%02d_src.mp4" % idx)
        vf = ["-vf", crop] if crop else []
        print("[%d/%d] cortando desde %.2f s" % (idx, nchunks, c * spc))
        sh("ffmpeg", "-y", "-v", "error", "-ss", str(c * spc), "-t", str(spc + 0.2),
           "-i", video, *vf, "-c:v", "libx264", "-crf", "12", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", src)

        # 2) una pasada por sujeto, encadenadas
        cur = src
        for s in subs:
            out = os.path.join(work, "trozo_%02d_%s.mp4" % (idx, s["name"]))
            if os.path.exists(out):
                print("   %s ya hecho, salto" % s["name"]); cur = out; continue
            print("   pasada: %s" % s["name"])
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
    faltan = [f for f in finales if not os.path.exists(f)]
    if faltan:
        print("\nfaltan %d trozos, no pego todavia." % len(faltan)); return

    lista = os.path.join(work, "lista.txt")
    with open(lista, "w", encoding="utf-8") as f:
        for p in finales:
            f.write("file '%s'\n" % p.replace("\\", "/"))
    out = os.path.join(base, "swap_COMPLETO.mp4")
    print("pegando %d trozos..." % len(finales))
    sh("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lista,
       "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out)
    print("LISTO -> %s" % out)


if __name__ == "__main__":
    main()
