#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe_tracks.py - find out WHICH SAM3 object index corresponds to which person in a
video, and how stable each track is. Cheap run: tracking only, no sampling.

  python probe_tracks.py --video seg.mp4
  python probe_tracks.py --video seg.mp4 --det-thr 0.25 --max-idx 5

Writes a PNG next to the video with the masks overlaid, plus per-frame coverage
percentages. A mask that FLICKERS on and off is unusable: the MVEx crop geometry
wanders and the repaint lands in the wrong place. A mask that grows or shrinks
monotonically (subject entering or leaving frame) is fine.
"""
import argparse, json, os, shutil, subprocess, time, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
PORTABLE = os.path.dirname(HERE)
COMFY = os.path.join(PORTABLE, "ComfyUI")
INPUT = os.path.join(COMFY, "input")
OUTPUT = os.path.join(COMFY, "output")
SERVER = "http://127.0.0.1:8188"
SAM3 = "sam3.1_multiplex_fp16.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detect", default="the head and face of a person")
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--max-idx", type=int, default=4)
    ap.add_argument("--frames", type=int, nargs="*", default=[20, 60, 100])
    ap.add_argument("--nframes", type=int, default=124)
    a = ap.parse_args()

    tag = "ptrk_" + uuid.uuid4().hex[:6]
    stage = tag + ".mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", a.video, "-r", "24",
                    "-frames:v", str(a.nframes), "-c:v", "libx264", "-crf", "14",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-an",
                    os.path.join(INPUT, stage)], check=True)

    g = {
        "10": {"class_type": "VHS_LoadVideo",
               "inputs": {"video": stage, "force_rate": 0, "custom_width": 0,
                          "custom_height": 0, "frame_load_cap": a.nframes,
                          "skip_first_frames": 0, "select_every_nth": 1}},
        "20": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["20", 1], "text": a.detect}},
        "22": {"class_type": "SAM3_VideoTrack",
               "inputs": {"images": ["10", 0], "model": ["20", 0],
                          "detection_threshold": a.det_thr, "max_objects": 0,
                          "detect_interval": 1, "conditioning": ["21", 0]}},
    }
    nid = 30
    idxs = [str(i) for i in range(a.max_idx + 1)]
    for i in idxs:
        g[str(nid)] = {"class_type": "SAM3_TrackToMask",
                       "inputs": {"track_data": ["22", 0], "object_indices": i}}
        g[str(nid + 1)] = {"class_type": "MaskToImage", "inputs": {"mask": [str(nid), 0]}}
        g[str(nid + 2)] = {"class_type": "SaveImage",
                           "inputs": {"images": [str(nid + 1), 0],
                                      "filename_prefix": "%s_%s" % (tag, i)}}
        nid += 10

    req = urllib.request.Request(SERVER + "/prompt",
        data=json.dumps({"prompt": g, "client_id": str(uuid.uuid4())}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req).read().decode())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit("no valido:\n" + e.read().decode()[:2000])
    print("rastreando (umbral %.2f)..." % a.det_thr)
    t0 = time.time()
    while True:
        time.sleep(8)
        h = json.loads(urllib.request.urlopen(SERVER + "/history/" + pid).read().decode())
        if h:
            st = list(h.values())[0].get("status", {})
            if st.get("status_str") != "success":
                for m in st.get("messages", []):
                    if "error" in str(m[0]): print("ERROR:", json.dumps(m[1])[:600])
                raise SystemExit(1)
            print("tracking listo en %.0f s\n" % (time.time() - t0))
            break

    import av
    from PIL import Image, ImageDraw, ImageChops
    c = av.open(os.path.join(INPUT, stage)); bases = {}
    for i, f in enumerate(c.decode(video=0)):
        if i in a.frames: bases[i] = f.to_image().convert("RGB")
        if i > max(a.frames): break

    rows = []
    print("%-6s %s" % ("objeto", "  ".join("f%-7d" % f for f in a.frames)))
    for idx in idxs:
        row, cov = [], []
        for fr in a.frames:
            p = os.path.join(OUTPUT, "%s_%s_%05d_.png" % (tag, idx, fr + 1))
            if not os.path.exists(p) or fr not in bases:
                row.append(None); cov.append(0.0); continue
            base = bases[fr]
            m = Image.open(p).convert("L").resize(base.size)
            hist = m.histogram()
            cov.append(100.0 * sum(hist[129:]) / float(sum(hist)))
            tint = Image.new("RGB", base.size, (255, 40, 40))
            comp = Image.composite(ImageChops.blend(base, tint, 0.6), base, m)
            comp.thumbnail((260, 260)); row.append(comp)
        estable = "ESTABLE" if all(v > 2.0 for v in cov) else "intermitente" if any(v > 2.0 for v in cov) else "vacio"
        print("%-6s %s  -> %s" % (idx, "  ".join("%6.1f%% " % v for v in cov), estable))
        rows.append((idx, row, cov, estable))

    imgs = [im for _, r, _, _ in rows for im in r if im]
    if imgs:
        w, h = imgs[0].width, imgs[0].height
        sh = Image.new("RGB", (w * len(a.frames), (h + 20) * len(rows)), "black")
        d = ImageDraw.Draw(sh); y = 0
        for idx, row, cov, est in rows:
            d.text((6, y + 4), "obj %s  %s  %s" % (idx, " ".join("%.1f%%" % v for v in cov), est), fill="yellow")
            x = 0
            for im in row:
                if im: sh.paste(im, (x, y + 20))
                x += w
            y += h + 20
        out = os.path.splitext(os.path.abspath(a.video))[0] + "_tracks.png"
        sh.thumbnail((1400, 1400)); sh.save(out)
        print("\nmapa visual -> %s" % out)
    try:
        os.remove(os.path.join(INPUT, stage))
    except OSError:
        pass


if __name__ == "__main__":
    main()
