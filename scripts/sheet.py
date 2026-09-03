#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sheet.py - pull a contact strip of frames from a video so you can review it at a
glance without playing it. Writes a PNG next to the video.

  python sheet.py video.mp4              # 8 frames
  python sheet.py video.mp4 --n 12       # 12 frames
  python sheet.py video.mp4 --crop face  # crop the upper area (faces)
"""
import argparse, os, sys

try:
    import av
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Run this with the portable python:\n"
             "  ..\\python_embeded\\python.exe sheet.py video.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=8, help="how many frames")
    ap.add_argument("--crop", choices=["no", "face"], default="no")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    c = av.open(a.video)
    vs = c.streams.video[0]
    total = vs.frames or 0
    has_audio = any(s.type == "audio" for s in c.streams)
    print("%dx%d | %d frames | %.2f fps | audio: %s"
          % (vs.width, vs.height, total, float(vs.average_rate), "yes" if has_audio else "NO"))

    want = [int(i * (total - 1) / max(1, a.n - 1)) for i in range(a.n)] if total else list(range(a.n))
    wset = set(want)
    imgs = []
    for i, f in enumerate(c.decode(video=0)):
        if i in wset:
            im = f.to_image()
            if a.crop == "face":
                W, H = im.size
                im = im.crop((0, 0, W, int(H * 0.45)))
            im.thumbnail((260, 460))
            imgs.append((i, im))
        if total and i > max(want):
            break

    if not imgs:
        sys.exit("could not decode any frames")

    W = sum(im.width for _, im in imgs)
    H = max(im.height for _, im in imgs)
    sh = Image.new("RGB", (W, H + 16), "black")
    d = ImageDraw.Draw(sh)
    x = 0
    fps = float(vs.average_rate) or 24.0
    for idx, im in imgs:
        sh.paste(im, (x, 16))
        d.text((x + 3, 3), "%.1fs" % (idx / fps), fill="yellow")
        x += im.width

    out = a.out or os.path.splitext(a.video)[0] + "_frames.png"
    sh.save(out)
    print("strip saved to: %s" % out)


if __name__ == "__main__":
    main()
