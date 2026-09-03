#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
h3_swap.py - replace a person inside existing footage, preserving the original
frames. Only the masked region is repainted.

Chain:
  VHS_LoadVideo -> SAM3_VideoTrack -> SAM3_TrackToMask -> MVEx_SubjectCrop
    -> NKDMaskOps -> NKDAVLatent (your footage BAKED into the latent, with a mask)
    -> KSampler (conditioning from MiniMaxH3ReferenceToVideo) -> VAEDecode
    -> MVEx_SubjectUncrop (pastes back into the original frames)

Two ways to get the mask:
  --detect "the face and the beard of a person"   text-grounded SAM 3.1
  --mask-box "308,128,60,74"                      a fixed rectangle, for subjects
                                                  segmentation cannot isolate

Run probe_tracks.py FIRST to learn which object index is which person.

For talking shots use --denoise 0.85, not 1.0: at 1.0 the masked region is repainted
from scratch every frame and lip sync is destroyed.

Example:
  python h3_swap.py --video seg.mp4 --detect "the face and the beard of a person" \
      --img andres.png --prompt-file swap.txt --steps 8 --denoise 1.0
"""
import argparse, json, os, subprocess, sys, time, urllib.request, uuid

PORTABLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY  = os.path.join(PORTABLE, "ComfyUI")
INPUT  = os.path.join(COMFY, "input")
OUTPUT = os.path.join(COMFY, "output")
SERVER = "http://127.0.0.1:8188"

SAM3      = "sam3.1_multiplex_fp16.safetensors"
REF2VA    = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TE        = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
LORA      = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


def ffmpeg(*a):
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + list(a), check=True)


def probe_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def post(path, payload):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60).read().decode()


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path, timeout=60).read().decode())


def build(video_name, nframes, detect_text, prompt, img_names, steps, denoise,
          seed, cfg, use_lora, crop_scale, expand, feather, ref_size, upscale_mp,
          object_indices="", det_thr=0.5, temporal_expand=1, temporal_smooth=0,
          mask_video=None):
    g = {
        # ---- source ----
        "10": {"class_type": "VHS_LoadVideo",
               "inputs": {"video": video_name, "force_rate": 0, "custom_width": 0,
                          "custom_height": 0, "frame_load_cap": nframes,
                          "skip_first_frames": 0, "select_every_nth": 1}},
        # ---- crop around the subject ----
        "30": {"class_type": "MVEx_SubjectCrop",
               "inputs": {"original_images": ["10", 0], "masks": ["23", 0],
                          "mode": "tracked", "mode.crop_scale": crop_scale,
                          "mode.padding": "firm", "mode.prefer": "stillness",
                          "mode.aspect_ratio": 0.0, "mode.seamless_loop": False,
                          "divisible_by": 32, "upscale_megapixels": upscale_mp}},
        # ---- VAEs and model ----
        "40": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "41": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "42": {"class_type": "UNETLoader",
               "inputs": {"unet_name": REF2VA, "weight_dtype": "default"}},
        "43": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": TE, "type": "minimax", "device": "default"}},
        # ---- clean mask for the repaint ----
        "50": {"class_type": "NKDMaskOps",
               "inputs": {"mask": ["30", 1], "vae": ["40", 0], "model": ["42", 0],
                          "invert": False, "black_point": 0.0, "white_point": 1.0,
                          "despeckle": 2, "fill_holes": False, "close_gaps": 0,
                          "temporal_expand": temporal_expand, "temporal_smooth": temporal_smooth,
                          "expand": expand, "blockify": 0, "blockify_threshold": 0.05,
                          "feather": 0, "edge_low": 0.0, "edge_high": 1.0}},
        # ---- soft mask for the final paste-back ----
        "51": {"class_type": "NKDMaskOpsLean",
               "inputs": {"mask": ["50", 0], "fill_holes": False,
                          "expand": 10, "feather": feather}},
        # ---- THE CORE: your footage baked into the latent, with a mask ----
        "60": {"class_type": "NKDAVLatent",
               "inputs": {"images": ["30", 0], "audio": ["10", 2],
                          "video_vae": ["40", 0], "audio_vae": ["41", 0],
                          "audio_mode": "keep", "fps": 24.0,
                          "ramp_ticks": 0, "ramp_out_ticks": 0, "ramp_shape": "cosine",
                          # slot 2 = latent_mask (slot 0 is the pixel mask: wrong)
                          "latent_mask": ["50", 2]}},
        # ---- crop size, to condition at that resolution ----
        "70": {"class_type": "GetImageSize", "inputs": {"image": ["30", 0]}},
    }

    # ---- where the mask comes from ----
    if mask_video:
        # Fixed rectangle: the mask arrives as a black-and-white video with the
        # same frame count and size as the source. For when SAM3 cannot isolate
        # the subject (e.g. a 20 px figure seen from overhead in a crowd all
        # wearing the same thing).
        g["20"] = {"class_type": "VHS_LoadVideo",
                   "inputs": {"video": mask_video, "force_rate": 0, "custom_width": 0,
                              "custom_height": 0, "frame_load_cap": nframes,
                              "skip_first_frames": 0, "select_every_nth": 1}}
        g["23"] = {"class_type": "ImageToMask",
                   "inputs": {"image": ["20", 0], "channel": "red"}}
    else:
        # SAM 3.1: detect and track the subject from text
        g["20"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3}}
        g["21"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": ["20", 1], "text": detect_text}}
        g["22"] = {"class_type": "SAM3_VideoTrack",
                   "inputs": {"images": ["10", 0], "model": ["20", 0],
                              "detection_threshold": det_thr, "max_objects": 0,
                              "detect_interval": 1, "conditioning": ["21", 0]}}
        g["23"] = {"class_type": "SAM3_TrackToMask",
                   "inputs": {"track_data": ["22", 0], "object_indices": object_indices}}

    # ---- conditioning from the new character's references ----
    cond = {"clip": ["43", 0], "vae": ["40", 0], "audio_vae": ["41", 0],
            "prompt": prompt, "width": ["70", 0], "height": ["70", 1],
            "length": nframes, "ref_image_size": ref_size}
    nid = 80
    for i, name in enumerate(img_names):
        g[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        cond["ref_images.ref_image_%d" % i] = [str(nid), 0]
        nid += 1
    g["90"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond}
    g["91"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["90", 0]}}

    model_src = ["42", 0]
    if use_lora:
        g["44"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_src, "lora_name": LORA, "strength_model": 1.0}}
        model_src = ["44", 0]
    g["45"] = {"class_type": "MiniMaxH3SigmaShift",
               "inputs": {"model": model_src, "shift_video": 12.0, "shift_audio": 3.0}}

    # ---- sampling: KSampler, cfg 1. Use denoise 0.85 for talking shots ----
    g["100"] = {"class_type": "KSampler",
                "inputs": {"model": ["45", 0], "positive": ["90", 0], "negative": ["91", 0],
                           "latent_image": ["60", 0], "seed": seed, "steps": steps,
                           "cfg": cfg, "sampler_name": "res_multistep",
                           "scheduler": "simple", "denoise": denoise}}
    g["110"] = {"class_type": "VAEDecode", "inputs": {"samples": ["100", 0], "vae": ["40", 0]}}
    # ---- paste back into the original frames ----
    g["120"] = {"class_type": "MVEx_SubjectUncrop",
                "inputs": {"cropped_images": ["110", 0], "original_images": ["10", 0],
                           "bboxes": ["30", 2], "feather": 16, "cropped_masks": ["51", 0]}}
    g["130"] = {"class_type": "CreateVideo",
                "inputs": {"images": ["120", 0], "fps": 24.0, "audio": ["10", 2]}}
    g["140"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["130", 0], "filename_prefix": "video/h3_swap",
                           "format": "auto", "codec": "auto"}}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="source segment (converted to 24 fps)")
    ap.add_argument("--detect", default="", help="what SAM3 should look for, e.g. 'the face and the beard of a person'")
    ap.add_argument("-p", "--prompt-file", required=True)
    ap.add_argument("--img", nargs="+", required=True, help="references for the new character")
    ap.add_argument("--seconds", type=float, default=5.17)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--quality", action="store_true", help="no turbo LoRA")
    ap.add_argument("--crop-scale", type=float, default=1.75)
    ap.add_argument("--expand", type=int, default=30, help="how far to grow the mask. On a face mask: 25 a goatee, 40 fills the jaw")
    ap.add_argument("--feather", type=int, default=20)
    ap.add_argument("--upscale-mp", type=float, default=0.8)
    ap.add_argument("--ref-size", choices=["match", "max"], default="max")
    ap.add_argument("--object-indices", default="", help="which tracked object to use: '0' or '1'. Empty = all")
    ap.add_argument("--det-thr", type=float, default=0.5, help="SAM3 detection threshold. Lowering it usually makes things worse")
    ap.add_argument("--temporal-expand", type=int, default=1, help="fill mask gaps between frames")
    ap.add_argument("--temporal-smooth", type=int, default=0, help="smooth the mask over time")
    ap.add_argument("--mask-box", default="",
                    help="fixed rectangle 'x,y,w,h' in source pixels, instead of SAM3. "
                         "For subjects that text segmentation cannot isolate.")
    ap.add_argument("--dry-run", action="store_true", help="only validate the graph, do not generate")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    n = int(round(a.seconds * 24))
    n = 17 * max(0, round((n - 5) / 17.0)) + 5

    if not a.detect and not a.mask_box:
        print("you need either --detect (SAM3) or --mask-box 'x,y,w,h'.")
        sys.exit(1)

    vname = "swap_src.mp4"
    print("preparing source -> 24 fps, %d frames..." % n)
    ffmpeg("-i", a.video, "-r", "24", "-frames:v", str(n),
           "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", os.path.join(INPUT, vname))

    mask_video = None
    if a.mask_box:
        bx, by, bw, bh = [int(v) for v in a.mask_box.replace(" ", "").split(",")]
        w, h = probe_size(os.path.join(INPUT, vname))
        # the rectangle is painted white on black, one frame per source frame
        mask_video = "swap_mask.mp4"
        print("box mask: %dx%d at (%d,%d) over %dx%d" % (bw, bh, bx, by, w, h))
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", "color=c=black:s=%dx%d:r=24" % (w, h),
                        "-vf", "drawbox=x=%d:y=%d:w=%d:h=%d:color=white:t=fill" % (bx, by, bw, bh),
                        "-frames:v", str(n), "-c:v", "libx264", "-crf", "0",
                        "-preset", "medium", "-pix_fmt", "yuv420p",
                        os.path.join(INPUT, mask_video)], check=True)

    imgs = []
    for p in a.img:
        base = "swap_ref_%d.png" % len(imgs)
        ffmpeg("-i", p, "-frames:v", "1", os.path.join(INPUT, base))
        imgs.append(base)

    prompt = open(a.prompt_file, encoding="utf-8").read()
    g = build(vname, n, a.detect, prompt, imgs, a.steps, a.denoise, a.seed, a.cfg,
              not a.quality, a.crop_scale, a.expand, a.feather, a.ref_size, a.upscale_mp,
              a.object_indices, a.det_thr, a.temporal_expand, a.temporal_smooth,
              mask_video)

    try:
        r = json.loads(post("/prompt", {"prompt": g, "client_id": str(uuid.uuid4())}))
    except urllib.error.HTTPError as e:
        print("INVALID GRAPH:")
        print(e.read().decode()[:3000])
        sys.exit(1)
    print("graph valid. prompt_id:", r["prompt_id"])
    if a.dry_run:
        post("/queue", {"delete": [r["prompt_id"]]})
        print("dry-run: removed from the queue, nothing was generated.")
        return

    t0 = time.time()
    while True:
        time.sleep(20)
        try:
            h = get("/history/" + r["prompt_id"])
        except Exception:
            continue
        if h:
            v = list(h.values())[0]; st = v.get("status", {})
            if st.get("status_str") != "success":
                for m in st.get("messages", []):
                    if "error" in str(m[0]):
                        print("ERROR:", json.dumps(m[1])[:1200])
                sys.exit(1)
            o = v["outputs"]["140"]["images"][0]
            src = os.path.join(OUTPUT, o.get("subfolder", ""), o["filename"])
            dst = a.out or os.path.join(os.path.dirname(os.path.abspath(a.video)), "swap_resultado.mp4")
            ffmpeg("-i", src, "-c:v", "libx264", "-crf", "15", "-preset", "slow",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dst)
            print("done in %.1f min -> %s" % ((time.time() - t0) / 60.0, dst))
            return
        print("   ... %.1f min" % ((time.time() - t0) / 60.0), end="\r", flush=True)


if __name__ == "__main__":
    main()
