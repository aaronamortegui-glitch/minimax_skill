#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
h3_pose.py - POSE transfer: take the motion from a video and apply it to your
character. This is not inpainting: it generates a new video whose motion is driven
by the source video's skeleton. The background is reinvented, not preserved.

Chain:
  VHS_LoadVideo -> DWPreprocessor (skeleton)
    UNETLoader ref2va -> turbo LoRA -> H3FunControlApply(control_video=skeleton)
    MiniMaxH3ReferenceToVideo(character references) -> conditioning + latent
    -> SamplerCustomAdvanced -> VAEDecode -> SaveVideo

--strength goes up to 2.0. At 1.0 the pose is followed loosely; 1.8 does follow it,
at the cost of detail and identity. DWPose extracts body, hands and face, but NOT
objects, so props do not survive.

Example:
  python h3_pose.py --video dance.mp4 --img character.png -p prompt.txt --seconds 15 --vertical
"""
import argparse, json, os, subprocess, sys, time, urllib.request, uuid

PORTABLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY  = os.path.join(PORTABLE, "ComfyUI")
INPUT  = os.path.join(COMFY, "input")
OUTPUT = os.path.join(COMFY, "output")
SERVER = "http://127.0.0.1:8188"

REF2VA  = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TE      = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_V   = "minimax_h3_video_vae_fp16.safetensors"
VAE_A   = "minimax_h3_audio_vae_fp32.safetensors"
LORA    = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
CONTROL = "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors"


def frames_for(seconds):
    t = round(seconds * 24)
    return 17 * max(0, round((t - 5) / 17.0)) + 5


def ffmpeg(*a):
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + list(a), check=True)


def post(path, payload):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60).read().decode()


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path, timeout=60).read().decode())


def build(vid, n, w, h, prompt, imgs, steps, seed, strength, start_pct, end_pct,
          use_lora, ref_size, pose_res, keep_audio):
    g = {
        "10": {"class_type": "VHS_LoadVideo",
               "inputs": {"video": vid, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                          "frame_load_cap": n, "skip_first_frames": 0, "select_every_nth": 1}},
        # skeleton: body + hands + face
        "11": {"class_type": "DWPreprocessor",
               "inputs": {"image": ["10", 0], "detect_hand": "enable", "detect_body": "enable",
                          "detect_face": "enable", "resolution": pose_res}},
        # the skeleton MUST match the generation size exactly
        "11c": {"class_type": "ImageScale",
                "inputs": {"image": ["11", 0], "width": w, "height": h,
                           "upscale_method": "lanczos", "crop": "disabled"}},
        "40": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_V}},
        "41": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_A}},
        "42": {"class_type": "UNETLoader", "inputs": {"unet_name": REF2VA, "weight_dtype": "default"}},
        "43": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "minimax", "device": "default"}},
        "46": {"class_type": "H3FunControlLoader", "inputs": {"control_net_name": CONTROL}},
        "8":  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10b":{"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }
    src = ["42", 0]
    if use_lora:
        g["44"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": src, "lora_name": LORA, "strength_model": 1.0}}
        src = ["44", 0]
    # the ControlNet patches the model with the skeleton video
    g["47"] = {"class_type": "H3FunControlApply",
               "inputs": {"model": src, "control_net": ["46", 0], "vae": ["40", 0],
                          "control_video": ["11c", 0], "strength": strength,
                          "start_percent": start_pct, "end_percent": end_pct}}
    g["45"] = {"class_type": "MiniMaxH3SigmaShift",
               "inputs": {"model": ["47", 0], "shift_video": 12.0, "shift_audio": 3.0}}

    cond = {"clip": ["43", 0], "vae": ["40", 0], "audio_vae": ["41", 0], "prompt": prompt,
            "width": w, "height": h, "length": n, "ref_image_size": ref_size}
    nid = 80
    for i, name in enumerate(imgs):
        g[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        cond["ref_images.ref_image_%d" % i] = [str(nid), 0]; nid += 1
    g["90"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond}

    g["7"]  = {"class_type": "BasicGuider", "inputs": {"model": ["45", 0], "conditioning": ["90", 0]}}
    g["9"]  = {"class_type": "BasicScheduler",
               "inputs": {"model": ["45", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    g["11b"]= {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10b", 0], "guider": ["7", 0], "sampler": ["8", 0],
                          "sigmas": ["9", 0], "latent_image": ["90", 1]}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11b", 0], "vae": ["40", 0]}}
    audio = ["10", 2] if keep_audio else ["13", 0]
    if not keep_audio:
        g["13"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11b", 0], "vae": ["41", 0]}}
    g["14"] = {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": 24.0, "audio": audio}}
    g["15"] = {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0], "filename_prefix": "video/h3_pose",
                          "format": "auto", "codec": "auto"}}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video that supplies the MOTION")
    ap.add_argument("--img", nargs="+", required=True, help="references for the character")
    ap.add_argument("-p", "--prompt-file", required=True)
    ap.add_argument("--seconds", type=float, default=5.17)
    ap.add_argument("--vertical", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--strength", type=float, default=1.0, help="ControlNet strength, up to 2.0")
    ap.add_argument("--start-pct", type=float, default=0.0)
    ap.add_argument("--end-pct", type=float, default=1.0)
    ap.add_argument("--pose-res", type=int, default=768)
    ap.add_argument("--quality", action="store_true", help="no turbo LoRA, 20 steps")
    ap.add_argument("--ref-size", choices=["match", "max"], default="max")
    ap.add_argument("--keep-audio", action="store_true", help="use the source video's audio")
    ap.add_argument("--save-pose", action="store_true", help="also save the skeleton")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    n = frames_for(a.seconds)
    w, h = (768, 1344) if a.vertical else (1344, 768)
    steps = 20 if a.quality else a.steps

    vid = "pose_src.mp4"
    print("source -> 24 fps, %d frames (%.2f s)" % (n, n / 24.0))
    ffmpeg("-i", a.video, "-r", "24", "-frames:v", str(n), "-c:v", "libx264", "-crf", "14",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           os.path.join(INPUT, vid))
    imgs = []
    for p in a.img:
        b = "pose_ref_%d.png" % len(imgs)
        ffmpeg("-i", p, "-frames:v", "1", os.path.join(INPUT, b)); imgs.append(b)

    prompt = open(a.prompt_file, encoding="utf-8").read()
    g = build(vid, n, w, h, prompt, imgs, steps, a.seed, a.strength, a.start_pct, a.end_pct,
              not a.quality, a.ref_size, a.pose_res, a.keep_audio)
    if a.save_pose:
        g["16"] = {"class_type": "SaveVideo",
                   "inputs": {"video": ["17", 0], "filename_prefix": "video/h3_pose_skel",
                              "format": "auto", "codec": "auto"}}
        g["17"] = {"class_type": "CreateVideo", "inputs": {"images": ["11c", 0], "fps": 24.0}}

    try:
        r = json.loads(post("/prompt", {"prompt": g, "client_id": str(uuid.uuid4())}))
    except urllib.error.HTTPError as e:
        print("INVALID GRAPH:"); print(e.read().decode()[:3000]); sys.exit(1)
    print("graph valid:", r["prompt_id"])
    if a.dry_run:
        post("/queue", {"delete": [r["prompt_id"]]}); print("dry-run, nothing generated."); return

    t0 = time.time()
    while True:
        time.sleep(20)
        try: hist = get("/history/" + r["prompt_id"])
        except Exception: continue
        if hist:
            v = list(hist.values())[0]; st = v.get("status", {})
            if st.get("status_str") != "success":
                for m in st.get("messages", []):
                    if "error" in str(m[0]): print("ERROR:", json.dumps(m[1])[:1500])
                sys.exit(1)
            o = v["outputs"]["15"]["images"][0]
            src = os.path.join(OUTPUT, o.get("subfolder", ""), o["filename"])
            dst = a.out or os.path.join(os.path.dirname(os.path.abspath(a.video)), "pose_resultado.mp4")
            ffmpeg("-i", src, "-c:v", "libx264", "-crf", "15", "-preset", "slow",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dst)
            print("done in %.1f min -> %s" % ((time.time() - t0) / 60.0, dst))
            if a.save_pose and "16" in v.get("outputs", {}):
                so = v["outputs"]["16"]["images"][0]
                print("skeleton -> %s" % os.path.join(OUTPUT, so.get("subfolder",""), so["filename"]))
            return
        print("   ... %.1f min" % ((time.time() - t0) / 60.0), end="\r", flush=True)


if __name__ == "__main__":
    main()
