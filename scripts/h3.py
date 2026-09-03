#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
h3.py - CLI for generating video+audio with MiniMax H3 on a local ComfyUI.
No external dependencies: it uses ComfyUI's HTTP API and ffmpeg.

EXAMPLES
--------
# text -> video (no references, uses fl2va)
python h3.py -p prompt.txt --seconds 5 --vertical

# image references + cloned voice (ref2va)
python h3.py -p prompt.txt --img avatar.png face.png --audio voice.mp3 --seconds 8 --vertical

# VIDEO reference with its own audio (identity + voice paired)
python h3.py -p prompt.txt --video source.mp4 --video-audio --seconds 8 --vertical

# maximum quality (no turbo LoRA, 20 steps) and several seeds
python h3.py -p prompt.txt --img a.png --seconds 5 --quality --seeds 1 2 3

In the prompt, references are named <Picture 1>..<Picture 9>, <Video 1>..<Video 3>,
<Audio 1>..<Audio 3>. Label order is: images, then videos (each video's soundtrack
takes its <Audio j> BEFORE its <Video k>), then standalone audios.
"""

import argparse, json, os, shutil, subprocess, sys, time, urllib.request, urllib.error, uuid

# ------------------------------------------------------------------- paths
PORTABLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY    = os.path.join(PORTABLE, "ComfyUI")
INPUT    = os.path.join(COMFY, "input")
OUTPUT   = os.path.join(COMFY, "output")
PY       = os.path.join(PORTABLE, "python_embeded", "python.exe")
SERVER   = "http://127.0.0.1:8188"

MODELS = {
    "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "fl2va":  "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
}
LORAS = {
    "ref2va": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "fl2va":  "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
}
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO    = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO    = "minimax_h3_audio_vae_fp32.safetensors"


# --------------------------------------------------------------- utilities
def frames_for(seconds):
    """H3 only accepts durations on the 17k+5 frame grid at 24 fps."""
    target = round(seconds * 24)
    k = max(0, round((target - 5) / 17.0))
    return 17 * k + 5


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + list(args), check=True)


def post(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30).read().decode()


def get(path):
    return json.loads(urllib.request.urlopen(SERVER + path, timeout=30).read().decode())


def server_up():
    try:
        get("/system_stats"); return True
    except Exception:
        return False


def stage_image(src):
    """Copy an image into ComfyUI's input folder and return its name."""
    name = "h3_" + os.path.splitext(os.path.basename(src))[0].replace(" ", "_") + ".png"
    from_ext = os.path.splitext(src)[1].lower()
    dst = os.path.join(INPUT, name)
    if from_ext == ".png":
        shutil.copy(src, dst)
    else:
        ffmpeg("-i", src, "-frames:v", "1", dst)
    return name


def stage_audio(src, seconds=15):
    """Trim to `seconds`, normalize, and drop the audio in the input folder."""
    name = "h3_" + os.path.splitext(os.path.basename(src))[0].replace(" ", "_") + ".mp3"
    dst = os.path.join(INPUT, name)
    ffmpeg("-t", str(seconds), "-i", src,
           "-af", "loudnorm=I=-18:TP=-2", "-ar", "48000", "-ac", "1", dst)
    return name


def stage_video(src, seconds=5.17):
    """H3 wants video references at 24 fps with frames = 17k+5."""
    n = frames_for(seconds)
    name = "h3_" + os.path.splitext(os.path.basename(src))[0].replace(" ", "_") + "_ref.mp4"
    dst = os.path.join(INPUT, name)
    ffmpeg("-i", src, "-r", "24", "-frames:v", str(n),
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", dst)
    return name, n


# --------------------------------------------------------------- the graph
def build(prompt, w, h, frames, steps, seed, imgs, auds, vid, vid_audio,
          use_lora, ref_size):
    mode = "ref2va" if (imgs or auds or vid) else "fl2va"

    g = {
        "1":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": MODELS[mode], "weight_dtype": "default"}},
        "2":  {"class_type": "CLIPLoader",
               "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
        "4":  {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "5":  {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "8":  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }

    # model -> (optional turbo LoRA) -> SigmaShift
    model_src = ["1", 0]
    if use_lora:
        g["3"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model_src, "lora_name": LORAS[mode], "strength_model": 1.0}}
        model_src = ["3", 0]
    g["16"] = {"class_type": "MiniMaxH3SigmaShift",
               "inputs": {"model": model_src, "shift_video": 12.0, "shift_audio": 3.0}}

    # reference loader nodes
    cond_inputs = {"clip": ["2", 0], "vae": ["4", 0], "prompt": prompt,
                   "width": w, "height": h, "length": frames}
    nid = 20
    if mode == "ref2va":
        cond_inputs["audio_vae"] = ["5", 0]
        cond_inputs["ref_image_size"] = ref_size
        for i, name in enumerate(imgs):
            g[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond_inputs["ref_images.ref_image_%d" % i] = [str(nid), 0]; nid += 1
        if vid:
            vname, vframes = vid
            g[str(nid)] = {"class_type": "VHS_LoadVideo",
                           "inputs": {"video": vname, "force_rate": 0, "custom_width": 0,
                                      "custom_height": 0, "frame_load_cap": vframes,
                                      "skip_first_frames": 0, "select_every_nth": 1}}
            cond_inputs["ref_videos.ref_video_0"] = [str(nid), 0]
            if vid_audio:
                cond_inputs["ref_video_audios.ref_video_audio_0"] = [str(nid), 2]
            nid += 1
        for i, name in enumerate(auds):
            g[str(nid)] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
            cond_inputs["ref_audios.ref_audio_%d" % i] = [str(nid), 0]; nid += 1
        g["6"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    else:
        g["6"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}

    g["7"]  = {"class_type": "BasicGuider", "inputs": {"model": ["16", 0], "conditioning": ["6", 0]}}
    g["9"]  = {"class_type": "BasicScheduler",
               "inputs": {"model": ["16", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    g["11"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0],
                          "sigmas": ["9", 0], "latent_image": ["6", 1]}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}}
    g["13"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}}
    g["14"] = {"class_type": "CreateVideo",
               "inputs": {"images": ["12", 0], "fps": 24.0, "audio": ["13", 0]}}
    g["15"] = {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0], "filename_prefix": "video/h3_cli",
                          "format": "auto", "codec": "auto"}}
    return g, mode


def run_one(graph, label):
    r = json.loads(post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())}))
    pid = r["prompt_id"]
    print("  [%s] en cola: %s" % (label, pid))
    t0 = time.time()
    while True:
        time.sleep(15)
        try:
            hist = get("/history/" + pid)
        except Exception:
            continue
        if hist:
            v = list(hist.values())[0]
            st = v.get("status", {})
            if st.get("status_str") != "success":
                for m in st.get("messages", []):
                    if "error" in str(m[0]):
                        print("  ERROR:", str(m[1].get("exception_message", ""))[:400])
                return None
            outs = v.get("outputs", {}).get("15", {}).get("images", [])
            if not outs:
                print("  ERROR: no output"); return None
            print("  [%s] done in %.1f min" % (label, (time.time() - t0) / 60.0))
            return os.path.join(OUTPUT, outs[0].get("subfolder", ""), outs[0]["filename"])
        mins = (time.time() - t0) / 60.0
        if int(mins * 4) % 8 == 0:
            print("    ... %.1f min" % mins, end="\r", flush=True)


def export(src, dst, w, h):
    """Crop to the exact aspect (9:16 or 16:9) without rescaling, then re-encode."""
    if h > w:   # vertical -> 9:16
        cw = (int(h * 9 / 16) // 2) * 2
        vf = "crop=%d:%d:%d:0" % (cw, h, (w - cw) // 2)
    else:       # horizontal -> 16:9
        ch = (int(w * 9 / 16) // 2) * 2
        vf = "crop=%d:%d:0:%d" % (w, ch, (h - ch) // 2)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    ffmpeg("-i", src, "-vf", vf, "-c:v", "libx264", "-crf", "15", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", dst)
    return dst


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="MiniMax H3 from the command line")
    ap.add_argument("-p", "--prompt-file", required=True, help=".txt file holding the prompt")
    ap.add_argument("--img", nargs="*", default=[], help="reference images -> <Picture i>")
    ap.add_argument("--audio", nargs="*", default=[], help="standalone audios -> <Audio j>")
    ap.add_argument("--video", help="reference video -> <Video 1>")
    ap.add_argument("--video-audio", action="store_true",
                    help="also use the reference video's soundtrack (takes <Audio 1>)")
    ap.add_argument("--video-seconds", type=float, default=5.17,
                    help="how much of the reference video to use (default 5.17; more = much slower)")
    ap.add_argument("--seconds", type=float, default=5.17, help="output duration in seconds")
    ap.add_argument("--vertical", action="store_true", help="768x1344 instead of 1344x768")
    ap.add_argument("--steps", type=int, help="steps (default 10 with turbo, 20 with --quality)")
    ap.add_argument("--quality", action="store_true", help="no turbo LoRA, 20 steps")
    ap.add_argument("--ref-size", choices=["match", "max"], default="match",
                    help="'max' = better identity, several times slower")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1], help="one or more seeds")
    ap.add_argument("-o", "--out", help="path of the final mp4 (default: next to the inputs)")
    a = ap.parse_args()

    if not server_up():
        sys.exit("ComfyUI is not answering on %s. Start START_COMFY_H3_FAST.bat first." % SERVER)

    prompt = open(a.prompt_file, encoding="utf-8").read()
    w, h = (768, 1344) if a.vertical else (1344, 768)
    frames = frames_for(a.seconds)
    use_lora = not a.quality
    steps = a.steps if a.steps else (20 if a.quality else 10)

    print("preparing references...")
    imgs = [stage_image(x) for x in a.img]
    auds = [stage_audio(x) for x in a.audio]
    vid = stage_video(a.video, a.video_seconds) if a.video else None

    # default destination: next to the inputs
    if a.out:
        out_base = a.out
    else:
        anchor = (a.img + a.audio + ([a.video] if a.video else []) + [a.prompt_file])[0]
        folder = os.path.dirname(os.path.abspath(anchor))
        stem = os.path.splitext(os.path.basename(a.prompt_file))[0]
        out_base = os.path.join(folder, stem + ".mp4")

    print("mode:      %s" % ("ref2va" if (imgs or auds or vid) else "fl2va"))
    print("output:    %dx%d, %d frames (%.2f s at 24 fps)" % (w, h, frames, frames / 24.0))
    print("sampling:  %d steps, turbo LoRA %s, ref_image_size=%s"
          % (steps, "ON" if use_lora else "OFF", a.ref_size))
    print("refs:      %d images, %d audios, video=%s%s"
          % (len(imgs), len(auds), "yes" if vid else "no",
             " (+its soundtrack)" if (vid and a.video_audio) else ""))
    print("output to: %s" % out_base)
    print()

    for i, seed in enumerate(a.seeds):
        g, _ = build(prompt, w, h, frames, steps, seed, imgs, auds, vid,
                     a.video_audio, use_lora, a.ref_size)
        src = run_one(g, "seed %d" % seed)
        if not src:
            continue
        dst = out_base if len(a.seeds) == 1 else \
              "%s_seed%d%s" % (os.path.splitext(out_base)[0], seed, os.path.splitext(out_base)[1])
        export(src, dst, w, h)
        print("  -> %s" % dst)

    print("\ndone. To review without opening the videos: python sheet.py <file.mp4>")


if __name__ == "__main__":
    main()
