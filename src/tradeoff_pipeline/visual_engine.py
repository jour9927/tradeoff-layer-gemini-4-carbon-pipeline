#!/usr/bin/env python3
"""
Tradeoff Layer — PyTorch Diffusion AI Illustration & Character Cutout Generator.
Optimized for the dedicated RTX PRO 6000 96GB GPU node (`stabilityai/stable-diffusion-xl-base-1.0`
at native 1344x768 / 768x1344 in FP16), with automatic fallback to `sdxl-turbo` if VRAM < 8 GiB.
Releases all GPU memory immediately on exit so shared nodes never retain idle VRAM.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

TRADEOFF_STYLE_PREFIX = (
    "Clean 2D flat cel-shaded vector editorial animation background illustration in the style of "
    "modern YouTube tech explainers, crisp dark navy ink outlines, smooth flat vibrant color blocks, "
    "warm daylight architectural studio lighting, clean geometric perspective, high clarity, "
    "zero text, no letters, no words, no watermark: "
)

CHAR_STYLE_PREFIX = (
    "Single isolated 2D flat cel-shaded cartoon upper-body waist-up character portrait centered on "
    "pure solid #FFFFFF white background, crisp black ink outlines, clean flat vector colors, "
    "friendly expressive face, modern tech explainer cartoon style, no legs, no background scenery, "
    "no text, no watermark: "
)

NEGATIVE_PROMPT = (
    "text, letters, words, watermark, signature, blurry, photorealistic skin pores, messy sketch, "
    "3d render clutter, dark horror, deformed, noisy grain, low quality, typography, extra limbs"
)


def cutout_white_bg_to_rgba(pil_rgb: Image.Image) -> Image.Image:
    """Flood-fills border-connected white background into clean anti-aliased transparent RGBA."""
    arr = np.asarray(pil_rgb.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    corners = np.concatenate([
        arr[:14, :14].reshape(-1, 3),
        arr[:14, -14:].reshape(-1, 3),
        arr[-14:, :14].reshape(-1, 3),
        arr[-14:, -14:].reshape(-1, 3),
    ], axis=0)
    bg_col = np.median(corners, axis=0)
    dist = np.linalg.norm(arr - bg_col[None, None, :], axis=-1)
    bright_neutral = (arr.min(axis=-1) > 222) & ((arr.max(axis=-1) - arr.min(axis=-1)) < 24)
    cand_bg = (dist < 34.0) | bright_neutral

    labeled, _ = ndimage.label(cand_bg)
    border_labels = (
        set(np.unique(labeled[0, :]))
        | set(np.unique(labeled[-1, :]))
        | set(np.unique(labeled[:, 0]))
        | set(np.unique(labeled[:, -1]))
    )
    border_labels.discard(0)
    outer_bg = np.isin(labeled, list(border_labels))

    fg = (~outer_bg).astype(np.float32)
    fg = ndimage.binary_opening(fg > 0.5, iterations=1).astype(np.float32)
    alpha_img = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0))
    alpha_np = np.asarray(alpha_img)

    rgba = np.dstack([arr.astype(np.uint8), alpha_np])
    ys, xs = np.where(alpha_np > 25)
    if len(xs) > 80 and len(ys) > 80:
        pad = 12
        x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
        y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
        rgba = rgba[y1:y2, x1:x2]
    return Image.fromarray(rgba, mode="RGBA")


def load_best_pipeline(preferred: str = "cuda") -> tuple[str, str, Any]:
    import torch
    from diffusers import AutoPipelineForText2Image

    if preferred == "cuda" and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_gib = free_b / (1024**3)
        print(f"[visual_engine] CUDA GPU detected: {torch.cuda.get_device_name(0)} ({free_gib:.1f}/{total_b/(1024**3):.1f} GiB free)", flush=True)
        if free_gib >= 8.0:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                ).to("cuda")
                pipe.set_progress_bar_config(disable=True)
                return "cuda", "sdxl_base", pipe
            except Exception as exc:
                print(f"[visual_engine] SDXL-Base load fallback ({exc}), using sdxl-turbo on CUDA", flush=True)
        if free_gib >= 4.0:
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            return "cuda", "sdxl_turbo", pipe

    n_threads = min(36, max(4, (os.cpu_count() or 8) - 6))
    torch.set_num_threads(n_threads)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.bfloat16,
    ).to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return "cpu", "sdxl_turbo", pipe


def generate_illustrations_for_plan(
    plan_path: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    t0 = time.time()

    use_dev, mode_name, pipe = load_best_pipeline(preferred=device)
    print(f"[visual_engine] Active generator: {mode_name} on {use_dev}", flush=True)

    tasks: list[dict[str, Any]] = []
    thumb = plan.get("thumbnail") or {}
    if thumb.get("visual_prompt"):
        tasks.append({"key": "thumbnail_bg", "prompt": thumb["visual_prompt"], "aspect": "16:9", "seed": 1001, "kind": "bg"})

    for idx, sc in enumerate(plan.get("long_scenes", [])):
        vmode = sc.get("visual_mode", "vector_whiteboard_scene")
        prompt = sc.get("ai_image_prompt") or sc.get("headline") or ""
        if prompt and vmode in ("vector_whiteboard_scene", "editorial_illustration"):
            tasks.append({"key": f"long_{sc['id']}", "prompt": prompt, "aspect": "16:9", "seed": 2000 + idx * 19, "kind": "bg"})

    for s_idx, sh in enumerate(plan.get("shorts", [])):
        for c_idx, sc in enumerate(sh.get("scenes", [])):
            vmode = sc.get("visual_mode", "dark_metric_card")
            prompt = sc.get("ai_image_prompt") or sc.get("headline") or ""
            if prompt and vmode in ("vector_whiteboard_scene", "editorial_illustration"):
                tasks.append({"key": f"{sh['id']}_{sc['id']}", "prompt": prompt, "aspect": "9:16", "seed": 5000 + s_idx * 100 + c_idx * 23, "kind": "bg"})

    # Also generate @TradeoffLayer signature cel-shaded character busts on pure white for RGBA cutouts!
    char_archetypes = {
        "char_engineer": "young male semiconductor engineer with curly brown hair, rectangular black glasses, light grey zip hoodie over dark navy t-shirt, friendly smile",
        "char_ceo": "tech executive with silver-grey hair, black rectangular glasses, sleek black leather collar jacket over dark t-shirt, confident warm smile",
        "char_analyst": "tech researcher with neat short hair, thin glasses, light sky-blue collared polo shirt, friendly expression",
    }
    for c_idx, (ckey, cprompt) in enumerate(char_archetypes.items()):
        tasks.append({"key": ckey, "prompt": cprompt, "aspect": "1:1", "seed": 8100 + c_idx * 31, "kind": "char"})

    generated: dict[str, str] = {}
    steps = 22 if mode_name == "sdxl_base" else 4
    cfg = 6.5 if mode_name == "sdxl_base" else 0.0

    for i, item in enumerate(tasks):
        out_png = output_dir / f"{item['key']}.png"
        if item["aspect"] == "16:9":
            gen_w, gen_h = (1344, 768) if mode_name == "sdxl_base" else ((896, 512) if use_dev == "cuda" else (640, 384))
            target_w, target_h = (1920, 1080)
        elif item["aspect"] == "9:16":
            gen_w, gen_h = (768, 1344) if mode_name == "sdxl_base" else ((512, 896) if use_dev == "cuda" else (384, 640))
            target_w, target_h = (1080, 1920)
        else:
            gen_w, gen_h = (1024, 1024) if mode_name == "sdxl_base" else (512, 512)
            target_w, target_h = (512, 512)

        prefix = CHAR_STYLE_PREFIX if item["kind"] == "char" else TRADEOFF_STYLE_PREFIX
        full_prompt = (prefix + item["prompt"])[:380]
        gen = torch.Generator(device=use_dev).manual_seed(int(item["seed"]))
        t_item = time.time()
        with torch.inference_mode():
            img = pipe(
                prompt=full_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=gen_w,
                height=gen_h,
                generator=gen,
            ).images[0]

        if item["kind"] == "char":
            cutout = cutout_white_bg_to_rgba(img)
            cutout.save(str(out_png))
        else:
            img_hi = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
            img_hi.save(str(out_png))

        generated[item["key"]] = out_png.name
        print(f"[visual_engine] ({i+1}/{len(tasks)}) {item['key']} ({item['kind']}) in {time.time()-t_item:.2f}s", flush=True)

    del pipe
    gc.collect()
    if use_dev == "cuda":
        torch.cuda.empty_cache()

    manifest = {
        "ok": True,
        "device": use_dev,
        "model": mode_name,
        "count": len(generated),
        "elapsed_sec": round(time.time() - t0, 2),
        "assets": generated,
    }
    (output_dir / "visual_assets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer PyTorch Diffusion Visual Asset Generator")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res = generate_illustrations_for_plan(plan_path=args.plan, output_dir=args.output_dir, device=args.device)
    print(f"[visual_engine] Generated {res['count']} assets ({res['model']} on {res['device']}) in {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
