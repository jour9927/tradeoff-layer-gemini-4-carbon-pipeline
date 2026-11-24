#!/usr/bin/env python3
"""
Tradeoff Layer — Full-Color Flat-Vector Background, Character Cutout & Hardware Prop Generator.
Optimized for the dedicated G4 · RTX PRO 6000 96GB node (`Untitled10.ipynb`) using SDXL-Base 1.0 FP16.

Enforces @TradeoffLayer's exact bright flat-pastel cartoon aesthetic (`sony_intro`, `tpu_intro`):
  - Full-color cel-shaded vector backgrounds (sky blue, mint green, warm light-oak wood, crisp outlines)
  - Full-color waist-up cartoon character sprites on pure #FFFFFF white -> clean RGBA alpha cutouts
  - Full-color hardware/analogy prop sprites on pure #FFFFFF white -> clean RGBA alpha cutouts
  - Automatic HSV saturation quality gate (re-rolls seed if SDXL ever outputs monochrome/greyscale)
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

TRADEOFF_BG_PREFIX = (
    "Bright 2D flat vector cartoon background illustration, educational explainer animation style, "
    "vibrant sky blue and mint green and warm light-oak wood pastel palette, clean solid flat color fills, "
    "crisp thin dark navy outlines, sunny cheerful daylight, clean minimal background scenery, "
    "Kurzgesagt and modern YouTube animated explainer style, full color, zero text, no words: "
)

CHAR_SPRITE_PREFIX = (
    "Single solo 2D flat cartoon character upper-body waist-up portrait centered alone on pure solid "
    "#FFFFFF white background with wide empty white margin on all sides, warm peach skin tone, colorful "
    "vibrant shirt and jacket, clean uniform black ink outline, smooth flat cel-shaded colors, friendly "
    "expressive eyes and smile, modern 2D animated explainer cartoon style, one person only, no legs, "
    "no desk, no chair, no background scenery, zero text: "
)

PROP_SPRITE_PREFIX = (
    "Single isolated vibrant full-color 2D flat vector cartoon object icon centered on pure solid "
    "#FFFFFF white background with wide empty white margin, crisp dark navy ink outline, clean flat "
    "cel-shaded colors, modern tech explainer animation prop, no background, zero text, no letters: "
)

NEGATIVE_PROMPT = (
    "black and white, monochrome, greyscale, grayscale, crosshatching, pencil sketch, charcoal, "
    "manga screentone, dark moody shadows, orange teal painterly concept art, oil painting, "
    "photorealistic, 3d render clutter, crowd, multiple people, two people, group, audience, collage, "
    "poster frame, colored background, room background, text, letters, words, watermark, blurry, extra limbs, chair, desk"
)


def cutout_white_bg_to_rgba(pil_rgb: Image.Image) -> tuple[Image.Image, float]:
    """Flood-fills border-connected white background into smooth anti-aliased RGBA sprite + returns bg ratio."""
    arr = np.asarray(pil_rgb.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    corners = np.concatenate([
        arr[:16, :16].reshape(-1, 3),
        arr[:16, -16:].reshape(-1, 3),
        arr[-16:, :16].reshape(-1, 3),
        arr[-16:, -16:].reshape(-1, 3),
    ], axis=0)
    bg_col = np.median(corners, axis=0)
    dist = np.linalg.norm(arr - bg_col[None, None, :], axis=-1)
    bright_neutral = (arr.min(axis=-1) > 220) & ((arr.max(axis=-1) - arr.min(axis=-1)) < 26)
    cand_bg = (dist < 38.0) | bright_neutral

    labeled, _ = ndimage.label(cand_bg)
    border_labels = (
        set(np.unique(labeled[0, :]))
        | set(np.unique(labeled[-1, :]))
        | set(np.unique(labeled[:, 0]))
        | set(np.unique(labeled[:, -1]))
    )
    border_labels.discard(0)
    outer_bg = np.isin(labeled, list(border_labels))
    bg_ratio = float(outer_bg.mean())

    fg = (~outer_bg).astype(np.float32)
    fg = ndimage.binary_opening(fg > 0.5, iterations=1).astype(np.float32)
    # Keep largest connected foreground component(s) so stray background flecks are removed
    fg_lbl, n_fg = ndimage.label(fg > 0.5)
    if n_fg > 1:
        sizes = ndimage.sum(fg > 0.5, fg_lbl, index=range(1, n_fg + 1))
        max_sz = float(np.max(sizes)) if len(sizes) else 0.0
        keep_ids = [idx + 1 for idx, sz in enumerate(sizes) if sz >= max_sz * 0.12]
        fg = np.isin(fg_lbl, keep_ids).astype(np.float32)

    alpha_img = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.1))
    alpha_np = np.asarray(alpha_img)

    rgba = np.dstack([arr.astype(np.uint8), alpha_np])
    ys, xs = np.where(alpha_np > 28)
    if len(xs) > 80 and len(ys) > 80:
        pad = 10
        x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
        y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
        rgba = rgba[y1:y2, x1:x2]
    return Image.fromarray(rgba, mode="RGBA"), bg_ratio


def _foreground_saturation(pil_img: Image.Image) -> float:
    hsv = np.asarray(pil_img.convert("HSV"), dtype=np.float32) / 255.0
    if pil_img.mode == "RGBA":
        alpha = np.asarray(pil_img)[..., 3] > 60
        if alpha.sum() > 100:
            return float(hsv[..., 1][alpha].mean())
    return float(hsv[..., 1].mean())


def load_best_pipeline(preferred: str = "cuda") -> tuple[str, str, Any]:
    import torch
    from diffusers import AutoPipelineForText2Image

    if preferred == "cuda" and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_gib = free_b / (1024**3)
        print(f"[visual_engine] CUDA GPU: {torch.cuda.get_device_name(0)} ({free_gib:.1f}/{total_b/(1024**3):.1f} GiB free)", flush=True)
        if free_gib >= 8.0:
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            return "cuda", "sdxl_base", pipe
        if free_gib >= 4.0:
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            return "cuda", "sdxl_turbo", pipe

    torch.set_num_threads(min(36, max(4, (os.cpu_count() or 8) - 6)))
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
    only_keys: set[str] | None = None,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    t0 = time.time()

    use_dev, mode_name, pipe = load_best_pipeline(preferred=device)
    steps = 24 if mode_name == "sdxl_base" else 4
    cfg = 7.0 if mode_name == "sdxl_base" else 0.0

    tasks: list[dict[str, Any]] = []
    thumb = plan.get("thumbnail") or {}
    tasks.append({
        "key": "thumbnail_bg",
        "prompt": thumb.get("visual_prompt") or "sunny modern semiconductor campus exterior with blue glass windows and green trees",
        "aspect": "16:9",
        "seed": 1101,
        "kind": "bg",
    })

    # Clean flat-pastel cartoon stage backgrounds used by character & whiteboard animated scenes
    default_stage_bgs = {
        "stage_courtyard": "sunny modern tech company courtyard with sky blue glass building facade, bright green trees, white clouds in blue sky",
        "stage_cleanroom": "bright cheerful semiconductor lab interior with large window, warm light wooden workbench, sky blue wall, glowing wafer display",
        "stage_datacenter": "bright clean modern cloud data center corridor with sky blue and white server racks, glowing cyan status LEDs, bright floor",
        "stage_studio": "bright sunny modern tech presenter room with floor-to-ceiling windows, green potted plants, light wood floor, soft pastel colors",
    }
    for idx, (bkey, bprompt) in enumerate(default_stage_bgs.items()):
        tasks.append({"key": bkey, "prompt": bprompt, "aspect": "16:9", "seed": 3100 + idx * 29, "kind": "bg"})

    for idx, sc in enumerate(plan.get("long_scenes", [])):
        prompt = sc.get("ai_image_prompt") or ""
        if prompt:
            tasks.append({"key": f"long_{sc['id']}", "prompt": prompt, "aspect": "16:9", "seed": 2100 + idx * 19, "kind": "bg"})

    for s_idx, sh in enumerate(plan.get("shorts", [])):
        for c_idx, sc in enumerate(sh.get("scenes", [])):
            prompt = sc.get("ai_image_prompt") or ""
            if prompt:
                tasks.append({"key": f"{sh['id']}_{sc['id']}", "prompt": prompt, "aspect": "9:16", "seed": 5100 + s_idx * 100 + c_idx * 23, "kind": "bg"})

    # Full-color @TradeoffLayer character torso cutouts (warm skin + colorful outfits, NO chairs/desks!)
    char_archetypes = {
        "char_engineer": "single solo young male tech engineer with curly brown hair, warm peach skin, black glasses, bright royal-blue zip hoodie over white t-shirt, friendly smile",
        "char_ceo": "single solo senior male semiconductor architect with neat silver hair, warm peach skin, black glasses, bright emerald-green zip jacket over navy shirt, friendly smile",
        "char_analyst": "single solo female semiconductor researcher with neat bob dark-brown hair, warm peach skin, bright cobalt-blue polo shirt, cheerful expression",
    }
    for c_idx, (ckey, cprompt) in enumerate(char_archetypes.items()):
        tasks.append({"key": ckey, "prompt": cprompt, "aspect": "1:1", "seed": 8205 + c_idx * 37, "kind": "char"})

    # Full-color hardware prop cutouts carried/handed off by characters!
    prop_archetypes = {
        "prop_chip_stack": "glowing golden and sapphire-blue 3D stacked HBM memory microchip cube with tiny copper pins",
        "prop_gpu_board": "sleek green and gold AI GPU processor circuit board module with heatsink fins",
        "prop_wafer": "shiny iridescent rainbow-blue circular silicon semiconductor wafer disc with grid pattern",
    }
    for p_idx, (pkey, pprompt) in enumerate(prop_archetypes.items()):
        tasks.append({"key": pkey, "prompt": pprompt, "aspect": "1:1", "seed": 9105 + p_idx * 41, "kind": "prop"})

    if only_keys:
        tasks = [t for t in tasks if t["key"] in only_keys]

    generated: dict[str, str] = {}
    for i, item in enumerate(tasks):
        out_png = output_dir / f"{item['key']}.png"
        if item["aspect"] == "16:9":
            gen_w, gen_h = (1344, 768) if mode_name == "sdxl_base" else (896, 512)
            target_w, target_h = (1920, 1080)
        elif item["aspect"] == "9:16":
            gen_w, gen_h = (768, 1344) if mode_name == "sdxl_base" else (512, 896)
            target_w, target_h = (1080, 1920)
        else:
            gen_w, gen_h = (1024, 1024) if mode_name == "sdxl_base" else (512, 512)
            target_w, target_h = (512, 512)

        prefix = (
            CHAR_SPRITE_PREFIX if item["kind"] == "char"
            else PROP_SPRITE_PREFIX if item["kind"] == "prop"
            else TRADEOFF_BG_PREFIX
        )
        full_prompt = (prefix + item["prompt"])[:410]

        best_img = None
        best_score = -1.0
        best_sat = 0.0
        best_bg_ratio = 1.0
        max_tries = 5 if item["kind"] in ("char", "prop") else 2
        t_item = time.time()

        for attempt in range(max_tries):
            seed_try = int(item["seed"]) + attempt * 137
            gen = torch.Generator(device=use_dev).manual_seed(seed_try)
            with torch.inference_mode():
                raw_img = pipe(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=gen_w,
                    height=gen_h,
                    generator=gen,
                ).images[0]

            if item["kind"] in ("char", "prop"):
                cand, bg_ratio = cutout_white_bg_to_rgba(raw_img)
                sat = _foreground_saturation(cand)
                isolated_ok = 0.25 <= bg_ratio <= 0.92
                score = sat + (1.0 if isolated_ok else 0.0)
            else:
                cand = raw_img
                bg_ratio = 1.0
                sat = _foreground_saturation(cand)
                isolated_ok = True
                score = sat

            if score > best_score:
                best_score = score
                best_sat = sat
                best_bg_ratio = bg_ratio
                best_img = cand
            if sat >= 0.16 and isolated_ok:
                break

        assert best_img is not None
        if item["kind"] in ("char", "prop"):
            best_img.save(str(out_png))
        else:
            best_img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS).save(str(out_png))

        generated[item["key"]] = out_png.name
        print(
            f"[visual_engine] ({i+1}/{len(tasks)}) {item['key']} ({item['kind']}, sat={best_sat:.2f}, bg={best_bg_ratio:.2f}) in {time.time()-t_item:.2f}s",
            flush=True,
        )

    del pipe
    gc.collect()
    if use_dev == "cuda":
        torch.cuda.empty_cache()

    manifest_path = output_dir / "visual_assets_manifest.json"
    existing_assets: dict[str, str] = {}
    if manifest_path.exists():
        try:
            existing_assets = json.loads(manifest_path.read_text(encoding="utf-8")).get("assets", {})
        except Exception:
            existing_assets = {}
    existing_assets.update(generated)

    manifest = {
        "ok": True,
        "device": use_dev,
        "model": mode_name,
        "count": len(existing_assets),
        "elapsed_sec": round(time.time() - t0, 2),
        "assets": existing_assets,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer Full-Color Flat-Vector & Cutout Asset Generator")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-keys", default="", help="Optional comma-separated asset keys to regenerate")
    args = ap.parse_args()

    only_set = {k.strip() for k in args.only_keys.split(",") if k.strip()} or None
    res = generate_illustrations_for_plan(
        plan_path=args.plan,
        output_dir=args.output_dir,
        device=args.device,
        only_keys=only_set,
    )
    print(f"[visual_engine] Generated {len(only_set) if only_set else res['count']} full-color vector & sprite assets in {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
