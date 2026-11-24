#!/usr/bin/env python3
"""
Tradeoff Layer v3 — SOTA Flat-Cartoon Visual Set, Multi-Pose Cast & Neural Cutout Engine.

Replaces legacy SDXL-Base 1.0 + flood-fill with:
  1. `Tongyi-MAI/Z-Image-Turbo` (BF16 on RTX PRO 6000 96GB, 8–9 steps) for crisp, uniform black-ink
     2D flat cel-shaded cartoon sets, props, and consistent recurring episode characters — plus
     accurate native brand lettering (`SK hynix`, `SAMSUNG`, `NVIDIA`, `TSMC`, `AMD`).
     (Automatic fallback to SDXL if running on a legacy low-VRAM fallback node.)
  2. `ZhengPeng7/BiRefNet` neural dichotomous segmentation at 1024x1024 for studio-grade,
     halo-free RGBA alpha cutouts of multi-pose characters and movable props.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

STYLE_SET_PREFIX = (
    "Clean 2D flat cel-shaded comic cartoon background illustration, uniform crisp black ink outlines, "
    "solid flat pastel and vibrant color fills, zero painterly brush strokes, zero oil texture, "
    "Tintin ligne claire and modern YouTube animated tech explainer style, clean architectural geometry, "
    "no people in background: "
)

STYLE_CHAR_PREFIX = (
    "Single solo 2D flat cel-shaded comic cartoon character waist-up portrait centered alone on pure "
    "solid #FFFFFF white background with wide empty white margin on all sides, uniform bold clean black "
    "ink outlines, simple expressive oval black eyes, flat solid cel-shaded skin and clothing colors, "
    "both arms and hands clearly visible in expressive pose, modern YouTube animated business explainer "
    "cartoon style, one person only, no desk, no chair, no background scenery, no text: "
)

STYLE_PROP_PREFIX = (
    "Single isolated 2D flat cel-shaded comic cartoon object centered alone on pure solid #FFFFFF white "
    "background with wide empty white margin, uniform crisp bold black ink outlines, clean flat solid "
    "colors, modern tech explainer animation prop, no background scenery: "
)

STYLE_THUMB_PREFIX = (
    "High-impact 2D flat cel-shaded comic cartoon YouTube thumbnail illustration, bold thick black ink "
    "outlines, dramatic saturated colors, expressive comic style, clean composition: "
)

NEGATIVE_FALLBACK = (
    "photorealistic, 3d render, oil painting, watercolor, sketchy, crosshatching, monochrome, "
    "blurry, deformed hands, crowd, multiple people, clutter, watermark"
)

# Default recurring stage sets & multi-pose character cast for semiconductor / AI infrastructure episodes
DEFAULT_STAGE_SETS: dict[str, str] = {
    "set_nvidia_hq": "sunny modern Silicon Valley tech headquarters glass building facade with green trees, blue sky, white clouds, clean flat sign reading NVIDIA above glass entrance",
    "set_sk_cleanroom": "bright modern semiconductor cleanroom interior with large glass window, glowing yellow-white wafer bay, sleek lithography equipment, clean flat sign reading SK hynix on wall",
    "set_samsung_office": "sleek high-rise executive boardroom interior with floor-to-ceiling city window, polished conference wall, clean flat blue sign reading SAMSUNG on wall, no people",
    "set_amd_2013_lab": "warm 2013 hardware engineering lab interior with oscilloscopes on shelves, whiteboard, sunny window, red accent stripe on wall, no people",
    "set_tsmc_fab": "clean futuristic semiconductor packaging factory floor with overhead yellow robotic rail, silver silicon wafer pods, clean sign reading TSMC on wall, no people",
    "set_datacenter_hall": "wide modern AI cloud data center corridor with symmetrical rows of dark server racks, cool sky-blue floor reflection, overhead cable trays, no people",
    "set_sunny_office": "bright cheerful tech company office room with warm wooden cabinets, green potted plant, wall clock, large sunny window with green trees outside, no people",
    "set_courtyard_road": "sunny tech campus walkway with green lawn, leafy trees, modern glass office building in distance, blue sky with soft white clouds, no people",
}

DEFAULT_CAST_SPRITES: dict[str, str] = {
    "char_jensen_talk": "charismatic silver-haired East Asian tech CEO in his late 50s, neat swept-back silver-white hair, black rectangular glasses, iconic black leather zip jacket over black t-shirt, confident friendly smile, gesturing with open hands",
    "char_jensen_shock": "silver-haired East Asian tech CEO in his late 50s, neat silver-white hair, black rectangular glasses, black leather zip jacket, worried surprised expression with raised eyebrows and sweat drop, holding hands palms up",
    "char_sk_eng_talk": "focused Korean semiconductor lead engineer in his 40s, neat black hair, round black glasses, crisp white lab coat over royal-blue collared shirt, warm intelligent smile, explaining with hand gesture",
    "char_sk_eng_proud": "focused Korean semiconductor lead engineer in his 40s, neat black hair, round black glasses, crisp white lab coat over royal-blue collared shirt, proud cheerful smile giving thumbs up",
    "char_samsung_dismiss": "senior corporate executive in his 50s, sleek parted dark grey hair, sharp dark navy business suit with blue tie, skeptical unimpressed expression waving one hand dismissively",
    "char_samsung_sweat": "senior corporate executive in his 50s, sleek parted dark grey hair, sharp dark navy business suit with blue tie, alarmed stressed expression with sweat drop looking down at report",
    "char_amd_eng_talk": "earnest hardware architect in his late 30s, wavy brown hair, glasses, burgundy-red collared button shirt with rolled sleeves, friendly focused expression gesturing with hands",
    "char_tsmc_eng_talk": "precise Taiwanese packaging engineer in her late 30s, neat dark bob hair, glasses, light sky-blue cleanroom smock jacket, calm confident smile gesturing",
}

DEFAULT_PROP_SPRITES: dict[str, str] = {
    "prop_hbm_tower": "glowing 3D-stacked gold and sapphire-blue HBM microchip cube with visible layered silicon slices and tiny golden solder bumps at base",
    "prop_gpu_package": "sleek emerald-green and black AI GPU processor module on golden interposer board",
    "prop_silicon_wafer": "shiny iridescent rainbow-blue circular 300mm silicon wafer disc with fine grid pattern",
    "prop_server_rack": "modern matte-black AI data center server rack cabinet on small caster wheels with glowing cyan LED bars",
    "prop_thumb_hero": "comic cartoon close-up of a silver-haired tech CEO in black leather jacket holding an empty glowing green GPU socket while reaching toward a shining golden stacked HBM memory chip cube, bold black outlines",
}


def _fallback_cutout_rgba(pil_rgb: Image.Image) -> Image.Image:
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
    bright_neutral = (arr.min(axis=-1) > 222) & ((arr.max(axis=-1) - arr.min(axis=-1)) < 24)
    cand_bg = (dist < 36.0) | bright_neutral
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
    alpha_np = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0)))
    return _crop_rgba(np.dstack([arr.astype(np.uint8), alpha_np]))


def _crop_rgba(rgba: np.ndarray, pad: int = 12) -> Image.Image:
    h, w, _ = rgba.shape
    ys, xs = np.where(rgba[..., 3] > 28)
    if len(xs) > 100 and len(ys) > 100:
        x1, x2 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
        y1, y2 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
        rgba = rgba[y1:y2, x1:x2]
    return Image.fromarray(rgba, mode="RGBA")


class NeuralSegmenter:
    """Wraps ZhengPeng7/BiRefNet on CUDA for surgical halo-free RGBA cutouts."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model = None
        if device == "cuda":
            try:
                import torch
                from transformers import AutoModelForImageSegmentation

                self.model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet",
                    trust_remote_code=True,
                ).to("cuda").eval()
                self.model.half()
                print("[visual_engine] Loaded ZhengPeng7/BiRefNet neural matting on CUDA FP16.", flush=True)
            except Exception as exc:
                print(f"[visual_engine] BiRefNet load fallback to border flood-fill ({exc})", flush=True)
                self.model = None

    def cutout(self, pil_rgb: Image.Image) -> Image.Image:
        if self.model is None:
            return _fallback_cutout_rgba(pil_rgb)

        import torch
        rgb = pil_rgb.convert("RGB")
        orig_w, orig_h = rgb.size
        resized = rgb.resize((1024, 1024), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = ((arr - mean) / std).transpose(2, 0, 1)[None, ...]
        inp = torch.from_numpy(norm).to("cuda", dtype=torch.float16)

        with torch.inference_mode():
            preds = self.model(inp)[-1].sigmoid().float().cpu().numpy()[0, 0]

        mask_pil = Image.fromarray((np.clip(preds, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
        alpha = np.asarray(mask_pil, dtype=np.float32)
        # Keep strictly the single largest connected foreground component so detached chair/prop flecks vanish
        fg_bin = alpha > 90
        lbl, n_cc = ndimage.label(fg_bin)
        if n_cc > 1:
            sizes = ndimage.sum(fg_bin, lbl, index=range(1, n_cc + 1))
            largest_id = int(np.argmax(sizes)) + 1
            alpha = np.where(lbl == largest_id, alpha, 0.0)

        rgb_np = np.asarray(rgb, dtype=np.float32)
        # Defringe semi-transparent border pixels against #FFFFFF white background
        a_norm = np.clip(alpha / 255.0, 0.0, 1.0)[..., None]
        edge = (a_norm > 0.08) & (a_norm < 0.92)
        defringed = np.where(edge, np.clip((rgb_np - (1.0 - a_norm) * 255.0) / np.maximum(a_norm, 0.15), 0.0, 255.0), rgb_np)
        rgba = np.dstack([defringed.astype(np.uint8), alpha.astype(np.uint8)])
        if int((alpha > 128).sum()) < 2500:
            return _fallback_cutout_rgba(pil_rgb)
        return _crop_rgba(rgba)

    def close(self) -> None:
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_best_pipeline(preferred: str = "cuda") -> tuple[str, str, Any]:
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_gib = free_b / (1024**3)
        print(f"[visual_engine] GPU: {torch.cuda.get_device_name(0)} ({free_gib:.1f}/{total_b/(1024**3):.1f} GiB free)", flush=True)
        if free_gib >= 18.0:
            try:
                from diffusers import ZImagePipeline
                pipe = ZImagePipeline.from_pretrained(
                    "Tongyi-MAI/Z-Image-Turbo",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                ).to("cuda")
                pipe.set_progress_bar_config(disable=True)
                print("[visual_engine] Loaded SOTA Tongyi-MAI/Z-Image-Turbo (BF16 CUDA).", flush=True)
                return "cuda", "z_image_turbo", pipe
            except Exception as exc:
                print(f"[visual_engine] Z-Image-Turbo unavailable ({exc}); falling back to SDXL.", flush=True)

        if free_gib >= 8.0:
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            return "cuda", "sdxl_base", pipe

    from diffusers import AutoPipelineForText2Image
    torch.set_num_threads(min(36, max(4, (os.cpu_count() or 8) - 6)))
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16).to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return "cpu", "sdxl_turbo", pipe


def generate_illustrations_for_plan(
    plan_path: Path,
    output_dir: Path,
    device: str = "cuda",
    only_keys: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    t0 = time.time()

    tasks: list[dict[str, Any]] = []

    # 1. Stage sets (from plan `stage_sets` merged with defaults, both 16:9 and 9:16 if requested)
    stage_sets = {**DEFAULT_STAGE_SETS, **(plan.get("stage_sets") or {})}
    for idx, (skey, sprompt) in enumerate(stage_sets.items()):
        tasks.append({"key": skey, "prompt": sprompt, "aspect": "16:9", "seed": 3100 + idx * 31, "kind": "set"})
        tasks.append({"key": f"{skey}_9x16", "prompt": sprompt + ", vertical portrait framing", "aspect": "9:16", "seed": 4100 + idx * 31, "kind": "set"})

    # 2. Custom per-shot illustrated backgrounds if specified
    seen_keys = {t["key"] for t in tasks}
    for idx, sc in enumerate(plan.get("long_scenes", [])):
        bg_key = sc.get("bg_key") or ""
        prompt = sc.get("ai_image_prompt") or ""
        if prompt and bg_key and bg_key not in seen_keys:
            seen_keys.add(bg_key)
            tasks.append({"key": bg_key, "prompt": prompt, "aspect": "16:9", "seed": 5200 + idx * 17, "kind": "set"})

    # 3. Thumbnail artwork & hero cutout
    thumb = plan.get("thumbnail") or {}
    tasks.append({
        "key": "thumbnail_bg",
        "prompt": thumb.get("visual_prompt") or "dramatic comic close-up split background blue vs dark slate semiconductor battle",
        "aspect": "16:9",
        "seed": 1109,
        "kind": "thumb",
    })

    # 4. Recurring Cast Multi-Pose RGBA Sprites
    cast_defs = {**DEFAULT_CAST_SPRITES, **(plan.get("cast_sprites") or {})}
    for c_idx, (ckey, cprompt) in enumerate(cast_defs.items()):
        tasks.append({"key": ckey, "prompt": cprompt, "aspect": "1:1", "seed": 8200 + c_idx * 43, "kind": "char"})

    # 5. Movable Prop RGBA Sprites
    prop_defs = {**DEFAULT_PROP_SPRITES, **(plan.get("prop_sprites") or {})}
    for p_idx, (pkey, pprompt) in enumerate(prop_defs.items()):
        tasks.append({"key": pkey, "prompt": pprompt, "aspect": "1:1", "seed": 9300 + p_idx * 47, "kind": "prop"})

    if only_keys:
        tasks = [t for t in tasks if t["key"] in only_keys]

    # Filter out already-cached valid assets unless overwrite=True
    pending_tasks: list[dict[str, Any]] = []
    cached_assets: dict[str, str] = {}
    for item in tasks:
        out_png = output_dir / f"{item['key']}.png"
        sig_file = output_dir / f".{item['key']}.sig"
        sig = hashlib.sha1(f"v3|{item['kind']}|{item['aspect']}|{item['prompt']}".encode("utf-8")).hexdigest()[:14]
        if not overwrite and not only_keys and out_png.exists() and sig_file.exists() and sig_file.read_text().strip() == sig:
            cached_assets[item["key"]] = out_png.name
        else:
            item["sig"] = sig
            pending_tasks.append(item)

    generated: dict[str, str] = dict(cached_assets)
    use_dev = device
    mode_name = "cached"

    if pending_tasks:
        use_dev, mode_name, pipe = load_best_pipeline(preferred=device)
        raw_cutouts: list[tuple[dict[str, Any], Image.Image]] = []

        for i, item in enumerate(pending_tasks):
            t_item = time.time()
            if item["aspect"] == "16:9":
                gen_w, gen_h = (1280, 720) if mode_name == "z_image_turbo" else (1344, 768)
                target_w, target_h = (1920, 1080)
            elif item["aspect"] == "9:16":
                gen_w, gen_h = (720, 1280) if mode_name == "z_image_turbo" else (768, 1344)
                target_w, target_h = (1080, 1920)
            else:
                gen_w, gen_h = (1024, 1024)
                target_w, target_h = (1024, 1024)

            prefix = (
                STYLE_CHAR_PREFIX if item["kind"] == "char"
                else STYLE_PROP_PREFIX if item["kind"] == "prop"
                else STYLE_THUMB_PREFIX if item["kind"] == "thumb"
                else STYLE_SET_PREFIX
            )
            full_prompt = prefix + item["prompt"]
            gen = torch.Generator(device=use_dev).manual_seed(int(item["seed"]))

            with torch.inference_mode():
                if mode_name == "z_image_turbo":
                    raw_img = pipe(
                        prompt=full_prompt,
                        height=gen_h,
                        width=gen_w,
                        num_inference_steps=9,
                        guidance_scale=0.0,
                        generator=gen,
                    ).images[0]
                else:
                    steps = 24 if mode_name == "sdxl_base" else 4
                    cfg = 7.0 if mode_name == "sdxl_base" else 0.0
                    raw_img = pipe(
                        prompt=full_prompt[:420],
                        negative_prompt=NEGATIVE_FALLBACK,
                        height=gen_h,
                        width=gen_w,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=gen,
                    ).images[0]

            if item["kind"] in ("char", "prop"):
                raw_cutouts.append((item, raw_img))
                print(f"[visual_engine] ({i+1}/{len(pending_tasks)}) synth {item['key']} ({mode_name}) in {time.time()-t_item:.2f}s", flush=True)
            else:
                out_png = output_dir / f"{item['key']}.png"
                raw_img.resize((target_w, target_h), Image.Resampling.LANCZOS).save(str(out_png))
                (output_dir / f".{item['key']}.sig").write_text(item["sig"], encoding="utf-8")
                generated[item["key"]] = out_png.name
                print(f"[visual_engine] ({i+1}/{len(pending_tasks)}) saved {item['key']} ({target_w}x{target_h}) in {time.time()-t_item:.2f}s", flush=True)

        del pipe
        gc.collect()
        if use_dev == "cuda":
            torch.cuda.empty_cache()

        # Run neural BiRefNet alpha matting across all character & prop sprites
        if raw_cutouts:
            seg = NeuralSegmenter(device=use_dev)
            for item, raw_img in raw_cutouts:
                t_cut = time.time()
                rgba_img = seg.cutout(raw_img)
                out_png = output_dir / f"{item['key']}.png"
                rgba_img.save(str(out_png))
                (output_dir / f".{item['key']}.sig").write_text(item["sig"], encoding="utf-8")
                generated[item["key"]] = out_png.name
                print(f"[visual_engine] cutout {item['key']} -> {rgba_img.size[0]}x{rgba_img.size[1]} RGBA in {time.time()-t_cut:.2f}s", flush=True)
            seg.close()

    manifest_path = output_dir / "visual_assets_manifest.json"
    manifest = {
        "ok": True,
        "version": "v3",
        "device": use_dev,
        "model": mode_name,
        "count": len(generated),
        "elapsed_sec": round(time.time() - t0, 2),
        "assets": generated,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer v3 Z-Image-Turbo + BiRefNet Asset Engine")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-keys", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    only_set = {k.strip() for k in args.only_keys.split(",") if k.strip()} or None
    res = generate_illustrations_for_plan(
        plan_path=args.plan,
        output_dir=args.output_dir,
        device=args.device,
        only_keys=only_set,
        overwrite=args.overwrite,
    )
    print(f"[visual_engine] Ready: {res['count']} v3 assets ({res['model']}) in {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
