#!/usr/bin/env python3
"""
Tradeoff Layer — PyTorch Diffusion AI Illustration & Editorial Scene Asset Generator.
Runs on the remote RTX PRO 6000 worker (or b850 CUDA node) to generate authentic AI
flat cel-shaded vector backgrounds & metaphor illustrations matching @TradeoffLayer's
signature visual aesthetic (`yB9SC4HbH5o`, `2d883UK7p9E`, `KTY7TnGIVNY`).

Automatically checks free CUDA VRAM (`torch.cuda.mem_get_info`) so it uses CUDA when
available (>= 4.5 GiB free) and gracefully runs 4-step SDXL-Turbo across 48 AVX-512 vCPUs
when another session holds static GPU VRAM.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

TRADEOFF_STYLE_PREFIX = (
    "Clean 2D flat cel-shaded vector editorial animation background illustration in the style of "
    "Kurzgesagt and modern tech explainer animation, crisp dark navy ink outlines, smooth flat "
    "color blocks, subtle warm architectural lighting, clean geometric perspective, high clarity, "
    "minimalist background detail, zero text, no letters, no words, no watermark: "
)

NEGATIVE_PROMPT = (
    "text, letters, words, watermark, signature, blurry, photorealistic skin pores, messy sketch, "
    "3d render clutter, dark horror, deformed, noisy grain, low quality, typography"
)


def select_compute_device(preferred: str = "cuda", min_vram_gib: float = 4.5) -> tuple[str, Any]:
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            if (free_b / (1024**3)) >= min_vram_gib:
                return "cuda", torch.float16
            print(
                f"[visual_engine] GPU VRAM currently held by another session "
                f"({free_b/(1024**3):.2f} GiB free / {total_b/(1024**3):.1f} GiB); "
                f"using {os.cpu_count()}-vCPU bfloat16 SDXL-Turbo execution.",
                flush=True,
            )
        except Exception:
            pass
    n_threads = min(36, max(4, (os.cpu_count() or 8) - 6))
    torch.set_num_threads(n_threads)
    return "cpu", torch.bfloat16


def generate_illustrations_for_plan(
    plan_path: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    import torch
    from diffusers import AutoPipelineForText2Image
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    use_dev, dtype = select_compute_device(preferred=device)
    model_id = "stabilityai/sdxl-turbo"
    print(f"[visual_engine] Loading {model_id} on {use_dev} (dtype={dtype})...", flush=True)
    t0 = time.time()

    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if use_dev == "cuda":
        load_kwargs["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **load_kwargs).to(use_dev)
    pipe.set_progress_bar_config(disable=True)

    tasks: list[dict[str, Any]] = []
    # Thumbnail background
    thumb = plan.get("thumbnail") or {}
    if thumb.get("visual_prompt"):
        tasks.append({
            "key": "thumbnail_bg",
            "prompt": thumb["visual_prompt"],
            "aspect": "16:9",
            "seed": 1001,
        })

    # Long-form scenes that use AI cel-shaded backgrounds/illustrations
    for idx, sc in enumerate(plan.get("long_scenes", [])):
        vmode = sc.get("visual_mode", "vector_whiteboard_scene")
        prompt = sc.get("ai_image_prompt") or sc.get("headline") or ""
        if prompt and vmode in ("vector_whiteboard_scene", "editorial_illustration"):
            tasks.append({
                "key": f"long_{sc['id']}",
                "prompt": prompt,
                "aspect": "16:9",
                "seed": 2000 + idx * 19,
            })

    # Short scenes that use AI cel-shaded backgrounds/illustrations
    for s_idx, sh in enumerate(plan.get("shorts", [])):
        for c_idx, sc in enumerate(sh.get("scenes", [])):
            vmode = sc.get("visual_mode", "dark_metric_card")
            prompt = sc.get("ai_image_prompt") or sc.get("headline") or ""
            if prompt and vmode in ("vector_whiteboard_scene", "editorial_illustration"):
                tasks.append({
                    "key": f"{sh['id']}_{sc['id']}",
                    "prompt": prompt,
                    "aspect": "9:16",
                    "seed": 5000 + s_idx * 100 + c_idx * 23,
                })

    generated: dict[str, str] = {}
    for i, item in enumerate(tasks):
        out_png = output_dir / f"{item['key']}.png"
        if out_png.exists() and out_png.stat().st_size > 10_000:
            generated[item["key"]] = out_png.name
            continue

        if item["aspect"] == "16:9":
            gen_w, gen_h = (896, 512) if use_dev == "cuda" else (640, 384)
            target_w, target_h = (1920, 1080)
        else:
            gen_w, gen_h = (512, 896) if use_dev == "cuda" else (384, 640)
            target_w, target_h = (1080, 1920)

        full_prompt = TRADEOFF_STYLE_PREFIX + item["prompt"]
        gen = torch.Generator(device=use_dev).manual_seed(int(item["seed"]))
        t_item = time.time()
        with torch.inference_mode():
            img = pipe(
                prompt=full_prompt[:360],
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=4,
                guidance_scale=0.0,
                width=gen_w,
                height=gen_h,
                generator=gen,
            ).images[0]

        img_hi = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        img_hi.save(str(out_png), optimize=False)
        generated[item["key"]] = out_png.name
        print(
            f"[visual_engine] ({i+1}/{len(tasks)}) {item['key']} ({target_w}x{target_h}) in {time.time()-t_item:.2f}s",
            flush=True,
        )

    manifest = {
        "ok": True,
        "device": use_dev,
        "model": model_id,
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
    print(f"[visual_engine] Generated {res['count']} AI editorial illustrations in {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
