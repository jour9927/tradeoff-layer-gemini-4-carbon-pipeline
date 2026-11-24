#!/usr/bin/env python3
"""
Tradeoff Layer — True 60fps 2D Vector, Physics & Rigged Sprite Video Compositor (`v2`).
Delegates per-scene 60fps frame synthesis to `skia_animator.js` (Google Skia `@napi-rs/canvas`
streaming raw RGBA buffers directly into FFmpeg) across parallel multi-core workers on the
dedicated RTX PRO 6000 node (`Untitled10.ipynb`) or `b850` Tailscale SSH worker.

Strictly enforces the `alan-hub` anti-rendering hardware guard.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import socket
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SKIA_SCRIPT = Path(__file__).resolve().parent / "skia_animator.js"


def _guard_not_alan_hub(allow_override: bool = False) -> None:
    host = socket.gethostname().lower()
    if ("alan-hub" in host or "alan_hub" in host) and not allow_override:
        raise RuntimeError(
            "Policy guard: Video rendering on local 'alan-hub' is strictly disabled! "
            "Dispatch rendering to the dedicated G4 RTX PRO 6000 worker or b850 node."
        )


def _find_node_binary() -> str:
    for cand in [
        shutil.which("node"),
        "/content/tradeoff_layer_carbon/opt/node/bin/node",
        "/usr/local/bin/node",
        "/usr/bin/node",
    ]:
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("Node.js binary not found for skia_animator.js")


def _find_font(bold: bool = False, mono: bool = False, size: int = 36) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        ]
        if mono
        else [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    )
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines = [words[0]]
    for w in words[1:]:
        cand = f"{lines[-1]} {w}"
        bb = draw.textbbox((0, 0), cand, font=font)
        if (bb[2] - bb[0]) <= max_w:
            lines[-1] = cand
        else:
            lines.append(w)
    return lines


def _compute_audio_envelope(wav_path: Path, fps: int) -> list[float]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        samples = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    total_frames = max(1, int(math.ceil((n / sr) * fps)))
    hop = max(1, int(sr / fps))
    env: list[float] = []
    for f in range(total_frames):
        chunk = samples[f * hop : (f + 1) * hop]
        val = min(1.0, float(np.sqrt(np.mean(chunk**2))) * 4.5) if len(chunk) > 0 else 0.0
        env.append(round(val, 4))
    return env


def _add_subtle_sound_design(voice_wav: Path, out_wav: Path, is_short: bool = False) -> Path:
    with wave.open(str(voice_wav), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        voice = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

    t = np.arange(n, dtype=np.float32) / sr
    f0 = 110.0 if not is_short else 130.81
    pad = (
        0.55 * np.sin(2 * np.pi * f0 * t)
        + 0.30 * np.sin(2 * np.pi * (f0 * 1.5) * t)
        + 0.15 * np.sin(2 * np.pi * (f0 * 2.0) * t)
    ) * (0.012 * (0.75 + 0.25 * np.sin(2 * np.pi * 0.12 * t)))

    sw_len = min(n, int(sr * 0.18))
    if sw_len > 10:
        st = np.linspace(0.0, 1.0, sw_len, dtype=np.float32)
        whoosh = np.sin(2 * np.pi * (190 + 240 * st) * (st * 0.18)) * np.hanning(sw_len) * 0.038
        pad[:sw_len] += whoosh

    mixed = np.clip(voice + pad, -0.98, 0.98)
    pcm = (mixed * 32767.0).astype(np.int16)
    with wave.open(str(out_wav), "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(sr)
        out_wf.writeframes(pcm.tobytes())
    return out_wav


def _render_single_scene_worker(job: dict[str, Any]) -> str:
    node_bin = _find_node_binary()
    wav_path = Path(job["wav_path"])
    out_mp4 = Path(job["out_mp4"])
    fps = int(job["fps"])
    is_short = bool(job["is_short"])

    scored_wav = out_mp4.with_suffix(".scored.wav")
    _add_subtle_sound_design(wav_path, scored_wav, is_short=is_short)
    envelope = _compute_audio_envelope(wav_path, fps=fps)

    skia_job = {
        "scene": job["scene"],
        "index": int(job["index"]),
        "ttsMeta": job["tts_meta"],
        "wavPath": str(scored_wav),
        "visualDir": str(job["visual_dir"]),
        "bgFile": job.get("bg_file"),
        "outMp4": str(out_mp4),
        "w": int(job["w"]),
        "h": int(job["h"]),
        "fps": fps,
        "isShort": is_short,
        "envelope": envelope,
    }
    job_json = out_mp4.with_suffix(".skia_job.json")
    job_json.write_text(json.dumps(skia_job), encoding="utf-8")

    env = dict(os.environ)
    extra_node_mods = [
        "/content/tradeoff_layer_carbon/animator_env/node_modules",
        str(Path.home() / "tradeoff-layer-gemini-4-carbon" / "node_modules"),
    ]
    env["NODE_PATH"] = ":".join(p for p in extra_node_mods if os.path.exists(p))

    try:
        subprocess.run([node_bin, str(SKIA_SCRIPT), str(job_json)], check=True, env=env)
    finally:
        scored_wav.unlink(missing_ok=True)
        job_json.unlink(missing_ok=True)
    return str(out_mp4)


def _format_srt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt_subtitles(tts_report: dict[str, Any], out_srt: Path) -> None:
    entries: list[str] = []
    idx = 1
    for sc in tts_report.get("scenes", []):
        words = sc.get("words") or []
        for i in range(0, len(words), 8):
            chunk = words[i : i + 8]
            st = chunk[0]["global_start"]
            et = chunk[-1]["global_end"]
            txt = " ".join(w["word"] for w in chunk)
            entries.append(f"{idx}\n{_format_srt_ts(st)} --> {_format_srt_ts(et)}\n{txt}\n")
            idx += 1
    out_srt.write_text("\n".join(entries), encoding="utf-8")


def generate_thumbnail(plan: dict[str, Any], visual_dir: Path, out_png: Path) -> Path:
    w, h = 1280, 720
    bg_p = visual_dir / "thumbnail_bg.png"
    if bg_p.exists():
        img = Image.open(bg_p).convert("RGB").resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    else:
        img = Image.new("RGBA", (w, h), (186, 230, 253, 255))

    grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grad, "RGBA")
    for x in range(w):
        a = int(195 * max(0.0, 1.0 - (x / (w * 0.68))))
        gdraw.line([(x, 0), (x, h)], fill=(12, 18, 32, a))
    img = Image.alpha_composite(img, grad)

    draw = ImageDraw.Draw(img, "RGBA")
    thumb = plan.get("thumbnail") or {}
    headline = (thumb.get("headline") or plan.get("title", "THE TRADEOFF"))[:28].upper()
    subbadge = (thumb.get("subbadge") or "")[:18]

    f_big = _find_font(bold=True, size=90)
    f_badge = _find_font(bold=True, mono=True, size=42)
    lines = _wrap_text(draw, headline, f_big, 730)
    cy = 148
    for ln in lines[:3]:
        draw.text((66, cy + 5), ln, font=f_big, fill=(0, 0, 0, 210))
        draw.text((62, cy), ln, font=f_big, fill=(255, 255, 255, 255))
        cy += 100

    if subbadge:
        bb = draw.textbbox((0, 0), subbadge, font=f_badge)
        bw = (bb[2] - bb[0]) + 48
        draw.rounded_rectangle([62, cy + 20, 62 + bw, cy + 92], radius=16, fill=(239, 68, 68, 248))
        draw.text((86, cy + 32), subbadge, font=f_badge, fill=(255, 255, 255, 255))

    # Composite full-color character sprite + prop sprite on right
    char_p = visual_dir / "char_engineer.png"
    prop_p = visual_dir / "prop_chip_stack.png"
    if char_p.exists():
        cb = Image.open(char_p).convert("RGBA")
        ch_h = 340
        ch_w = int(ch_h * (cb.width / max(1, cb.height)))
        cb_r = cb.resize((ch_w, ch_h), Image.Resampling.LANCZOS)
        # Stick legs beneath character bust
        cx, hip_y, foot_y = 1045, 545, 698
        ink = (20, 24, 33, 255)
        draw.line([(cx - 26, hip_y), (cx - 26, foot_y)], fill=ink, width=6)
        draw.line([(cx + 26, hip_y), (cx + 26, foot_y)], fill=ink, width=6)
        draw.pieslice([cx - 42, foot_y - 12, cx - 10, foot_y + 12], 180, 360, fill=ink)
        draw.pieslice([cx + 10, foot_y - 12, cx + 42, foot_y + 12], 180, 360, fill=ink)
        img.alpha_composite(cb_r, dest=(cx - ch_w // 2, hip_y - ch_h + 18))
    if prop_p.exists():
        pb = Image.open(prop_p).convert("RGBA").resize((175, 175), Image.Resampling.LANCZOS)
        img.alpha_composite(pb, dest=(815, 435))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(out_png), quality=95)
    return out_png


def build_youtube_metadata_and_citations(
    plan: dict[str, Any],
    long_tts: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    seo = plan.get("seo") or {}
    tts_by_id = {s["id"]: s for s in long_tts.get("scenes", [])}

    chapters_lines = []
    for i, sc in enumerate(plan.get("long_scenes", [])):
        ch = (sc.get("chapter_title") or "").strip()
        if i == 0 and not ch:
            ch = "The hook"
        if ch and sc["id"] in tts_by_id:
            st = int(tts_by_id[sc["id"]]["start_sec"])
            mm, ss = divmod(st, 60)
            chapters_lines.append(f"{mm}:{ss:02d} {ch}")

    sources = plan.get("research_metadata", {}).get("sources", [])
    source_lines = [
        f"[{s['id']}] {s['publisher']} ({s['date']}) — {s['title']}\n    {s['url']}"
        for s in sources
    ]

    bullets = "\n".join(f"- {b}" for b in (seo.get("learning_bullets") or []))
    description = (
        f"{seo.get('summary_hook', plan.get('core_paradox', ''))}\n\n"
        f"You'll learn:\n{bullets}\n\n"
        f"CHAPTERS\n" + "\n".join(chapters_lines) + "\n\n"
        f"SOURCES & PUBLIC DATA REFERENCES\n" + "\n".join(source_lines) + "\n"
    )

    shorts_meta = []
    for sh in plan.get("shorts", []):
        shorts_meta.append({
            "id": sh["id"],
            "title": sh.get("title"),
            "description": (
                f"{sh.get('hook_angle', '')}\n\n"
                f"Full animated breakdown on @TradeoffLayer: \"{plan.get('title')}\"\n\n"
                f"Sources:\n" + "\n".join(source_lines[:4])
            ),
            "related_long_video_title": plan.get("title"),
        })

    meta = {
        "channel": "@TradeoffLayer",
        "long_video": {
            "title": plan.get("title"),
            "description": description,
            "tags": seo.get("tags") or [],
            "chapters": chapters_lines,
        },
        "shorts": shorts_meta,
    }
    (output_dir / "youtube_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "description.txt").write_text(description, encoding="utf-8")

    audit = {
        "title": plan.get("title"),
        "retrieved_at": plan.get("research_metadata", {}).get("retrieved_at"),
        "sources": sources,
        "scene_citations": [
            {
                "scene_id": sc["id"],
                "animation_type": sc.get("animation_type"),
                "citation_fact": sc.get("citation_fact"),
                "sources": sc.get("resolved_sources", []),
            }
            for sc in plan.get("long_scenes", [])
        ],
    }
    (output_dir / "citations_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def concat_scene_mp4s(scene_mp4s: list[Path], out_mp4: Path) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    concat_txt = out_mp4.with_suffix(".concat.txt")
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_mp4s) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "copy",
            "-af", "loudnorm=I=-14.0:TP=-1.5:LRA=6.5",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_mp4),
        ],
        check=True,
    )
    concat_txt.unlink(missing_ok=True)
    return out_mp4


def render_all(
    plan_path: Path,
    visual_dir: Path,
    long_tts_dir: Path,
    shorts_tts_root: Path,
    output_dir: Path,
    fps: int = 60,
    max_workers: int = 12,
    allow_local_override: bool = False,
) -> dict[str, Any]:
    _guard_not_alan_hub(allow_override=allow_local_override)
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_scenes_dir = output_dir / "_scene_segments"
    tmp_scenes_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    long_tts = json.loads((long_tts_dir / "tts_alignment_report.json").read_text(encoding="utf-8"))
    long_tts_map = {s["id"]: s for s in long_tts["scenes"]}
    long_total_dur = float(long_tts["total_duration_sec"])

    workers = min(max_workers, max(2, (os.cpu_count() or 4) // 2))
    print(f"[video_composer] Animating Long-Form (1920x1080@{fps}fps Skia 2D Engine) across {workers} workers...", flush=True)

    long_jobs: list[dict[str, Any]] = []
    long_part_paths: list[Path] = []
    for idx, sc in enumerate(plan.get("long_scenes", [])):
        sid = sc["id"]
        tmeta = long_tts_map[sid]
        seg_mp4 = tmp_scenes_dir / f"long_{sid}.mp4"
        long_part_paths.append(seg_mp4)
        bg_name = f"long_{sid}.png"
        long_jobs.append({
            "index": idx,
            "scene": sc,
            "tts_meta": tmeta,
            "wav_path": str(long_tts_dir / tmeta["wav_file"]),
            "visual_dir": str(visual_dir),
            "bg_file": bg_name if (visual_dir / bg_name).exists() else None,
            "out_mp4": str(seg_mp4),
            "w": 1920,
            "h": 1080,
            "fps": fps,
            "is_short": False,
        })

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_render_single_scene_worker, long_jobs))

    long_mp4 = output_dir / "long_video_1080p60.mp4"
    concat_scene_mp4s(long_part_paths, long_mp4)
    write_srt_subtitles(long_tts, output_dir / "long_video.srt")

    short_outputs: list[str] = []
    short_anim_cycle = [
        "dark_cross_section_meter",
        "rain_buckets_analogy",
        "chip_3d_flip_stack",
        "dark_gpu_partition_split",
        "character_walk_handoff",
    ]
    for sh in plan.get("shorts", []):
        sh_id = sh["id"]
        sh_tts_dir = shorts_tts_root / sh_id
        sh_tts = json.loads((sh_tts_dir / "tts_alignment_report.json").read_text(encoding="utf-8"))
        sh_map = {s["id"]: s for s in sh_tts["scenes"]}

        print(f"[video_composer] Animating Vertical Short '{sh_id}' (1080x1920@{fps}fps Skia 2D Engine)...", flush=True)
        sh_jobs: list[dict[str, Any]] = []
        sh_parts: list[Path] = []
        for idx, sc in enumerate(sh.get("scenes", [])):
            sid = sc["id"]
            if not sc.get("animation_type"):
                sc = {**sc, "animation_type": short_anim_cycle[idx % len(short_anim_cycle)]}
            tmeta = sh_map[sid]
            seg_mp4 = tmp_scenes_dir / f"{sh_id}_{sid}.mp4"
            sh_parts.append(seg_mp4)
            bg_name = f"{sh_id}_{sid}.png"
            sh_jobs.append({
                "index": idx,
                "scene": sc,
                "tts_meta": tmeta,
                "wav_path": str(sh_tts_dir / tmeta["wav_file"]),
                "visual_dir": str(visual_dir),
                "bg_file": bg_name if (visual_dir / bg_name).exists() else None,
                "out_mp4": str(seg_mp4),
                "w": 1080,
                "h": 1920,
                "fps": fps,
                "is_short": True,
            })

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_render_single_scene_worker, sh_jobs))

        sh_mp4 = output_dir / f"{sh_id}_1080x1920_60fps.mp4"
        concat_scene_mp4s(sh_parts, sh_mp4)
        write_srt_subtitles(sh_tts, output_dir / f"{sh_id}.srt")
        short_outputs.append(sh_mp4.name)

    thumb_png = generate_thumbnail(plan, visual_dir, output_dir / "thumbnail_1280x720.png")
    build_youtube_metadata_and_citations(plan, long_tts, output_dir)

    for p in tmp_scenes_dir.glob("*"):
        p.unlink(missing_ok=True)
    tmp_scenes_dir.rmdir()

    summary = {
        "ok": True,
        "engine": "skia-2d-60fps",
        "elapsed_sec": round(time.time() - t0, 2),
        "long_video": long_mp4.name,
        "long_duration_sec": long_total_dur,
        "shorts": short_outputs,
        "thumbnail": thumb_png.name,
    }
    (output_dir / "render_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer Skia 60fps 2D Animation Video & Shorts Compositor")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--visual-dir", required=True, type=Path)
    ap.add_argument("--long-tts-dir", required=True, type=Path)
    ap.add_argument("--shorts-tts-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--allow-local-override", action="store_true")
    args = ap.parse_args()

    res = render_all(
        plan_path=args.plan,
        visual_dir=args.visual_dir,
        long_tts_dir=args.long_tts_dir,
        shorts_tts_root=args.shorts_tts_root,
        output_dir=args.output_dir,
        fps=args.fps,
        max_workers=args.max_workers,
        allow_local_override=args.allow_local_override,
    )
    print(f"[video_composer] Animated & rendered in {res['elapsed_sec']}s -> {res['long_video']} + {res['shorts']}")


if __name__ == "__main__":
    main()
