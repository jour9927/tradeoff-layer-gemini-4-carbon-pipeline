#!/usr/bin/env python3
"""
Tradeoff Layer v3 — Multi-Layer 60fps Video, Hook-Excerpt Vertical Short, Foley & Comic Thumbnail Compositor.

Key v3 Upgrades:
  1. Uses `sfx_engine.mix_foley_with_voice` for frame-accurate Foley transients with clean <-66 dB speech pauses
     (zero muddy synthetic drone).
  2. Supports @TradeoffLayer's authentic `hook_prefix_excerpt` Short pipeline: natively re-composes the
     Cold Open + Act I cliffhanger (`L001..L025`, ~120–145s) at 1080x1920 @ 60fps using the exact mastered
     studio voiceover WAVs from the Long-Form video, with zero burned-in KTV subtitle clutter.
  3. Generates @TradeoffLayer-style comic thumbnails (angled giant bold comic-outlined display headline +
     brand badge + neural-matted comic hero artwork).
  4. Strictly enforces the `alan-hub` anti-rendering hardware guard.
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

from tradeoff_pipeline.sfx_engine import mix_foley_with_voice

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
        "/content/tradeoff_layer_carbon/node22/bin/node",
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
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    )
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


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
        val = min(1.0, float(np.sqrt(np.mean(chunk**2))) * 4.2) if len(chunk) > 0 else 0.0
        env.append(round(val, 4))
    return env


def _write_foley_scored_wav(voice_wav: Path, out_wav: Path, scene: dict[str, Any], index: int) -> Path:
    with wave.open(str(voice_wav), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        voice = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

    mixed = mix_foley_with_voice(voice, sr=sr, scene=scene, index=index)
    pcm = (np.clip(mixed, -0.98, 0.98) * 32767.0).astype(np.int16)
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
    idx = int(job["index"])
    scene = job["scene"]

    scored_wav = out_mp4.with_suffix(".foley.wav")
    _write_foley_scored_wav(wav_path, scored_wav, scene=scene, index=idx)
    envelope = _compute_audio_envelope(wav_path, fps=fps)

    skia_job = {
        "scene": scene,
        "index": idx,
        "ttsMeta": job["tts_meta"],
        "wavPath": str(scored_wav),
        "visualDir": str(job["visual_dir"]),
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
    existing_mods = [p for p in extra_node_mods if os.path.exists(p)]
    if existing_mods:
        env["NODE_PATH"] = ":".join(existing_mods)

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


def write_srt_subtitles(scenes_tts: list[dict[str, Any]], out_srt: Path) -> None:
    entries: list[str] = []
    idx = 1
    for sc in scenes_tts:
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
    """
    Replicates @TradeoffLayer's signature comic YouTube thumbnail style (`PAY UP`, `STILL ALIVE`, `TPU VS GPU`):
    bold diagonal comic split + giant angled black-stroked comic headline + clean brand badge + neural cutouts.
    """
    w, h = 1280, 720
    bg_p = visual_dir / "thumbnail_bg.png"
    if bg_p.exists():
        img = Image.open(bg_p).convert("RGB").resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    else:
        img = Image.new("RGBA", (w, h), (24, 43, 89, 255))

    # Crisp comic dark-navy angled left panel (`STILL ALIVE` / `TPU VS GPU` split style)
    wedge = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    wdraw = ImageDraw.Draw(wedge, "RGBA")
    wdraw.polygon([(0, 0), (575, 0), (465, h), (0, h)], fill=(15, 23, 42, 246))
    wdraw.line([(575, 0), (465, h)], fill=(255, 255, 255, 255), width=9)
    img = Image.alpha_composite(img, wedge)

    # Optional crisp stacked HBM cube accent bridging the comic divider line
    hbm_p = visual_dir / "prop_hbm_tower.png"
    if hbm_p.exists():
        hb_im = Image.open(hbm_p).convert("RGBA").resize((250, 250), Image.Resampling.LANCZOS)
        img.alpha_composite(hb_im, dest=(375, 435))

    thumb = plan.get("thumbnail") or {}
    headline = (thumb.get("headline") or "SOLD OUT").upper()
    accent_word = (thumb.get("accent_word") or "SK hynix").strip()

    # Angled giant comic text layer (`-5 deg` tilt, thick black comic outline!)
    txt_layer = Image.new("RGBA", (620, 400), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(txt_layer, "RGBA")
    words = headline.split()
    f_giant = _find_font(bold=True, size=124 if len(headline) <= 10 else 98)

    cy = 48
    for i, wd in enumerate(words[:2]):
        fill_col = (255, 255, 255, 255) if i == 0 else (253, 224, 71, 255)
        tdraw.text((34, cy), wd, font=f_giant, fill=fill_col, stroke_width=14, stroke_fill=(9, 13, 24, 255))
        cy += 128

    rotated_txt = txt_layer.rotate(5.0, resample=Image.Resampling.BICUBIC, expand=True)
    img.alpha_composite(rotated_txt, dest=(14, 26))

    # Clean brand signplate badge (`SK hynix` / `ARM` / `ASML` style!)
    draw = ImageDraw.Draw(img, "RGBA")
    if accent_word:
        f_brand = _find_font(bold=True, size=58)
        bb = draw.textbbox((0, 0), accent_word, font=f_brand)
        bw = (bb[2] - bb[0]) + 64
        bh = 98
        bx, by = 42, 365
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=18, fill=(255, 255, 255, 255), outline=(15, 23, 42, 255), width=6)
        draw.text((bx + 32, by + 15), accent_word, font=f_brand, fill=(220, 38, 38, 255))

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
            "mode": sh.get("mode", "hook_prefix_excerpt"),
            "title": sh.get("title"),
            "description": (
                f"{sh.get('hook_angle', '')}\n\n"
                f"Watch the full 8-minute animated story on @TradeoffLayer: \"{plan.get('title')}\"\n\n"
                f"Sources:\n" + "\n".join(source_lines[:4])
            ),
            "related_long_video_title": plan.get("title"),
        })

    meta = {
        "channel": "@TradeoffLayer",
        "version": "v3",
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
        "version": "v3",
        "retrieved_at": plan.get("research_metadata", {}).get("retrieved_at"),
        "sources": sources,
        "shot_count": len(plan.get("long_scenes", [])),
        "scene_citations": [
            {
                "scene_id": sc["id"],
                "template": sc.get("template") or sc.get("animation_type"),
                "citation_note": sc.get("citation_note") or sc.get("citation_fact"),
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
    shorts_tts_root: Path | None,
    output_dir: Path,
    fps: int = 60,
    max_workers: int = 16,
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

    workers = min(max_workers, max(2, (os.cpu_count() or 8) - 2))
    long_scenes = plan.get("long_scenes", [])
    print(f"[video_composer v3] Rendering {len(long_scenes)} Long-Form shots (1920x1080@{fps}fps) across {workers} workers...", flush=True)

    long_jobs: list[dict[str, Any]] = []
    long_part_paths: list[Path] = []
    for idx, sc in enumerate(long_scenes):
        sid = sc["id"]
        tmeta = long_tts_map[sid]
        seg_mp4 = tmp_scenes_dir / f"long_{sid}.mp4"
        long_part_paths.append(seg_mp4)
        long_jobs.append({
            "index": idx,
            "scene": sc,
            "tts_meta": tmeta,
            "wav_path": str(long_tts_dir / tmeta["wav_file"]),
            "visual_dir": str(visual_dir),
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
    write_srt_subtitles(long_tts["scenes"], output_dir / "long_video.srt")

    # Render Paired Traffic-Driving Vertical Short(s)
    short_outputs: list[dict[str, Any]] = []
    for sh in plan.get("shorts", []):
        sh_id = sh["id"]
        mode = sh.get("mode", "hook_prefix_excerpt")
        if mode == "hook_prefix_excerpt":
            n_ex = min(int(sh.get("excerpt_shot_count", 25)), len(long_scenes))
            sh_scenes = long_scenes[:n_ex]
            sh_tts_list = [long_tts_map[sc["id"]] for sc in sh_scenes]
            sh_wav_dir = long_tts_dir
        else:
            assert shorts_tts_root is not None
            sh_wav_dir = shorts_tts_root / sh_id
            sh_rep = json.loads((sh_wav_dir / "tts_alignment_report.json").read_text(encoding="utf-8"))
            sh_map = {s["id"]: s for s in sh_rep["scenes"]}
            sh_scenes = sh.get("scenes", [])
            sh_tts_list = [sh_map[sc["id"]] for sc in sh_scenes]

        print(f"[video_composer v3] Rendering Vertical Short '{sh_id}' ({len(sh_scenes)} shots, mode={mode}, 1080x1920@{fps}fps)...", flush=True)
        sh_jobs: list[dict[str, Any]] = []
        sh_parts: list[Path] = []
        for idx, (sc, tmeta) in enumerate(zip(sh_scenes, sh_tts_list)):
            sid = sc["id"]
            seg_mp4 = tmp_scenes_dir / f"{sh_id}_{sid}.mp4"
            sh_parts.append(seg_mp4)
            sh_jobs.append({
                "index": idx,
                "scene": sc,
                "tts_meta": tmeta,
                "wav_path": str(sh_wav_dir / tmeta["wav_file"]),
                "visual_dir": str(visual_dir),
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
        write_srt_subtitles(sh_tts_list, output_dir / f"{sh_id}.srt")
        sh_dur = round(sum(float(t["duration_sec"]) for t in sh_tts_list), 2)
        short_outputs.append({"file": sh_mp4.name, "shots": len(sh_scenes), "duration_sec": sh_dur, "mode": mode})

    thumb_png = generate_thumbnail(plan, visual_dir, output_dir / "thumbnail_1280x720.png")
    build_youtube_metadata_and_citations(plan, long_tts, output_dir)

    shutil.rmtree(tmp_scenes_dir, ignore_errors=True)

    summary = {
        "ok": True,
        "engine": "skia-v3-multilayer-60fps",
        "elapsed_sec": round(time.time() - t0, 2),
        "long_video": long_mp4.name,
        "long_shots": len(long_scenes),
        "long_duration_sec": long_total_dur,
        "avg_shot_sec": round(long_total_dur / max(1, len(long_scenes)), 2),
        "shorts": short_outputs,
        "thumbnail": thumb_png.name,
    }
    (output_dir / "render_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer v3 Multi-Layer 60fps Video & Shorts Compositor")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--visual-dir", required=True, type=Path)
    ap.add_argument("--long-tts-dir", required=True, type=Path)
    ap.add_argument("--shorts-tts-root", default=None, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--max-workers", type=int, default=16)
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
    print(
        f"[video_composer v3] Rendered {res['long_shots']} shots ({res['long_duration_sec']}s, avg {res['avg_shot_sec']}s/shot) "
        f"in {res['elapsed_sec']}s -> {res['long_video']} + {res['shorts']}"
    )


if __name__ == "__main__":
    main()
