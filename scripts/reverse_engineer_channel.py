#!/usr/bin/env python3
"""
Tradeoff Layer — Comprehensive Channel Reverse-Engineering & Style Bible Extractor.
Analyzes all downloaded long-form videos and Shorts from @TradeoffLayer:
  - Video/audio codecs, resolution, FPS, loudness (LUFS)
  - Transcript pacing (WPM), hook patterns, analogy patterns, closing Tradeoff structure
  - Chapter durations and title patterns
  - Short-to-Long video mapping and Shorts structure
  - Extracts visual keyframes from representative videos and Shorts for style inspection
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "reference_channel"
VIDEOS_DIR = REF_DIR / "videos"
SHORTS_DIR = REF_DIR / "shorts"
FRAMES_DIR = REF_DIR / "frames"
REPORT_JSON = REF_DIR / "channel_style_analysis.json"


def parse_vtt_clean(vtt_path: Path) -> str:
    raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if "-->" in s or re.match(r"^\d+$", s):
            continue
        clean = re.sub(r"<[^>]+>", "", s).strip()
        clean = re.sub(r"\s+", " ", clean)
        if clean and (not lines or lines[-1] != clean):
            # Avoid rolling caption duplication
            if lines and clean.startswith(lines[-1]):
                lines[-1] = clean
            elif lines and lines[-1].endswith(clean):
                continue
            else:
                lines.append(clean)
    return " ".join(lines)


def ffprobe_media(mp4_path: Path) -> dict:
    out = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(mp4_path)],
        text=True,
    )
    data = json.loads(out)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "video_codec": v.get("codec_name"),
        "profile": v.get("profile"),
        "width": v.get("width"),
        "height": v.get("height"),
        "r_frame_rate": v.get("r_frame_rate"),
        "pix_fmt": v.get("pix_fmt"),
        "audio_codec": a.get("codec_name"),
        "sample_rate": a.get("sample_rate"),
        "channels": a.get("channels"),
        "duration_sec": round(float(data.get("format", {}).get("duration", 0)), 2),
        "size_mb": round(int(data.get("format", {}).get("size", 0)) / (1024 * 1024), 2),
    }


def extract_sample_frames(mp4_path: Path, out_dir: Path, timestamps: list[float]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ts in timestamps:
        out_png = out_dir / f"{mp4_path.stem}_t{int(ts):04d}s.jpg"
        if not out_png.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(ts), "-i", str(mp4_path),
                    "-frames:v", "1", "-q:v", "2", str(out_png),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if out_png.exists():
            saved.append(str(out_png))
    return saved


def analyze_collection(folder: Path, kind: str) -> list[dict]:
    items = []
    for mp4 in sorted(folder.glob("*.mp4")):
        vid_id = mp4.stem
        info_path = folder / f"{vid_id}.info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        vtts = sorted(folder.glob(f"{vid_id}*.vtt"))
        transcript = parse_vtt_clean(vtts[0]) if vtts else ""
        word_count = len(transcript.split())
        probe = ffprobe_media(mp4)
        dur_min = max(probe["duration_sec"] / 60.0, 0.01)
        chapters = info.get("chapters") or []
        items.append({
            "id": vid_id,
            "kind": kind,
            "title": info.get("title"),
            "upload_date": info.get("upload_date"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "probe": probe,
            "word_count": word_count,
            "wpm": round(word_count / dur_min, 1) if word_count else 0,
            "chapter_count": len(chapters),
            "avg_chapter_sec": round(probe["duration_sec"] / len(chapters), 1) if chapters else None,
            "chapters": chapters,
            "tags": info.get("tags") or [],
            "description": info.get("description") or "",
            "transcript_opening": " ".join(transcript.split()[:90]),
            "transcript_closing": " ".join(transcript.split()[-90:]),
            "transcript_full": transcript,
        })
    return items


def main() -> None:
    videos = analyze_collection(VIDEOS_DIR, "long")
    shorts = analyze_collection(SHORTS_DIR, "short")

    # Extract sample keyframes from 3 long-form videos and 3 shorts
    sampled_frames = {"long": {}, "short": {}}
    for vid_id in ("0vZCvW7IzmA", "2d883UK7p9E", "yB9SC4HbH5o"):
        mp4 = VIDEOS_DIR / f"{vid_id}.mp4"
        if mp4.exists():
            sampled_frames["long"][vid_id] = extract_sample_frames(
                mp4, FRAMES_DIR / "long", [3, 14, 32, 65, 120, 210, 310]
            )
    for sh_id in ("bmFUEo6A_1U", "OtNoAtR33o0", "2xcY7K8GL3M", "MnOOAvY_2NQ"):
        mp4 = SHORTS_DIR / f"{sh_id}.mp4"
        if mp4.exists():
            sampled_frames["short"][sh_id] = extract_sample_frames(
                mp4, FRAMES_DIR / "short", [1, 5, 12, 22, 35]
            )

    summary = {
        "video_count": len(videos),
        "short_count": len(shorts),
        "avg_long_wpm": round(sum(v["wpm"] for v in videos if v["wpm"]) / max(1, sum(1 for v in videos if v["wpm"])), 1),
        "avg_short_wpm": round(sum(s["wpm"] for s in shorts if s["wpm"]) / max(1, sum(1 for s in shorts if s["wpm"])), 1),
        "sampled_frames": sampled_frames,
        "videos": videos,
        "shorts": shorts,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Analyzed {len(videos)} long videos and {len(shorts)} shorts.")
    print(f"Avg Long WPM: {summary['avg_long_wpm']} | Avg Short WPM: {summary['avg_short_wpm']}")
    for v in videos:
        p = v["probe"]
        print(f"[LONG]  {v['id']} | {p['width']}x{p['height']}@{p['r_frame_rate']} | {p['duration_sec']}s | {v['wpm']} WPM | ch={v['chapter_count']} | {v['title']}")
    for s in shorts:
        p = s["probe"]
        print(f"[SHORT] {s['id']} | {p['width']}x{p['height']}@{p['r_frame_rate']} | {p['duration_sec']}s | {s['wpm']} WPM | {s['title']}")


if __name__ == "__main__":
    main()
