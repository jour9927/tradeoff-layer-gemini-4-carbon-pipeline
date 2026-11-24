#!/usr/bin/env python3
"""
Tradeoff Layer — Deliverable QA & Frame Extractor.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "20261124_sk_hynix_hbm"
    run_dir = ROOT / "runs" / run_id
    deliv = run_dir / "deliverables"
    qa_frames = run_dir / "qa_frames"
    qa_frames.mkdir(parents=True, exist_ok=True)

    for mp4_name in ["long_video_1080p60.mp4", "short_01_1080x1920_60fps.mp4"]:
        mp4 = deliv / mp4_name
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(mp4)],
            text=True,
        )
        d = json.loads(out)
        v = next(s for s in d["streams"] if s["codec_type"] == "video")
        a = next(s for s in d["streams"] if s["codec_type"] == "audio")
        lufs_res = subprocess.run(
            ["ffmpeg", "-t", "90", "-i", str(mp4), "-af", "ebur128", "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        lufs_lines = [l.strip() for l in lufs_res.stderr.splitlines() if "I:" in l or "LRA:" in l]
        lufs_summary = " | ".join(lufs_lines[-2:]) if lufs_lines else "n/a"
        dur_s = float(d["format"]["duration"])
        sz_mb = int(d["format"]["size"]) / 1048576.0
        print(f"=== {mp4_name} ===")
        print(
            f"Video: {v['codec_name']} {v['width']}x{v['height']} @ {v['r_frame_rate']}fps pix={v['pix_fmt']} "
            f"| Audio: {a['codec_name']} {a['sample_rate']}Hz ch={a['channels']}"
        )
        print(f"Duration: {dur_s:.2f}s | Size: {sz_mb:.2f} MB | Loudness: {lufs_summary}")

    long_tts = json.loads((run_dir / "tts_long" / "tts_alignment_report.json").read_text(encoding="utf-8"))
    for sc in long_tts["scenes"]:
        mid_t = sc["start_sec"] + min(2.6, sc["duration_sec"] * 0.5)
        out_jpg = qa_frames / f"qa_long_{sc['id']}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{mid_t:.2f}",
                "-i", str(deliv / "long_video_1080p60.mp4"),
                "-frames:v", "1", "-q:v", "2", str(out_jpg),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Long {sc['id']}: {sc['duration_sec']:.2f}s | words={len(sc['words'])} | ASR={sc['asr_transcript'][:90]}")

    short_tts = json.loads((run_dir / "tts_shorts" / "short_01" / "tts_alignment_report.json").read_text(encoding="utf-8"))
    for sc in short_tts["scenes"]:
        mid_t = sc["start_sec"] + min(2.2, sc["duration_sec"] * 0.45)
        out_jpg = qa_frames / f"qa_short_{sc['id']}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{mid_t:.2f}",
                "-i", str(deliv / "short_01_1080x1920_60fps.mp4"),
                "-frames:v", "1", "-q:v", "2", str(out_jpg),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Short {sc['id']}: {sc['duration_sec']:.2f}s | words={len(sc['words'])} | ASR={sc['asr_transcript'][:90]}")


if __name__ == "__main__":
    main()
