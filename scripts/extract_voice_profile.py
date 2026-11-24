#!/usr/bin/env python3
"""
Tradeoff Layer — Clean Multi-Style Voice Profile Extractor.
Extracts pristine 24kHz mono studio reference clips + exact VTT transcripts
from the channel owner's authorized reference videos for PyTorch Voice Cloning (F5-TTS).
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "reference_channel" / "videos"
OUT_DIR = ROOT / "assets" / "voice_profile"


def parse_vtt_cues(vtt_path: Path) -> list[dict]:
    text = vtt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text)
    cues = []
    ts_re = re.compile(r"(\d+):(\d+):(\d+\.\d+)\s*-->\s*(\d+):(\d+):(\d+\.\d+)")

    def to_sec(h: str, m: str, s: str) -> float:
        return int(h) * 3600 + int(m) * 60 + float(s)

    for blk in blocks:
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        m = None
        idx_line = 0
        for i, l in enumerate(lines):
            m = ts_re.search(l)
            if m:
                idx_line = i
                break
        if not m:
            continue
        st = to_sec(m.group(1), m.group(2), m.group(3))
        et = to_sec(m.group(4), m.group(5), m.group(6))
        if et - st < 0.4:
            continue
        body = " ".join(lines[idx_line + 1 :])
        if "<c>" in body:
            continue
        clean = html.unescape(re.sub(r"<[^>]+>", "", body)).replace("\xa0", " ").strip()
        clean = re.sub(r"\s+", " ", clean)
        if clean and (not cues or cues[-1]["text"] != clean):
            cues.append({"start": st, "end": et, "text": clean})
    return cues


def extract_normalized_wav(src_mp4: Path, start: float, duration: float, dst_wav: Path) -> dict:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(src_mp4),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-af",
            "highpass=f=65,loudnorm=I=-16.0:TP=-1.5:LRA=7.0",
            "-c:a",
            "pcm_s16le",
            str(dst_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with wave.open(str(dst_wav), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        arr = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * 0.05)
    frames = arr[: (len(arr) // hop) * hop].reshape(-1, hop)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-7)
    p5, p90 = float(np.percentile(rms_db, 5)), float(np.percentile(rms_db, 90))
    return {
        "duration_sec": round(n / sr, 2),
        "sample_rate": sr,
        "noise_floor_db": round(p5, 1),
        "speech_p90_db": round(p90, 1),
        "snr_db": round(p90 - p5, 1),
    }


def build_window_from_cues(cues: list[dict], min_start: float, target_dur: float = 9.5) -> tuple[float, float, str]:
    for i, c in enumerate(cues):
        if c["start"] < min_start:
            continue
        acc = [c["text"]]
        end = c["end"]
        j = i + 1
        while j < len(cues) and (end - c["start"]) < target_dur:
            # Ensure contiguous speech without huge music gaps
            if cues[j]["start"] - end > 1.2:
                break
            acc.append(cues[j]["text"])
            end = cues[j]["end"]
            j += 1
        dur = end - c["start"]
        joined = " ".join(acc)
        # Prefer sentences ending cleanly and 7.5s <= dur <= 11.5s (ideal for F5-TTS!)
        if 7.5 <= dur <= 11.8 and joined[-1:] in (".", "?", "!"):
            return c["start"], dur, joined
    raise RuntimeError(f"Could not find clean cue window after {min_start}s")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles = []

    # 1. Narrative flagship tone from 2d883UK7p9E (Sony Sensor — manual clean en-US captions!)
    vtt_sony = VIDEOS_DIR / "2d883UK7p9E.en-US.vtt"
    mp4_sony = VIDEOS_DIR / "2d883UK7p9E.mp4"
    cues_sony = parse_vtt_cues(vtt_sony)

    for style_name, min_t in [("narrative", 10.0), ("analytical", 92.0), ("hook", 0.2)]:
        st, dur, txt = build_window_from_cues(cues_sony, min_t, target_dur=9.2)
        wav_path = OUT_DIR / f"ref_{style_name}.wav"
        txt_path = OUT_DIR / f"ref_{style_name}.txt"
        stats = extract_normalized_wav(mp4_sony, st, dur, wav_path)
        txt_path.write_text(txt, encoding="utf-8")
        profiles.append({
            "style": style_name,
            "source_video": "2d883UK7p9E",
            "start_sec": round(st, 2),
            "wav_file": wav_path.name,
            "txt_file": txt_path.name,
            "ref_text": txt,
            **stats,
        })

    # 2. Fast crisp Short/Explainer tone from 0vZCvW7IzmA (AI Costs Explained — manual en-US captions!)
    vtt_ai = VIDEOS_DIR / "0vZCvW7IzmA.en-US.vtt"
    mp4_ai = VIDEOS_DIR / "0vZCvW7IzmA.mp4"
    cues_ai = parse_vtt_cues(vtt_ai)
    st, dur, txt = build_window_from_cues(cues_ai, 5.0, target_dur=8.5)
    wav_path = OUT_DIR / "ref_short_punchy.wav"
    txt_path = OUT_DIR / "ref_short_punchy.txt"
    stats = extract_normalized_wav(mp4_ai, st, dur, wav_path)
    txt_path.write_text(txt, encoding="utf-8")
    profiles.append({
        "style": "short_punchy",
        "source_video": "0vZCvW7IzmA",
        "start_sec": round(st, 2),
        "wav_file": wav_path.name,
        "txt_file": txt_path.name,
        "ref_text": txt,
        **stats,
    })

    manifest = {
        "channel": "@TradeoffLayer",
        "default_long_style": "narrative",
        "default_short_style": "short_punchy",
        "profiles": profiles,
    }
    (OUT_DIR / "voice_profile_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
