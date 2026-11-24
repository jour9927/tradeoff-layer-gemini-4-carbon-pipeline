#!/usr/bin/env python3
"""
Tradeoff Layer — Unattended Recurring Channel Daemon (`無人值守排程守護進程`).
Periodically scrapes public tech/AI/semiconductor feeds, deduplicates against all
previously generated episodes in `runs/history.json`, and executes the full
Long-Form + Paired Shorts production pipeline on RTX PRO 6000 or b850 (Tailscale SSH).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tradeoff_pipeline.orchestrator import run_pipeline
from tradeoff_pipeline.research_scraper import discover_topics

HISTORY_FILE = ROOT / "runs" / "published_history.json"


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return []


def save_history(items: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def run_once_unattended(worker: str = "auto", fps: int = 60, nfe_step: int = 24) -> Path:
    history = load_history()
    done_queries = {h.get("query", "").lower() for h in history}
    candidates = discover_topics(count=6)
    chosen = next((c for c in candidates if c["query"].lower() not in done_queries), None)
    if chosen is None and candidates:
        chosen = candidates[0]
    if chosen is None:
        raise RuntimeError("No candidate topics discovered from live public feeds")

    topic = chosen["lead_story"]["title"] or chosen["query"]
    deliverables = run_pipeline(
        topic=topic,
        queries=[chosen["query"], topic[:60]],
        worker=worker,
        nfe_step=nfe_step,
        fps=fps,
    )
    history.append({
        "query": chosen["query"],
        "topic": topic,
        "deliverables_dir": str(deliverables),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    save_history(history)
    return deliverables


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer Unattended Autonomous Production Scheduler")
    ap.add_argument("--worker", choices=["auto", "colab-rtx6000", "b850"], default="auto")
    ap.add_argument("--every-hours", type=float, default=0.0, help="If > 0, loops continuously every N hours")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--nfe-step", type=int, default=24)
    args = ap.parse_args()

    while True:
        out = run_once_unattended(worker=args.worker, fps=args.fps, nfe_step=args.nfe_step)
        print(f"[unattended_scheduler] Episode ready at: {out}", flush=True)
        if args.every_hours <= 0:
            break
        sleep_s = int(args.every_hours * 3600)
        print(f"[unattended_scheduler] Sleeping {sleep_s}s until next unattended episode...", flush=True)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
