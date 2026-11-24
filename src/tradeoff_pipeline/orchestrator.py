#!/usr/bin/env python3
"""
Tradeoff Layer — Fully Unattended End-to-End Video & Shorts Production Orchestrator.

Master scripts and run state ALWAYS live locally first (`runs/<run_id>/`), while heavy
compute (LLM scriptwriting when no cloud key is set, PyTorch F5-TTS voice cloning,
Whisper word-level forced alignment, PyTorch SDXL-Turbo editorial illustration generation,
and 60fps 1080p Long-Form + 1080x1920 Shorts rendering) is dispatched to:
  1. Primary: Temporary High-Speed RTX PRO 6000 Colab Worker (`colab-rtx6000`)
  2. Fallback / Permanent Node: `b850` over Tailscale SSH (`alan-eagle@b850-ubuntu`)

Never executes heavy video rendering on local `alan-hub`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from remote_client import CONN_FILE, ColabWorkerClient, sync_to_b850
from tradeoff_pipeline.research_scraper import build_research_dossier, discover_topics


def _select_worker(preferred: str = "auto") -> str:
    if preferred in ("colab-rtx6000", "b850"):
        return preferred
    if CONN_FILE.exists():
        try:
            c = ColabWorkerClient.from_config(CONN_FILE)
            h = c.health()
            if h.get("ok"):
                return "colab-rtx6000"
        except Exception:
            pass
    return "b850"


def _write_tts_manifests_from_plan(plan: dict[str, Any], manifests_dir: Path) -> dict[str, Path]:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    long_manifest = {
        "variant": "long",
        "default_style": "narrative",
        "scenes": [
            {
                "id": sc["id"],
                "voice_style": sc.get("voice_style", "narrative"),
                "speed": float(sc.get("speed", 1.0)),
                "narration": sc["narration"],
            }
            for sc in plan.get("long_scenes", [])
        ],
    }
    lp = manifests_dir / "long_tts_manifest.json"
    lp.write_text(json.dumps(long_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    out["long"] = lp

    for sh in plan.get("shorts", []):
        sh_id = sh["id"]
        sh_manifest = {
            "variant": "short",
            "default_style": "short_punchy",
            "scenes": [
                {
                    "id": sc["id"],
                    "voice_style": sc.get("voice_style", "short_punchy"),
                    "speed": float(sc.get("speed", 1.04)),
                    "narration": sc["narration"],
                }
                for sc in sh.get("scenes", [])
            ],
        }
        sp = manifests_dir / f"{sh_id}_tts_manifest.json"
        sp.write_text(json.dumps(sh_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        out[sh_id] = sp

    return out


def run_pipeline(
    topic: str = "",
    queries: list[str] | None = None,
    run_id: str = "",
    worker: str = "auto",
    preauthored_plan: Path | None = None,
    long_scenes: int = 10,
    nfe_step: int = 32,
    fps: int = 60,
) -> Path:
    t0 = time.time()
    chosen_worker = _select_worker(worker)
    if not run_id:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(ch if ch.isalnum() else "_" for ch in (topic[:28].lower() or "unattended")).strip("_")
        run_id = f"{ts}_{slug}"

    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orchestrator] Run={run_id} | Selected compute worker={chosen_worker}", flush=True)

    # Stage 1: Public Web Scraper & Source Dossier (runs locally & saves immediately)
    dossier_path = run_dir / "research_dossier.json"
    if not dossier_path.exists():
        if not topic:
            print("[orchestrator] Stage 1/5: Discovering trending engineering topic via public feeds...", flush=True)
            discovered = discover_topics(count=5)
            (run_dir / "discovered_topics.json").write_text(
                json.dumps(discovered, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if not discovered:
                raise RuntimeError("No candidate topics returned from public feeds")
            top = discovered[0]
            topic = top["lead_story"]["title"] or top["query"]
            queries = [top["query"], topic]
            print(f"[orchestrator] Unattended auto-selected topic: {topic} (score={top['score']})", flush=True)

        print(f"[orchestrator] Stage 1/5: Scraping verified public sources for '{topic}'...", flush=True)
        dossier = build_research_dossier(topic=topic, search_queries=queries, max_sources=10)
        dossier_path.write_text(json.dumps(dossier, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[orchestrator] Saved {dossier['source_count']} verified public sources -> {dossier_path}", flush=True)
    else:
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))

    # Stage 2: Production Plan & Paired Shorts Script
    plan_path = run_dir / "production_plan.json"
    if preauthored_plan and preauthored_plan.exists():
        from tradeoff_pipeline.script_generator import enrich_plan_with_sources
        raw_plan = json.loads(preauthored_plan.read_text(encoding="utf-8"))
        plan = enrich_plan_with_sources(raw_plan, dossier)
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    if chosen_worker == "colab-rtx6000":
        client = ColabWorkerClient.from_config(CONN_FILE)
        print("[orchestrator] Syncing local master repository to RTX PRO 6000 worker...", flush=True)
        client.upload_dir(ROOT, "repo")
        remote_run = f"jobs/{run_id}"
        client.upload_file(dossier_path, f"{remote_run}/research_dossier.json")

        if not plan_path.exists():
            print("[orchestrator] Stage 2/5: Generating Long-Form + Shorts production plan on remote worker...", flush=True)
            env_pass = {}
            for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
                if os.environ.get(k):
                    env_pass[k] = os.environ[k]
            cmd_plan = (
                f"PYTHONPATH=/content/tradeoff_layer_carbon/repo/src "
                f"/content/tradeoff_layer_carbon/.venv/bin/python -m tradeoff_pipeline.script_generator "
                f"--dossier /content/tradeoff_layer_carbon/{remote_run}/research_dossier.json "
                f"--output /content/tradeoff_layer_carbon/{remote_run}/production_plan.json "
                f"--long-scenes {long_scenes}"
            )
            client.run_and_wait(cmd_plan, cwd="repo", env=env_pass, timeout_sec=1800)
            client.download_file(f"{remote_run}/production_plan.json", plan_path)
        else:
            client.upload_file(plan_path, f"{remote_run}/production_plan.json")

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        manifests = _write_tts_manifests_from_plan(plan, run_dir / "manifests")
        for key, mp in manifests.items():
            client.upload_file(mp, f"{remote_run}/manifests/{mp.name}")

        # Stage 3: PyTorch Voice Cloning (F5-TTS) + GPU/48-vCPU Whisper Word Alignment
        print("[orchestrator] Stage 3/5: Synthesizing cloned voiceover & word-level timestamps on RTX PRO 6000...", flush=True)
        tts_cmds = [
            f"PYTHONPATH=/content/tradeoff_layer_carbon/repo/src "
            f"/content/tradeoff_layer_carbon/.venv/bin/python -m tradeoff_pipeline.voice_engine "
            f"--manifest /content/tradeoff_layer_carbon/{remote_run}/manifests/long_tts_manifest.json "
            f"--voice-profile-dir /content/tradeoff_layer_carbon/repo/assets/voice_profile "
            f"--output-dir /content/tradeoff_layer_carbon/{remote_run}/tts_long "
            f"--nfe-step {nfe_step}"
        ]
        for sh in plan.get("shorts", []):
            sh_id = sh["id"]
            tts_cmds.append(
                f"PYTHONPATH=/content/tradeoff_layer_carbon/repo/src "
                f"/content/tradeoff_layer_carbon/.venv/bin/python -m tradeoff_pipeline.voice_engine "
                f"--manifest /content/tradeoff_layer_carbon/{remote_run}/manifests/{sh_id}_tts_manifest.json "
                f"--voice-profile-dir /content/tradeoff_layer_carbon/repo/assets/voice_profile "
                f"--output-dir /content/tradeoff_layer_carbon/{remote_run}/tts_shorts/{sh_id} "
                f"--nfe-step {nfe_step}"
            )
        client.run_and_wait(" && ".join(tts_cmds), cwd="repo", timeout_sec=3600)
        # Immediately pull cloned audio & word timestamps back to local host!
        client.download_dir(f"{remote_run}/tts_long", run_dir / "tts_long")
        for sh in plan.get("shorts", []):
            sh_id = sh["id"]
            client.download_dir(f"{remote_run}/tts_shorts/{sh_id}", run_dir / "tts_shorts" / sh_id)
        print("[orchestrator] Pulled mastered cloned voice WAVs & word timestamps to local host.", flush=True)

        # Stage 4: PyTorch Diffusion Editorial Visual Generation
        print("[orchestrator] Stage 4/5: Generating AI cel-shaded illustrations on RTX PRO 6000...", flush=True)
        cmd_vis = (
            f"PYTHONPATH=/content/tradeoff_layer_carbon/repo/src "
            f"/content/tradeoff_layer_carbon/.venv/bin/python -m tradeoff_pipeline.visual_engine "
            f"--plan /content/tradeoff_layer_carbon/{remote_run}/production_plan.json "
            f"--output-dir /content/tradeoff_layer_carbon/{remote_run}/visuals"
        )
        client.run_and_wait(cmd_vis, cwd="repo", timeout_sec=2400)
        client.download_dir(f"{remote_run}/visuals", run_dir / "visuals")
        print("[orchestrator] Pulled AI illustrations back to local host.", flush=True)

        # Stage 5: 60fps Long-Form & Vertical Shorts Rendering on RTX PRO 6000
        print(f"[orchestrator] Stage 5/5: Rendering {fps}fps 1080p Long-Form + 1080x1920 Shorts on RTX PRO 6000...", flush=True)
        cmd_render = (
            f"PYTHONPATH=/content/tradeoff_layer_carbon/repo/src "
            f"/content/tradeoff_layer_carbon/.venv/bin/python -m tradeoff_pipeline.video_composer "
            f"--plan /content/tradeoff_layer_carbon/{remote_run}/production_plan.json "
            f"--visual-dir /content/tradeoff_layer_carbon/{remote_run}/visuals "
            f"--long-tts-dir /content/tradeoff_layer_carbon/{remote_run}/tts_long "
            f"--shorts-tts-root /content/tradeoff_layer_carbon/{remote_run}/tts_shorts "
            f"--output-dir /content/tradeoff_layer_carbon/{remote_run}/deliverables "
            f"--fps {fps} --max-workers 16"
        )
        client.run_and_wait(cmd_render, cwd="repo", timeout_sec=2400)
        client.download_dir(f"{remote_run}/deliverables", run_dir / "deliverables")

    else:
        # Permanent b850 Tailscale SSH execution path (`alan-eagle@b850-ubuntu`)
        remote_host = "alan-eagle@b850-ubuntu"
        remote_repo = "~/tradeoff-layer-gemini-4-carbon"
        sync_to_b850(ROOT, remote_host=remote_host, remote_dir=remote_repo)
        r_run = f"{remote_repo}/runs/{run_id}"
        subprocess.run(["ssh", "-o", "BatchMode=yes", remote_host, f"mkdir -p {r_run}"], check=True)
        subprocess.run(["rsync", "-az", f"{run_dir}/", f"{remote_host}:{r_run}/"], check=True)

        if not plan_path.exists():
            subprocess.run(
                [
                    "ssh", "-o", "BatchMode=yes", remote_host,
                    f"cd {remote_repo} && PYTHONPATH=src .venv/bin/python -m tradeoff_pipeline.script_generator "
                    f"--dossier {r_run}/research_dossier.json --output {r_run}/production_plan.json --long-scenes {long_scenes}",
                ],
                check=True,
            )
            subprocess.run(["scp", f"{remote_host}:{r_run}/production_plan.json", str(plan_path)], check=True)

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        _write_tts_manifests_from_plan(plan, run_dir / "manifests")
        subprocess.run(["rsync", "-az", f"{run_dir / 'manifests'}/", f"{remote_host}:{r_run}/manifests/"], check=True)

        short_tts_chain = " && ".join(
            f"PYTHONPATH=src .venv/bin/python -m tradeoff_pipeline.voice_engine "
            f"--manifest {r_run}/manifests/{sh['id']}_tts_manifest.json "
            f"--voice-profile-dir assets/voice_profile --output-dir {r_run}/tts_shorts/{sh['id']} --nfe-step {nfe_step}"
            for sh in plan.get("shorts", [])
        )
        remote_pipeline_cmd = (
            f"cd {remote_repo} && "
            f"PYTHONPATH=src .venv/bin/python -m tradeoff_pipeline.voice_engine "
            f"--manifest {r_run}/manifests/long_tts_manifest.json "
            f"--voice-profile-dir assets/voice_profile --output-dir {r_run}/tts_long --nfe-step {nfe_step}"
            + (f" && {short_tts_chain}" if short_tts_chain else "")
            + f" && PYTHONPATH=src .venv/bin/python -m tradeoff_pipeline.visual_engine "
            f"--plan {r_run}/production_plan.json --output-dir {r_run}/visuals "
            f"&& PYTHONPATH=src .venv/bin/python -m tradeoff_pipeline.video_composer "
            f"--plan {r_run}/production_plan.json --visual-dir {r_run}/visuals "
            f"--long-tts-dir {r_run}/tts_long --shorts-tts-root {r_run}/tts_shorts "
            f"--output-dir {r_run}/deliverables --fps {fps} --max-workers 6"
        )
        subprocess.run(["ssh", "-o", "BatchMode=yes", remote_host, remote_pipeline_cmd], check=True)
        subprocess.run(["rsync", "-az", f"{remote_host}:{r_run}/", f"{run_dir}/"], check=True)

    elapsed = round(time.time() - t0, 1)
    print(f"[orchestrator] COMPLETE in {elapsed}s! Deliverables saved locally at: {run_dir / 'deliverables'}", flush=True)
    return run_dir / "deliverables"


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer Unattended Video + Shorts Production Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_disc = sub.add_parser("discover", help="Discover candidate topics from live public feeds")
    p_disc.add_argument("--count", type=int, default=5)

    p_run = sub.add_parser("run", help="Run full unattended Long-Form + Shorts production pipeline")
    p_run.add_argument("--topic", default="", help="Optional topic; if omitted, auto-discovers & selects #1 topic")
    p_run.add_argument("--queries", nargs="*", help="Optional targeted web search queries")
    p_run.add_argument("--run-id", default="")
    p_run.add_argument("--worker", choices=["auto", "colab-rtx6000", "b850"], default="auto")
    p_run.add_argument("--plan", type=Path, default=None, help="Optional pre-generated production plan JSON")
    p_run.add_argument("--long-scenes", type=int, default=10)
    p_run.add_argument("--nfe-step", type=int, default=32)
    p_run.add_argument("--fps", type=int, default=60)

    args = ap.parse_args()
    if args.cmd == "discover":
        topics = discover_topics(count=args.count)
        print(json.dumps(topics, indent=2, ensure_ascii=False))
    elif args.cmd == "run":
        run_pipeline(
            topic=args.topic,
            queries=args.queries,
            run_id=args.run_id,
            worker=args.worker,
            preauthored_plan=args.plan,
            long_scenes=args.long_scenes,
            nfe_step=args.nfe_step,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
