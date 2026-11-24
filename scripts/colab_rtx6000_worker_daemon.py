#!/usr/bin/env python3
"""
Tradeoff Layer — Isolated Remote GPU Worker Daemon (Colab RTX PRO 6000 / b850 HTTP Worker)
Strictly isolated in /content/tradeoff_layer_carbon on port 19531.
Never touches any other session's directories, ports, or processes.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WORK: Path = Path("/content/tradeoff_layer_carbon").resolve()
TOKEN: str = ""
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _safe_path(rel: str) -> Path:
    p = (WORK / rel.lstrip("/")).resolve()
    if p != WORK and WORK not in p.parents:
        raise ValueError(f"Path escapes work root: {rel}")
    return p


def _build_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    venv_bin = WORK / ".venv" / "bin"
    path_parts = [
        str(venv_bin) if venv_bin.exists() else "",
        "/usr/local/cuda/bin",
        "/usr/local/bin",
        env.get("PATH", ""),
    ]
    env["PATH"] = ":".join(p for p in path_parts if p)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(WORK / "hf_cache")
    env["TORCH_HOME"] = str(WORK / "torch_cache")
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def _query_gpu() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
        return {"available": True, "summary": out}
    except Exception as exc:
        return {"available": False, "summary": str(exc)}


def _run_job(jid: str, cmd: str, cwd: Path, log_path: Path, extra_env: dict[str, str] | None) -> None:
    t0 = time.time()
    try:
        with log_path.open("wb") as lf:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                cwd=str(cwd),
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=_build_env(extra_env),
                start_new_session=True,
            )
            with LOCK:
                JOBS[jid]["pid"] = proc.pid
            rc = proc.wait()
        with LOCK:
            JOBS[jid].update(
                status="done" if rc == 0 else "failed",
                rc=rc,
                elapsed_sec=round(time.time() - t0, 2),
            )
    except Exception as exc:
        with LOCK:
            JOBS[jid].update(
                status="failed",
                rc=-1,
                error=repr(exc),
                elapsed_sec=round(time.time() - t0, 2),
            )


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _auth_ok(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {TOKEN}":
            return True
        self._send_json(401, {"ok": False, "error": "unauthorized"})
        return False

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._auth_ok():
            return
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)

        if parsed.path == "/health":
            with LOCK:
                running = [k for k, v in JOBS.items() if v.get("status") == "running"]
            du = shutil.disk_usage(str(WORK))
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "tradeoff-layer-rtx6000-worker",
                    "work": str(WORK),
                    "cpus": os.cpu_count(),
                    "disk_free_gb": round(du.free / (1024**3), 2),
                    "gpu": _query_gpu(),
                    "running_jobs": running,
                },
            )
            return

        if parsed.path == "/job":
            jid = (q.get("id") or [""])[0]
            tail = int((q.get("tail") or ["8000"])[0])
            with LOCK:
                info = dict(JOBS.get(jid) or {})
            if not info:
                self._send_json(404, {"ok": False, "error": f"unknown job {jid}"})
                return
            lp = Path(info["log"])
            output = lp.read_text(errors="replace")[-tail:] if lp.exists() else ""
            self._send_json(200, {"ok": True, **info, "output": output})
            return

        if parsed.path == "/get":
            try:
                target = _safe_path((q.get("path") or [""])[0])
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            if not target.exists():
                self._send_json(404, {"ok": False, "error": f"not found: {target}"})
                return
            if (q.get("tar") or ["0"])[0] == "1":
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                    if target.is_dir():
                        for child in sorted(target.iterdir()):
                            tf.add(str(child), arcname=child.name)
                    else:
                        tf.add(str(target), arcname=target.name)
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            size = target.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with target.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile, length=1024 * 1024)
            return

        self._send_json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self) -> None:
        if not self._auth_ok():
            return
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length) if length > 0 else b""

        if parsed.path == "/put":
            try:
                target = _safe_path((q.get("path") or [""])[0])
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            if (q.get("untar") or ["0"])[0] == "1":
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
                    tf.extractall(path=str(target), filter="data")
                self._send_json(200, {"ok": True, "extracted_to": str(target), "bytes": len(body)})
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            self._send_json(200, {"ok": True, "saved": str(target), "bytes": len(body)})
            return

        if parsed.path == "/exec":
            req = json.loads(body.decode("utf-8") or "{}")
            jid = req.get("id") or f"job_{uuid.uuid4().hex[:10]}"
            try:
                cwd = _safe_path(req.get("cwd") or "repo")
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            cwd.mkdir(parents=True, exist_ok=True)
            logs_dir = WORK / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            lp = logs_dir / f"{jid}.log"
            with LOCK:
                JOBS[jid] = {
                    "id": jid,
                    "cmd": req["cmd"],
                    "cwd": str(cwd),
                    "status": "running",
                    "rc": None,
                    "pid": None,
                    "started_at": time.time(),
                    "log": str(lp),
                }
            threading.Thread(
                target=_run_job,
                args=(jid, req["cmd"], cwd, lp, req.get("env")),
                daemon=True,
            ).start()
            self._send_json(200, {"ok": True, "id": jid})
            return

        if parsed.path == "/kill_job":
            req = json.loads(body.decode("utf-8") or "{}")
            jid = req.get("id", "")
            with LOCK:
                info = dict(JOBS.get(jid) or {})
            pid = info.get("pid")
            if pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except Exception:
                    pass
            self._send_json(200, {"ok": True, "killed": jid, "pid": pid})
            return

        self._send_json(404, {"ok": False, "error": "unknown endpoint"})


def main() -> None:
    global WORK, TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19531)
    ap.add_argument("--token", required=True)
    ap.add_argument("--work", default="/content/tradeoff_layer_carbon")
    args = ap.parse_args()
    TOKEN = args.token
    WORK = Path(args.work).resolve()
    for sub in ("repo", "logs", "hf_cache", "torch_cache", "jobs", "bin"):
        (WORK / sub).mkdir(parents=True, exist_ok=True)
    print(f"[tradeoff-worker] listening on 127.0.0.1:{args.port} work={WORK}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), WorkerHandler).serve_forever()


if __name__ == "__main__":
    main()
