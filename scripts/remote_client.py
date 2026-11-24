#!/usr/bin/env python3
"""
Unified Remote Compute Client for Tradeoff Layer Pipeline.
Supports:
  1. Temporary High-Speed Colab RTX PRO 6000 Worker (HTTP/Cloudflared tunnel) — PRIMARY
  2. Permanent Tailscale SSH Worker on b850 (alan-eagle@b850-ubuntu / 100.100.230.119) — FALLBACK / DEPLOY

All master scripts always reside on the local host first and are synced before remote execution.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONN_FILE = ROOT / "config" / "colab_rtx6000_connection.json"


class ColabWorkerClient:
    def __init__(self, base_url: str, token: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    @classmethod
    def from_config(cls, path: Path = CONN_FILE) -> "ColabWorkerClient":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(base_url=data["url"], token=data["token"])

    def _req(
        self,
        method: str,
        rel_url: str,
        body: bytes | None = None,
        content_type: str = "application/octet-stream",
        timeout: int | None = None,
    ) -> bytes:
        req = urllib.request.Request(
            f"{self.base_url}{rel_url}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": content_type,
                "User-Agent": "TradeoffLayerPipeline/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return resp.read()

    def health(self) -> dict:
        return json.loads(self._req("GET", "/health", timeout=15).decode("utf-8"))

    def upload_file(self, local_path: Path, remote_rel_path: str) -> dict:
        q = urllib.parse.urlencode({"path": remote_rel_path})
        raw = self._req("POST", f"/put?{q}", body=local_path.read_bytes(), timeout=300)
        return json.loads(raw.decode("utf-8"))

    def upload_dir(
        self,
        local_dir: Path,
        remote_rel_dir: str = "repo",
        exclude_dirs: tuple[str, ...] = (".git", ".venv", "__pycache__", "reference_channel", "output", "runs"),
    ) -> dict:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for item in sorted(local_dir.rglob("*")):
                rel = item.relative_to(local_dir)
                if any(part in exclude_dirs for part in rel.parts):
                    continue
                if item.is_file():
                    tf.add(str(item), arcname=str(rel))
        q = urllib.parse.urlencode({"path": remote_rel_dir, "untar": "1"})
        raw = self._req("POST", f"/put?{q}", body=buf.getvalue(), content_type="application/gzip", timeout=300)
        return json.loads(raw.decode("utf-8"))

    def download_file(self, remote_rel_path: str, local_path: Path) -> Path:
        q = urllib.parse.urlencode({"path": remote_rel_path})
        data = self._req("GET", f"/get?{q}", timeout=600)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return local_path

    def download_dir(self, remote_rel_dir: str, local_dir: Path) -> Path:
        q = urllib.parse.urlencode({"path": remote_rel_dir, "tar": "1"})
        data = self._req("GET", f"/get?{q}", timeout=600)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(path=str(local_dir), filter="data")
        return local_dir

    def start_job(self, cmd: str, cwd: str = "repo", job_id: str | None = None, env: dict | None = None) -> str:
        payload = {"cmd": cmd, "cwd": cwd}
        if job_id:
            payload["id"] = job_id
        if env:
            payload["env"] = env
        raw = self._req(
            "POST",
            "/exec",
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            timeout=30,
        )
        return json.loads(raw.decode("utf-8"))["id"]

    def job_status(self, job_id: str, tail: int = 8000) -> dict:
        q = urllib.parse.urlencode({"id": job_id, "tail": str(tail)})
        raw = self._req("GET", f"/job?{q}", timeout=30)
        return json.loads(raw.decode("utf-8"))

    def run_and_wait(
        self,
        cmd: str,
        cwd: str = "repo",
        poll_sec: float = 3.0,
        timeout_sec: float = 3600.0,
        env: dict | None = None,
    ) -> dict:
        jid = self.start_job(cmd=cmd, cwd=cwd, env=env)
        t0 = time.time()
        while True:
            st = self.job_status(jid, tail=12000)
            if st.get("status") in ("done", "failed"):
                if st.get("status") != "done":
                    raise RuntimeError(f"Remote job {jid} failed (rc={st.get('rc')}):\n{st.get('output', '')}")
                return st
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"Remote job {jid} timed out after {timeout_sec}s")
            time.sleep(poll_sec)


def sync_to_b850(local_root: Path = ROOT, remote_host: str = "alan-eagle@b850-ubuntu", remote_dir: str = "~/tradeoff-layer-gemini-4-carbon") -> None:
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", remote_host, f"mkdir -p {remote_dir}"],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude", ".git/",
            "--exclude", ".venv/",
            "--exclude", "__pycache__/",
            "--exclude", "reference_channel/videos/*.mp4",
            "--exclude", "reference_channel/shorts/*.mp4",
            "--exclude", "runs/",
            f"{local_root}/",
            f"{remote_host}:{remote_dir}/",
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Manage Colab RTX PRO 6000 & b850 Tailscale workers")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("health")
    sub.add_parser("sync-colab")
    sub.add_parser("sync-b850")
    p_exec = sub.add_parser("exec")
    p_exec.add_argument("cmd")
    p_exec.add_argument("--cwd", default="repo")
    args = ap.parse_args()

    if args.action == "sync-b850":
        sync_to_b850()
        print("Synced master repository to b850 via Tailscale SSH.")
        return

    client = ColabWorkerClient.from_config()
    if args.action == "health":
        print(json.dumps(client.health(), indent=2, ensure_ascii=False))
    elif args.action == "sync-colab":
        print(json.dumps(client.upload_dir(ROOT, "repo"), indent=2, ensure_ascii=False))
    elif args.action == "exec":
        res = client.run_and_wait(args.cmd, cwd=args.cwd)
        print(res.get("output", ""))


if __name__ == "__main__":
    main()
