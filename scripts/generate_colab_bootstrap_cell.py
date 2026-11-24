#!/usr/bin/env python3
"""
Generates the isolated Colab cell code from our local master script
scripts/colab_rtx6000_worker_daemon.py so the master logic always originates locally.
"""
from __future__ import annotations

import base64
import secrets
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAEMON_SRC = ROOT / "scripts" / "colab_rtx6000_worker_daemon.py"
OUT_CELL = ROOT / "scripts" / "_colab_bootstrap_cell.py"


def main() -> None:
    daemon_b64 = base64.b64encode(zlib.compress(DAEMON_SRC.read_bytes(), 9)).decode("ascii")
    token = f"tl_rtx6000_{secrets.token_urlsafe(24)}"

    cell_code = f'''# === Tradeoff Layer (@TradeoffLayer) Gemini 4 Carbon — Isolated RTX PRO 6000 Worker (:19531) ===
import base64, json, os, re, shutil, subprocess, sys, time, urllib.request, zlib
from pathlib import Path

candidates = [Path("/local-scratch"), Path("/content")]
best_mount = max(
    [p for p in candidates if p.exists() and os.access(p, os.W_OK)],
    key=lambda p: shutil.disk_usage(str(p)).free,
)
real_work = best_mount / "tradeoff_layer_carbon"
real_work.mkdir(parents=True, exist_ok=True)
work = Path("/content/tradeoff_layer_carbon")
if work != real_work and not work.exists():
    work.symlink_to(real_work)

for sub in ("bin", "logs", "repo", "hf_cache", "torch_cache", "jobs"):
    (real_work / sub).mkdir(parents=True, exist_ok=True)

tok = {token!r}
(real_work / "worker_token.txt").write_text(tok)

daemon_py = real_work / "tradeoff_daemon_19531.py"
daemon_py.write_bytes(zlib.decompress(base64.b64decode({daemon_b64!r})))
daemon_py.chmod(0o755)

subprocess.run("pkill -f '[t]radeoff_daemon_19531.py' || true", shell=True)
subprocess.run("pkill -f '[t]radeoff_cf_19531' || true", shell=True)
time.sleep(0.4)

cf = real_work / "bin" / "tradeoff_cf_19531"
if not cf.exists():
    src = Path("/usr/local/bin/cloudflared")
    if src.exists():
        shutil.copy2(src, cf)
    else:
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            str(cf),
        )
    cf.chmod(0o755)

subprocess.Popen(
    [sys.executable, "-u", str(daemon_py), "--port", "19531", "--token", tok, "--work", str(real_work)],
    stdin=subprocess.DEVNULL,
    stdout=open(real_work / "daemon.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
time.sleep(0.6)

cf_log = real_work / "cf_19531.log"
cf_log.write_text("")
subprocess.Popen(
    [str(cf), "tunnel", "--protocol", "http2", "--no-autoupdate", "--url", "http://127.0.0.1:19531"],
    stdin=subprocess.DEVNULL,
    stdout=open(cf_log, "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

url = ""
for _ in range(50):
    time.sleep(0.5)
    if cf_log.exists():
        txt = cf_log.read_text(errors="ignore")
        m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", txt)
        if m and "Registered tunnel connection" in txt:
            url = m.group(0)
            break
        elif m:
            url = m.group(0)

gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
    text=True,
).strip()
du = shutil.disk_usage(str(real_work))
info = {{
    "url": url,
    "token": tok,
    "port": 19531,
    "work": str(real_work),
    "gpu": gpu,
    "cpus": os.cpu_count(),
    "disk_free_gb": round(du.free / (1024**3), 2),
}}
(real_work / "connection_info.json").write_text(json.dumps(info, indent=2))
print("TRADEOFF_RTX6000_READY=" + json.dumps(info))
'''
    OUT_CELL.write_text(cell_code, encoding="utf-8")
    print(f"Wrote {OUT_CELL} ({len(cell_code)} chars)")


if __name__ == "__main__":
    main()
