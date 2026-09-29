"""Sobe o endpoint Kev (jaredpalmer/kev) na VM Colab — script roda NA VM.

Clona o repo de Kev, instala as dependencias de servico (uv sync --extra serve)
e deixa `kev.serve` rodando em background na porta 8009 (127.0.0.1). Espera o
GET /v1/models responder e imprime modelo/backend/precisao. O endpoint fica vivo
enquanto a sessao Colab existir (processo destacado com setsid).

Modelo default: Kev-0.8B (cabe em T4). Para outro checkpoint (ex.: 4B, se a GPU
comportar): KEV_RUN=jaredpalmer/kev-4b.

Envs:
  KEV_RUN       checkpoint a servir (default: jaredpalmer/kev-0.8b)
  KEV_PORT      porta local (default: 8009)
  KEV_BASE_URL  base para o poll de ready (default: http://127.0.0.1:<port>)
  KEV_TIMEOUT_S tempo maximo de espera pelo ready (default: 1800)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

KEV_REPO = Path(os.getenv("KEV_REPO", "/content/kev"))
KEV_RUN = os.getenv("KEV_RUN", "jaredpalmer/kev-0.8b")
PORT = int(os.getenv("KEV_PORT", "8009"))
BASE_URL = os.getenv("KEV_BASE_URL", f"http://127.0.0.1:{PORT}")
TIMEOUT_S = int(os.getenv("KEV_TIMEOUT_S", "1800"))
LOG = Path("/content/kev_serve.log")


def _sh(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _install_uv() -> None:
    if shutil.which("uv"):
        return
    _sh([sys.executable, "-m", "pip", "install", "-q", "uv"])


def _clone() -> None:
    if (KEV_REPO / "pyproject.toml").exists():
        print(f"repo ja presente em {KEV_REPO}", flush=True)
        return
    KEV_REPO.parent.mkdir(parents=True, exist_ok=True)
    _sh(["git", "clone", "--depth", "1", "https://github.com/jaredpalmer/kev.git", str(KEV_REPO)])


def _sync() -> None:
    _sh(["uv", "sync", "--extra", "serve"], cwd=KEV_REPO)


def _serve() -> None:
    cmd = ["uv", "run", "--extra", "serve", "python", "-m", "kev.serve",
           "--run", KEV_RUN, "--port", str(PORT)]
    print(f"> background: {' '.join(cmd)} (log: {LOG})", flush=True)
    with LOG.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, cwd=KEV_REPO, stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,  # sobrevive ao fim do script (setsid)
        )
    print(f"pid={proc.pid}", flush=True)


def _wait_ready() -> dict:
    url = f"{BASE_URL}/v1/models"
    t0 = time.time()
    while time.time() - t0 < TIMEOUT_S:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            _ = e
            time.sleep(5)
    print("TIMEOUT aguardando o Kev ficar pronto. Tail do log:", flush=True)
    if LOG.exists():
        print(LOG.read_text(encoding="utf-8")[-4000:], flush=True)
    raise SystemExit(1)


def main() -> None:
    _install_uv()
    _clone()
    _sync()
    _serve()
    info = _wait_ready()
    print("\n=== Kev pronto ===")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()