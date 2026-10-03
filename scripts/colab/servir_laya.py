"""Sobe o laya-serve (convaiinnovations/laya) na VM Colab — script roda NA VM.

Espelha `servir_kev.py`: instala `laya[serve]` com uv, deixa `laya-serve` rodando
em background na porta local e espera o /health responder (modelo/dispositivo).
O endpoint fica vivo enquanto a sessao Colab existir (processo destacado com
setsid). Mesma interface HTTP `POST /v1/systemone` do Kev, o que mantem os
avaliadores comparáveis.

Checkpoints (env `LAYA_MODELS`, separados por virgula):
  english           ModernBERT-large ~421M, ctx 512  (padrao do Router)
  multilingual      mmBERT-base       ~322M, ctx 1024 (PT-BR — o do projeto)
  typed-decisions   ModernBERT-large ~421M, ctx 1024

Envs:
  LAYA_PORT      porta local (default: 8010)
  LAYA_DEVICE    dispositivo (default: cuda; use cpu se a VM nao tiver GPU)
  LAYA_MODELS    checkpoints a precarregar (default: multilingual,typed-decisions)
  LAYA_TIMEOUT_S tempo maximo de espera pelo ready (default: 1800)
  LAYA_BASE_URL  base para o poll de ready (default: http://127.0.0.1:<port>)
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

PORT = int(os.getenv("LAYA_PORT", "8010"))
DEVICE = os.getenv("LAYA_DEVICE", "cuda")
MODELS = os.getenv("LAYA_MODELS", "multilingual,typed-decisions,english")
BASE_URL = os.getenv("LAYA_BASE_URL", f"http://127.0.0.1:{PORT}")
TIMEOUT_S = int(os.getenv("LAYA_TIMEOUT_S", "1800"))
LOG = Path("/content/laya_serve.log")


def _sh(cmd: list[str], env: dict | None = None) -> None:
    print(f"> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def _install_uv() -> None:
    if shutil.which("uv"):
        return
    _sh([sys.executable, "-m", "pip", "install", "-q", "uv"])


def _install() -> None:
    _sh([shutil.which("uv") or "uv", "pip", "install", "-q", "laya[serve]"])


def _amb() -> dict:
    env = dict(os.environ)
    env.update({
        "LAYA_HOST": "127.0.0.1",
        "LAYA_PORT": str(PORT),
        "LAYA_DEVICE": DEVICE,
        "LAYA_MODELS": MODELS,
        "LAYA_DEFAULT_MODEL": "multilingual",  # trafego PT-BR: nao cair no english
        "LAYA_PRELOAD": "1",
    })
    return env


def _serve() -> None:
    exe = shutil.which("laya-serve") or "laya-serve"
    cmd = [exe, "--host", "127.0.0.1", "--port", str(PORT)]
    print(f"> background: {' '.join(cmd)} (log: {LOG})", flush=True)
    with LOG.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, env=_amb(),
            start_new_session=True,  # sobrevive ao fim do script (setsid)
        )
    print(f"pid={proc.pid}", flush=True)


def _esperar_ready() -> dict:
    """Poll /health ate responder JSON (reporta checkpoints/dispositivo)."""
    url = f"{BASE_URL}/health"
    t0 = time.time()
    while time.time() - t0 < TIMEOUT_S:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            _ = e
            time.sleep(5)
    print("TIMEOUT aguardando o laya-serve ficar pronto. Tail do log:", flush=True)
    if LOG.exists():
        print(LOG.read_text(encoding="utf-8")[-4000:], flush=True)
    raise SystemExit(1)


def smoke() -> None:
    """Uma inferencia real (PT-BR) para fixar o contrato de resposta."""
    payload = {
        "state": "O que é RAG?",
        "model": "multilingual",
        "questions": {
            "rota": {
                "type": "choice",
                "instructions": "Classifique a pergunta como simples ou complexa.",
                "criteria": {
                    "simples": "Pergunta factual e direta que pode ser respondida com uma unica fonte.",
                    "complexa": "Pergunta composta, comparativa, ambigua ou que exige raciocinio sobre multiplas fontes.",
                },
            }
        },
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/systemone",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    print("\n=== smoke ===")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])


def main() -> None:
    _install_uv()
    _install()
    _serve()
    info = _esperar_ready()
    print("\n=== laya pronto ===")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    smoke()


if __name__ == "__main__":
    main()