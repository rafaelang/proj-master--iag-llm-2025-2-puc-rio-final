"""Sobe o laya-serve com um checkpoint FINE-TUNED registrado na VM Colab.

Espelha `servir_laya.py`, com uma diferenca: o `laya-serve` de CLI so precarrega
os checkpoints embutidos (LAYA_MODELS = nomes), e o caminho de um fine-tune
precisa do `Router.register(name, local_path)` (router.py: "a local directory
holding model.safetensors and rl_agent_config.json — a fine-tune exported by
the training recipe"). Este script monta o Router, registra o fine-tune como
`laya-ft` e sobe o MESMO FastAPI `POST /v1/systemone` do laya-serve — o que
mantem `avaliar_roteador_laya.py` integro (basta LAYA_BASE_URL + --modelo).

Tambem precarrega o `multilingual` zero-shot junto, para medir baseline e
fine-tune na MESMA sessao/mesma versao do laya (controle de versao).

Envs:
  LAYA_FT_DIR     diretorio do checkpoint (default: /content/laya_ft)
  LAYA_PORT       porta local (default: 8010)
  LAYA_DEVICE     dispositivo (default: cuda)
  LAYA_TIMEOUT_S  espera do ready (default: 1800)
  LAYA_BASE_URL   base do poll (default: http://127.0.0.1:<port>)

Uso na VM:
  python scripts/colab/servir_laya_ft.py            # sobe em background + smoke
  python scripts/colab/servir_laya_ft.py --serve    # uvicorn bloqueante (interno)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(os.getenv("LAYA_PORT", "8010"))
DEVICE = os.getenv("LAYA_DEVICE", "cuda")
FT_DIR = os.getenv("LAYA_FT_DIR", "/content/laya_ft")
BASE_URL = os.getenv("LAYA_BASE_URL", f"http://127.0.0.1:{PORT}")
TIMEOUT_S = int(os.getenv("LAYA_TIMEOUT_S", "1800"))
LOG = Path("/content/laya_serve_ft.log")


def _serve() -> None:
    import uvicorn
    from uvicorn.protocols.http.auto import AutoHTTPProtocol

    import socket

    from laya.router import Router
    from laya.serve import create_app

    # O laya-serve filtra o campo `model` do cliente (_resolve_model) e so aceita
    # nomes conhecidos — um checkpoint registrado localmente ("laya-ft") viraria
    # auto-roteamento (que cai no english nao-carregado). Para avaliar o fine-tune,
    # passamos o nome adiante e deixamos o Router.route() validar/escolher.
    import laya.serve as _serve_mod
    if getattr(_serve_mod, "_resolve_model", None) is not None:
        _serve_mod._resolve_model = lambda m: m

    if not (Path(FT_DIR) / "model.safetensors").exists():
        raise SystemExit(f"checkpoint fine-tuned nao encontrado: {FT_DIR}")

    class NoDelayHTTPProtocol(AutoHTTPProtocol):
        def connection_made(self, transport):
            sock = transport.get_extra_info("socket")
            if sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6):
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass
            super().connection_made(transport)

    router = Router(device=DEVICE)
    router.register("laya-ft", FT_DIR, "fine-tune do roteador (cenario v0.7)")
    router.preload(["multilingual", "laya-ft"])
    print(f"checkpoints residentes: {sorted(router.loaded)}", flush=True)

    app = create_app(router)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info",
                http=NoDelayHTTPProtocol)


def _esperar_ready() -> dict:
    url = f"{BASE_URL}/health"
    t0 = time.time()
    while time.time() - t0 < TIMEOUT_S:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            _ = e
            time.sleep(5)
    print("TIMEOUT aguardando o serve. Tail do log:", flush=True)
    if LOG.exists():
        print(LOG.read_text(encoding="utf-8")[-4000:], flush=True)
    raise SystemExit(1)


def smoke() -> None:
    payload = {
        "state": "O que e RAG?",
        "model": "laya-ft",
        "questions": {
            "rota": {
                "type": "choice",
                "instructions": "Classifique a pergunta do assistente do Master IAG e LLM da PUC-Rio como SIMPLES ou COMPLEXA.",
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
    print("\n=== smoke (laya-ft) ===")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true",
                    help="roda o uvicorn bloqueante (chamado pelo proprio script)")
    a = ap.parse_args()
    if a.serve:
        _serve()
        return

    cmd = [sys.executable, str(Path(__file__).resolve()), "--serve"]
    print(f"> background: {' '.join(cmd)} (log: {LOG})", flush=True)
    with LOG.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env=dict(os.environ), start_new_session=True,
        )
    print(f"pid={proc.pid}", flush=True)
    info = _esperar_ready()
    print("\n=== laya-ft pronto ===")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    smoke()


if __name__ == "__main__":
    main()
