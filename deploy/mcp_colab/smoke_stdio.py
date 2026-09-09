"""Smoke test manual: sobe o servidor MCP via stdio e fala JSON-RPC direto.

Uso:  uv run python smoke_stdio.py   (a partir deste diretório)

Valida o handshake MCP (initialize/notifications/initialized) e imprime a
lista de tools registradas. Nenhuma VM Colab é provisionada aqui.
"""

import json
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).parent / "server.py"


def _send(proc, obj: dict) -> None:
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()


def _recv(proc, msg_id: int) -> dict:
    while True:
        line = proc.stdout.readline()
        if not line:
            err = (proc.stderr.read() or "").strip()
            raise SystemExit(f"Servidor fechou o stdout. stderr={err}")
        msg = json.loads(line)
        if msg.get("id") == msg_id:
            return msg


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _send(proc, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "smoke_stdio", "version": "0.1.0"},
            },
        })
        init = _recv(proc, 1)
        assert "result" in init, f"initialize falhou: {init}"
        print(f"initialize ok — servidor: {init['result']['serverInfo']}")

        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = _recv(proc, 2)
        nomes = sorted(t["name"] for t in tools["result"]["tools"])
        print(f"tools/list ok — {len(nomes)} tools registradas:")
        for nome in nomes:
            print(f"  - {nome}")
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
