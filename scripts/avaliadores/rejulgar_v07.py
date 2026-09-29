"""v0.7 · Re-julga respostas com juiz ausente (fix do parse de veredito).

O parse anterior do juiz (`avaliar_v5.juiz`) quebrava em respostas longas
(justificativa/reasoning com chaves), deixando `juiz=None` em 280/516 itens da
fronteira v0.7. Este script re-chama o juiz (agora com `_extrair_veredito`
robusto) apenas para as linhas sem veredito, SEM regenerar respostas, e reescreve
os jsonl no lugar.

Uso:
  python scripts/avaliadores/rejulgar_v07.py [--frontier] [--cascade]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger
from openai import OpenAI

from projeto_final import config, llm as llm_mod

RAIZ = Path(__file__).resolve().parent.parent.parent
FRONTIER = RAIZ / "data/processed/v07_ab/frontier_v07.jsonl"
CASCADE = RAIZ / "data/processed/v05b_ab/cascade_v07_p2.jsonl"


def _juiz(pergunta: str, fontes: list[str], resposta: str, cliente):
    sys.path.insert(0, str(Path(__file__).parent))
    from avaliar_v5 import juiz  # type: ignore

    return juiz(pergunta, fontes, resposta, cliente)


def _sem_veredito(linha: dict) -> bool:
    j = linha.get("juiz")
    return not (isinstance(j, dict) and "correta" in j)


def rejulgar(caminho: Path, campo_resposta: str = "resposta") -> None:
    if not caminho.exists():
        logger.warning("sem arquivo: {}", caminho)
        return
    linhas = [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines()]
    alvo = [l for l in linhas if _sem_veredito(l)]
    if not alvo:
        print(f"{caminho.name}: nada a re-julgar ({len(linhas)} linhas)")
        return
    cliente = llm_mod.cliente()
    print(f"{caminho.name}: re-julgando {len(alvo)}/{len(linhas)}")
    for l in alvo:
        resp = (l.get(campo_resposta) or "").strip()
        fontes = sorted(l.get("docs_esperados") or [])
        j = _juiz(l["pergunta"], fontes, resp, cliente)
        l["juiz"] = j
        print(f"  #{l['id']:03d} [{l['estrato']:11s}] juiz={j and j['correta']}", flush=True)
    with caminho.open("w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"atualizado {caminho.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--cascade", action="store_true")
    a = ap.parse_args()
    if not a.frontier and not a.cascade:
        a.frontier = a.cascade = True
    if a.frontier:
        rejulgar(FRONTIER, "resposta")
    if a.cascade:
        rejulgar(CASCADE, "resposta")


if __name__ == "__main__":
    main()