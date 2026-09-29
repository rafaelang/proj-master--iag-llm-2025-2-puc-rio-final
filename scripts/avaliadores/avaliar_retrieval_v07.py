"""v0.7 · Retrieval isolado no golden expandido (sem LLM) — leg sem rerank.

Reusa a máquina de `avaliar_retrieval_v05b.py` apontando o golden para a v0.7
(RAG_GOLDEN_SET_FILE=perguntas_v07.json) e roda APENAS a leg RRF direto
(sem rerank) — o rerank cross-encoder ~160 s/pergunta em CPU tornaria 516Q
inviável, e a decisão P1 já o rejeitou. Mede recall@5 e MRR (pool 60, top_k 5),
por estrato.

Uso:
  python scripts/avaliadores/avaliar_retrieval_v07.py

Saida (fora do Git): data/processed/rag/retrieval_v07.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["RAG_GOLDEN_SET_FILE"] = "perguntas_v07.json"

from avaliar_retrieval_v05b import RESULTADOS_JSON, avaliar  # noqa: E402


def main() -> None:
    sem = avaliar(rerank=False)
    RESULTADOS_JSON.write_text(
        json.dumps({"sem_rerank": sem}, ensure_ascii=False, indent=2), encoding="utf-8")
    s = sem["resumo"]
    print("=== RETRIEVAL v0.7 (sem rerank, pool 60) ===")
    print(f"perguntas: {s['n_perguntas']} (com docs: {s['n_com_docs_esperados']})")
    print(f"Recall@5: {s['recall@5']:.3f} · MRR: {s['mrr']:.3f}")
    print(f"por estrato: {json.dumps(s['por_estrato'], ensure_ascii=False)}")
    print(f"json: {RESULTADOS_JSON}")


if __name__ == "__main__":
    main()