"""Auditoria P1 v0.5b DEFINITIVA - fluxo de producao completo com pool variavel.

Reproduz EXATAMENTE o recuperar() de producao (fusao RRF ponderada + dedup
jaccard 0.85), variando apenas candidatos_n (o pool de over-fetch por lado).
Para as 12 dificeis (fora do top-5 com rerank no golden v0.5b), reporta:
  - pre: posicao no ranking fusionado bruto
  - dedup: posicao no ranking APOS _deduplicar (como o recuperar entrega)
  - top5: entrou no top-5 de entrega (=recall fix)?
Decisao P1-combinado: se pool>=60 coloca o doc no top-5 de entrega SEM rerank,
a fix e "so subir POOL_RERANK" (constante, zero custo). Se nem pool=400,
o problema e golden/corpus (nao e pool) -> auditamos golden.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

logging.disable(logging.CRITICAL)

from projeto_final.rag.index import carregar_indices
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import (
    PESO_BM25,
    PESO_DENSO,
    PESO_IMAGEM_RRF,
    _deduplicar,
    _ranking_bm25,
    _ranking_embeddings,
)

BASE = Path("data/processed/rag")
GOLDEN = Path("data/golden_set/rag/perguntas_v05b.json")
POOLS = (30, 60, 100, 200, 400)
DIFICUIS = [1, 8, 11, 12, 13, 28, 30, 34, 35, 38, 40, 41]


def _ranking_fusionado(
    pergunta: str,
    chunks: list[dict],
    bm25,
    embeddings,
    chunk_ids,
    pool: int,
) -> list[int]:
    r_b = _ranking_bm25(pergunta, bm25, chunks, top_k=pool)
    r_d = _ranking_embeddings(pergunta, embeddings, chunk_ids, chunks, top_k=pool)
    tipo = {c["id"]: c.get("tipo", "texto") for c in chunks}
    scores: dict[int, float] = {}
    for cid in set(r_b) | set(r_d):
        s = PESO_BM25 * r_b.get(cid, 0.0) + PESO_DENSO * r_d.get(cid, 0.0)
        if tipo.get(cid) == "imagem":
            s *= PESO_IMAGEM_RRF
        scores[cid] = s
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _pos_pre_dedup(ranking: list[int], chunks: list[dict], doc: str) -> int | None:
    by_id = {c["id"]: c for c in chunks}
    for i, cid in enumerate(ranking):
        if by_id[cid].get("arquivo") == doc:
            return i
    return None


def _pos_pos_dedup(ranking: list[int], chunks: list[dict], doc: str, pool: int) -> int | None:
    by_id = {c["id"]: c for c in chunks}
    candidatos = [by_id[cid] for cid in ranking[:pool] if cid in by_id]
    dedup = _deduplicar(candidatos)
    for i, c in enumerate(dedup):
        if c.get("arquivo") == doc:
            return i
    return None


def main() -> None:
    chunks = carregar_chunks()
    bm25, embeddings, chunk_ids = carregar_indices(BASE)
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    pergam = gold["perguntas"]
    por_id = {p["id"]: p for p in pergam}

    print("DIFICIL  | pool | pre | dedup | top5? | (posicoes do doc esperado)")
    for qid in DIFICUIS:
        q = por_id[qid]
        pergunta = q["pergunta"]
        estrato = q["estrato"]
        doc_esp = q["docs_esperados"][0]
        for pool in POOLS:
            ranking = _ranking_fusionado(pergunta, chunks, bm25, embeddings, chunk_ids, pool)
            pre = _pos_pre_dedup(ranking, chunks, doc_esp)
            pos = _pos_pos_dedup(ranking, chunks, doc_esp, pool)
            ok = "SIM" if pos is not None and pos < 5 else "nao"
            print("#%02d[%-9s] P%-3d | pre=%s | dedup=%s | %s" % (
                qid, estrato, pool,
                "-" if pre is None else str(pre),
                "-" if pos is None else str(pos),
                ok,
            ))
    print()
    print("pre = posicao bruta (antes do dedup); dedup = posicao final entregue.")
    print("top5=SIM => subir POOL_RERANK sozinho resgataria essa pergunta.")


if __name__ == "__main__":
    main()