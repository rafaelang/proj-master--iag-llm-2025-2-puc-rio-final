"""Auditoria P1 combinado v0.5b - pool RRF ponderado variavel.

Reproduz a fusao RRF ponderada de producao de forma FIEL (mesmas funcoes
privadas _ranking_bm25/_ranking_embeddings e mesmas constantes PESO_BM25/
PESO_DENSO/PESO_IMAGEM_RRF/K_RRF), variando somente o numero de candidatos
por lado (pool) de 30 (padrao de producao) ate 400.

Para as 12 perguntas que ficaram FORA do top-5 com rerank no golden oficial
v0.5b, reporta a posicao (0-based) do DOC ESPERADO no ranking ponderado.

Decisao P1-combinado que essa rodada informa:
  * Se pool >= X traz o doc esperado para o top-5 -> "so subir POOL_RERANK"
    (mudanca de constante, zero treino, risco nulo) SEMPRRE sem cross-encoder.
  * Se nem pool=400 traz -> o doc esperado nao chega ao pool de candidatos;
    entao o problema NAO e o pool e sim golden/cross-encoder -> auditamos golden.
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
    K_RRF,
    PESO_BM25,
    PESO_DENSO,
    PESO_IMAGEM_RRF,
    _ranking_bm25,
    _ranking_embeddings,
)

BASE = Path("data/processed/rag")
GOLDEN = Path("data/golden_set/rag/perguntas_v05b.json")

# As 12 dificeis (fora do top-5 oficial v0.5b COM rerank):
DIFICUIS = [1, 8, 11, 12, 13, 28, 30, 34, 35, 38, 40, 41]
POOLS = (30, 60, 100, 200, 400)


def _rrf_ponderado(
    pergunta: str,
    chunks: list[dict],
    bm25,
    embeddings,
    chunk_ids,
    pool: int,
) -> dict[int, float]:
    r_b = _ranking_bm25(pergunta, bm25, chunks, top_k=pool)
    r_d = _ranking_embeddings(pergunta, embeddings, chunk_ids, chunks, top_k=pool)
    tipo = {c["id"]: c.get("tipo", "texto") for c in chunks}
    scores: dict[int, float] = {}
    for cid in set(r_b) | set(r_d):
        s = PESO_BM25 * r_b.get(cid, 0.0) + PESO_DENSO * r_d.get(cid, 0.0)
        if tipo.get(cid) == "imagem":
            s *= PESO_IMAGEM_RRF
        scores[cid] = s
    return scores


def _posicao_esperado(
    pergunta: str,
    chunks: list[dict],
    bm25,
    embeddings,
    chunk_ids,
    doc_esperado: str,
    pool: int,
) -> int | None:
    fusao = _rrf_ponderado(pergunta, chunks, bm25, embeddings, chunk_ids, pool)
    ordem = sorted(fusao.items(), key=lambda kv: kv[1], reverse=True)
    for pos, (cid, _s) in enumerate(ordem):
        ch = next(c for c in chunks if c["id"] == cid)
        if ch.get("arquivo") == doc_esperado:
            return pos
    return None


def main() -> None:
    chunks = carregar_chunks()
    bm25, embeddings, chunk_ids = carregar_indices(BASE)
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    pergam = gold["perguntas"]
    por_id = {p["id"]: p for p in pergam}

    print("# estrato       | pool30 | pool60 | pool100 | pool200 | pool400  (posicao do doc esperado)")
    for qid in DIFICUIS:
        q = por_id[qid]
        pergunta = q["pergunta"]
        estrato = q["estrato"]
        doc_esp = q["docs_esperados"][0]
        linha = "#%02d  %-11s|" % (qid, estrato)
        for pool in POOLS:
            pos = _posicao_esperado(pergunta, chunks, bm25, embeddings, chunk_ids, doc_esp, pool)
            if pos is None:
                linha += "   n/d  |"
            elif pos < 5:
                linha += "   @%d ok|" % pos
            else:
                linha += "   @%d  |" % pos
        print(linha)

    print()
    print("n/d = o doc esperado NAO chega ao pool de candidatos naquele tamanho.")
    print("@0-@4 = entraria no top-5 se o pool fosse aquele (fix = soba POOL_RERANK).")


if __name__ == "__main__":
    main()
