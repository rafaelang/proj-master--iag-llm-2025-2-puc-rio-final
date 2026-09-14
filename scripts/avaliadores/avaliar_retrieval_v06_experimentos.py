"""v0.6 · P3 (plano do orientador) — título no índice + boilerplate/dedup + régua doc vs página.

Mede recall@5 em DUAS réguas no mesmo golden 132Q:
- DOC: doc esperado (docs_esperados) presente nos 5 documentos recuperados.
- PÁGINA (88 novas, com _auditoria.fonte): o trecho-âncora (fonte[:200] normalizado)
  aparece no texto de algum chunk do top-5.

Experimentos isolados (NÃO tocam o índice de produção `data/processed/rag/`; índices
experimentais persistem em `data/processed/rag/exp_*` — fora do Git, como data/processed):
- E0_base: re-medição do índice atual (sanity, deve ~0.763 doc).
- E1_titulos: +1 chunk sintético por doc ("TITULO: <filename> · <titulo>") — rota para
  a p.1/titulo do doc.
- E2_sem_boilerplate: remove os ~30 chunks do rodapé "Transformando Dados em Percepção".
- E3_combinado: E1 + E2.

Uso:
  python scripts/avaliadores/avaliar_retrieval_v06_experimentos.py [E0_base|E1_titulos|E2_sem_boilerplate|E3_combinado ...]
(sem argumentos roda os 4; cada um usa o índice cacheado em exp_<nome>).

Saída: data/processed/v05b_ab/retrieval_v06_<experimento>.jsonl (uma linha por pergunta)
e resumo impresso (doc-recall / page-recall por experimento e por estrato).
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from loguru import logger

from projeto_final.rag import index
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import recuperar

RAIZ = Path(__file__).resolve().parent.parent.parent
GOLDEN = RAIZ / "data/golden_set/rag/perguntas_v06.json"
SAIDA_DIR = RAIZ / "data/processed/v05b_ab"
EXP_BASE = RAIZ / "data/processed/rag"

BOILERPLATE = ("transformando dados em percep", "transformando dados em p")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[-\u00ad]\s*", "", t.lower())
    return re.sub(r"\s+", " ", t)


def is_boilerplate(c: dict) -> bool:
    t = norm(c.get("texto") or "")
    return any(m in t for m in BOILERPLATE) and len(t) < 260


def chunks_experimento(nome: str, chunks: list[dict]) -> list[dict]:
    if nome == "E0_base":
        return chunks
    base = chunks
    if "E2_sem_boilerplate" in nome:
        base = [c for c in base if not is_boilerplate(c)]
    if "E1_titulos" in nome or nome == "E3_combinado":
        vistos = set()
        extras = []
        for c in base:
            if c["arquivo"] in vistos:
                continue
            vistos.add(c["arquivo"])
            titulo = (c.get("titulo") or "").strip()
            nome_doc = Path(c["arquivo"]).stem.replace("_", " ")
            texto = f"TITULO: {nome_doc}"
            if titulo:
                texto += f" · {titulo}"
            extras.append({"id": None, "doc_id": c["doc_id"], "arquivo": c["arquivo"],
                           "pagina": 0, "titulo": titulo, "texto": texto,
                           "tokens": len(texto.split()), "chars": len(texto)})
        base = base + extras
    return base


def docs_do_top5(top5: list[dict]) -> set[str]:
    return {c.get("arquivo") for c in top5}


def anchor_no_top5(anchor: str, top5: list[dict]) -> bool:
    a = norm(anchor)[:200].strip()
    if not a or len(a) < 30:
        return False
    return any(a in norm(c.get("texto") or "") for c in top5)


def main() -> None:
    exp_args = sys.argv[1:] or ["E0_base", "E1_titulos", "E2_sem_boilerplate", "E3_combinado"]
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    with_anchor = [p for p in golden if (p.get("_auditoria") or {}).get("fonte")]
    print(f"golden 132Q · com ancora para régua de página: {len(with_anchor)}")
    chunks = carregar_chunks()

    for nome in exp_args:
        ch_exp = chunks_experimento(nome, chunks)
        base_dir = EXP_BASE / f"exp_{nome}"
        t0 = time.time()
        bm25, emb, ids = index.construir_indices(ch_exp, force=True, base=base_dir)
        print(f"[{nome}] índice pronto em {time.time()-t0:.0f}s ({len(ch_exp)} chunks)")

        saida = SAIDA_DIR / f"retrieval_v06_{nome}.jsonl"
        f = open(saida, "w")
        ac_doc = pg_ac = pg_n = 0
        por_estr = {}
        for p in golden:
            pergunta, qid = p["pergunta"], p["id"]
            estr = p.get("estrato") or "?"
            t1 = time.time()
            top5 = recuperar(pergunta, ch_exp, top_k=5, rerank=False, base=base_dir)
            docs = docs_do_top5(top5)
            doc_hit = bool(set(p.get("docs_esperados") or []) & docs)
            anchor = (p.get("_auditoria") or {}).get("fonte")
            page_hit = anchor_no_top5(anchor, top5) if anchor else None
            ac_doc += int(doc_hit)
            if page_hit is not None:
                pg_n += 1
                pg_ac += int(page_hit)
            cel = por_estr.setdefault(estr, {"n": 0, "doc": 0, "pag": 0, "pagn": 0})
            cel["n"] += 1
            cel["doc"] += int(doc_hit)
            if page_hit is not None:
                cel["pagn"] += 1
                cel["pag"] += int(page_hit)
            f.write(json.dumps({
                "id": qid, "estrato": estr, "pergunta": pergunta,
                "docs_esperados": p.get("docs_esperados"),
                "doc_hit": doc_hit, "page_hit": page_hit,
                "docs_recuperados": sorted(docs),
                "latencia_s": round(time.time() - t1, 2),
            }, ensure_ascii=False) + "\n")
        f.close()

        print(f"[{nome}] recall@5 DOC = {ac_doc}/132 = {ac_doc/132:.3f}"
              f" · PÁGINA = {pg_ac}/{pg_n} = {(pg_ac/pg_n if pg_n else 0):.3f}")
        for e, c in sorted(por_estr.items()):
            pdoc = c["doc"] / c["n"]
            ppag = c["pag"] / c["pagn"] if c["pagn"] else -1
            print(f"    {e:11s} n={c['n']:3d} doc={c['doc']:3d} ({pdoc:.3f})"
                  f" página={c['pag']:3d}/{c['pagn']} ({ppag:.3f})")
        print(f"    -> {saida}")


if __name__ == "__main__":
    main()