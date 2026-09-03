"""Avaliacao v0.3 — imagens (OCR local) integradas ao RAG.

Padrao de avaliadores: scripts/avaliadores/avaliar_v{n}.py.

- Dataset unico da v0.3: data/golden_set/imagem/perguntas.json.
- Metricas (sem LLM): o conteudo da FIGURA (termos_esperados, presentes apenas
  no OCR) esta no top-5 recuperado?
    * corpus texto-only (v0.2)  -> contrafactual: NAO deve estar;
    * corpus texto+imagem (v3)  -> DEVE estar (e o chunk que acerta e imagem).
- --e2e (opcional, usa LLM): executa responder (texto) vs responder_v3 (texto+
  imagem) para documentar o "caso em que a visao corrigiu/agregou o texto".
- Evidencia oficial: docs/v03_evidencia.md (Markdown, secoes manuais preservadas
  pelo marcador). JSON intermediario: data/processed/rag_v3/avaliacao_v3.json.

Uso:
  python scripts/avaliadores/avaliar_v3.py          # retrieval (sem LLM)
  python scripts/avaliadores/avaliar_v3.py --e2e    # + respostas LLM (caso doc.)
  python scripts/avaliadores/avaliar_v3.py --no-evidencia
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from loguru import logger

from projeto_final import config
from projeto_final.bm25 import normalizar
from projeto_final.rag.pipeline import carregar_chunks, carregar_corpus_v3
from projeto_final.rag.retrieve import recuperar

GOLDEN = config.RAG_IMG_GOLDEN_SET
JSON_SAIDA = config.RAG_V3_DIR / "avaliacao_v3.json"
HISTORICO = config.RAG_V3_DIR / "avaliacao_historico_v3.json"
EVIDENCIA_MD = config.DOCS_DIR / "v03_evidencia.md"
MARKER = "<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->"
TOP_K = 5


def _termos_em_chunks(top: list[dict], termos: list[str]) -> bool:
    """True se TODOS os termos-alvo aparecem (normalizados) em algum chunk do top."""
    alvo = {t.lower() for t in termos}
    for c in top:
        toks = set(normalizar(c["texto"]))
        if alvo <= toks:
            return True
    return False


def _chunk_imagem_com_termos(top: list[dict], termos: list[str]) -> dict | None:
    for c in top:
        if c.get("tipo") != "imagem":
            continue
        toks = set(normalizar(c["texto"]))
        if all(t.lower() in toks for t in termos):
            return c
    return None


def avaliar_retrieval() -> dict:
    """Mede, por pergunta do golden set de imagem, se o conteudo da figura
    chega ao top-5 no corpus texto-only vs corpus texto+imagem (v3)."""
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks_texto = carregar_chunks()
    chunks_v3 = carregar_corpus_v3()

    # garante o indice combinado (v3) construido em disco
    from projeto_final.rag.index import construir_indices

    construir_indices(chunks_v3, base=config.RAG_V3_DIR)

    linhas = []
    for q in perguntas:
        termos = q.get("termos_esperados", [])
        top_t = recuperar(q["pergunta"], chunks_texto, top_k=TOP_K, rerank=False)
        top_v = recuperar(q["pergunta"], chunks_v3, top_k=TOP_K, rerank=False, base=config.RAG_V3_DIR)
        imagem = _chunk_imagem_com_termos(top_v, termos)
        linhas.append({
            "id": q["id"],
            "estrato": q["estrato"],
            "pergunta": q["pergunta"],
            "doc_esperado": q["doc_esperado"],
            "pagina_figura": q["pagina_figura"],
            "termos_esperados": termos,
            "conteudo_top5_texto": _termos_em_chunks(top_t, termos),
            "conteudo_top5_v3": _termos_em_chunks(top_v, termos),
            "chunk_imagem": (
                {"pagina": imagem["pagina"], "doc_id": imagem["doc_id"]} if imagem else None
            ),
            "docs_top5_texto": sorted({c["doc_id"] for c in top_t}),
            "docs_top5_v3": sorted({c["doc_id"] for c in top_v}),
        })

    n = len(linhas)
    casos_visao = [
        l for l in linhas
        if (not l["conteudo_top5_texto"]) and l["conteudo_top5_v3"]
    ]
    resumo = {
        "perguntas": n,
        "conteudo_top5_texto": round(sum(1 for l in linhas if l["conteudo_top5_texto"]) / n, 3),
        "conteudo_top5_v3": round(sum(1 for l in linhas if l["conteudo_top5_v3"]) / n, 3),
        "casos_visao_corrige_texto": len(casos_visao),
        "ids_casos_visao": [l["id"] for l in casos_visao],
        "top_k": TOP_K,
        "corpus": {"texto": len(chunks_texto), "v3_texto_imagem": len(chunks_v3)},
    }
    logger.info("Resumo retrieval v0.3: {}", resumo)
    return {"resumo": resumo, "perguntas": linhas}


def _e2e_pergunta(q: dict) -> dict:
    """Respostas LLM: texto-only (v0.2) vs texto+imagem (v0.3)."""
    from projeto_final.rag.pipeline import responder, responder_v3

    def _resumo(r: dict) -> dict:
        return {
            "resposta": r["resposta"],
            "abstencao": r["abstencao"],
            "tem_termos": all(
                t.lower() in normalizar(r["resposta"]) for t in q.get("termos_esperados", [])
            ),
            "chunks_tipos": sorted({c.get("tipo", "texto") for c in r["chunks"]}),
        }

    try:
        r_t = _resumo(responder(q["pergunta"]))
    except Exception as e:
        r_t = {"erro": str(e)}
    try:
        r_v = _resumo(responder_v3(q["pergunta"]))
    except Exception as e:
        r_v = {"erro": str(e)}
    return {"id": q["id"], "texto_only": r_t, "v3_texto_imagem": r_v}


def avaliar_e2e(perguntas: list[dict]) -> list[dict]:
    logger.info("Executando e2e (LLM) nas {} perguntas do golden set de imagem...", len(perguntas))
    return [_e2e_pergunta(q) for q in perguntas]


def gerar_evidencia(out: dict, e2e: list[dict] | None = None) -> str:
    r = out["resumo"]
    md = ["# v0.3 · Imagem — evidência (OCR local de figuras integrado ao RAG)", ""]
    md.append(f"> Gerado por `scripts/avaliadores/avaliar_v3.py` em {datetime.now().isoformat(timespec='seconds')}.")
    md.append(f"> Dataset: `data/golden_set/imagem/perguntas.json` ({r['perguntas']} perguntas).")
    md.append(f"> Corpus: texto-only {r['corpus']['texto']} chunks · texto+imagem {r['corpus']['v3_texto_imagem']} chunks.")
    md += ["", "## Resumo honesto (retrieval)", "",
           "| Métrica | Valor |", "|---|---|",
           f"| Conteúdo da figura no top-5 — corpus TEXTO-only | **{r['conteudo_top5_texto']:.3f}** |",
           f"| Conteúdo da figura no top-5 — corpus TEXTO+IMAGEM | **{r['conteudo_top5_v3']:.3f}** |",
           f"| Casos em que a visão (OCR) corrigiu/agregou o texto | **{r['casos_visao_corrige_texto']}** (ids {r['ids_casos_visao']}) |",
           ""]
    md += ["## Por pergunta", "",
           "| # | Estrato | Conteúdo figura no top-5 (texto-only) | (texto+imagem) | Chunk imagem |",
           "|---|---|---|---|---|"]
    for l in out["perguntas"]:
        ch = l["chunk_imagem"]
        img_txt = f"p.{ch['pagina']} · {ch['doc_id'][:26]}" if ch else "-"
        md.append(f"| {l['id']} | {l['estrato']} | {'SIM' if l['conteudo_top5_texto'] else 'NAO'} "
                  f"| {'SIM' if l['conteudo_top5_v3'] else 'NAO'} "
                  f"| {img_txt} |")
    md.append("")
    if e2e:
        md += ["## Respostas (LLM) — caso documentado: visão corrige o texto", ""]
        for l in e2e:
            md += [f"### #{l['id']}", ""]
            rt, rv = l["texto_only"], l["v3_texto_imagem"]
            if "erro" in rt:
                md += [f"- **texto-only (v0.2):** erro — {rt['erro']}", ""]
            else:
                md += [f"- **texto-only (v0.2):** abstenção={rt['abstencao']} · contém termos={rt['tem_termos']}",
                       f"  > {rt['resposta']}", ""]
            if "erro" in rv:
                md += [f"- **texto+imagem (v0.3):** erro — {rv['erro']}", ""]
            else:
                md += [f"- **texto+imagem (v0.3):** abstenção={rv['abstencao']} · contém termos={rv['tem_termos']} · tipos de chunk={rv['chunks_tipos']}",
                       f"  > {rv['resposta']}", ""]
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliador v0.3 (imagens/OCR no RAG)")
    ap.add_argument("--e2e", action="store_true", help="executa respostas LLM (caso documentado)")
    ap.add_argument("--no-evidencia", action="store_true", help="nao escreve docs/v03_evidencia.md")
    args = ap.parse_args()

    config.RAG_V3_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.RAG_V3_DIR / "avaliacao_v3.log", rotation="1 MB", level="DEBUG")
    logger.info("Avaliacao v0.3 iniciada: dataset {}", GOLDEN)

    out = avaliar_retrieval()
    e2e = avaliar_e2e(out["perguntas"]) if args.e2e else None

    historico = []
    if HISTORICO.exists():
        try:
            historico = json.loads(HISTORICO.read_text(encoding="utf-8"))
        except Exception:
            pass
    historico.append({
        "data": datetime.now().isoformat(timespec="seconds"),
        "e2e": bool(args.e2e),
        "resumo": out["resumo"],
    })
    HISTORICO.write_text(json.dumps(historico, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {"resumo": out["resumo"], "perguntas": out["perguntas"], "e2e": e2e}
    JSON_SAIDA.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_evidencia:
        novo_md = gerar_evidencia(out, e2e)
        manual = ""
        if EVIDENCIA_MD.exists():
            atual = EVIDENCIA_MD.read_text(encoding="utf-8")
            if MARKER in atual:
                manual = atual.split(MARKER, 1)[1]
        EVIDENCIA_MD.write_text(novo_md + MARKER + "\n\n" + manual, encoding="utf-8")

    print("RESUMO:", json.dumps(out["resumo"], ensure_ascii=False, indent=2))
    print(f"json intermediario: {JSON_SAIDA}")
    print(f"evidencia (markdown): {EVIDENCIA_MD}")


if __name__ == "__main__":
    main()
