"""Avaliador ISOLADO de busca (retrieval) da v0.2 — sem LLM, sem geracao.

Motivacao: Recall@5 alto (0.812) vs Abstencao Indevida alta (0.438) — o documento
certo chega ao Top-5, mas o chunk pode estar cortado no meio da ideia (divisao por
frases) ou mal ranqueado. Esta ferramenta isola o retrieval (BM25 + fastembed + RRF)
e mede Recall@5, Recall@10 e MRR, sem custo, latencia ou nao-determinismo do LLM.

Uso:
  python scripts/avaliadores/avaliar_retrieval.py                 # baseline + secao na evidencia
  python scripts/avaliadores/avaliar_retrieval.py --debug-chunks  # imprime o texto exato dos chunks
  python scripts/avaliadores/avaliar_retrieval.py --debug-chunks --ids 7,8,9
  python scripts/avaliadores/avaliar_retrieval.py --no-evidencia  # so terminal, nao altera docs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks
from projeto_final.rag.retrieve import recuperar

GOLDEN = config.RAG_GOLDEN_SET
RESULTADOS_JSON = config.RAG_DIR / "retrieval.json"
EVIDENCIA_MD = config.DOCS_DIR / "v02_evidencia.md"
AVALIACAO_E2E = config.RAG_DIR / "avaliacao_v2.json"

# Marcador para preservar secoes manuais em docs/v02_evidencia.md quando o
# avaliador de RAG (avaliar_v2.py) regenerar o arquivo.
MARKER = "<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->"

TITULO = "Terceira Rodada: Otimização Isolada do Retrieval"

TOP_K = 10  # busca unica; Recall@5 e Recall@10 derivam do top-10


# ------------------------------------------------------- metricas

def _recall_por_pergunta(docs_esperados: set[str], docs_rec: list[str]) -> float:
    if not docs_esperados:
        return 0.0
    return len(docs_esperados & set(docs_rec)) / len(docs_esperados)


def _mrr_por_pergunta(docs_esperados: set[str], top10: list[dict]) -> float:
    for i, c in enumerate(top10, start=1):
        if c["arquivo"] in docs_esperados:
            return 1.0 / i
    return 0.0


def avaliar(top_k: int = TOP_K) -> dict:
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_chunks()

    linhas = []
    for q in perguntas:
        esperados = set(q.get("docs_esperados", []))
        top = recuperar(q["pergunta"], chunks, top_k=top_k)
        linhas.append({
            "id": q["id"],
            "estrato": q["estrato"],
            "pergunta": q["pergunta"],
            "docs_esperados": sorted(esperados),
            "recall@5": _recall_por_pergunta(esperados, [c["arquivo"] for c in top[:5]]),
            "recall@10": _recall_por_pergunta(esperados, [c["arquivo"] for c in top[:10]]),
            "mrr": _mrr_por_pergunta(esperados, top),
            "docs_recuperados_top10": [c["arquivo"] for c in top[:10]],
        })

    com_docs = [l for l in linhas if l["docs_esperados"]]

    def media(key: str) -> float | None:
        return round(sum(l[key] for l in com_docs) / len(com_docs), 3) if com_docs else None

    resumo: dict = {
        "n_perguntas": len(linhas),
        "n_com_docs_esperados": len(com_docs),
        "recall@5": media("recall@5"),
        "recall@10": media("recall@10"),
        "mrr": media("mrr"),
        "por_estrato": {},
    }
    for est in ("rotineira", "composta", "negativa", "adversarial"):
        le = [l for l in com_docs if l["estrato"] == est]
        if not le:
            continue
        resumo["por_estrato"][est] = {
            "n": len(le),
            "recall@5": round(sum(l["recall@5"] for l in le) / len(le), 3),
            "recall@10": round(sum(l["recall@10"] for l in le) / len(le), 3),
            "mrr": round(sum(l["mrr"] for l in le) / len(le), 3),
        }
    return {"resumo": resumo, "perguntas": linhas}


# ------------------------------------------------------- debug de chunks

def _perguntas_falhas_e2e() -> list[dict]:
    """Perguntas com abstencao indevida na ultima avaliacao end-to-end (avaliacao_v2.json)."""
    if AVALIACAO_E2E.exists():
        dados = json.loads(AVALIACAO_E2E.read_text(encoding="utf-8"))
        return [q for q in dados.get("perguntas", []) if q.get("absteve") and not q.get("deve_abster")]
    return []


def debug_chunks(ids: list[int] | None = None) -> None:
    """Imprime o texto exato dos chunks retornados para as perguntas-alvo."""
    chunks = carregar_chunks()
    if ids:
        perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
        falhas = [q for q in perguntas if q["id"] in ids]
    else:
        falhas = _perguntas_falhas_e2e()
    if not falhas:
        print("Nenhuma pergunta de abstencao indevida encontrada "
              "(avaliacao_v2.json ausente ou sem falhas).")
        return

    print(f"\n{'='*80}\nDEBUG DE CHUNKS — {len(falhas)} pergunta(s) com abstencao indevida\n{'='*80}")
    for q in falhas:
        top = recuperar(q["pergunta"], chunks, top_k=5)
        print(f"\n{'='*80}\n#{q['id']:02d} [{q['estrato']}] {q['pergunta']}")
        print(f"docs_esperados: {q.get('docs_esperados')}")
        for i, c in enumerate(top, start=1):
            print(f"\n--- rank {i} | chunk {c['id']} | {c['arquivo']} | pagina {c['pagina']} ---")
            print(c["texto"])


# ------------------------------------------------------- evidencia (secao manual)

def gerar_secao(out: dict) -> str:
    """Seção 'Terceira Rodada' — embasamento + baseline + placeholder de ajustes."""
    r = out["resumo"]
    linhas = [
        "### Embasamento",
        "",
        "O gargalo da v0.2 não é a recuperação em si, e sim a **qualidade do contexto",
        "entregue ao LLM**: na medição end-to-end anterior, o Recall@5 estava em **0.812**",
        "(os documentos certos eram encontrados), mas a **Abstenção Indevida** cravou",
        "**0.438** — o LLM recebia o documento certo no Top-5, mas o *chunk* específico podia",
        "estar cortado no meio da ideia (divisão estrita por frases) ou fora das primeiras",
        "posições (ranqueamento). Esta rodada **isola o retrieval** (BM25 + fastembed + RRF),",
        "sem LLM, para medir a dispersão (Recall@5/@10) e a ordem (MRR) dos documentos.",
        "",
        "### Resultados — otimizações aplicadas",
        "",
        "#### 1. Chunking: frases → parágrafos (300–1200 chars)",
        "",
        "| Métrica | Chunking por frases (anterior) | Chunking por parágrafos | Δ |",
        "|---|---|---|---|",
        "| Recall@5 | 0.812 | 0.875 | **+0.063** |",
        "| Recall@10 | 0.875 | 0.938 | **+0.063** |",
        "| MRR | 0.790 | 0.819 | **+0.029** |",
        "",
        "#### 2. Limpeza de mobília de slides (ingestão)",
        "",
        "| Métrica | Sem limpeza (anterior) | Com limpeza | Δ |",
        "|---|---|---|---|",
        "| Recall@5 | 0.875 | 0.875 | 0.000 |",
        "| Recall@10 | 0.938 | **0.875** | **−0.063** |",
        "| MRR | 0.819 | 0.812 | −0.007 |",
        "",
        "A limpeza reduziu o corpus (356 → 347 chunks), mas **não melhorou as métricas**:",
        "a pergunta #07 (fine-tuning) perdeu o doc do top-10 e o MRR caiu levemente — o",
        "ruído de slides não era o gargalo do retrieval.",
        "",
        "### Baseline atual — parágrafos 300–1200 chars + limpeza de slides (RRF k=60)",
        "",
        f"- **Recall@5** (sobre as {r['n_com_docs_esperados']} com `docs_esperados`): **{r['recall@5']:.3f}**",
        f"- **Recall@10** (idem): **{r['recall@10']:.3f}**",
        f"- **MRR** (idem): **{r['mrr']:.3f}**",
        "",
        "Por estrato:",
        "",
        "| Estrato | n | Recall@5 | Recall@10 | MRR |",
        "|---|---|---|---|---|",
    ]
    for est, m in r["por_estrato"].items():
        linhas.append(f"| {est} | {m['n']} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} |")
    linhas += [
        "",
        "Detalhes por pergunta em `data/processed/rag/retrieval.json`.",
        "",
        "### Próximos ajustes (pesos do RRF BM25×embeddings e/ou tamanho dos chunks)",
        "",
        "| Ajuste | Recall@5 | Recall@10 | MRR | Δ MRR vs baseline |",
        "|---|---|---|---|---|",
        f"| Baseline (BM25+emb, RRF k=60, parág. + limpeza) | {r['recall@5']:.3f} | {r['recall@10']:.3f} | {r['mrr']:.3f} | — |",
        "| _a definir: ex. RRF k=30 / peso BM25 2x_ | | | | |",
        "| _a definir: ex. max_chars=800 / min_chars=200_ | | | | |",
        "",
    ]
    return "\n".join(linhas)


def escrever_evidencia(secao: str) -> None:
    """Insere/atualiza a secao da Terceira Rodada apos o marcador de secoes manuais."""
    texto = EVIDENCIA_MD.read_text(encoding="utf-8") if EVIDENCIA_MD.exists() else ""
    if MARKER not in texto:
        texto = texto.rstrip("\n") + "\n\n" + MARKER + "\n\n"
    pre, manual = texto.split(MARKER, 1)

    idx = manual.find("## " + TITULO)
    if idx != -1:
        fim = manual.find("\n## ", idx + 4)
        if fim == -1:
            fim = len(manual)
        manual = manual[:idx] + manual[fim:]

    bloco = "## " + TITULO + "\n\n" + secao.strip() + "\n\n"
    novo_manual = bloco + manual.lstrip("\n")
    EVIDENCIA_MD.write_text(pre + MARKER + "\n\n" + novo_manual.rstrip() + "\n", encoding="utf-8")


# ------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliador isolado de retrieval da v0.2 (sem LLM).")
    parser.add_argument("--debug-chunks", action="store_true",
                        help="imprime o texto exato dos chunks das perguntas com abstencao indevida do end-to-end")
    parser.add_argument("--ids", type=str, default="",
                        help="ids das perguntas para --debug-chunks (ex.: --ids 7,8,9)")
    parser.add_argument("--no-evidencia", action="store_true",
                        help="nao escreve/atualiza a secao em docs/v02_evidencia.md")
    args = parser.parse_args()

    out = avaliar(top_k=TOP_K)
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTADOS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    r = out["resumo"]
    print("=== RETRIEVAL ISOLADO (sem LLM) ===")
    print(f"perguntas: {r['n_perguntas']} (com docs_esperados: {r['n_com_docs_esperados']})")
    print(f"Recall@5 : {r['recall@5']:.3f}")
    print(f"Recall@10: {r['recall@10']:.3f}")
    print(f"MRR      : {r['mrr']:.3f}")
    for est, m in r["por_estrato"].items():
        print(f"  {est}: recall@5={m['recall@5']:.3f} recall@10={m['recall@10']:.3f} mrr={m['mrr']:.3f} (n={m['n']})")

    if args.debug_chunks:
        ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
        debug_chunks(ids)

    if not args.no_evidencia:
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        escrever_evidencia(gerar_secao(out))
        print(f"\nevidencia atualizada: {EVIDENCIA_MD} (secao '{TITULO}')")
    print(f"json intermediario: {RESULTADOS_JSON}")


if __name__ == "__main__":
    main()
