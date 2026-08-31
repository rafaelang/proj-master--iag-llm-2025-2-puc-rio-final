"""Avaliacao do RAG: recall@k e taxa de abstencao correta."""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from projeto_final import config
from projeto_final.rag.pipeline import responder


def carregar_golden_set(caminho: Path) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)["perguntas"]


def _extrair_citacoes(resposta: str) -> list[tuple[int, str, int]]:
    """Extrai citacoes no formato [N] doc_id, pagina X."""
    padrao = r"\[(\d+)\]\s*([^,]+),\s*pagina\s*(\d+)"
    return [(int(n), doc_id.strip(), int(pagina)) for n, doc_id, pagina in re.findall(padrao, resposta)]


def _recall_k(pergunta: dict, chunks: list[dict], k: int = 5) -> bool:
    docs_esperados = {d["doc_id"] for d in pergunta.get("docs_esperados", [])}
    if not docs_esperados:
        return False
    docs_recuperados = {c["doc_id"] for c in chunks[:k]}
    return bool(docs_esperados & docs_recuperados)


def avaliar(golden_path: Path | None = None, top_k: int = 5) -> dict:
    golden_path = golden_path or config.GOLDEN_SET_DIR / "rag" / "golden_set.json"
    perguntas = carregar_golden_set(golden_path)

    resultados = []
    for p in perguntas:
        logger.info("Avaliando pergunta: {}", p["pergunta"])
        resposta = responder(p["pergunta"], top_k=10)
        citacoes = _extrair_citacoes(resposta["resposta"])
        recall = _recall_k(p, resposta["chunks"], top_k)
        item = {
            "id": p["id"],
            "pergunta": p["pergunta"],
            "tipo": p["tipo"],
            "resposta": resposta["resposta"],
            "abstencao": resposta["abstencao"],
            "recall_k5": recall,
            "citacoes": citacoes,
            "docs_esperados": p.get("docs_esperados", []),
            "chunks": [{"id": c["id"], "doc_id": c["doc_id"], "pagina": c["pagina"]} for c in resposta["chunks"][:top_k]],
        }
        resultados.append(item)

    # Metricas por estrato
    metricas = {}
    for tipo in ["rotineira", "composta", "negativa", "adversarial"]:
        itens = [r for r in resultados if r["tipo"] == tipo]
        if not itens:
            continue
        metricas[tipo] = {
            "n": len(itens),
            "recall_k5": sum(r["recall_k5"] for r in itens) / len(itens),
            "abstencao_correta": sum(r["abstencao"] for r in itens) / len(itens) if tipo in ("negativa", "adversarial") else None,
            "citacao_presente": sum(bool(r["citacoes"]) for r in itens if not r["abstencao"]) / max(1, len([r for r in itens if not r["abstencao"]])),
        }

    # Medias gerais: recall sobre perguntas que esperam docs; abstencao sobre negativas/adversariais; citacao sobre respostas nao-abstidas
    respondidas = [r for r in resultados if not r["abstencao"]]
    com_docs_esperados = [r for r in resultados if r.get("docs_esperados")]
    neg_adv = [r for r in resultados if r["tipo"] in ("negativa", "adversarial")]

    recall_total = sum(r["recall_k5"] for r in com_docs_esperados) / max(1, len(com_docs_esperados))
    abstencao_total = sum(r["abstencao"] for r in neg_adv) / max(1, len(neg_adv))
    citacao_total = sum(bool(r["citacoes"]) for r in respondidas) / max(1, len(respondidas))

    resumo = {
        "n_perguntas": len(resultados),
        "recall_k5": round(recall_total, 3),
        "abstencao_correta": round(abstencao_total, 3),
        "citacao_presente": round(citacao_total, 3),
        "top_k": top_k,
        "por_estrato": metricas,
    }

    logger.info("Avaliacao RAG concluida: {}", resumo)
    return {"resumo": resumo, "perguntas": resultados}


def gerar_evidencia(out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r = out["resumo"]
    md = [
        "# v0.2 - Evidencia - RAG com citacao e abstencao",
        "",
        f"- **Perguntas avaliadas:** {r['n_perguntas']}",
        f"- **recall@5 (sobre perguntas com docs esperados):** {r['recall_k5']:.3f}",
        f"- **Taxa de abstencao correta (negativas + adversariais):** {r['abstencao_correta']:.3f}",
        f"- **Citacao presente (sobre respostas nao-abstidas):** {r['citacao_presente']:.3f}",
        f"- **Top-k usado:** {r['top_k']}",
        "",
        "## Metricas por estrato",
        "",
        "| Estrato | n | recall@5 | abstencao correta | citacao presente |",
        "|---|---|---|---|---|",
    ]
    for tipo, m in r["por_estrato"].items():
        abst = f"{m['abstencao_correta']:.3f}" if m["abstencao_correta"] is not None else "-"
        md.append(f"| {tipo} | {m['n']} | {m['recall_k5']:.3f} | {abst} | {m['citacao_presente']:.3f} |")
    md.extend([
        "",
        "## Detalhes por pergunta",
        "",
    ])
    for r in out["perguntas"]:
        md.append(f"### {r['id']}. ({r['tipo']}) {r['pergunta']}")
        md.append(f"- **Resposta:** {r['resposta']}")
        md.append(f"- **Abstencao:** {r['abstencao']}")
        md.append(f"- **recall@5:** {r['recall_k5']}")
        md.append(f"- **Citacoes:** {r['citacoes']}")
        md.append("")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Evidencia RAG salva em: {}", path)


def main() -> None:
    logger.add(config.RAG_DIR / "avaliacao.log", rotation="1 MB", level="DEBUG")
    out = avaliar()
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAG_DIR / "avaliacao.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    gerar_evidencia(out, config.DOCS_DIR / "v02_evidencia.md")
