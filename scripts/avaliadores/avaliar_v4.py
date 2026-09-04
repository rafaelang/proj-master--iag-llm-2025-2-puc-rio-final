"""Avaliacao v0.4 — fluxo multiagente (roteador + SLM local + LLM remoto).

Padrao de avaliadores: scripts/avaliadores/avaliar_v{n}.py.

- Dataset unico da v0.4: data/golden_set/rag/perguntas.json (mesmo das releases
  RAG — nao cria outro).
- Metricas: abstencao correta sobre 20 (absteve == deve_abster), distribuicao de
  rotas (simples/complexa), agente usado, fallbacks, latencia media/por pergunta,
  tokens de API consumidos (SLM = US$ 0.00).
- Evidencia oficial em Markdown: docs/v04_evidencia.md (secoes manuais
  preservadas pelo marcador). JSON intermediario: data/processed/rag_v4/
  (fora do Git).
- O nucleo da avaliacao vive em projeto_final.agentes.avaliar_golden (modelos
  parametrizaveis: --roteador/--simples/--complexa).

Uso:
  python scripts/avaliadores/avaliar_v4.py                 # config padrao
  python scripts/avaliadores/avaliar_v4.py --no-evidencia
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from loguru import logger

from projeto_final import agentes, config

JSON_SAIDA = config.PROCESSED_DIR / "rag_v4" / "avaliacao_v4.json"
HISTORICO = config.PROCESSED_DIR / "rag_v4" / "avaliacao_historico_v4.json"
EVIDENCIA_MD = config.DOCS_DIR / "v04_evidencia.md"
MARKER = "<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->"


def _gerar_evidencia(out: dict) -> str:
    r = out["resumo"]
    md = [
        "# v0.4 · Agentes — evidência (roteador SIMPLES/COMPLEXA + SLM local + LLM remoto)",
        "",
        f"> Gerado por `scripts/avaliadores/avaliar_v4.py` em "
        f"{datetime.now().isoformat(timespec='seconds')}.",
        f"> Dataset: `data/golden_set/rag/perguntas.json` (20 perguntas, inalterado).",
        f"> Config: roteador=`{r['config']['roteador']}` · simples=`{r['config']['simples']}` "
        f"· complexa=`{r['config']['complexa']}` · SLM local: `{r['slm_local']}`.",
        "",
        "## Resumo honesto",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Abstenção correta (20) | **{r['abstencao_correta']:.3f}** |",
        f"| Rotas | simples {r['rotas'].get('simples', 0)} · complexa {r['rotas'].get('complexa', 0)} |",
        f"| Agentes | {r['agentes']} |",
        f"| Fallbacks | **{r['fallbacks']}** |",
        f"| Latência média | **{r['latencia_media_s']} s** |",
        f"| Tokens API (rotas pagas) | prompt {r['tokens_api']['prompt']} · completion {r['tokens_api']['completion']} |",
        f"| Custo do SLM (rotas simples locais) | US$ 0.00 |",
        "",
        "## Por pergunta",
        "",
        "| # | Estrato | Rota | Agente | Fallback | Absteve | Deve abster | Acerto | Latência (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for l in out["perguntas"]:
        md.append(
            f"| {l['id']} | {l['estrato']} | {l['rota']} | {l['agente']} "
            f"| {'sim' if l['fallback'] else 'nao'} | {l['absteve']} "
            f"| {l['deve_abster']} | {'sim' if l['acerto'] else 'nao'} "
            f"| {l['latencia_s']} |"
        )
    md.append("")
    md.append("## Respostas (resumo)")
    md.append("")
    for l in out["perguntas"]:
        md.append(f"### #{l['id']} [{l['estrato']}] {l['pergunta']}")
        md.append(f"- **Rota:** {l['rota']} · **Agente:** {l['agente']} · "
                  f"**Fallback:** {l['fallback']}")
        md.append(f"- **Resposta:** {l['resposta']}")
        md.append("")
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliador v0.4 (multiagentes)")
    ap.add_argument("--roteador", choices=["slm", "flash", "tfidf", "cascade"], default=None)
    ap.add_argument("--simples", choices=["slm", "flash", "pro"], default=None)
    ap.add_argument("--complexa", choices=["slm", "flash", "pro"], default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-evidencia", action="store_true",
                    help="nao escreve docs/v04_evidencia.md")
    args = ap.parse_args()

    saida_dir = JSON_SAIDA.parent
    saida_dir.mkdir(parents=True, exist_ok=True)
    logger.add(saida_dir / "avaliacao_v4.log", rotation="1 MB", level="DEBUG")
    logger.info("Avaliacao v0.4 iniciada: dataset {}", config.RAG_GOLDEN_SET)

    out = agentes.avaliar_golden(
        top_k=args.top_k, roteador=args.roteador,
        simples=args.simples, complexa=args.complexa,
    )

    historico = []
    if HISTORICO.exists():
        try:
            historico = json.loads(HISTORICO.read_text(encoding="utf-8"))
        except Exception:
            pass
    historico.append({
        "data": datetime.now().isoformat(timespec="seconds"),
        "resumo": out["resumo"],
    })
    HISTORICO.write_text(json.dumps(historico, ensure_ascii=False, indent=2), encoding="utf-8")
    JSON_SAIDA.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_evidencia:
        novo_md = _gerar_evidencia(out)
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
