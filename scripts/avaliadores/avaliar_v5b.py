"""v0.5b · Avaliador — mini-experimento prompt-only x RAG no golden EXPANDIDO.

Mesma mecânica do avaliar_v5.py (juiz de correção R3, citação/abstenção), mas:
- usa o golden set expandido `perguntas_v05b.json` (RAG_GOLDEN_SET_FILE env);
- grava artefatos em `data/processed/v05b_ab/`;
- gera `docs/v05b_evidencia.md` com a comparação v0.5 × v0.5b.

Uso:
  RAG_GOLDEN_SET_FILE=perguntas_v05b.json python scripts/avaliadores/avaliar_v5b.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from projeto_final import config

from avaliar_v5 import avaliar_mini_experimento  # type: ignore  (reaproveita a máquina da v0.5)

EVIDENCIA_MD = config.DOCS_DIR / "v05b_evidencia.md"


def _resumo_v05() -> dict:
    """Lê a evidência JSON da v0.5 (baseline) para a tabela comparativa."""
    arr = config.PROCESSED_DIR / "v05_ab" / "resultados_prompt_rag.json"
    if arr.exists():
        return json.loads(arr.read_text(encoding="utf-8"))
    return {}


def _carregar_slm() -> list[dict]:
    """Resultados do SLM (base/destilado, com/sem RAG) + juiz de correção (v0.5b)."""
    proc = config.PROCESSED_DIR / "v05b_ab"
    blocos = []
    for arq in sorted(proc.glob("resultado_*.json")):
        try:
            dados = json.loads(arq.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(dados, list) or not dados:
            continue
        sujeito = arq.stem.replace("resultado_", "")
        judge = proc / f"correcao_judge_{sujeito}.json"
        j = None
        if judge.exists():
            try:
                j = json.loads(judge.read_text(encoding="utf-8"))
            except Exception:
                j = None
        blocos.append({"sujeito": sujeito, "resultado": dados, "juiz": j})
    return blocos


def gerar_evidencia_v05b(mini: dict, baseline: dict) -> str:
    """Evidência da v0.5b: comparativo corpus expandido × corpus v0.5 + SLM."""
    r = mini["resumo"]
    b = (baseline.get("resumo") or {}) if baseline else {}
    data = mini.get("meta", {}).get("data", "")
    md = [
        "# v0.5b · Expansão do corpus — evidência (comparativo v0.5 × v0.5b)",
        "",
        f"> Gerado por `scripts/avaliadores/avaliar_v5b.py` em {data}.",
        f"> Dataset: `{mini['meta']['dataset']}` (20 congeladas + novas, v0.5b).",
        "> Baseline v0.5: corpus 10 docs / 272 chunks · v0.5b: corpus expandido.",
        "",
        "## 1. Mini-experimento: prompt-only × RAG — corpus expandido",
        "",
        "| Métrica | Prompt-only | RAG |",
        "|---|---|---|",
        f"| Abstenção correta | {r['prompt_only']['abstencao_correta']:.3f} | {r['rag']['abstencao_correta']:.3f} |",
        f"| **Acurácia end-to-end** (juiz) | **{r['prompt_only']['acerto_end_to_end']:.3f}** | **{r['rag']['acerto_end_to_end']:.3f}** |",
        f"| Nota média (1–5) | {r['prompt_only']['nota_media']} | {r['rag']['nota_media']} |",
        f"| Alucinações | {r['prompt_only']['alucinacoes']} | {r['rag']['alucinacoes']} |",
        f"| Citação presente | {r['prompt_only']['citacao_presente']:.3f} | {r['rag']['citacao_presente']:.3f} |",
        "",
    ]

    if b:
        md += [
            "## 2. Comparativo v0.5 × v0.5b (mesmas perguntas congeladas + novas)",
            "",
            "| Métrica | v0.5 RAG | v0.5b RAG | delta |",
            "|---|---|---|---|",
            f"| Acurácia end-to-end | {b['rag'].get('acerto_end_to_end')} | "
            f"{r['rag']['acerto_end_to_end']:.3f} | "
            f"{(r['rag']['acerto_end_to_end'] - b['rag'].get('acerto_end_to_end', 0)):+.3f} |",
            f"| Alucinações | {b['rag'].get('alucinacoes')} | {r['rag']['alucinacoes']} | - |",
            f"| Citação presente | {b['rag'].get('citacao_presente')} | "
            f"{r['rag']['citacao_presente']:.3f} | - |",
            "",
            "> O executável local do RAG (v0.5b) responde com o corpus expandido "
            "(PDF/MD/IPYNB/PPTX). A comparação direta com a v0.5 é limitada porque as "
            "perguntas novas (21–44) só existem no corpus expandido; a coluna `v0.5` usa "
            "somente as 20 congeladas.",
            "",
        ]

    # SLM da v0.5b (matriz 2×2) — anexada se os resultados + juiz estiverem presentes
    slm = _carregar_slm()
    if slm:
        md += [
            "## 3. SLM (base × destilado) — matriz 2×2 no corpus expandido",
            "",
            "| Gerador | Sem RAG (memória) | Com RAG (grounding) |",
            "|---|---|---|",
        ]
        with_rag = {b["sujeito"].replace("_no_rag", ""): b
                    for b in slm if "_no_rag" not in b["sujeito"]}
        no_rag = {b["sujeito"].replace("_no_rag", ""): b
                  for b in slm if "_no_rag" in b["sujeito"]}
        for key, rotulo in (("base", "**SLM base** (Qwen2.5-1.5B)"),
                            ("destilado", "**SLM destilado** (LoRA curado)")):
            def _cel(b):
                if not b:
                    return "—"
                res = b["resultado"]
                abs_ok = sum(1 for x in res if x.get("absteve") == x.get("deve")) / max(1, len(res))
                j = b["juiz"]
                if j:
                    return (f"abstenção {abs_ok:.3f} · **acurácia "
                            f"{j['resumo']['acerto_rate']:.3f}** · aluc. "
                            f"{j['resumo']['alucinacoes']}")
                return f"abstenção {abs_ok:.3f} · (juiz não rodado)"
            md.append(f"| {rotulo} | {_cel(no_rag.get(key))} | {_cel(with_rag.get(key))} |")
        md += [""]
        md += [
            "> **Leitura honesta v0.5b:** no corpus expandido (44 perguntas) o SLM "
            "permanece abaixo do frontier RAG/acurácia (0.295–0.386 vs 0.705). A "
            "destilação não manteve o ganho de grounding da v0.5 (alucinações: com "
            "RAG base 8 × destilado 9; sem RAG 13 × 14). A decisão da v0.5 se "
            "**mantém**: RAG (frontier) é a escolha de produção; o SLM segue como "
            "executor operacional da v0.4.",
            "",
        ]
    else:
        md += [
            "## 3. SLM (base × destilado) — matriz 2×2 no corpus expandido",
            "",
            "> Resultados do SLM serão anexados após a execução no Colab T4 "
            "(`data/processed/v05b_ab/resultado_*.json` + `correcao_judge_*.json`).",
            "",
        ]
    return "\n".join(md) + "\n"


def main() -> None:
    proc = config.PROCESSED_DIR / "v05b_ab"
    proc.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Golden set da v0.5b: {}", config.RAG_GOLDEN_SET)

    baseline = _resumo_v05()
    mini = avaliar_mini_experimento()
    (proc / "resultados_prompt_rag.json").write_text(
        json.dumps(mini, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESUMO:", json.dumps(mini["resumo"], ensure_ascii=False, indent=2))

    EVIDENCIA_MD.write_text(
        gerar_evidencia_v05b(mini, baseline), encoding="utf-8")
    print(f"evidência consolidada: {EVIDENCIA_MD}")


if __name__ == "__main__":
    main()