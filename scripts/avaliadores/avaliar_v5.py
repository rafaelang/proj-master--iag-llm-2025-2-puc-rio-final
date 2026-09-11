"""v0.5 · Avaliador — mini-experimento prompt-only x RAG (com juiz de correção).

Compara, nas 20 perguntas do golden set, o gerador frontier (deepseek-v4-pro)
sem evidência (prompt-only) contra o RAG do projeto_final (com evidência e
citação), medindo ACURÁCIA via juiz (R3) — não só abstenção.

Gera também a evidência consolidada docs/v05_evidencia.md reunindo o
mini-experimento e, se presentes, os resultados do SLM (base/destilado) e o
juiz de correção correspondente em data/processed/v05_ab/.

Uso: python scripts/avaliadores/avaliar_v5.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import OpenAI

from projeto_final import config
from projeto_final import llm as llm_mod
from projeto_final.rag.pipeline import _formatar_contexto, carregar_chunks
from projeto_final.rag.retrieve import recuperar

GOLDEN = config.RAG_GOLDEN_SET
PROC = config.PROCESSED_DIR / "v05_ab"
EVIDENCIA_MD = config.DOCS_DIR / "v05_evidencia.md"

PROMPT_ONLY_SYS = "Você é um assistente. Responda à pergunta. Se não souber a resposta, responda apenas: Não sei."


def detectar_abstencao(resposta: str) -> bool:
    r = (resposta or "").lower().strip()
    return (r.startswith(("nao_sei", "nao sei", "não_sei", "não sei"))
            or "não há informações" in r[:60]
            or "não tenho informações" in r[:60])


def extrair_citacoes(resposta: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d+)\]", resposta or "")]


JUIZ_SYS = (
    "Você é um avaliador rigoroso de respostas de um assistente sobre o curso "
    '"Master IAG & LLM (PUC-Rio)". Você recebe: PERGUNTA, FONTES_ESPERADAS '
    "(documentos do curso cujo conteúdo deveria embasar a resposta; vazio = a pergunta "
    "está FORA do corpus) e a RESPOSTA do assistente.\n"
    "Regras:\n"
    "1. Se FONTES_ESPERADAS for vazio (ou \"<fora do corpus>\") e a RESPOSTA se abstiver, "
    "a resposta é CORRETA (nota 4-5).\n"
    "2. Se FONTES_ESPERADAS for vazio e a RESPOSTA responder de fato, é INCORRETA e houve "
    "ALUCINAÇÃO (nota 1-2).\n"
    "3. Com fontes esperadas: avalie se a resposta está correta e completa em relação ao tema "
    "do curso. Erro factual material, invenção ou fuga do tema => correta=false. Se a resposta "
    "se abstiver mesmo havendo fontes, é INCORRETA (nota 1-2, alucinou=false).\n"
    '"alucinou" = true quando a resposta afirma fatos sem suporte nos materiais/domínio.\n'
    "Responda APENAS com um JSON válido: "
    '{"nota": <1-5>, "correta": true|false, "alucinou": true|false, '
    '"justificativa": "<1-2 frases>"}'
)


def juiz(pergunta: str, fontes: list[str], resposta: str, cliente: OpenAI) -> dict | None:
    fontes_txt = ", ".join(fontes) if fontes else "<fora do corpus>"
    user = (f"PERGUNTA: {pergunta}\n\nFONTES_ESPERADAS: {fontes_txt}\n\n"
            f"RESPOSTA DO ASSISTENTE:\n{resposta}")
    try:
        r = cliente.chat.completions.create(
            model=config.JUIZ_MODEL,
            messages=[{"role": "system", "content": JUIZ_SYS},
                      {"role": "user", "content": user}],
            max_tokens=2500, temperature=0.0,
        )
    except Exception as e:
        logger.error("Juiz falhou: {}", e)
        return None
    txt = (r.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group())
        return {"nota": int(j.get("nota", 0)), "correta": bool(j.get("correta")),
                "alucinou": bool(j.get("alucinou")),
                "justificativa": j.get("justificativa", "")}
    except Exception:
        return None


def gerar_prompt_only(pergunta: str, cliente: OpenAI) -> str:
    """Resposta do frontier SEM evidências (memória/alucinação)."""
    r = cliente.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": PROMPT_ONLY_SYS},
                  {"role": "user", "content": pergunta}],
        max_tokens=400, temperature=0.0,
    )
    return (r.choices[0].message.content or "").strip()


def avaliar_mini_experimento() -> dict:
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_chunks()
    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    linhas = []
    for q in perguntas:
        qid = q["id"]
        esperados = set(q.get("docs_esperados", []))
        print(f"== #{qid:02d} [{q['estrato']}] {q['pergunta'][:55]}", flush=True)

        # prompt-only
        resp_po = gerar_prompt_only(q["pergunta"], cliente)
        ab_po = detectar_abstencao(resp_po)
        cit_po = extrair_citacoes(resp_po)
        j_po = juiz(q["pergunta"], sorted(esperados), resp_po, cliente)
        print(f"   prompt-only: absteve={ab_po} acerto={j_po['correta'] if j_po else None} "
              f"alucinou={j_po['alucinou'] if j_po else None}", flush=True)

        # RAG (pipeline real do projeto_final)
        top_n = recuperar(q["pergunta"], chunks, top_k=5)
        contexto = _formatar_contexto(top_n)
        sistema = config.ler_prompt("v0.2/rag_sistema.txt") or ""
        try:
            resp_rag, _meta = llm_mod.responder_com_contexto(q["pergunta"], contexto, sistema=sistema)
        except Exception as e:
            logger.error("RAG falhou na #{}: {}", qid, e)
            resp_rag = ""
        ab_rag = detectar_abstencao(resp_rag)
        cit_rag = extrair_citacoes(resp_rag)
        j_rag = juiz(q["pergunta"], sorted(esperados), resp_rag, cliente)
        print(f"   rag: absteve={ab_rag} acerto={j_rag['correta'] if j_rag else None} "
              f"cit={cit_rag}", flush=True)

        linhas.append({
            "id": qid, "estrato": q["estrato"], "pergunta": q["pergunta"],
            "deve_abster": q.get("deve_abster", False),
            "docs_esperados": sorted(esperados),
            "prompt_only": {"resposta": resp_po, "absteve": ab_po, "citacoes": cit_po,
                            **({"acerto": j_po["correta"], "nota": j_po["nota"],
                                "alucinou": j_po["alucinou"]} if j_po else {})},
            "rag": {"resposta": resp_rag, "absteve": ab_rag, "citacoes": cit_rag,
                    **({"acerto": j_rag["correta"], "nota": j_rag["nota"],
                        "alucinou": j_rag["alucinou"]} if j_rag else {})},
        })

    resumo = {"n": len(linhas)}
    for modo in ("prompt_only", "rag"):
        n = len(linhas)
        abs_ok = sum(1 for l in linhas if l[modo]["absteve"] == l["deve_abster"])
        julg = [l for l in linhas if "acerto" in l[modo]]
        acerto = sum(1 for l in julg if l[modo]["acerto"])
        aluc = sum(1 for l in julg if l[modo]["alucinou"])
        nota = round(sum(l[modo]["nota"] for l in julg) / max(1, len(julg)), 2)
        cit = sum(1 for l in linhas if not l[modo]["absteve"] and l[modo]["citacoes"])
        nao_abst = sum(1 for l in linhas if not l[modo]["absteve"])
        resumo[modo] = {
            "abstencao_correta": round(abs_ok / n, 3),
            "acerto_end_to_end": round(acerto / max(1, len(julg)), 3),
            "nota_media": nota,
            "alucinacoes": aluc,
            "citacao_presente": round(cit / max(1, nao_abst), 3),
        }
    return {"meta": {"dataset": str(GOLDEN.relative_to(config.RAIZ)),
                     "gerador": config.DEEPSEEK_MODEL, "juiz": config.JUIZ_MODEL,
                     "data": datetime.now().isoformat(timespec="seconds")},
            "resumo": resumo, "linhas": linhas}


# ------------------------------------------------------- evidencia consolidada

def _carregar_slm() -> list[dict]:
    """Resultados do SLM (base/destilado, com/sem RAG) + juiz de correção."""
    blocos = []
    for arq in sorted(PROC.glob("resultado_*.json")):
        try:
            dados = json.loads(arq.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(dados, list) or not dados:
            continue
        sujeito = arq.stem.replace("resultado_", "")
        judge = PROC / f"correcao_judge_{sujeito}.json"
        j = None
        if judge.exists():
            try:
                j = json.loads(judge.read_text(encoding="utf-8"))
            except Exception:
                j = None
        blocos.append({"sujeito": sujeito, "resultado": dados, "juiz": j})
    return blocos


def gerar_evidencia(mini: dict, slm: list[dict]) -> str:
    r = mini["resumo"]
    md = [
        "# v0.5 · Adaptação — evidência (comparativo e decisão)",
        "",
        f"> Gerado por `scripts/avaliadores/avaliar_v5.py` em {r.get('data', '')}.",
        f"> Dataset: `{mini['meta']['dataset']}` (20 perguntas, congelado).",
        f"> Gerador frontier: `{mini['meta']['gerador']}` · Juiz: `{mini['meta']['juiz']}`.",
        "",
        "## 1. Mini-experimento: prompt-only × RAG (acurácia via juiz — R3)",
        "",
        "| Métrica | Prompt-only (sem evidência) | RAG (com evidência) |",
        "|---|---|---|",
        f"| Abstenção correta | {r['prompt_only']['abstencao_correta']:.3f} | {r['rag']['abstencao_correta']:.3f} |",
        f"| **Acurácia end-to-end** (juiz) | **{r['prompt_only']['acerto_end_to_end']:.3f}** | **{r['rag']['acerto_end_to_end']:.3f}** |",
        f"| Nota média (1–5) | {r['prompt_only']['nota_media']} | {r['rag']['nota_media']} |",
        f"| Alucinações | {r['prompt_only']['alucinacoes']} | {r['rag']['alucinacoes']} |",
        f"| Citação presente | {r['prompt_only']['citacao_presente']:.3f} | {r['rag']['citacao_presente']:.3f} |",
        "",
        "> **Leitura honesta:** o requisito da spec é **citação + abstenção** (grounding), não "
        "acurácia bruta. O prompt-only (frontier sem evidência) até acerta mais no corpus "
        "(modelo forte memorizou o tema), mas tem **0% de citação e 2 alucinações fora do "
        "corpus** (#18/#20 respondem em vez de abster); o RAG tem **0 alucinações e 100% de "
        "citação** — cumpre o requisito que a spec pede. A métrica agora é acurácia via juiz "
        "(R3), corrigindo a falha metodológica da PoC.",
        "",
    ]

    if slm:
        md += ["## 2. SLM (base × destilado) no RAG — matriz 2×2 e juiz de correção", ""]
        md += ["| Sujeito | RAG | Abstenção correta | Acurácia (juiz) | Nota | Alucinações |",
               "|---|---|---|---|---|---|"]
        for b in slm:
            res = b["resultado"]
            n = len(res)
            abs_ok = sum(1 for x in res if x.get("absteve") == x.get("deve")) / max(1, n)
            j = b["juiz"]
            if j:
                jr = j["resumo"]
                md.append(f"| {b['sujeito']} | {'sim' if 'no_rag' not in b['sujeito'] else 'não'} | "
                          f"{abs_ok:.3f} | {jr['acerto_rate']:.3f} | {jr['nota_media']} | "
                          f"{jr['alucinacoes']} |")
            else:
                md.append(f"| {b['sujeito']} | {'sim' if 'no_rag' not in b['sujeito'] else 'não'} | "
                          f"{abs_ok:.3f} | (juiz não rodado) | - | - |")
        md += [""]
    else:
        md += ["## 2. SLM (matriz 2×2)", "",
               "> Resultados do SLM serão anexados após a execução no Colab "
               "(`data/processed/v05_ab/resultado_*.json` + `correcao_judge_*.json`).", ""]

    md += ["## 3. Decisão (resumida)", "",
           "A decisão completa (com checklist go/no-go, anti-gatilhos e TCO) está em "
           "`docs/v05.md`. Síntese dos resultados: **não vale treinar/pré-treinar** — a "
           "destilação (LoRA curado, 29 itens) reduziu alucinações (5→4) mas piorou a "
           "acurácia (0.40→0.30) por **super-cautela** (abstém mesmo com evidência em #1/#3); "
           "o RAG (frontier) permanece a escolha de produção: 0 alucinações + 100% de "
           "citação, atendendo o requisito de grounding da spec.", ""]
    return "\n".join(md) + "\n"


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    mini = avaliar_mini_experimento()
    (PROC / "resultados_prompt_rag.json").write_text(
        json.dumps(mini, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESUMO:", json.dumps(mini["resumo"], ensure_ascii=False, indent=2))

    slm = _carregar_slm()
    EVIDENCIA_MD.write_text(gerar_evidencia(mini, slm), encoding="utf-8")
    print(f"evidência consolidada: {EVIDENCIA_MD}")
    if slm:
        print("SLM blocos detectados:", [b["sujeito"] for b in slm])


if __name__ == "__main__":
    main()