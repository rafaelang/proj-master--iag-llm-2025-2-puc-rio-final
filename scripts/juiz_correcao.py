"""v0.5 · Juiz de correção (LLM-as-judge) + auditoria do juiz (R3).

Mede ACURÁCIA (não só abstenção) das respostas do SLM base/destilado no golden
set, com o mesmo prompt do juiz do projeto2 (v0.6). Inclui a AUDITORIA do juiz
(protocolo `rlaif, ato 4`): consistência, viés de comprimento e obediência à
rubrica.

Uso:
  python scripts/juiz_correcao.py data/processed/v05_ab/resultado_base.json
  python scripts/juiz_correcao.py data/processed/v05_ab/resultado_destilado.json

Saída: data/processed/v05_ab/correcao_judge.json + auditoria.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from projeto_final import config

RAIZ = Path(__file__).resolve().parent.parent

JUIZ_SYS = (
    "Você é um avaliador rigoroso de respostas de um assistente sobre o curso "
    '"Master IAG & LLM (PUC-Rio)". Você recebe: PERGUNTA, FONTES_ESPERADAS '
    "(documentos do curso cujo conteúdo deveria embasar a resposta; vazio = a pergunta "
    "está FORA do corpus) e a RESPOSTA do assistente.\n"
    "Regras:\n"
    "1. Se FONTES_ESPERADAS for vazio (ou \"<fora do corpus>\") e a RESPOSTA se abstiver "
    '(ex.: "NAO_SEI", "Não sei", "não tenho informações"), a resposta é CORRETA (nota 4-5).\n'
    "2. Se FONTES_ESPERADAS for vazio e a RESPOSTA responder de fato, é INCORRETA e houve "
    "ALUCINAÇÃO (nota 1-2).\n"
    "3. Com fontes esperadas: avalie se a resposta está correta e completa em relação ao tema "
    "do curso. Erro factual material, invenção ou fuga do tema => correta=false. Se a resposta "
    "se abstiver mesmo havendo fontes, é INCORRETA (nota 1-2, alucinou=false).\n"
    '"alucinou" = true quando a resposta afirma fatos sem suporte nos materiais/domínio '
    "do curso.\n"
    "Responda APENAS com um JSON válido neste formato exato:\n"
    '{"nota": <1-5>, "correta": true|false, "alucinou": true|false, '
    '"justificativa": "<1-2 frases>"}\n'
    "Não escreva NENHUM texto fora do JSON."
)


def _cliente() -> OpenAI:
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def avaliar(pergunta: str, fontes: list[str], resposta: str, cliente: OpenAI) -> dict | None:
    fontes_txt = ", ".join(fontes) if fontes else "<fora do corpus>"
    user = (f"PERGUNTA: {pergunta}\n\nFONTES_ESPERADAS: {fontes_txt}\n\n"
            f"RESPOSTA DO ASSISTENTE:\n{resposta}")
    for _ in range(2):
        try:
            r = cliente.chat.completions.create(
                model=config.JUIZ_MODEL,
                messages=[{"role": "system", "content": JUIZ_SYS},
                          {"role": "user", "content": user}],
                max_tokens=2500, temperature=0.0,
            )
        except Exception as e:
            print(f"  juiz falhou (chamada): {e}", flush=True)
            time.sleep(1)
            continue
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                j = json.loads(m.group())
                return {
                    "nota": int(j.get("nota", 0)),
                    "correta": bool(j.get("correta")),
                    "alucinou": bool(j.get("alucinou")),
                    "justificativa": j.get("justificativa", ""),
                }
            except Exception:
                pass
        time.sleep(1)
    return None


# ------------------------------------------------------- auditoria do juiz (R3)

def auditar_juiz(cliente: OpenAI) -> dict:
    """Protocolo rlaif ato 4 adaptado ao juiz de correção."""
    print("== auditoria do juiz ==")

    # 1) consistência: mesma resposta julgada 2x -> veredito igual?
    perg = "O que é RAG?"
    resp = "RAG é uma técnica que combina recuperação de informações com geração de texto [1]. É apresentado no curso como um mecanismo de busca associado a modelos de linguagem [1]."
    j1 = avaliar(perg, ["nlp_aula06_rag_avancado_ocr.pdf"], resp, cliente)
    time.sleep(1)
    j2 = avaliar(perg, ["nlp_aula06_rag_avancado_ocr.pdf"], resp, cliente)
    consistente = bool(j1 and j2 and j1["correta"] == j2["correta"] and j1["nota"] == j2["nota"])
    print(f"  consistência (mesma resposta 2x): {j1['correta'] if j1 else '?'} vs "
          f"{j2['correta'] if j2 else '?'} -> {'OK' if consistente else 'FALHOU'}")

    # 2) viés de comprimento: mesmo conteúdo, curto x longo (com fonte esperada)
    resp_curta = "RAG combina recuperação de informação com geração, com citação [1]."
    resp_longa = ("RAG, que significa Retrieval-Augmented Generation, é uma técnica de "
                  "inteligência artificial que combina um sistema de recuperação de "
                  "informações com um modelo de linguagem generativo. Na prática, ele "
                  "busca trechos relevantes em um corpus e usa esses trechos como "
                  "contexto para gerar a resposta, permitindo citar a fonte e evitar "
                  "alucinações. Essa abordagem está evoluindo para um sistema de "
                  "raciocínio iterativo sobre dados privados, conforme apresentado nas "
                  "aulas do curso [1].")
    jc = avaliar(perg, ["nlp_aula06_rag_avancado_ocr.pdf"], resp_curta, cliente)
    time.sleep(1)
    jl = avaliar(perg, ["nlp_aula06_rag_avancado_ocr.pdf"], resp_longa, cliente)
    # correto não deve PUNIR a curta nem exagerar a longa; viés = longa nota 5 e curta <=2
    tendencia = bool(jc and jl and jl["nota"] - jc["nota"] >= 2)
    print(f"  viés de comprimento: curta nota {jc['nota'] if jc else '?'} x longa "
          f"nota {jl['nota'] if jl else '?'} -> {'ALERTA (viés)' if tendencia else 'OK'}")

    # 3) obediência à rubrica: fora do corpus que ABSTÉM (deve ser correta) x que RESPONDE
    perg_fora = "Quem ganhou a Copa do Mundo de 2022?"
    jab = avaliar(perg_fora, [], "Não sei.", cliente)
    time.sleep(1)
    jre = avaliar(perg_fora, [], "A Argentina venceu a final.", cliente)
    rubrica_ok = bool(jab and jab["correta"] and jre and not jre["correta"] and jre["alucinou"])
    print(f"  rubrica: abstenção fora->correta={jab['correta'] if jab else '?'} ; "
          f"resposta fora->correta={jre['correta'] if jre else '?'} alucinou={jre['alucinou'] if jre else '?'} "
          f"-> {'OK' if rubrica_ok else 'FALHOU'}")

    return {
        "consistencia": {"ok": consistente, "j1": j1, "j2": j2},
        "vies_comprimento": {"alerta": tendencia, "curta": jc, "longa": jl},
        "rubrica": {"ok": rubrica_ok, "abstencao": jab, "resposta": jre},
        "aprovado": consistente and not tendencia and rubrica_ok,
    }


# ------------------------------------------------------- avaliação

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: python scripts/juiz_correcao.py <resultado_*.json>")
    fonte = Path(sys.argv[1])
    if not fonte.exists():
        raise SystemExit(f"arquivo não encontrado: {fonte}")
    respostas = json.loads(fonte.read_text(encoding="utf-8"))
    respostas = {r["id"]: r for r in respostas} if isinstance(respostas, list) else respostas

    perguntas = json.loads(config.RAG_GOLDEN_SET.read_text(encoding="utf-8"))["perguntas"]
    cliente = _cliente()

    linhas = []
    for q in perguntas:
        qid = q["id"]
        r = respostas.get(qid)
        if not r:
            print(f"  #{qid:02d} sem resposta no arquivo; pulando", flush=True)
            continue
        resp = r.get("resposta", "") or ""
        print(f"=== #{qid:02d} [{q['estrato']}] {q['pergunta'][:55]}", flush=True)
        j = avaliar(q["pergunta"], q.get("docs_esperados", []), resp, cliente)
        if not j:
            print(f"  #{qid:02d} FALHA do juiz", flush=True)
            continue
        linha = {"id": qid, "pergunta": q["pergunta"], "estrato": q["estrato"],
                 "deve_abster": q.get("deve_abster", False),
                 "absteve": r.get("absteve", False), **j}
        linhas.append(linha)
        print(f"  nota={j['nota']} correta={j['correta']} alucinou={j['alucinou']} "
              f"| {j['justificativa'][:70]}", flush=True)

    # auditoria
    auditoria = auditar_juiz(cliente)

    resumo = {
        "n": len(linhas),
        "acertos": sum(1 for x in linhas if x["correta"]),
        "acerto_rate": round(sum(1 for x in linhas if x["correta"]) / max(1, len(linhas)), 3),
        "nota_media": round(sum(x["nota"] for x in linhas) / max(1, len(linhas)), 2),
        "alucinacoes": sum(1 for x in linhas if x["alucinou"]),
        "abstencao_correta": round(sum(1 for x in linhas if x["absteve"] == x["deve_abster"]) / max(1, len(linhas)), 3),
        "por_estrato": {},
    }
    for est in ("rotineira", "composta", "negativa", "adversarial"):
        le = [x for x in linhas if x["estrato"] == est]
        if le:
            resumo["por_estrato"][est] = {
                "n": len(le),
                "acerto_rate": round(sum(1 for x in le if x["correta"]) / len(le), 3),
                "nota_media": round(sum(x["nota"] for x in le) / len(le), 2),
                "alucinacoes": sum(1 for x in le if x["alucinou"]),
            }
    print(f"--- {fonte.stem}: acerto {resumo['acerto_rate']} · nota {resumo['nota_media']} "
          f"· aluc. {resumo['alucinacoes']} · abstenção ok {resumo['abstencao_correta']}")

    dest = config.PROCESSED_DIR / os.getenv("JUIZ_PROC_DIR", "v05_ab")
    dest.mkdir(parents=True, exist_ok=True)
    saída = dest / f"correcao_judge_{fonte.stem.replace('resultado_', '')}.json"
    saída.write_text(json.dumps({
        "meta": {"fonte": str(fonte), "juiz": config.JUIZ_MODEL,
                 "data": datetime.now().isoformat(timespec="seconds")},
        "resumo": resumo, "linhas": linhas}, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "auditoria_juiz.json").write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"salvo: {saída} + auditoria_juiz.json")


if __name__ == "__main__":
    main()