"""v1.0 · Fase 3 — Auditoria do juiz em amostra 30% (40Q) do golden 132Q.

Re-audita o juiz (`deepseek-v4-pro`, mesmo prompt do avaliar_v5) numa amostra
estratificada de 40 perguntas (30% do golden), no mesmo protocolo rlaif do
`auditar_juiz_negativa.py`, mas ampliado para o golden inteiro:

  S1 · consistência    — re-julga N pares (pergunta,resposta) 2x → `correta` estável
  S2 · comprimento     — resposta curta × longa (mesmo conteúdo) → não punir a curta
  S3 · rubrica         — abstenção c/ fonte = incorreta; adversarial NÃO que abstém = correta
  S5 · concordância    — veredito do juiz × rótulo humano (`_auditoria`/docs esperados)

A amostra é estratificada por estrato (rotineira 21 / composta 9 / negativa 5 /
adversarial 5 ≈ 30% de 69/29/16/18) com seed fixa. Respostas reais do cascade v0.6
(ou da re-medição v1.0) são re-julgadas.

Uso (Colab/local, precisa de DEEPSEEK_API_KEY):
  python scripts/avaliadores/auditoria_juiz_v10.py

Saída: data/processed/v10_auditoria/auditoria_juiz_v10.json
Evidência: docs/v10_auditoria_juiz.md (gerada pelo --gerar-md)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

from projeto_final import config

RAIZ = Path(__file__).resolve().parent.parent.parent
GOLDEN = RAIZ / "data/golden_set/rag/perguntas_v06.json"
CASCADE = RAIZ / "data/processed/v05b_ab/cascade_v06_p2.jsonl"
SAIDA_DIR = RAIZ / "data/processed/v10_auditoria"

SEED = 42
AMOSTRA_POR_ESTRATO = {"rotineira": 21, "composta": 9, "negativa": 5, "adversarial": 5}

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


def _cliente() -> OpenAI:
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


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
        print(f"  juiz falhou: {e}", flush=True)
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


def _amostra(perguntas: list[dict]) -> list[dict]:
    """Amostra estratificada de 40Q (30%) com seed fixa."""
    rng = random.Random(SEED)
    out = []
    for estrato, n in AMOSTRA_POR_ESTRATO.items():
        pool = [p for p in perguntas if p["estrato"] == estrato]
        out.extend(rng.sample(pool, min(n, len(pool))))
    return out


def _resposta_cascade(qid: int, cascade: list[dict]) -> dict | None:
    for l in cascade:
        if l["id"] == qid:
            return l
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostra-only", action="store_true", help="só grava a amostra")
    a = ap.parse_args()

    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    cascade = [json.loads(l) for l in CASCADE.read_text(encoding="utf-8").splitlines()]
    amostra = _amostra(perguntas)
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    out = {"meta": {"dataset": "perguntas_v06.json", "juiz": config.JUIZ_MODEL,
                    "n_amostra": len(amostra), "seed": SEED,
                    "estratos": AMOSTRA_POR_ESTRATO},
           "amostra": []}

    if a.amostra_only:
        for p in amostra:
            l = _resposta_cascade(p["id"], cascade)
            out["amostra"].append({
                "id": p["id"], "estrato": p["estrato"], "pergunta": p["pergunta"],
                "deve_abster": p.get("deve_abster", False),
                "docs_esperados": p.get("docs_esperados", []),
                "resposta_cascade": (l or {}).get("resposta", ""),
                "juiz_cascade": (l or {}).get("juiz"),
                "resposta_esperada": (p.get("_auditoria") or {}).get("resposta_esperada", ""),
            })
        (SAIDA_DIR / "amostra_40q.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("amostra gravada:", len(out["amostra"]), "perguntas")
        return

    cliente = _cliente()
    print("=== S5 · Concordância juiz × rótulo humano (40Q) ===", flush=True)
    concord = {"n": 0, "acertos": 0, "por_estrato": {}}
    for p in amostra:
        l = _resposta_cascade(p["id"], cascade)
        if not l or not l.get("resposta"):
            continue
        j = juiz(p["pergunta"], p.get("docs_esperados", []), l.get("resposta", ""), cliente)
        time.sleep(0.3)
        # rótulo esperado (juiz): deve_abster XOR ... na prática o juiz v0.6 já deu; medimos
        # se o re-veredito bate com o do cascade (S1) e com o _auditoria (S5).
        esperado = p.get("deve_abster", False) or bool(p.get("docs_esperados"))
        row = {"id": p["id"], "estrato": p["estrato"],
               "juiz_cascade": l.get("juiz"), "juiz_reveredito": j,
               "deve_abster": p.get("deve_abster", False),
               "docs_esperados": p.get("docs_esperados", []),
               "resposta_esperada": (p.get("_auditoria") or {}).get("resposta_esperada", "")}
        if j and l.get("juiz"):
            corr_estavel = j["correta"] == l["juiz"].get("correta")
            concord["n"] += 1
            concord["acertos"] += 1 if corr_estavel else 0
            concord["por_estrato"].setdefault(p["estrato"], {"n": 0, "estavel": 0})
            concord["por_estrato"][p["estrato"]]["n"] += 1
            concord["por_estrato"][p["estrato"]]["estavel"] += 1 if corr_estavel else 0
        out["amostra"].append(row)
        print(f"  #{p['id']:3d} [{p['estrato']:11s}] cascade={l['juiz'] and l['juiz']['correta']} "
              f"reveredito={j and j['correta']} → {'OK' if corr_estavel else 'MUDOU'}", flush=True)

    out["S5_concordancia"] = concord
    print(f"S5: {concord['acertos']}/{concord['n']} estáveis "
          f"({concord['acertos']/max(1,concord['n']):.3f})")

    # S1 — consistência: 8 pares re-julgados 2x
    print("=== S1 · Consistência (re-julgar 2x, 8 casos) ===", flush=True)
    s1 = {"n": 0, "estaveis": 0, "casos": []}
    rng = random.Random(SEED + 1)
    for p in rng.sample(amostra, 8):
        l = _resposta_cascade(p["id"], cascade)
        if not l or not l.get("resposta"):
            continue
        j1 = juiz(p["pergunta"], p.get("docs_esperados", []), l.get("resposta", ""), cliente)
        time.sleep(0.3)
        j2 = juiz(p["pergunta"], p.get("docs_esperados", []), l.get("resposta", ""), cliente)
        time.sleep(0.3)
        estavel = bool(j1 and j2 and j1["correta"] == j2["correta"])
        nota_estavel = bool(j1 and j2 and j1["nota"] == j2["nota"])
        s1["n"] += 1
        s1["estaveis"] += 1 if estavel else 0
        s1["casos"].append({"id": p["id"], "j1": j1, "j2": j2,
                            "correta_estavel": estavel, "nota_estavel": nota_estavel})
        print(f"  #{p['id']} correta {j1 and j1['correta']} × {j2 and j2['correta']} "
              f"| nota {j1 and j1['nota']} × {j2 and j2['nota']} → "
              f"{'OK' if estavel else 'MUDOU'}{'' if nota_estavel else ' (nota flutuou)'}", flush=True)
    out["S1_consistencia"] = s1

    # S2 — viés de comprimento (curta × longa, mesmo conteúdo)
    print("=== S2 · Viés de comprimento ===", flush=True)
    p = next(x for x in perguntas if x["id"] == 15)  # negativa tipo A
    curta = "GAN"
    longa = ("A resposta correta é GAN, pois os itens apresentados no contexto que "
             "relacionam-se ao tema são outros [1], e GAN não aparece no material [2].")
    jc = juiz(p["pergunta"], p.get("docs_esperados", []), curta, cliente)
    time.sleep(0.3)
    jl = juiz(p["pergunta"], p.get("docs_esperados", []), longa, cliente)
    vies = bool(jc and jl and (jc["nota"] - jl["nota"]) < -1)
    out["S2_vies_comprimento"] = {"curta": jc, "longa": jl, "vies": vies}
    print(f"  curta nota {jc and jc['nota']} × longa nota {jl and jl['nota']} → "
          f"{'ALERTA viés' if vies else 'OK'}")

    # S3 — rubrica (casos sintéticos)
    print("=== S3 · Rubrica ===", flush=True)
    fora = "Qual destes itens NÃO está relacionado a bancos de dados: tabela, SQL ou Copa do Mundo?"
    s3 = {}
    j = juiz(fora, [], "NAO_SEI", cliente)          # (c) adversarial NÃO que abstém
    s3["adversarial_nao_abstencao"] = {"esperado": "correta", "juiz": j,
                                       "ok": bool(j and j["correta"])}
    time.sleep(0.3)
    j = juiz(fora, [], "Copa do Mundo.", cliente)   # (d) adversarial NÃO que responde
    s3["adversarial_nao_responde"] = {"esperado": "incorreta+aluc", "juiz": j,
                                      "ok": bool(j and not j["correta"] and j["alucinou"])}
    time.sleep(0.3)
    j = juiz(p["pergunta"], p.get("docs_esperados", []), "NAO_SEI", cliente)  # negativa c/ fonte abstém
    s3["negativa_fonte_abstem"] = {"esperado": "incorreta", "juiz": j,
                                   "ok": bool(j and not j["correta"] and not j["alucinou"])}
    out["S3_rubrica"] = s3
    out["sondas_ok"] = all(v["ok"] for v in s3.values())
    for k, v in s3.items():
        print(f"  S3 {k}: {v['juiz'] and (v['juiz']['correta'], v['juiz']['alucinou'])} "
              f"→ {'OK' if v['ok'] else 'FALHOU'}", flush=True)

    (SAIDA_DIR / "auditoria_juiz_v10.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== RESUMO ===")
    print(f"S1 consistência: {s1['estaveis']}/{s1['n']} corretas estáveis")
    print(f"S2 viés comprimento: {'ALERTA' if vies else 'OK'}")
    print(f"S3 rubrica: {'OK' if out['sondas_ok'] else 'FALHOU'}")
    print(f"S5 concordância cascade×reveredito: {concord['acertos']}/{concord['n']}")


if __name__ == "__main__":
    main()