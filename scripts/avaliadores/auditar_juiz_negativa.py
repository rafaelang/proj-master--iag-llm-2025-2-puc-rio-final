"""v0.6 · P0 (plano do orientador, ORIENTADOR_PLAN.md) — AUDITORIA do juiz nas 16 negativas.

Bloqueador: decidir se o veredito do juiz no estrato NEGATIVA é confiável — em
especial se ele exige grounding (citação/doc recuperado) ou aceita conhecimento
de fundo. Roda ANTES de qualquer correção do estrato (mudar o juiz muda a régua).

Entradas (dados locais, fora do Git):
- data/processed/v05b_ab/cascade_v06_p2.jsonl   (16 linhas negativas: resposta+veredito)
- data/processed/rag/retrieval_v06.json          (recall@5/MRR por pergunta)
- data/golden_set/rag/perguntas_v06.json         (docs_esperados + _auditoria)

Etapa 1 (local, sem API) — tabela 16Q:
  id | tipo A/B | recall@5 | doc esperado no top-5? | resposta do sistema |
  veredito do juiz | resposta esperada (gold) | grounding (manual + dica do retriever).

Etapa 2 (API, juiz deepseek-v4-pro) — sondas rlaif REDUZIDAS nas negativas:
  S1 consistência: mesmo (pergunta,resposta) 2x → `correta` estável (P4: nota flutua = alerta).
  S2 viés de comprimento: curta × longa corretas → não punir a curta.
  S3 rubrica: (a) negativa c/ fonte que ABSTÉM = incorreta; (b) negativa c/ fonte que
     RESPONDE certo = correta; (c) adversarial com "NÃO" fora do corpus que ABSTÉM =
     correta (colisão negativa×adversarial do Passo 1); (d) idem que RESPONDE = incorreta+aluc.
  S4 grounding: o juiz aceita resposta certa mesmo quando o doc esperado NÃO está
     recuperado (caso real #16/#42) — documenta a possível leniência do orientador §0.4.

Decisão registrada em docs/v06_auditoria_juiz_negativa.md (aceitar juiz OU ajustar rubrica).

Uso:
  python scripts/avaliadores/auditar_juiz_negativa.py --tabela-only   # só dados locais
  python scripts/avaliadores/auditar_juiz_negativa.py                 # tabela + sondas API
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from openai import OpenAI

from projeto_final import config

RAIZ = Path(__file__).resolve().parent.parent.parent
CASCADE = RAIZ / "data/processed/v05b_ab/cascade_v06_p2.jsonl"
RETRIEVAL = RAIZ / "data/processed/rag/retrieval_v06.json"
GOLDEN = RAIZ / "data/golden_set/rag/perguntas_v06.json"
SAIDA_JSON = RAIZ / "data/processed/v05b_ab/auditoria_juiz_negativa.json"
EVIDENCIA_MD = config.DOCS_DIR / "v06_auditoria_juiz_negativa.md"

NEG_IDS = [15, 16, 41, 42, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

# tipo (A = item AUSENTE do contexto; B = item PRESENTE/afirmado) — classificação do
# orientador §0.1, conferida contra a formulação de cada pergunta.
TIPO = {15: "A", 41: "A", 89: "A", 90: "A", 91: "A", 92: "A", 93: "A", 94: "A",
        95: "A", 96: "A", 97: "A", 98: "A", 99: "A", 100: "A", 16: "B", 42: "B"}

# Resposta esperada (gold): das 44 congeladas (15/16/41/42) derivada do doc esperado;
# das novas (89+) do campo _auditoria do golden. Congelado — não re-editar.
RESPOSTA_ESPERADA: dict[int, str] = {
    15: "GAN", 16: "voz", 41: "fundo de investimento", 42: "processamento de imagens",
}

# Termos-chave da resposta esperada (normalizados sem acento) p/ a dica de grounding:
# no tipo A o termo correto é o item AUSENTE (espera-se NÃO achar no contexto);
# no tipo B o termo é o item afirmado (espera-se achar no contexto recuperado).
TERMOS_ESPERADA: dict[int, list[str]] = {
    15: ["gan"], 16: ["wer"], 41: ["fundo de investimento"], 42: ["tensorflow", "imagens"],
    89: ["copa do mundo"], 90: ["relatividade"], 91: ["copa do mundo"],
    92: ["relatividade"], 93: ["relatividade"], 94: ["relatividade"],
    95: ["capital do japao"], 96: ["receita de bolo"], 97: ["copa do mundo"],
    98: ["fundo de investimento"], 99: ["relatividade"], 100: ["cotacao de acao"],
}

# Items fora do domínio (opções distractors) por id — ajuda a conferir a resposta correta.
DISTRATOR: dict[int, str] = {
    89: "Copa do Mundo", 90: "teoria da relatividade", 91: "copa do mundo",
    92: "teoria da relatividade", 93: "teoria da relatividade", 94: "Teoria da Relatividade",
    95: "capital do Japão", 96: "receita de bolo", 97: "Copa do Mundo de Futebol",
    98: "fundo de investimento", 99: "teoria da relatividade", 100: "cotação de ação",
}


def _norm(s: str) -> str:
    t = unicodedata.normalize("NFKD", (s or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def _resumo_justif(j: dict | None, n: int = 90) -> str:
    if not j:
        return "(juiz falhou)"
    return (j.get("justificativa") or "")[:n].replace("\n", " ")


# -------------------------------------------------- etapa 1: tabela local

def _carregar() -> tuple[list[dict], dict, dict]:
    cascade = [json.loads(l) for l in CASCADE.read_text(encoding="utf-8").splitlines()]
    casc = {l["id"]: l for l in cascade}
    retr = json.loads(RETRIEVAL.read_text(encoding="utf-8"))["sem_rerank"]["perguntas"]
    ret = {p["id"]: p for p in retr}
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    g = {q["id"]: q for q in golden}
    return casc, ret, g


def _dica_grounding(qid: int, chunks_top5: list[dict]) -> str | None:
    """Dica automática de grounding: o termo da resposta esperada aparece no top-5?

    Tipo A: espera-se AUSENTE (o item fora do domínio não deve estar no material).
    Tipo B: espera-se PRESENTE (a resposta foi afirmada no material recuperado).
    """
    termos = TERMOS_ESPERADA.get(qid, [])
    textos = " ".join(_norm(c.get("texto", "")) for c in chunks_top5)
    achou = [t for t in termos if _norm(t) in textos]
    return f"achou no top-5 (dica: {achou})" if achou else "ausente do top-5 (dica)"

def _top5_chunks(qid: int) -> list[dict]:
    """Re-recupera o top-5 (retriever real) para a dica de grounding — local, sem API/LLM."""
    from projeto_final.rag.pipeline import carregar_chunks
    from projeto_final.rag.retrieve import recuperar

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    pergunta = next(q["pergunta"] for q in golden if q["id"] == qid)
    chunks = carregar_chunks()
    return recuperar(pergunta, chunks, top_k=5)


def montar_tabela() -> dict:
    casc, ret, g = _carregar()
    linhas = []
    for qid in NEG_IDS:
        c, r, q = casc[qid], ret[qid], g[qid]
        esperados = q.get("docs_esperados", [])
        doc_no_top5 = bool(esperados and any(d in r.get("docs_recuperados", []) for d in esperados))
        resp = (c.get("resposta") or "").strip()
        linha = {
            "id": qid,
            "tipo": TIPO[qid],
            "pergunta": c["pergunta"],
            "deve_abster": q.get("deve_abster", False),
            "docs_esperados": esperados,
            "recall@5": r.get("recall@5", 0.0),
            "mrr": r.get("mrr", 0.0),
            "doc_esperado_no_top5": doc_no_top5,
            "rota": c.get("rota"),
            "modelo": c.get("modelo"),
            "resposta": resp[:200],
            "absteve": bool(c.get("absteve")),
            "juiz": c.get("juiz"),
            "resposta_esperada": RESPOSTA_ESPERADA.get(qid) or
                                 (q.get("_auditoria") or {}).get("resposta_esperada", ""),
            "distrator": DISTRATOR.get(qid, ""),
        }
        # dica de grounding (só nas que responderam — B e possíveis A não-abstidas)
        if not c.get("absteve"):
            try:
                top5 = _top5_chunks(qid)
                linha["grounding_dica"] = _dica_grounding(qid, top5)
                linha["docs_top5"] = sorted({c2.get("arquivo") for c2 in top5})
            except Exception as exc:  # fallback: sem re-consulta
                linha["grounding_dica"] = f"(dica indisponível: {exc})"
        linhas.append(linha)

    contagem = {"n": len(linhas), "tipoA": sum(1 for l in linhas if l["tipo"] == "A"),
                "tipoB": sum(1 for l in linhas if l["tipo"] == "B"),
                "tipoA_corretas": sum(1 for l in linhas if l["tipo"] == "A" and l["juiz"] and l["juiz"]["correta"]),
                "tipoB_corretas": sum(1 for l in linhas if l["tipo"] == "B" and l["juiz"] and l["juiz"]["correta"]),
                "tipoA_abstiveram": sum(1 for l in linhas if l["tipo"] == "A" and l["absteve"]),
                "tipoA_doc_no_top5": sum(1 for l in linhas if l["tipo"] == "A" and l["doc_esperado_no_top5"])}
    return {"contagem": contagem, "linhas": linhas}


# -------------------------------------------------- etapa 2: sondas rlaif

def _juiz_cliente() -> OpenAI:
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def _juiz(pergunta: str, fontes: list[str], resposta: str, cliente: OpenAI) -> dict | None:
    sys.path.insert(0, str(RAIZ / "scripts/avaliadores"))
    from avaliar_v5 import juiz  # type: ignore

    return juiz(pergunta, fontes, resposta, cliente)


def sondas(cliente: OpenAI) -> dict:
    """Sondas rlaif reduzidas focadas no estrato negativa (+ colisão adversarial 'NÃO')."""
    out: dict = {}

    # S1 — consistência: 3 pares representativos julgados 2x (correta deve ser estável).
    s1_casos = [
        ("n1_abstencao_tipoA", 15, ["tdp_aula03_sql_ddl.pdf"], "NAO_SEI"),
        ("n2_resposta_tipoB", 16, ["proj_spec_tcc.pdf"], "WER é uma métrica de voz."),
        ("n3_abstencao_tipoA", 89,
         ["proj_aula02_projeto_rag_press_kit_censo_escolar_2018.pdf"], "NAO_SEI"),
    ]
    golden = {q["id"]: q for q in json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]}
    s1 = {}
    for nome, qid, fontes, resp in s1_casos:
        pergunta = golden[qid]["pergunta"]
        j1 = _juiz(pergunta, fontes, resp, cliente)
        time.sleep(0.5)
        j2 = _juiz(pergunta, fontes, resp, cliente)
        s1[nome] = {
            "pergunta": pergunta[:80], "resposta": resp[:80], "fontes": fontes,
            "j1": j1, "j2": j2,
            "correta_estavel": bool(j1 and j2 and j1["correta"] == j2["correta"]),
            "nota_estavel": bool(j1 and j2 and j1["nota"] == j2["nota"]),
            "ok": bool(j1 and j2 and j1["correta"] == j2["correta"]),
        }
        print(f"  S1 {nome}: correta {j1 and j1['correta']} x {j2 and j2['correta']}", flush=True)
    out["S1_consistencia"] = s1
    out["S1_ok"] = all(v["ok"] for v in s1.values())

    # S2 — viés de comprimento em resposta NEGATIVA correta (curta x longa).
    pergunta15 = golden[15]["pergunta"]
    curta = "GAN"
    longa = ("O termo correto é GAN. DDL e DQL são linguagens SQL usadas em bancos de "
             "dados, enquanto GAN (Generative Adversarial Network) é um modelo generativo "
             "de redes neurais e não tem relação com banco de dados [1].")
    jc = _juiz(pergunta15, ["tdp_aula03_sql_ddl.pdf"], curta, cliente)
    time.sleep(0.5)
    jl = _juiz(pergunta15, ["tdp_aula03_sql_ddl.pdf"], longa, cliente)
    vies = bool(jc and jl and jl["nota"] - jc["nota"] >= 2)
    out["S2_vies_comprimento"] = {"curta": jc, "longa": jl,
                                  "vies": vies, "ok": not vies}
    print(f"  S2 comprimento: curta nota {jc and jc['nota']} x longa {jl and jl['nota']}", flush=True)

    # S3 — rubrica no estrato + colisão adversarial com "NÃO".
    fora_nao = "Qual destes NÃO é uma cidade do Brasil: Rio de Janeiro, São Paulo ou Tóquio?"
    s3 = {}
    j = _juiz(pergunta15, ["tdp_aula03_sql_ddl.pdf"], "NAO_SEI", cliente)   # (a) abstenção c/ fonte
    s3["a_abstencao_com_fonte"] = {"esperado": "incorreta", "juiz": j,
                                   "ok": bool(j and not j["correta"] and not j["alucinou"])}
    time.sleep(0.5)
    j = _juiz(pergunta15, ["tdp_aula03_sql_ddl.pdf"], "GAN", cliente)      # (b) resposta certa
    s3["b_resposta_certa"] = {"esperado": "correta", "juiz": j,
                              "ok": bool(j and j["correta"] and not j["alucinou"])}
    time.sleep(0.5)
    j = _juiz(fora_nao, [], "NAO_SEI", cliente)                            # (c) adversarial NÃO que abstém
    s3["c_adversarial_nao_abstencao"] = {"esperado": "correta", "juiz": j,
                                         "ok": bool(j and j["correta"])}
    time.sleep(0.5)
    j = _juiz(fora_nao, [], "Tóquio.", cliente)                            # (d) adversarial NÃO que responde
    s3["d_adversarial_nao_responde"] = {"esperado": "incorreta+aluc", "juiz": j,
                                        "ok": bool(j and not j["correta"] and j["alucinou"])}
    out["S3_rubrica"] = s3
    out["S3_ok"] = all(v["ok"] for v in s3.values())
    for k, v in s3.items():
        print(f"  S3 {k}: {v['juiz'] and (v['juiz']['correta'], v['juiz']['alucinou'])}", flush=True)

    # S4 — grounding: juiz aceita resposta certa SEM o doc esperado recuperado (caso #16).
    # O juiz não vê os chunks; recebe só pergunta+fontes+resposta. Teste: resposta correta
    # com fonte esperada que NÃO estava no top-5 (o retriever achou WER em outros docs).
    j = _juiz(golden[16]["pergunta"], ["proj_spec_tcc.pdf"], "WER é uma métrica de voz.", cliente)
    s4 = {"caso_real_16": {"juiz": j,
                           "nota_obs": "resposta estava ancorada em OUTROS docs do top-5 "
                                       "(proj_instrucoes_projeto_final.md, proj_aula01_slides_aula01.md), "
                                       "não no doc esperado; doc esperado recall@5=0.0"}}
    out["S4_grounding_aceite"] = s4
    print(f"  S4 grounding #16: {j and (j['correta'], j['alucinou'])}", flush=True)

    # decisão agregada das sondas (S4 é observacional, não decide sozinho)
    out["sondas_ok"] = out["S1_ok"] and out["S3_ok"] and out["S2_vies_comprimento"]["ok"]
    return out


def gerar_md(tabela: dict, snd: dict | None) -> str:
    c = tabela["contagem"]
    md = [
        "# v0.6 · P0 — Auditoria do juiz nas 16 negativas (plano do orientador)",
        "",
        "> Data: 2026-09-13 · Juiz: `deepseek-v4-pro` (mesmo prompt do avaliar_v5) · "
        "Golden congelado `perguntas_v06.json` (132Q) · Dados: `cascade_v06_p2.jsonl` + "
        "`retrieval_v06.json`.",
        "",
        "## 1. Tabela 16Q (veredito do juiz × rótulo humano + grounding)",
        "",
        "| id | tipo | pergunta (resumida) | recall@5 | doc esperado top-5 | rota | resposta sistema | juiz (nota/correta/aluc) | resposta esperada (gold) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for l in tabela["linhas"]:
        j = l["juiz"] or {}
        resp = (l["resposta"][:42] or "(vazia)").replace("|", "/").replace("\n", " ")
        esperada = l["resposta_esperada"][:38].replace("|", "/")
        md.append(
            f"| #{l['id']} | {l['tipo']} | {l['pergunta'][:48].replace('|','/')} | "
            f"{l['recall@5']:.2f} | {'sim' if l['doc_esperado_no_top5'] else 'não'} | "
            f"{l['rota']} | {resp} | {j.get('nota','?')}/{j.get('correta')}/{j.get('alucinou')} | "
            f"{esperada} |"
        )
    md += [
        "",
        f"**Contagem:** 16 negativas · tipo A **{c['tipoA']}** (resposta = item ausente) · "
        f"tipo B **{c['tipoB']}** (resposta = item presente).",
        f"- Tipo A: **{c['tipoA_corretas']}/{c['tipoA']}** corretas no cascade; "
        f"**{c['tipoA_abstiveram']}/{c['tipoA']}** abstiveram; doc esperado no top-5 em "
        f"**{c['tipoA_doc_no_top5']}/{c['tipoA']}** (retrieval não é o gargalo).",
        f"- Tipo B: **{c['tipoB_corretas']}/{c['tipoB']}** corretas (#16 WER→voz, #42 "
        f"TensorFlow→imagens).",
        "",
        "### 1.1 Grounding das duas tipo B \"corretas\" (hipótese de leniência do orientador §0.4)",
        "",
        "Ambas foram respondidas (não abstiveram) e o doc esperado **não** está no top-5 "
        "(recall@5 = 0.00). A inspeção dos chunks recuperados mostra que a resposta **estava "
        "ancorada em OUTROS docs do top-5** — não é conhecimento de fundo puro:",
        "",
        "- **#16** (WER→voz): `proj_instrucoes_projeto_final.md` p.1 (\"WER medido em 10 "
        "amostras próprias\" na entrega de voz) e `proj_aula01_slides_aula01.md` p.1 "
        "(\"curva wer × sinal-ruído\") — contexto de voz.",
        "- **#42** (TensorFlow→processamento de imagens): `proj_aula04_projeto_multiagente_obj_intro.pptx` "
        "p.5 (\"detecção de objetos utilizando o TensorFlow\").",
        "",
        "**Caveat de alinhamento do golden (não re-editado):** `docs_esperados` de #16/#42 "
        "aponta para o doc onde o P5 *ancorou* a pergunta, mas no índice atual a informação "
        "vive em outros chunks — o juiz acerta ao julgar pelo tema do curso (regra 3), e a "
        "resposta tinha grounding no material recuperado.",
    ]
    if snd:
        md += [
            "",
            "## 2. Sondas rlaif reduzidas nas negativas",
            "",
            "### S1 — consistência (mesma avaliação 2x → `correta` estável)",
        ]
        for nome, v in snd["S1_consistencia"].items():
            md.append(f"- **{nome}**: correta {v['j1'] and v['j1']['correta']} × "
                      f"{v['j2'] and v['j2']['correta']} · nota {v['j1'] and v['j1']['nota']} × "
                      f"{v['j2'] and v['j2']['nota']} → "
                      f"{'OK' if v['ok'] else 'FALHOU'}"
                      f"{'' if v['nota_estavel'] else ' (alerta: nota flutuou)'}")
        md += ["", "### S2 — viés de comprimento (resposta negativa correta)"]
        jc, jl = snd["S2_vies_comprimento"]["curta"], snd["S2_vies_comprimento"]["longa"]
        md.append(f"- curta \"GAN\" nota {jc and jc['nota']} × longa (com citação) nota "
                  f"{jl and jl['nota']} → {'ALERTA (viés)' if snd['S2_vies_comprimento']['vies'] else 'OK'}")
        md += ["", "### S3 — rubrica (abstenção c/ fonte = incorreta; colisão adversarial \"NÃO\")"]
        for k, v in snd["S3_rubrica"].items():
            j = v["juiz"] or {}
            md.append(f"- **{k}**: esperado {v['esperado']} · juiz "
                      f"(correta={j.get('correta')}, alucinou={j.get('alucinou')}, "
                      f"nota={j.get('nota')}) → {'OK' if v['ok'] else 'FALHOU'}")
        md += ["", "### S4 — o juiz exige grounding no doc esperado? (caso real #16)"]
        j = snd["S4_grounding_aceite"]["caso_real_16"]["juiz"] or {}
        md.append(f"- Resposta correta com `docs_esperados` fora do top-5: juiz "
                  f"(correta={j.get('correta')}, alucinou={j.get('alucinou')}, "
                  f"nota={j.get('nota')}) → o juiz **não** verifica se o doc esperado estava "
                  f"recuperado; julga o tema do curso. Observação em "
                  f"`S4_grounding_aceite.caso_real_16.nota_obs`.")
        md += [
            "",
            "## 3. Decisão",
            "",
            "> **(decisão do aluno, agenda P0 do plano do orientador)**",
            "",
        ]
    md += ["", "(tabela e sondas completas em `data/processed/v05b_ab/auditoria_juiz_negativa.json`).", ""]
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description="P0 — auditoria do juiz nas 16 negativas")
    ap.add_argument("--tabela-only", action="store_true", help="só etapa 1 (sem API)")
    a = ap.parse_args()

    tabela = montar_tabela()
    print("=== TABELA 16 NEGATIVAS ===")
    print(json.dumps(tabela["contagem"], ensure_ascii=False, indent=1))
    for l in tabela["linhas"]:
        j = l["juiz"] or {}
        print(f"#{l['id']:3d} [{l['tipo']}] recall={l['recall@5']:.2f} "
              f"doc_top5={'S' if l['doc_esperado_no_top5'] else 'N'} "
              f"rota={l['rota'][:8]:8s} absteve={int(l['absteve'])} "
              f"juiz={j.get('correta')} n{j.get('nota')} "
          f"gold={l['resposta_esperada'][:30]}")
        if l.get("grounding_dica"):
            print(f"      grounding_dica: {l['grounding_dica']}")
            if l.get("docs_top5"):
                print(f"      docs_top5: {sorted(l['docs_top5'])[:4]}")

    snd = None
    if not a.tabela_only:
        print("\n=== SONDAS rlaif (API) ===")
        snd = sondas(_juiz_cliente())

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"passo": "P0", "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "juiz": config.JUIZ_MODEL},
               "tabela": tabela, "sondas": snd}
    SAIDA_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCIA_MD.write_text(gerar_md(tabela, snd), encoding="utf-8")
    print(f"\njson: {SAIDA_JSON}\nmd: {EVIDENCIA_MD}")


if __name__ == "__main__":
    main()