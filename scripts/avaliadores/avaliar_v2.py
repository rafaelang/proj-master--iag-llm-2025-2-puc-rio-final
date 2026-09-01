"""Avaliacao v0.2 do pipeline RAG alvo (projeto_final) com o dataset unico.

Padrao de avaliadores: scripts/avaliadores/avaliar_v{n}.py.

- Dataset unico da v0.2: data/golden_set/rag/perguntas.json (copia da tag v0.2 do projeto2).
- Evidencia oficial em Markdown: docs/v02_evidencia.md.
- JSON intermediario (diagnostico): data/processed/rag/avaliacao_v2.json (fora do Git).

Criterios (reaproveitados do projeto2, mais rigorosos e honestos):
- detectar_abstencao: heuristica robusta (nao_sei / nao sei / nao ha informacoes / nao tenho informacoes);
- extrair_citacoes + auditoria DETERMINISTICA de citacao (fiel / fora / fantasma), sem usar LLM;
- juiz LLM-as-judge para acerto end-to-end (mesmo prompt do juiz_correcao.py v0.6;
  modelo configurado por JUIZ_MODEL, padrao deepseek-v4-pro).

NAO altera o golden set. Apenas le data/golden_set/rag/perguntas.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime

from loguru import logger
from openai import OpenAI

from projeto_final import config
from projeto_final import llm as llm_mod
from projeto_final.rag.pipeline import _formatar_contexto, carregar_chunks
from projeto_final.rag.retrieve import recuperar

GOLDEN = config.RAG_GOLDEN_SET
JSON_SAIDA = config.RAG_DIR / "avaliacao_v2.json"
EVIDENCIA_MD = config.DOCS_DIR / "v02_evidencia.md"
HISTORICO = config.RAG_DIR / "avaliacao_historico.json"
# Marcador para preservar secoes manuais em docs/v02_evidencia.md (ex.: Terceira Rodada).
MARKER = "<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->"

TOP_K_RECALL = 5    # granularidade de documento, como a v0.2 real (recall@5)
TOP_K_GERACAO = 10  # blocos de contexto enviados ao LLM na geracao


# ------------------------------------------------------- criterios (referencia projeto2)

def detectar_abstencao(resposta: str) -> bool:
    """Heuristica robusta de abstencao (identica a do projeto2)."""
    r = (resposta or "").lower().strip()
    return (r.startswith(("nao_sei", "nao sei", "não_sei", "não sei"))
            or "não há informações" in r[:60]
            or "não tenho informações" in r[:60])


def extrair_citacoes(resposta: str) -> list[int]:
    """Numeros [n] presentes na resposta, na ordem de aparicao."""
    return [int(n) for n in re.findall(r"\[(\d+)\]", resposta or "")]


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


def juiz(pergunta: str, fontes: list[str], resposta: str, cliente: OpenAI) -> dict | None:
    """LLM-as-judge (mesmo prompt do projeto2) para acerto end-to-end."""
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
            logger.error("Juiz falhou (chamada): {}", e)
            return None
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


# ------------------------------------------------------- avaliacao

def _recall(l: dict) -> float | None:
    """Compatibilidade com chaves 'recall_k5' (nova) e 'recall@5' (antiga)."""
    return l.get("recall_k5", l.get("recall@5"))


def avaliar() -> dict:
    perguntas = json.loads(GOLDEN.read_text(encoding="utf-8"))["perguntas"]
    chunks = carregar_chunks()
    cliente = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    sistema = config.ler_prompt("v0.2/rag_sistema.txt") or ""

    linhas = []
    for q in perguntas:
        qid = q["id"]
        esperados = set(q.get("docs_esperados", []))
        deve_abster = q.get("deve_abster", False)
        logger.info("== #{} [{}] {}", qid, q["estrato"], q["pergunta"][:60])

        # 1) recuperacao top-K (uma unica busca; o top-5 deriva do top-10)
        top10 = recuperar(q["pergunta"], chunks, top_k=TOP_K_GERACAO)
        docs_rec5 = [c["arquivo"] for c in top10[:TOP_K_RECALL]]
        docs_rec5_set = set(docs_rec5)
        recall = (len(esperados & docs_rec5_set) / len(esperados)) if esperados else None

        # 2) geracao com citacoes (mesmo contexto do pipeline alvo)
        contexto = _formatar_contexto(top10)
        try:
            resposta, _meta = llm_mod.responder_com_contexto(q["pergunta"], contexto, sistema=sistema)
        except Exception as e:
            logger.error("Geracao falhou na #{}: {}", qid, e)
            resposta = ""

        absteve = detectar_abstencao(resposta)

        # 3) citacoes e auditoria DETERMINISTICA
        cit_n = extrair_citacoes(resposta)
        citacoes = []
        for n in cit_n:
            doc = top10[n - 1]["arquivo"] if 1 <= n <= len(top10) else None
            citacoes.append({"n": n, "doc": doc})

        # 4) acerto end-to-end (juiz)
        j = juiz(q["pergunta"], sorted(esperados), resposta, cliente)

        linha = {
            "id": qid,
            "estrato": q["estrato"],
            "pergunta": q["pergunta"],
            "resposta": resposta,
            "absteve": absteve,
            "deve_abster": deve_abster,
            "recall_k5": recall,
            "docs_esperados": sorted(esperados),
            "docs_recuperados_top5": docs_rec5,
            "citacoes": citacoes,
            "alucinou": j["alucinou"] if j else None,
            "acerto": j["correta"] if j else None,
            "juiz_nota": j["nota"] if j else None,
            "juiz_justificativa": j["justificativa"] if j else None,
        }
        linhas.append(linha)
        logger.info("   absteve={} deve={} recall={} acerto={} cit={}",
                    absteve, deve_abster, recall, linha["acerto"], cit_n)

    resumo = _agregar(linhas)
    meta = {
        "golden_set": str(GOLDEN.relative_to(config.RAIZ)),
        "pipeline_alvo": f"projeto_final (BM25 + fastembed + RRF, DeepSeek {config.DEEPSEEK_MODEL})",
        "criterios": "projeto2 (detectar_abstencao, extrair_citacoes, auditar_citacao, juiz LLM-as-judge)",
        "juiz_model": config.JUIZ_MODEL,
        "top_k_recall": TOP_K_RECALL,
        "top_k_geracao": TOP_K_GERACAO,
    }
    return {"meta": meta, "resumo": resumo, "perguntas": linhas}


def _agregar(linhas: list[dict]) -> dict:
    estrato_fixos = ["rotineira", "composta", "negativa", "adversarial"]

    com_docs = [l for l in linhas if l["docs_esperados"]]
    recalls = [_recall(l) for l in com_docs if _recall(l) is not None]

    abs_ok = [1 for l in linhas if l["absteve"] == l["deve_abster"]]

    devem_responder = [l for l in linhas if not l["deve_abster"]]
    abs_indevida = [l for l in devem_responder if l["absteve"]]

    julgadas = [l for l in linhas if l["acerto"] is not None]
    acertos = [l for l in julgadas if l["acerto"]]

    nao_abstidas = [l for l in linhas if not l["absteve"]]
    com_cit = [l for l in nao_abstidas if l["citacoes"]]

    aud = _auditar_citacao(linhas)

    por_estrato = {}
    for est in estrato_fixos:
        le = [l for l in linhas if l["estrato"] == est]
        if not le:
            continue
        le_rec = [_recall(l) for l in le if _recall(l) is not None]
        le_abs_ok = sum(1 for l in le if l["absteve"] == l["deve_abster"]) / len(le)
        le_nao_abst = [l for l in le if not l["absteve"]]
        le_com_cit = sum(1 for l in le_nao_abst if l["citacoes"]) / max(1, len(le_nao_abst))
        le_acertos = [l for l in le if l["acerto"]]
        por_estrato[est] = {
            "n": len(le),
            "recall_k5": round(sum(le_rec) / len(le_rec), 3) if le_rec else None,
            "acerto": round(len(le_acertos) / len(le), 3),
            "abstencao_correta": round(le_abs_ok, 3),
            "citacao_presente": round(le_com_cit, 3),
        }

    return {
        "n_perguntas": len(linhas),
        "recall_k5_media": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "acerto_end_to_end": round(len(acertos) / max(1, len(julgadas)), 3),
        "abstencao_correta_20": round(sum(abs_ok) / len(linhas), 3),
        "abstencao_indevida": {"n": len(abs_indevida),
                               "taxa": round(len(abs_indevida) / max(1, len(devem_responder)), 3)},
        "citacao_presente": round(len(com_cit) / max(1, len(nao_abstidas)), 3),
        "auditoria_citacao": aud,
        "por_estrato": por_estrato,
        "juiz_nao_julgadas": len(linhas) - len(julgadas),
    }


def _auditar_citacao(linhas: list[dict]) -> dict:
    """Auditoria deterministica (projeto2): fiel / fora / fantasma."""
    corpo = com_cit = fiel = fora = fantasma = 0
    for l in linhas:
        if l["absteve"]:
            continue
        corpo += 1
        cit = l["citacoes"] or []
        esperados = set(l["docs_esperados"] or [])
        if not cit or not esperados:
            continue
        com_cit += 1
        docs = {c["doc"] for c in cit if c.get("doc")}
        n_max = TOP_K_GERACAO
        if any(c["n"] > n_max for c in cit):
            fantasma += 1
        if docs and docs.issubset(esperados):
            fiel += 1
        else:
            fora += 1
    return {"respostas_corpo": corpo, "com_citacao": com_cit,
            "citacao_fiel": fiel, "citacao_fora": fora, "citacao_fantasma": fantasma}


def _falhas(linhas: list[dict]) -> list[dict]:
    falhas = []
    for l in linhas:
        causas = []
        if l["docs_esperados"] and _recall(l) == 0:
            causas.append("recuperacao falhou (nenhum doc esperado no top-5)")
        if not l["deve_abster"] and l["absteve"]:
            causas.append("abstencao indevida")
        if l["deve_abster"] and not l["absteve"]:
            causas.append("respondeu fora do corpus (deveria abster)")
        if l["acerto"] is False:
            causas.append("resposta incorreta (juiz): " + (l["juiz_justificativa"] or "")[:140])
        if l["acerto"] is None:
            causas.append("juiz nao julgou (erro de parse/chamada)")
        if l["citacoes"]:
            esperados = set(l["docs_esperados"] or [])
            docs = {c["doc"] for c in l["citacoes"] if c.get("doc")}
            if any(c["n"] > TOP_K_GERACAO for c in l["citacoes"]):
                causas.append("citacao fantasma")
            if docs and not docs.issubset(esperados):
                causas.append("citacao fora")
        if causas:
            falhas.append({"id": l["id"], "estrato": l["estrato"],
                           "pergunta": l["pergunta"], "causas": causas,
                           "resposta": (l["resposta"] or "")[:200]})
    return falhas


# ------------------------------------------------------- evidencia (Markdown)

def gerar_evidencia(out: dict, historico: list[dict] | None = None) -> str:
    meta = out.get("meta", {})
    r = out["resumo"]
    md = [
        "# v0.2 - Evidencia - RAG (dataset unico do projeto2)",
        "",
        f"> Pipeline alvo: `{meta.get('pipeline_alvo', 'projeto_final (BM25 + fastembed + RRF, DeepSeek)')}`",
        f"> Dataset: `{meta.get('golden_set', 'data/golden_set/rag/perguntas.json')}`.",
        f"> Criterios: {meta.get('criterios', 'detectar_abstencao + auditoria deterministica de citacao + juiz LLM-as-judge')}.",
        f"> Juiz: LLM-as-judge `{meta.get('juiz_model', config.JUIZ_MODEL)}` (mesmo prompt do 'juiz_correcao.py' v0.6).",
        f"> Prompt de geração: `rag_sistema.txt#{_prompt_hash()}`.",
        "",
        "## Metricas por estrato",
        "",
        "| Estrato | n | recall@5 | acerto | abstencao correta | citacao presente |",
        "|---|---|---|---|---|---|",
    ]
    for est, m in r["por_estrato"].items():
        rec = f"{m['recall_k5']:.3f}" if m.get("recall_k5") is not None else "-"
        md.append(f"| {est} | {m['n']} | {rec} | {m['acerto']:.3f} | {m['abstencao_correta']:.3f} | {m['citacao_presente']:.3f} |")

    md += ["", "## Resumo honesto", ""]
    n_devem = len([l for l in out["perguntas"] if not l["deve_abster"]])
    rec_txt = f"{r['recall_k5_media']:.3f}" if r.get("recall_k5_media") is not None else "-"
    md += [
        f"- **recall@5** (sobre as {n_devem} com `docs_esperados`): **{rec_txt}**",
        f"- **acurácia end-to-end** (sobre {r['n_perguntas']}, juiz LLM-as-judge): **{r['acerto_end_to_end']:.3f}**",
        f"- **abstenção correta** (sobre {r['n_perguntas']}): **{r['abstencao_correta_20']:.3f}**",
        f"- **abstenção indevida** (sobre as {n_devem} com `deve_abster`: false): **{r['abstencao_indevida']['taxa']:.3f}** ({r['abstencao_indevida']['n']})",
        f"- **citação presente** (respostas não-abstidas com `[n]`): **{r['citacao_presente']:.3f}**",
        f"- **auditoria de citação** — fiel: {r['auditoria_citacao']['citacao_fiel']} · fora: {r['auditoria_citacao']['citacao_fora']} · fantasma: {r['auditoria_citacao']['citacao_fantasma']}",
        f"- **alucinações** (juiz flag \"alucinou\"): {sum(1 for l in out['perguntas'] if l.get('alucinou'))}",
        f"- juiz não julgou (erro de parse): {r['juiz_nao_julgadas']}",
        "",
    ]

    falhas = _falhas(out["perguntas"])
    md += ["## Falhas detectadas", ""]
    if falhas:
        for f in falhas:
            md.append(f"### #{f['id']:02d} [{f['estrato']}] {f['pergunta']}")
            for c in f["causas"]:
                md.append(f"- **causa:** {c}")
            md.append(f"- **resposta:** {f['resposta']}")
            md.append("")
    else:
        md.append("Nenhuma falha detectada.")
        md.append("")

    md += ["## Detalhes por pergunta", ""]
    for l in out["perguntas"]:
        rec = _recall(l)
        rec_txt = f"{rec}" if rec is not None else "-"
        cit = "; ".join(f"[{c['n']}] {c['doc']}" for c in (l.get("citacoes") or []) if c.get("doc")) or "-"
        acerto = "true" if l.get("acerto") is True else ("false" if l.get("acerto") is False else "-")
        aluc = "true" if l.get("alucinou") is True else ("false" if l.get("alucinou") is False else "-")
        nota = l.get("juiz_nota")
        md.append(f"### #{l['id']:02d} [{l['estrato']}] {l['pergunta']}")
        md.append(f"- **Resposta:** {l['resposta']}")
        md.append(f"- **Absteve:** {l['absteve']} · **Deve abster:** {l['deve_abster']} · **recall@5:** {rec_txt}")
        md.append(f"- **Acerto (juiz):** {acerto} · **Alucinou:** {aluc} · **Nota:** {nota if nota is not None else '-'}")
        if l.get("juiz_justificativa"):
            md.append(f"- **Justificativa do juiz:** {l['juiz_justificativa']}")
        md.append(f"- **Citações:** {cit}")
        md.append("")

    if historico:
        md += ["## Histórico de medições", ""]
        md += [
            "| Data | Geração | Juiz | Prompt | recall@5 | acerto | abstenção correta | abstenção indevida | citação presente | auditoria fiel/fora/fantasma |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for h in historico:
            r = h["resumo"]
            rec = f"{r['recall_k5_media']:.3f}" if r.get("recall_k5_media") is not None else "-"
            aud = r["auditoria_citacao"]
            md.append(
                f"| {h['data']} | {h['modelo_geracao']} | {h['juiz_model']} | {h['prompt']} | {rec} "
                f"| {r['acerto_end_to_end']:.3f} | {r['abstencao_correta_20']:.3f} "
                f"| {r['abstencao_indevida']['taxa']:.3f} | {r['citacao_presente']:.3f} "
                f"| {aud['citacao_fiel']}/{aud['citacao_fora']}/{aud['citacao_fantasma']} |"
            )
        md.append("")

    return "\n".join(md) + "\n"


def _prompt_hash() -> str:
    """Hash curto do prompt de geracao atual (para identificar versoes no historico)."""
    sistema = config.ler_prompt("v0.2/rag_sistema.txt") or ""
    return hashlib.sha256(sistema.encode("utf-8")).hexdigest()[:8]


def _carregar_historico() -> list[dict]:
    if HISTORICO.exists():
        try:
            return json.loads(HISTORICO.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Historico invalido; reiniciando a lista de medicoes")
    return []


def main() -> None:
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(config.RAG_DIR / "avaliacao_v2.log", rotation="1 MB", level="DEBUG")
    logger.info("Avaliacao v0.2 iniciada: dataset unico {}", GOLDEN)
    out = avaliar()

    historico = _carregar_historico()
    historico.append({
        "data": datetime.now().isoformat(timespec="seconds"),
        "modelo_geracao": config.DEEPSEEK_MODEL,
        "juiz_model": config.JUIZ_MODEL,
        "prompt": f"rag_sistema.txt#{_prompt_hash()}",
        "resumo": out["resumo"],
    })
    HISTORICO.write_text(json.dumps(historico, ensure_ascii=False, indent=2), encoding="utf-8")

    JSON_SAIDA.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    novo_md = gerar_evidencia(out, historico)
    manual = ""
    if EVIDENCIA_MD.exists():
        atual = EVIDENCIA_MD.read_text(encoding="utf-8")
        if MARKER in atual:
            manual = atual.split(MARKER, 1)[1]
    EVIDENCIA_MD.write_text(novo_md + MARKER + "\n\n" + manual, encoding="utf-8")
    print("RESUMO:", json.dumps(out["resumo"], ensure_ascii=False, indent=2))
    print(f"json intermediario: {JSON_SAIDA}")
    print(f"historico de medicoes: {HISTORICO}")
    print(f"evidencia (markdown): {EVIDENCIA_MD}")


if __name__ == "__main__":
    main()
