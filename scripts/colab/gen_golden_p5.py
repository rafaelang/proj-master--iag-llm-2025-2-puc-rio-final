"""P5 · Lab — gerar candidatos do golden expandido (v0.6, ≥100 casos).

Expande o golden v0.5b (44Q) para >=100 casos com estratos balanceados, ancorados
no corpus real (chunks.json) e no mesmo criterio_rotulo.md (congelado). Gera:

  rotineira : pergunta direta respondivel por 1 chunk rico (docs_esperados=[doc])
  composta  : pergunta que junta/compara 2+ chunks de docs distintos
  negativa  : pergunta "qual NAO e / nao esta relacionado" (distrator fora do dominio)
  adversarial: pergunta FORA do corpus (docs_esperados=[], deve_abster=true)

Curadoria automatica (reuso do gen_dataset.py): juiz de ancoragem verifica que a
pergunta e respondivel pelo doc esperado; regras deterministicas filtram (tamanho,
formato de pergunta); dedup por pergunta normalizada. Cada candidato guarda a
RESPOSTA_ESPERADA e o trecho fonte (auditoria/revisao humana antes de congelar).

Saida: data/processed/v05b_ab/golden_v06_candidatos.json
  [{"pergunta","estrato","docs_esperados","deve_abster","resposta_esperada",
    "fonte_chunks","motivo"}]
Roda localmente (usa API DeepSeek pro + indice local). Uma execucao por estrato
com --alvo n.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from projeto_final import config
from projeto_final.rag.pipeline import carregar_chunks

load_dotenv()
API_KEY = config.DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise SystemExit("DEEPSEEK_API_KEY nao configurada")
cliente = OpenAI(api_key=API_KEY, base_url=config.DEEPSEEK_BASE_URL)

GEN_MODELO = os.environ.get("GEN_MODELO", config.AGENTE_MODELO_PRO)
JUIZ_MODELO = os.environ.get("JUIZ_MODELO", config.JUIZ_MODEL)
SEED = int(os.environ.get("SEED", "42"))
SAIDA = config.PROCESSED_DIR / "v05b_ab" / "golden_v06_candidatos.json"
MIN_CHUNK = 260          # chunk minimo para servir de fonte (texto substancial)

# extratores
NEG_DISTRATORES = [
    "fundo de investimento", "previsao do tempo", "receita de bolo",
    "cotacao de acao", "teoria da relatividade", "capital do Japao",
    "copa do mundo", "time de futebol",
]


def _chat(modelo: str, system: str, user: str, max_tokens: int = 1200,
          temperature: float = 0.3) -> str:
    for tentativa in range(3):
        try:
            r = cliente.chat.completions.create(
                model=modelo,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  [api] erro ({tentativa+1}/3): {e}", flush=True)
            time.sleep(2)
    return ""


def _extrair_json(txt: str) -> list[dict]:
    if not txt:
        return []
    t = re.sub(r"```[a-z]*", "", txt)
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        return []
    blob = re.sub(r",\s*([\]}])", r"\1", m.group())
    try:
        itens = json.loads(blob)
        return itens if isinstance(itens, list) else []
    except Exception:
        return []


def normalizar(q: str) -> str:
    t = re.sub(r"[^a-z0-9]", "", q.lower())
    return t


def _juiz_ancoragem(pergunta: str, trecho: str, doc: str) -> tuple[bool, str]:
    """Verifica se a pergunta e respondivel pelo trecho/doc (ancoragem)."""
    sys_cur = (
        "Voce e um curador de um dataset de avaliacao. Recebe PERGUNTA e TRECHO "
        "(de um material do curso). Responda se a pergunta pode ser respondida "
        "com INFORMACAO PRESENTE no trecho. Responda APENAS com JSON: "
        '{"ancorada": true|false, "motivo": "<curto>"}'
    )
    user = f"PERGUNTA:\n{pergunta}\n\nTRECHO (fonte: {doc}):\n{trecho[:4000]}"
    for _ in range(2):
        txt = _chat(JUIZ_MODELO, sys_cur, user, max_tokens=600, temperature=0.0)
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                j = json.loads(m.group())
                return bool(j.get("ancorada")), str(j.get("motivo", ""))[:90]
            except Exception:
                pass
        time.sleep(1)
    # fallback determinístico: aceita se termos do(s) doc(s) aparecem na pergunta
    termos = [t for t in re.findall(r"[a-záéíóúâêôçãõ]{4,}", doc.lower().replace(".pdf", "").replace("_", " "))]
    n = sum(1 for t in set(termos) if t in pergunta.lower())
    return (n >= 1, f"fallback: {n} termos do doc na pergunta")


# ----------------------------------------------------------- geradores por estrato

GEN_SYS = (
    "Voce gera PERGUNTAS DE AVALIACAO para um assistente sobre o curso Master IAG "
    "& LLM (PUC-Rio). A pergunta deve ser a que um ALUNO faria, ancorada no trecho "
    "do material fornecido. Regras:\n"
    "- Responda APENAS com JSON: [{\"pergunta\": \"...\", \"resposta_esperada\": \"...\"}]\n"
    "- A resposta_esperada deve estar ANCORADA no trecho (1-2 frases, citando o trecho).\n"
    "- Nao repita perguntas ja existentes nem invente fatos fora do trecho."
)


def gerar_rotineira(chunk: dict) -> list[dict]:
    user = (
        f"Trecho (fonte: {chunk['doc_id']}, pagina {chunk.get('pagina')}):\n\n"
        f"\"{chunk['texto']}\"\n\nGere UMA pergunta rotineira (factual, direta, "
        "respondivel em 1-2 frases com este trecho)."
    )
    return _extrair_json(_chat(GEN_MODELO, GEN_SYS, user, max_tokens=900))


def gerar_composta(c: dict) -> list[dict]:
    user = (
        f"Trecho (fonte: {c['doc_id']}, pagina {c.get('pagina')}):\n\n"
        f"\"{c['texto']}\"\n\nGere UMA pergunta COMPOSTA que exija JUNTAR/COMPARAR "
        "2+ conceitos MENCIONADOS NESTE trecho (diferenca, relacao, como se combinam)."
    )
    return _extrair_json(_chat(GEN_MODELO, GEN_SYS, user, max_tokens=1100))


def gerar_negativa(chunk: dict, distrator: str) -> list[dict]:
    user = (
        f"Trecho (fonte: {chunk['doc_id']}, pagina {chunk.get('pagina')}):\n\n"
        f"\"{chunk['texto']}\"\n\nGere UMA pergunta do tipo NEGATIVA: \"Qual destes "
        "itens NAO e / NAO esta relacionado a <topico do trecho>: <3 conceitos do "
        f"trecho> ou <distrator>{distrator}?\" O distrator deve ser a opcao fora do "
        "dominio do curso. A resposta_esperada deve dizer qual e o item fora do dominio."
    )
    return _extrair_json(_chat(GEN_MODELO, GEN_SYS, user, max_tokens=900))


ADVERSARIAIS = [
    "Qual a previsao do tempo para amanha em Sao Paulo?",
    "Quem venceu a ultima Copa America de futebol?",
    "Qual a cotacao atual do dolar frente ao real?",
    "Qual a receita de um bolo de cenoura?",
    "Explique o funcionamento do sistema imunologico humano.",
    "Qual a populacao atual da cidade de Nova York?",
    "Quem e o atual presidente da Argentina?",
    "Como calcular o imposto de renda no Brasil em 2026?",
    "Qual e o melhor celular do mercado atualmente?",
    "Como funciona a fotossintese das plantas?",
    "Qual o resultado do ultimo jogo do Flamengo?",
    "Onde fica a capital da Australia e qual sua populacao?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estrato", choices=["rotineira", "composta", "negativa", "adversarial"], required=True)
    ap.add_argument("--alvo", type=int, default=0, help="0 = gerar ate nao ter mais fonte")
    a = ap.parse_args()

    random.seed(SEED)
    chunks = carregar_chunks()
    ricos = [c for c in chunks if len(c.get("texto", "")) >= MIN_CHUNK]
    if a.estrato == "composta":
        # usa UM chunk rico por candidato (como a rotineira), mas o prompt pede
        # pergunta COMPARATIVA/RELACIONAL entre 2+ conceitos DENTRO do trecho
        # (padrão do golden atual: #12 "diferença prompt engineering x fine-tuning"
        # tem docs_esperados com 1 doc que cobre os dois conceitos).
        random.shuffle(ricos)
        fontes = [("composta", c) for c in ricos]
    else:
        random.shuffle(ricos)
        fontes = [(a.estrato, c) for c in ricos]

    candidatos = []
    n_gerados = 0
    if a.estrato == "adversarial":
        # fora do corpus: nao usa chunks, nao precisa ancoragem; deve abster
        random.shuffle(ADVERSARIAIS)
        for p in ADVERSARIAIS:
            candidatos.append({
                "pergunta": p, "estrato": "adversarial",
                "docs_esperados": [], "deve_abster": True,
                "resposta_esperada": "NAO_SEI", "fonte_chunks": [],
                "motivo": "fora do corpus",
            })
        n_gerados = len(candidatos)
        # pula o loop de chunks (adversarial nao gera a partir de trechos)
        fontes = []

    for item in fontes:
        if a.alvo and len(candidatos) >= a.alvo:
            break
        if item[0] == "composta":
            _, c1 = item
            itens = gerar_composta(c1)
        elif item[0] == "rotineira":
            _, c1 = item
            itens = gerar_rotineira(c1)
        else:  # negativa
            _, c1 = item
            dist = random.choice(NEG_DISTRATORES)
            itens = gerar_negativa(c1, dist)
        n_gerados += 1
        if not itens:
            continue
        for it in itens:
            pergunta = (it.get("pergunta") or "").strip()
            resp = (it.get("resposta_esperada") or "").strip()
            if len(pergunta) < 12 or "?" not in pergunta or len(resp) < 10:
                continue
            if len(pergunta) > 240:
                continue
            docs_esp = [c1["doc_id"]]
            fonte = [c1["texto"][:200]]
            trecho_juiz = c1["texto"][:4000]
            doc = c1["doc_id"]
            ancorada, motivo = _juiz_ancoragem(pergunta, trecho_juiz, doc)
            if not ancorada:
                print(f"  [rejeitada] nao ancorada: {pergunta[:60]} ({motivo})", flush=True)
                continue
            candidatos.append({
                "pergunta": pergunta, "estrato": a.estrato,
                "docs_esperados": docs_esp, "deve_abster": False,
                "resposta_esperada": resp, "fonte_chunks": fonte,
                "motivo": motivo,
            })
            print(f"  [ok] {pergunta[:70]}", flush=True)

    # dedup por pergunta normalizada
    vistos = {}
    for c in candidatos:
        k = normalizar(c["pergunta"])
        vistos.setdefault(k, c)
    final = list(vistos.values())
    random.seed(SEED)
    random.shuffle(final)

    existentes = []
    if SAIDA.exists():
        existentes = json.loads(SAIDA.read_text(encoding="utf-8"))
    outros = [c for c in existentes if c["estrato"] != a.estrato]
    todos = outros + final
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{a.estrato}] gerados={n_gerados} aprovados={len(final)} "
          f"total_acumulado={len(todos)} ({sum(1 for c in todos if c['estrato']=='rotineira')} rot, "
          f"{sum(1 for c in todos if c['estrato']=='composta')} comp, "
          f"{sum(1 for c in todos if c['estrato']=='negativa')} neg, "
          f"{sum(1 for c in todos if c['estrato']=='adversarial')} adv)")
    print(f"salvo em {SAIDA}")


if __name__ == "__main__":
    main()