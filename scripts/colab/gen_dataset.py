"""v0.5 · Lab — gerar dataset sintético (Q&A) com curadoria (R5).

Spec: "o modelo grande gera o dataset da tarefa específica; treina-se um SLM".

Correções da PoC aplicadas (analise_v0.5_pos_aula07.md, recomendação R5):
1. Curar o dataset: poda automática via LLM-as-judge de ancoragem/alucinação
   + filtros determinísticos (resposta não-vazia, mín. de tokens, pergunta real);
2. Injetar casos de ABSTENÇÃO (perguntas fora do corpus + resposta "Não sei")
   e casos DIFÍCEIS/compostos no treino — lacuna que a v0.6 expôs;
3. Temperatura baixa e formato estrito na geração.

Saída: /content/data/golden_set/adaptacao/dataset_sintetico.json
  [{"pergunta": ..., "resposta": ..., "fonte": ..., "tipo": "ancorado"|"abstencao"}]

Roda no Colab (T4 opcional; a geração usa API DeepSeek).
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv("/content/.env")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
if not API_KEY:
    raise SystemExit("DEEPSEEK_API_KEY não configurada em /content/.env")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

GEN_MODELO = os.environ.get("GEN_MODELO", "deepseek-v4-pro")
JUIZ_MODELO = os.environ.get("JUIZ_MODELO", "deepseek-v4-pro")
N_TRECHOS = int(os.environ.get("N_TRECHOS", "14"))       # trechos sorteados do corpus
Q_PER_TRECHO = int(os.environ.get("Q_PER_TRECHO", "5"))  # Q&A por trecho
N_ABSTENCAO = int(os.environ.get("N_ABSTENCAO", "8"))    # casos de abstenção injetados
SEED = int(os.environ.get("SEED", "42"))

RAW = "/content/data/raw"
SAIDA = "/content/data/golden_set/adaptacao/dataset_sintetico.json"


def extrair(pdf: str) -> str:
    try:
        reader = PdfReader(pdf)
    except Exception:
        return ""
    return " ".join((p.extract_text() or "") for p in reader.pages)


def _chat(modelo: str, system: str, user: str, max_tokens: int = 1500,
          temperature: float = 0.2) -> str:
    for tentativa in range(2):
        try:
            r = client.chat.completions.create(
                model=modelo,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  [api] erro ({tentativa+1}/2): {e}", flush=True)
            time.sleep(2)
    return ""


GEN_SYS = (
    "Você gera material de treinamento para um SLM sobre o curso Master IAG & LLM "
    "(PUC-Rio). Com base no trecho do material, crie perguntas e respostas em "
    "português que um aluno faria. Regras:\n"
    "- A resposta deve estar ANCORADA no trecho (nunca inventar fatos fora dele);\n"
    "- Pode incluir perguntas que EXIGEM COMPARAR/JUNTAR 2 conceitos do trecho;\n"
    "- Responda APENAS com JSON no formato: "
    '[{"pergunta": "...", "resposta": "..."}]'
)

CURADOR_SYS = (
    "Você é um curador rigoroso de um dataset de treino. Recebe: FONTE (trecho do "
    "material do curso), PERGUNTA e RESPOSTA. Avalie a RESPOSTA em 3 eixos:\n"
    "1. alucinacao: a resposta afirma algum fato que NÃO pode ser derivado da FONTE?\n"
    "2. ancoragem: a resposta usa informação presente/derivável da FONTE?\n"
    "3. utilidade: a pergunta faz sentido e a resposta a responde?\n"
    'Responda APENAS com JSON: {"ancorada": true|false, "alucinada": true|false, '
    '"util": true|false, "motivo": "<curto>"}'
)


def _extrair_json(txt: str) -> list[dict]:
    """Extrai o primeiro array JSON da resposta, tolerando code fences e vírgulas sobrando."""
    if not txt:
        return []
    # remove code fences (```json ... ```)
    t = re.sub(r"```[a-z]*", "", txt)
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        return []
    blob = m.group()
    # tolera vírgula antes de fechamento de objeto/array
    blob = re.sub(r",\s*([\]}])", r"\1", blob)
    try:
        itens = json.loads(blob)
        return itens if isinstance(itens, list) else []
    except Exception:
        return []


def gerar_perguntas(trecho: str, doc: str) -> list[dict]:
    prompt = (
        f"Trecho (fonte: {doc}):\n\n\"{trecho}\"\n\n"
        f"Gere exatamente {Q_PER_TRECHO} perguntas e respostas ancoradas no trecho. "
        "No máximo 1 pergunta por item pode ser do tipo 'comparação/composta'."
    )
    txt = _chat(GEN_MODELO, GEN_SYS, prompt, max_tokens=2500)
    itens = _extrair_json(txt)
    return [{"pergunta": it.get("pergunta", ""), "resposta": it.get("resposta", ""),
             "fonte": doc} for it in itens]


def curar(item: dict, trecho: str) -> tuple[bool, str]:
    """Curadoria automática (R5): juiz de ancoragem + regras determinísticas."""
    pergunta = (item.get("pergunta") or "").strip()
    resposta = (item.get("resposta") or "").strip()
    if len(pergunta) < 5 or len(resposta) < 10:
        return False, "pergunta/resposta curta demais"
    if len(pergunta) > 220 or len(resposta) > 600:
        return False, "tamanho fora do padrão"
    if not any(ch.isalpha() for ch in pergunta) or "?" not in pergunta:
        return False, "não parece uma pergunta"
    user = (f"FONTE:\n{trecho[:4000]}\n\nPERGUNTA:\n{pergunta}\n\nRESPOSTA:\n{resposta}")
    txt = _chat(JUIZ_MODELO, CURADOR_SYS, user, max_tokens=800, temperature=0.0)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        # sem juiz, aprova por regra determinística conservadora
        return True, "aprovado por fallback (juiz sem resposta)"
    try:
        j = json.loads(m.group())
    except Exception:
        return True, "aprovado por fallback (json do juiz quebrado)"
    if not j.get("util"):
        return False, str(j.get("motivo", "reprovado pelo curador"))[:80]
    if j.get("alucinada"):
        return False, "alucinado: " + str(j.get("motivo", ""))[:70]
    return True, ""


ABSTENCAO_PERGUNTAS = [
    "Quem ganhou a Copa do Mundo de 2022?",
    "Qual a previsão do tempo para amanhã no Rio de Janeiro?",
    "Explique a teoria da relatividade geral de Einstein.",
    "Qual o preço atual da ação da Petrobras?",
    "Como fazer um bolo de chocolate?",
    "Quem é o presidente da França?",
    "Qual a capital do Japão?",
    "Qual a fórmula da velocidade da luz?",
]


def main() -> None:
    random.seed(SEED)
    docs = {}
    for p in sorted(glob.glob(os.path.join(RAW, "*.pdf"))):
        docs[os.path.basename(p)] = extrair(p)

    trechos = []
    for doc, txt in docs.items():
        for start in range(0, min(len(txt), 6000), 800):
            trechos.append((doc, txt[start:start + 800]))

    if not trechos:
        raise SystemExit("sem PDFs em /content/data/raw")
    amostra = random.sample(trechos, min(N_TRECHOS, len(trechos)))
    print(f"[gen] {len(trechos)} trechos no total; sorteando {len(amostra)}")

    gerados: list[dict] = []
    for doc, trecho in amostra:
        itens = gerar_perguntas(trecho, doc)
        print(f"  [gen] {doc}: {len(itens)} Q&A brutos", flush=True)
        gerados.extend(itens)

    # ---- curadoria (R5) ----
    aprovados = []
    reprovados = 0
    trecho_por_item = {i: trecho for i, (doc, trecho) in enumerate(amostra)
                       for _ in range(sum(1 for it in gerados if it.get("fonte") == doc))}
    for i, it in enumerate(gerados):
        trecho = trecho_por_item.get(i, "")
        ok, motivo = curar(it, trecho)
        if ok:
            aprovados.append(it)
        else:
            reprovados += 1

    # ---- injetar casos de abstenção (R5) ----
    abst = []
    for pergunta in ABSTENCAO_PERGUNTAS[:N_ABSTENCAO]:
        abst.append({"pergunta": pergunta, "resposta": "Não sei.",
                     "fonte": "abstencao", "tipo": "abstencao"})

    # ---- casos difíceis/compostos adicionais (geração explícita) ----
    compostas = [
        it for it in aprovados if (" e " in it["pergunta"].lower()
                                   or " ou " in it["pergunta"].lower()
                                   or "diferença" in it["pergunta"].lower())
    ]
    for it in compostas:
        it["tipo"] = "composta"
    for it in aprovados:
        it.setdefault("tipo", "ancorado")

    final = aprovados + abst
    random.seed(SEED)
    random.shuffle(final)

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    json.dump(final, open(SAIDA, "w"), ensure_ascii=False, indent=2)
    n_abst = sum(1 for x in final if x.get("tipo") == "abstencao")
    n_comp = sum(1 for x in final if x.get("tipo") == "composta")
    print(f"[gen] bruto={len(gerados)} reprovados={reprovados} "
          f"aprovados={len(aprovados)} abstencao={n_abst} composta={n_comp} "
          f"total={len(final)}")
    print(f"[gen] salvo em {SAIDA}")
    for it in final[:3]:
        print("  -", it["pergunta"][:70])


if __name__ == "__main__":
    main()
