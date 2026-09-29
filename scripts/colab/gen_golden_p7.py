"""v0.7 · Gerar candidatos do golden expandido (~500 casos) — receita P5.

Mesma mecanica do P5 (`gen_golden_p5.py`), evoluida para o golden v0.7:
  - preserva a receita: ancoragem em chunks reais + juiz de ancoragem (pro) +
    fallback deterministico + dedup por pergunta normalizada;
  - saida propria (`data/processed/v07_ab/golden_v07_candidatos.json`), nao toca
    o P5/v0.6;
  - lista adversarial expandida (fora do corpus, escrita a mao — sem API).

Estratos e alvo (novos, alem das 132 congeladas da v0.6):
  rotineira 200 · composta 80 · negativa 50 · adversarial 50  (total ~512)

Saida (dado intermediario, fora do Git):
  data/processed/v07_ab/golden_v07_candidatos.json
  [{"pergunta","estrato","docs_esperados","deve_abster","resposta_esperada",
    "fonte_chunks","motivo"}]

Roda localmente (API DeepSeek pro + indice local). Uma execucao por estrato com --alvo n:
  python scripts/colab/gen_golden_p7.py --estrato rotineira --alvo 200
  python scripts/colab/gen_golden_p7.py --estrato composta  --alvo 80
  python scripts/colab/gen_golden_p7.py --estrato negativa  --alvo 50
  python scripts/colab/gen_golden_p7.py --estrato adversarial --alvo 50
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
cliente = OpenAI(api_key=API_KEY, base_url=config.DEEPSEEK_BASE_URL,
                 default_headers={"x-opencode-session": os.getenv("OPENCODE_SESSION_ID", "projeto-final-assistente"),
                                  "User-Agent": "projeto-final-assistente/1.0"})

GEN_MODELO = os.environ.get("GEN_MODELO", config.AGENTE_MODELO_PRO)
JUIZ_MODELO = os.environ.get("JUIZ_MODELO", config.JUIZ_MODEL)
SEED = int(os.environ.get("SEED", "42"))
SAIDA = config.PROCESSED_DIR / "v07_ab" / "golden_v07_candidatos.json"
MIN_CHUNK = 260  # chunk minimo para servir de fonte (texto substancial)

NEG_DISTRATORES = [
    "fundo de investimento", "previsao do tempo", "receita de bolo",
    "cotacao de acao", "teoria da relatividade", "capital do Japao",
    "copa do mundo", "time de futebol",
]

# Fora do corpus (deve abster). Expandido p/ o golden v0.7 (~70 itens).
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
    "Qual o melhor investimento para renda fixa em 2026?",
    "Quais os sintomas da dengue?",
    "Como trocar o pneu de um carro?",
    "Qual a idade minima para dirigir na California?",
    "Quem escreveu Dom Casmurro?",
    "Qual a formula da agua e por que e importante?",
    "Como preparar um currículo para o LinkedIn?",
    "Qual o valor do salario minimo em Portugal?",
    "Como funciona a energia solar residencial?",
    "Quem foi a primeira pessoa a pisar na Lua?",
    "Qual a capital da Bolivia?",
    "Como plantar e cuidar de um bonsai?",
    "Qual a historia da rede social Instagram?",
    "O que e o efeito estufa e quais suas causas?",
    "Qual o codigo de area telefonico do Rio de Janeiro?",
    "Como fazer uma viagem internacional com pouco dinheiro?",
    "Quais as regras do volei de praia?",
    "Qual o nome do maior deserto do mundo?",
    "Como funciona o seguro desemprego no Brasil?",
    "Qual o melhor exercicio para emagrecer?",
    "Quem fundou a Apple e em que ano?",
    "Qual a diferenca entre cafe arabica e robusta?",
    "Como resolver problemas de conexao wifi em casa?",
    "Qual a taxa de desemprego atual nos Estados Unidos?",
    "O que e a teoria da evolucao de Darwin?",
    "Como declarar criptomoedas no imposto de renda?",
    "Qual a receita do pato no tucupi?",
    "Como escolher um curso de ingles online?",
    "Qual a populacao mundial em 2026?",
    "Quais os beneficios do jejum intermitente?",
    "Como funciona o sistema de pontos do SUS?",
    "Qual o autor do livro O Pequeno Principe?",
    "Como montar um planejamento financeiro pessoal?",
    "Qual a capital da Nova Zelandia?",
    "Como limpar e conservar panelas de ferro?",
    "Quais os efeitos do cafe na saude?",
    "O que e o Bolsa Familia e quem tem direito?",
    "Como configurar um roteador de internet?",
    "Qual a melhor epoca para plantar milho no Brasil?",
    "Como funciona o pagamento por aproximacao (NFC)?",
    "Quem ganhou o Premio Nobel da Paz em 2024?",
    "Qual a altura da Torre Eiffel?",
    "Como tratar uma gripe resfriado em casa?",
    "Qual o pais com mais campeonatos de Formula 1?",
    "Como funciona o transporte por aplicativo no transito?",
    "Qual a diferenca entre PIB e IDH?",
    "O que e a ginastica artistica e suas modalidades?",
    "Como economizar energia eletrica em casa?",
    "Qual o papel do FMI na economia mundial?",
    "Como fazer um orcamento de reforma?",
    "Qual a capital da Coreia do Sul?",
    "Quais os passos para abrir uma empresa no Brasil?",
    "Como funciona o voto em segundo turno?",
    "Qual o melhor time de basquete da NBA?",
    "Como escolher uma faculdade de medicina?",
    "Qual o valor do frete de uma encomenda internacional?",
    "Quais as especies de orquideas mais comuns no Brasil?",
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
    termos = [t for t in re.findall(r"[a-záéíóúâêôçãõ]{4,}", doc.lower().replace(".pdf", "").replace("_", " "))]
    n = sum(1 for t in set(termos) if t in pergunta.lower())
    return (n >= 1, f"fallback: {n} termos do doc na pergunta")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estrato", choices=["rotineira", "composta", "negativa", "adversarial"], required=True)
    ap.add_argument("--alvo", type=int, default=0, help="0 = gerar ate nao ter mais fonte")
    a = ap.parse_args()

    random.seed(SEED)
    chunks = carregar_chunks()
    ricos = [c for c in chunks if len(c.get("texto", "")) >= MIN_CHUNK]
    random.shuffle(ricos)
    fontes = [(a.estrato, c) for c in ricos]

    candidatos = []
    n_gerados = 0
    if a.estrato == "adversarial":
        random.shuffle(ADVERSARIAIS)
        for p in ADVERSARIAIS:
            candidatos.append({
                "pergunta": p, "estrato": "adversarial",
                "docs_esperados": [], "deve_abster": True,
                "resposta_esperada": "NAO_SEI", "fonte_chunks": [],
                "motivo": "fora do corpus",
            })
        n_gerados = len(candidatos)
        fontes = []

    for item in fontes:
        if a.alvo and len(candidatos) >= a.alvo:
            break
        _, c1 = item
        if a.estrato == "composta":
            itens = gerar_composta(c1)
        elif a.estrato == "rotineira":
            itens = gerar_rotineira(c1)
        else:  # negativa
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
            ancorada, motivo = _juiz_ancoragem(pergunta, c1["texto"][:4000], c1["doc_id"])
            if not ancorada:
                print(f"  [rejeitada] nao ancorada: {pergunta[:60]} ({motivo})", flush=True)
                continue
            candidatos.append({
                "pergunta": pergunta, "estrato": a.estrato,
                "docs_esperados": [c1["doc_id"]], "deve_abster": False,
                "resposta_esperada": resp, "fonte_chunks": [c1["texto"][:200]],
                "motivo": motivo,
            })
            print(f"  [ok] {pergunta[:70]}", flush=True)

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