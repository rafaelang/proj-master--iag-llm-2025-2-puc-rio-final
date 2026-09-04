"""Roteador classico R2 — TF-IDF (char_wb 3-5) + XGBoost — estudo v0.4.

Comparado ao roteador SLM (R1), roda **sem llama.cpp/torch**: ~<1 ms por
classificacao, ~10 MB de artefato, US$ 0.00 e robusto a typos de ASR (n-grams de
caracteres dentro de palavras absorvem "diferenca"/"diferença", trocas/quedas de
letras). Referencia: decisao v1.0 do prototipo projeto2 (0.945 em holdout).

Os ROTULOS abaixo sao o ground-truth do estudo (13 SIMPLES / 7 COMPLEXAS,
mesmo criterio de complexidade do prompt do roteador). Treino e LAZY (singleton,
sem dependencia de arquivo externo); o script de dataset
(scripts/avaliadores/gerar_dataset_roteador.py) apenas materializa os mesmos
rotulos + variacoes com typo em JSON para a avaliacao.
"""

from __future__ import annotations

import pickle
import threading
import unicodedata

from loguru import logger

NOME = "TF-IDF char_wb(3-5) + XGBoost"

# Ground-truth de complexidade (pergunta canonica do golden set -> rotulo).
# SIMPLES = factual/direta/1 fonte · COMPLEXA = composta/comparativa/ambigua/multifonte.
ROTULOS: dict[str, str] = {
    # rotineiras factuais
    "O que é RAG?": "SIMPLES",
    "O que significa DDL e para que serve?": "SIMPLES",
    "O que é DQL?": "SIMPLES",
    "Como o FastAPI é usado para servir um modelo?": "SIMPLES",
    "O que são GAN e VAE?": "SIMPLES",
    "O que é OCR e como ele ajuda na recuperação de documentos?": "SIMPLES",
    "O que é fine-tuning de um modelo de linguagem?": "SIMPLES",
    "O que é transfer learning?": "SIMPLES",
    # compostas / relacionais (definicao + uso, comparacao, multi-fonte) -> COMPLEXA
    "O que é WER e como ele é usado na avaliação de voz?": "COMPLEXA",
    "O que são sistemas multiagentes com small language models?": "COMPLEXA",
    "O que são embeddings e como são usados no RAG?": "COMPLEXA",
    "Qual a diferença entre prompt engineering e fine-tuning?": "COMPLEXA",
    "O que é atenção em modelos de linguagem e onde ela é usada?": "COMPLEXA",
    "Como Whisper e WER se relacionam com o protótipo de voz do projeto?": "COMPLEXA",
    # negativas de escolha unica (1 fonte resolve) -> SIMPLES
    "Qual destes termos NÃO está relacionado a bancos de dados: DDL, DQL ou GAN?": "SIMPLES",
    "O WER é uma métrica de qual modalidade: voz, imagem ou banco de dados?": "SIMPLES",
    # adversariais (fora do corpus): factuais diretas -> SIMPLES; explicacao ampla -> COMPLEXA
    "Qual a previsão do tempo para amanhã no Rio de Janeiro?": "SIMPLES",
    "Quem ganhou a Copa do Mundo de 2022?": "SIMPLES",
    "Explique a teoria da relatividade geral de Einstein.": "COMPLEXA",
    "Qual o preço atual da ação da Petrobras?": "SIMPLES",
}

_cache: dict = {}
_lock = threading.Lock()


def _normalizar(texto: str) -> str:
    """Minusculas, sem acentos e espacos colapsados (mesma base p/ treino e uso)."""
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def _treinar() -> None:
    if "clf" in _cache:
        return
    with _lock:
        if "clf" in _cache:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier

        pares = [(_normalizar(p), r) for p, r in ROTULOS.items()]
        textos = [p for p, _ in pares]
        # XGBoost exige classes numericas; SIMPLES=0, COMPLEXA=1
        num = {"SIMPLES": 0, "COMPLEXA": 1}
        rotulos = [num[r] for _, r in pares]
        vetor = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=1000
        )
        # XGB raso: n pequeno (20 exemplos) — profundidade 3 + colsample 1.0
        # memoriza o treino sem virar floresta profunda.
        clf = XGBClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            subsample=1.0, colsample_bytree=1.0,
            random_state=0, n_jobs=1, eval_metric="logloss",
        )
        clf.fit(vetor.fit_transform(textos), rotulos)
        _cache.update(vetor=vetor, clf=clf, n_exemplos=len(pares))
        logger.info("Roteador TF-IDF treinado com {} exemplos", len(pares))


def proba_classes(pergunta: str) -> tuple[float, float]:
    """Retorna (p_simples, p_complexa) via predict_proba do XGBoost."""
    texto = _normalizar(pergunta)
    if not texto:
        return (1.0, 0.0)
    _treinar()
    p = _cache["clf"].predict_proba(_cache["vetor"].transform([texto]))[0]
    p = [float(x) for x in p]  # colunas: 0=SIMPLES, 1=COMPLEXA
    if len(p) < 2:
        p = p + [0.0]
    return (p[0], p[1])


def classificar_limiar(pergunta: str, limiar: float) -> str:
    """Classifica com zona de rejeicao (R3/cascata).

    Se a margem (max p) ficar abaixo de `limiar`, retorna INDETERMINADO —
    quem chama (agentes.classificar_cascade) escala para o R1 (SLM) decidir.
    """
    ps, pc = proba_classes(pergunta)
    margem = max(ps, pc)
    if margem < limiar:
        return "INDETERMINADO"
    return "SIMPLES" if ps >= pc else "COMPLEXA"


def classificar(pergunta: str) -> str:
    """Classifica a pergunta em SIMPLES/COMPLEXA. Falha/desconhecido -> SIMPLES."""
    texto = _normalizar(pergunta)
    if not texto:
        return "SIMPLES"
    try:
        _treinar()
        pred = int(_cache["clf"].predict(_cache["vetor"].transform([texto]))[0])
        return "COMPLEXA" if pred == 1 else "SIMPLES"
    except Exception as exc:
        logger.warning("Roteador TF-IDF falhou ({}); assume SIMPLES", exc)
        return "SIMPLES"


def tamanho_artefato_bytes() -> int | None:
    """Tamanho serializado (pickle) do vetorizador+modelo — proxy de memoria."""
    if "clf" not in _cache:
        _treinar()
    try:
        return len(pickle.dumps((_cache["vetor"], _cache["clf"])))
    except Exception:
        return None
