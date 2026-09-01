"""BM25 Okapi proprio (Python puro) — tokenizacao pt-BR, stopwords e IDF suavizado.

Substitui rank_bm25.BM25Okapi:
- k1=1.5, b=0.75 (BM25 Okapi padrao);
- tokenizacao via normalizar (minusculas, sem acento, sem pontuacao);
- stopwords em pt-BR removidas da QUERY (a, o, de, para, que, e, um, ...);
- filtro de tokens curtos (len(t) > 1);
- IDF suavizado: log((n - df + 0.5) / (df + 0.5) + 1.0).
"""

from __future__ import annotations

import math
import re
import unicodedata


STOPWORDS_PT = frozenset({
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "entre", "era", "essa", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "ha", "isso", "isto", "ja", "la", "mais", "mas",
    "meu", "minha", "muito", "na", "nao", "nas", "nem", "no", "nos", "nossa", "nosso",
    "o", "os", "ou", "para", "pela", "pelo", "pelas", "pelos", "por", "qual", "quando",
    "que", "quem", "se", "sem", "sendo", "ser", "seu", "sua", "so", "sobre", "te",
    "tem", "ter", "um", "uma", "umas", "uns", "voce",
})


def normalizar(texto: str) -> list[str]:
    """Tokenizacao: minusculas, sem acento, sem pontuacao e sem tokens curtos (len <= 1)."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [tok for tok in t.split() if len(tok) > 1]


def preparar_query(pergunta: str) -> list[str]:
    """Tokens da query: normalizados, sem stopwords pt-BR e sem tokens curtos."""
    return [tok for tok in normalizar(pergunta) if tok not in STOPWORDS_PT]


class BM25Okapi:
    """BM25 Okapi puro em Python (k1=1.5, b=0.75), com IDF suavizado."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        # frequencia de documentos (df) por termo
        df: dict[str, int] = {}
        for doc in corpus:
            for termo in set(doc):
                df[termo] = df.get(termo, 0) + 1

        # IDF suavizado: log((n - df + 0.5) / (df + 0.5) + 1.0)
        n = len(corpus)
        self.idf = {
            termo: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for termo, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        """Score BM25 de cada documento do corpus para a query tokenizada."""
        scores = [0.0] * len(self.corpus)
        for termo in query:
            idf = self.idf.get(termo)
            if idf is None:
                continue
            for i, doc in enumerate(self.corpus):
                tf = doc.count(termo)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1.0) / denom
        return scores
