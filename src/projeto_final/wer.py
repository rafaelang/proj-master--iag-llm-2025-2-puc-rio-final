"""Metricas de qualidade de transcricao — v0.1 (Voz)."""

from __future__ import annotations

import re
import unicodedata


def normalizar(texto: str) -> list[str]:
    """Normaliza texto para comparacao: minusculas, sem acento, sem pontuacao."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return t.split()


def wer(referencia: str, hipotese: str) -> float:
    """Word Error Rate entre referencia e hipotese."""
    r = normalizar(referencia)
    h = normalizar(hipotese)
    n, m = len(r), len(h)

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            custo = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + custo,
            )
    return d[n][m] / max(n, 1)


def wer_por_frase(referencias: list[str], hipoteses: list[str]) -> list[float]:
    """WER individual para listas paralelas de referencia/hipotese."""
    if len(referencias) != len(hipoteses):
        raise ValueError("referencias e hipoteses devem ter o mesmo tamanho")
    return [wer(r, h) for r, h in zip(referencias, hipoteses)]
