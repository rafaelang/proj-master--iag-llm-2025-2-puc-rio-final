"""Configuracao global de testes: desliga o rate limit da API (testes dedicados
em test_api_rate.py reativam e controlam o estado explicitamente)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _rate_limit_desligado():
    from projeto_final import config
    from projeto_final.main import _rate_limpar

    config.RATELIMIT_HABILITADO = False
    _rate_limpar()
    yield
