"""Gera o dataset do ESTUDO do roteador (R1 x R2) — artefato de experimento.

NAO e golden set da release (o golden set unico continua sendo
data/golden_set/rag/perguntas.json, intocado). Saida (fora do Git):
  data/processed/rag_v4/roteador_rotulos.json

Conteudo:
  - 20 perguntas canonicas do perguntas.json rotuladas SIMPLES/COMPLEXA
    (13 S / 7 C) — ground truth = modulo roteador_tfidf.ROTULOS;
  - ~14 variacoes com typo simulando ASR pt-BR (sem acentos, "para"->"pra",
    queda/troca de letra no meio) — deterministicas.

Uso:
  python scripts/avaliadores/gerar_dataset_roteador.py
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from projeto_final import config
from projeto_final import roteador_tfidf

SAIDA = config.PROCESSED_DIR / "rag_v4" / "roteador_rotulos.json"


def _sem_acento(s: str) -> str:
    t = unicodedata.normalize("NFKD", s)
    return "".join(c for c in t if not unicodedata.combining(c))


def _para_pra(s: str) -> str:
    return _sem_acento(s).replace("para que serve", "pra que serve")


def _drop_meio(s: str) -> str:
    t = _sem_acento(s)
    i = len(t) // 2
    return t[:i] + t[i + 1:]


def _troca_meio(s: str) -> str:
    t = _sem_acento(s)
    i = len(t) // 2
    if i + 1 >= len(t):
        return _drop_meio(s)
    return t[:i] + t[i + 1] + t[i] + t[i + 2:]


# (id da pergunta, operador de typo) — deterministico
_VARIACOES: dict[int, object] = {
    1: _sem_acento, 2: _para_pra, 3: _drop_meio, 4: _troca_meio,
    5: _sem_acento, 6: _drop_meio, 7: _troca_meio, 8: _sem_acento,
    9: _drop_meio, 11: _troca_meio, 12: _drop_meio, 14: _troca_meio,
    15: _sem_acento, 19: _drop_meio,
}


def gerar() -> dict:
    from projeto_final import config as _cfg

    perguntas = json.loads(_cfg.RAG_GOLDEN_SET.read_text(encoding="utf-8"))["perguntas"]
    por_id = {q["id"]: q for q in perguntas}

    itens: list[dict] = []
    for q in perguntas:
        rotulo = roteador_tfidf.ROTULOS[q["pergunta"]]
        itens.append({"id": q["id"], "tipo": "canonica", "texto": q["pergunta"], "rotulo": rotulo})

    for qid, op in sorted(_VARIACOES.items()):
        base = por_id[qid]
        variado = op(base["pergunta"])
        assert variado != base["pergunta"], f"typo sem efeito em {qid}"
        itens.append({
            "id": qid, "tipo": "typo",
            "texto": variado, "rotulo": roteador_tfidf.ROTULOS[base["pergunta"]],
        })

    n_s = sum(1 for i in itens if i["rotulo"] == "SIMPLES")
    n_c = len(itens) - n_s
    return {
        "meta": {
            "descricao": "Estudo do roteador (R1 SLM x R2 TF-IDF+XGB) — artefato, nao golden set",
            "criterio": "SIMPLES=factual/direta/1 fonte · COMPLEXA=composta/comparativa/ambigua/multifonte",
            "fonte": str(config.RAG_GOLDEN_SET),
            "n_canonicas": len(perguntas),
            "n_typos": len(_VARIACOES),
        },
        "resumo_rotulos": {"SIMPLES": n_s, "COMPLEXA": n_c},
        "itens": itens,
    }


def main() -> None:
    dados = gerar()
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset do estudo salvo: {SAIDA}")
    print(f"itens: {len(dados['itens'])} "
          f"(canonicas={dados['meta']['n_canonicas']} + typos={dados['meta']['n_typos']})")
    print("rotulos:", dados["resumo_rotulos"])
    for i in dados["itens"][-4:]:
        print("  exemplo typo:", i["texto"], "->", i["rotulo"])


if __name__ == "__main__":
    main()
