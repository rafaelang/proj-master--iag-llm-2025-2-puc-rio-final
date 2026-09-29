"""v0.7 · Golden expandido (~500 casos): integridade e preservação das congeladas.

O golden v0.7 (perguntas_v07.json) preserva as 132Q da v0.6 textualmente (IDs 1-132)
e adiciona casos P7 (IDs 133+). As novas perguntas:
  - referenciam docs_esperados que existem no corpus;
  - doc esperado é ALCANÇÁVEL no pool (top-60) — descartou-se formulação genérica
    irrecuperável (evita golden injusto tipo as 5Q residuais);
  - não duplicam perguntas já existentes (normalizada);
  - adversarial => docs_esperados vazio e deve_abster=true.
"""
from __future__ import annotations

import json
import re

from projeto_final import config

RAG = config.GOLDEN_SET_DIR / "rag"


def _norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", q.lower())


class TestGoldenV07:
    def _golden(self):
        return json.loads((RAG / "perguntas_v07.json").read_text(encoding="utf-8"))

    def test_quase_500_casos(self):
        perg = self._golden()
        assert len(perg["perguntas"]) >= 450, f"golden v0.7 < 450: {len(perg['perguntas'])}"

    def test_docs_esperados_existem_no_corpus(self):
        perg = self._golden()
        manifest = json.loads(
            (config.GOLDEN_SET_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
        corpus = {e["arquivo"] for e in manifest["corpus"]}
        ausentes = {d for q in perg["perguntas"] for d in q["docs_esperados"] if d not in corpus}
        assert not ausentes, f"docs_esperados fora do corpus: {ausentes}"

    def test_preserva_v06_textualmente(self):
        perg = self._golden()
        v06 = json.loads((RAG / "perguntas_v06.json").read_text(encoding="utf-8"))
        orig = {q["id"]: q for q in v06["perguntas"]}
        for q in perg["perguntas"]:
            if q["id"] in orig:
                assert q["pergunta"] == orig[q["id"]]["pergunta"]
                assert q["docs_esperados"] == orig[q["id"]]["docs_esperados"]

    def test_sem_duplicata_normalizada(self):
        perg = self._golden()
        seen = {}
        for q in perg["perguntas"]:
            k = _norm(q["pergunta"])
            assert k not in seen, f"duplicata: #{q['id']} == #{seen[k]}"
            seen[k] = q["id"]

    def test_adversarial_docs_vazio_e_deve_abster(self):
        perg = self._golden()
        for q in perg["perguntas"]:
            if q["estrato"] == "adversarial":
                assert not q["docs_esperados"], f"adversarial com docs: #{q['id']}"
                assert q["deve_abster"], f"adversarial sem deve_abster: #{q['id']}"

    def test_estratos_representados(self):
        perg = self._golden()
        estratos = {q["estrato"] for q in perg["perguntas"]}
        assert {"rotineira", "composta", "negativa", "adversarial"} <= estratos

    def test_novos_sao_alcancaveis_no_pool(self):
        perg = self._golden()
        novas = [q for q in perg["perguntas"] if q["id"] > 132 and q["docs_esperados"]]
        assert novas
        from projeto_final.rag.pipeline import carregar_chunks
        from projeto_final.rag.retrieve import recuperar

        chunks = carregar_chunks()
        irrec = []
        for q in novas:
            rec = recuperar(q["pergunta"], chunks, top_k=60, base=None)
            if not any(c["doc_id"] in q["docs_esperados"] for c in rec):
                irrec.append(q["id"])
        assert not irrec, f"novas perguntas com doc irrecuperável no pool: {irrec}"

    def test_pelo_menos_um_novo_por_estrato(self):
        perg = self._golden()
        for est in ("rotineira", "composta", "negativa", "adversarial"):
            assert any(q["estrato"] == est and q["id"] > 132 for q in perg["perguntas"]), est