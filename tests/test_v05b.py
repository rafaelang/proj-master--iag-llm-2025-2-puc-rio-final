"""Testes da v0.5b — expansao do corpus (IPYNB/PPTX) e golden set expandido.

Rodam sem API/GPU:
- extração de texto de .ipynb e .pptx a partir de fixtures sintéticas;
- ingest() distribui por formato;
- manifesto do corpus (data/golden_set/corpus_manifest.json) referencia apenas
  arquivos presentes em data/raw;
- golden set expandido (perguntas_v05b.json) tem os docs_esperados todos no
  corpus e mantém as 20 perguntas congeladas da v0.5.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from projeto_final import config
from projeto_final.rag.ingest import (
    extrair_texto_ipynb,
    extrair_texto_pptx,
    ingest,
)

RAG = Path(__file__).resolve().parent.parent / "data" / "golden_set" / "rag"


# ---------------------------------------------------------- extratores de formato

class TestIngestIpynb:
    def test_extrai_markdown_e_code_sem_outputs(self, tmp_path):
        fixture = tmp_path / "nota.ipynb"
        celula = lambda tipo, src: {"cell_type": tipo, "source": src.splitlines(True)}
        fixture.write_text(json.dumps({
            "cells": [
                celula("markdown", "# RAG\\nCombina recuperação e geração."),
                celula("code", "import numpy\\nprint('hi')\\n"),
                celula("code", "data:image/png;base64,AAAA"),  # output base64 -> vira code
            ],
        }), encoding="utf-8")
        paginas = extrair_texto_ipynb(fixture)
        assert len(paginas) == 1
        assert paginas[0]["doc_id"] == "nota.ipynb"
        assert "RAG" in paginas[0]["texto"]
        assert "import numpy" in paginas[0]["texto"]

    def test_ipynb_invalido_retorna_vazio(self, tmp_path):
        fixture = tmp_path / "quebrado.ipynb"
        fixture.write_text("{nao é json", encoding="utf-8")
        assert extrair_texto_ipynb(fixture) == []


class TestIngestPptx:
    def test_extrai_por_slide(self, tmp_path):
        from xml.sax.saxutils import escape

        def slide(n, texto):
            return (f"ppt/slides/slide{n}.xml",
                    f'<p:sld xmlns:p="http://schemas.openxmlformats.org/'
                    f'presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody>'
                    f'<a:p xmlns:a="http://schemas.openxmlformats.org/'
                    f'drawingml/2006/main"><a:r><a:t>{escape(texto)}</a:t></a:r></a:p>'
                    f'</p:txBody></p:sp></p:spTree></p:cSld></p:sld>')

        fixture = tmp_path / "deck.pptx"
        with zipfile.ZipFile(fixture, "w") as z:
            z.writestr("[Content_Types].xml", "<?xml version='1.0'?>")
            z.writestr("ppt/presentation.xml", "<?xml version='1.0'?>")
            z.writestr(*slide(1, "Joins combinam tabelas"))
            z.writestr(*slide(2, "Subqueries aninham consultas"))

        paginas = extrair_texto_pptx(fixture)
        assert len(paginas) == 2
        assert paginas[0]["pagina"] == 1
        assert paginas[1]["pagina"] == 2
        assert "Joins" in paginas[0]["texto"]

    def test_pptx_vazio(self, tmp_path):
        fixture = tmp_path / "vazio.pptx"
        with zipfile.ZipFile(fixture, "w") as z:
            z.writestr("[Content_Types].xml", "<?xml version='1.0'?>")
        assert extrair_texto_pptx(fixture) == []


# ---------------------------------------------------------- dispatcher do ingest

class TestIngestDispatch:
    def test_ingest_cobre_quatro_formatos(self, tmp_path):
        # cria 1 arquivo de cada formato minimo
        (tmp_path / "a.md").write_text("texto md", encoding="utf-8")
        (tmp_path / "b.ipynb").write_text(
            json.dumps({"cells": [{"cell_type": "markdown",
                                   "source": ["# titulo\\nnumpy"]}]}),
            encoding="utf-8")
        with zipfile.ZipFile(tmp_path / "c.pptx", "w") as z:
            z.writestr("ppt/slides/slide1.xml",
                       '<p:sld xmlns:p="http://schemas.openxmlformats.org/'
                       'presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody>'
                       '<a:p xmlns:a="http://schemas.openxmlformats.org/'
                       'drawingml/2006/main"><a:r><a:t>Joins combinam tabelas</a:t></a:r></a:p>'
                       '</p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
        writer = PdfWriter()
        writer.merge_streams = lambda *_: None
        writer.add_blank_page(width=200, height=200)
        from io import BytesIO
        buf = BytesIO()
        writer.write(buf)
        (tmp_path / "d.pdf").write_bytes(buf.getvalue())

        paginas = ingest(tmp_path)
        sufixos = {Path(p["arquivo"]).suffix for p in paginas}
        assert ".md" in sufixos and ".ipynb" in sufixos
        assert ".pptx" in sufixos and ".pdf" in sufixos


# ---------------------------------------------------------- manifesto corpus

class TestManifesto:
    def test_docs_esperados_existem_no_corpus(self):
        manifest = json.loads(
            (config.GOLDEN_SET_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
        raws = {e["arquivo"] for e in manifest["corpus"]}
        # os 10 arquivos pre-v0.5 continuam presentes e documentados
        pre = [e for e in manifest["corpus"] if e["pre_v05"]]
        assert len(pre) >= 10, f"esperado >=10 pre_v05, veio {len(pre)}"
        # extensões aceitas
        for e in manifest["corpus"]:
            assert e["extensao"] in {".pdf", ".md", ".ipynb", ".pptx"}
        # nada de duplicatas de nome
        assert len(raws) == len(manifest["corpus"])
        assert config.RAW_DIR.is_dir()

    def test_golden_v05b_aponta_para_corpus(self):
        perg = json.loads((RAG / "perguntas_v05b.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (config.GOLDEN_SET_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
        corpus = {e["arquivo"] for e in manifest["corpus"]}
        ausentes = {
            d for q in perg["perguntas"] for d in q["docs_esperados"] if d not in corpus
        }
        assert not ausentes, f"docs_esperados fora do corpus: {ausentes}"
        # v0.5 continua intacta como dataset de referencia
        v05 = json.loads((RAG / "perguntas.json").read_text(encoding="utf-8"))
        ids_v05 = {q["id"] for q in v05["perguntas"]}
        ids_v05b = {q["id"] for q in perg["perguntas"]}
        assert ids_v05.issubset(ids_v05b)
        # Correcoes de docs_esperados feitas NA v0.5b (congeladas no golden v0.5b
        # e preservadas pela v0.6 — nunca re-editar): #13 pergunta sobre o artigo
        # "attention is all you need" apontava no v0.5 para
        # proj_aula04_multiagentes_slm.pdf (erro de mira); corrigida para o doc
        # que efetivamente contem o trecho (palavras-chave de vision transformers).
        CORRECOES_V05B = {
            13: ["pai_aula08_visiontransformers_attention_is_all_you_need.pdf"],
        }
        # perguntas congeladas preservadas textualmente
        orig = {q["id"]: q for q in v05["perguntas"]}
        for q in perg["perguntas"]:
            if q["id"] in orig:
                assert q["pergunta"] == orig[q["id"]]["pergunta"]
                assert q["docs_esperados"] == CORRECOES_V05B.get(q["id"], orig[q["id"]]["docs_esperados"])


# ---------------------------------------------------------- evidencia v0.5b (pura)

class TestEvidenciaV05b:
    def _importar(self):
        import importlib.util
        import sys

        caminho = Path(__file__).resolve().parent.parent / "scripts" / "avaliadores"
        sys.path.insert(0, str(caminho))
        spec = importlib.util.spec_from_file_location("avaliar_v5b", caminho / "avaliar_v5b.py")
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        return modulo

    def test_gera_matriz_2x2_e_leitura(self, monkeypatch):
        m = self._importar()
        # bloqueia a leitura de disco (_carregar_slm) com uma matriz fake
        fake = [
            {"sujeito": "base", "resultado": [
                {"absteve": True, "deve": True}, {"absteve": False, "deve": True}]},
            {"sujeito": "base_no_rag", "resultado": [
                {"absteve": True, "deve": True}, {"absteve": True, "deve": False}]},
            {"sujeito": "destilado", "resultado": [
                {"absteve": True, "deve": True}, {"absteve": True, "deve": True}]},
            {"sujeito": "destilado_no_rag", "resultado": [
                {"absteve": True, "deve": True}, {"absteve": False, "deve": True}]},
        ]
        juiz = {"resumo": {"acerto_rate": 0.500, "alucinacoes": 1}}
        for b in fake:
            b["juiz"] = juiz
        monkeypatch.setattr(m, "_carregar_slm", lambda: fake)

        mini = {"meta": {"dataset": "data/golden_set/rag/perguntas_v05b.json",
                         "data": "2026-09-11T00:00:00"},
                "resumo": {"prompt_only": {"abstencao_correta": 0.9, "acerto_end_to_end": 0.88,
                                           "nota_media": 4.4, "alucinacoes": 3,
                                           "citacao_presente": 0.0},
                           "rag": {"abstencao_correta": 0.84, "acerto_end_to_end": 0.70,
                                   "nota_media": 3.8, "alucinacoes": 2,
                                   "citacao_presente": 1.0}}}
        md = m.gerar_evidencia_v05b(mini, {})
        assert "SLM base" in md
        assert "SLM destilado" in md
        assert "acurácia 0.500" in md

    def test_sem_slm_detecta_pendente(self, monkeypatch):
        m = self._importar()
        monkeypatch.setattr(m, "_carregar_slm", lambda: [])
        mini = {"meta": {"dataset": "x", "data": "d"},
                "resumo": {"prompt_only": {"abstencao_correta": 1.0, "acerto_end_to_end": 1.0,
                                           "nota_media": 5.0, "alucinacoes": 0,
                                           "citacao_presente": 0.0},
                           "rag": {"abstencao_correta": 1.0, "acerto_end_to_end": 1.0,
                                   "nota_media": 5.0, "alucinacoes": 0,
                                   "citacao_presente": 1.0}}}
        md = m.gerar_evidencia_v05b(mini, {})
        assert "após a execução no Colab T4" in md