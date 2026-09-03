# v0.3 · Imagem — evidência (OCR local de figuras integrado ao RAG)

> Gerado por `scripts/avaliadores/avaliar_v3.py` em 2026-09-03T08:27:42.
> Dataset: `data/golden_set/imagem/perguntas.json` (3 perguntas).
> Corpus: texto-only 272 chunks · texto+imagem 294 chunks.

## Resumo honesto (retrieval)

| Métrica | Valor |
|---|---|
| Conteúdo da figura no top-5 — corpus TEXTO-only | **0.000** |
| Conteúdo da figura no top-5 — corpus TEXTO+IMAGEM | **1.000** |
| Casos em que a visão (OCR) corrigiu/agregou o texto | **3** (ids [1, 2, 3]) |

## Por pergunta

| # | Estrato | Conteúdo figura no top-5 (texto-only) | (texto+imagem) | Chunk imagem |
|---|---|---|---|---|
| 1 | modelo_em_figura | NAO | SIM | p.22 · tdp_aula03_sql_ddl.pdf |
| 2 | modelo_em_figura | NAO | SIM | p.18 · tdp_aula03_sql_ddl.pdf |
| 3 | figura_explicativa | NAO | SIM | p.39 · dpia_aula01_intro_ia.pdf |

## Respostas (LLM) — caso documentado: visão corrige o texto

### #1

- **texto-only (v0.2):** abstenção=True · contém termos=False
  > NAO_SEI

- **texto+imagem (v0.3):** abstenção=False · contém termos=True · tipos de chunk=['imagem', 'texto']
  > Na figura da tabela cliente, alem do codcliente, existem as colunas nomecliente, sexo, estcivil, uf e dtcadastro, conforme especificado no modelo apresentado. [1] tdp_aula03_sql_ddl.pdf

### #2

- **texto-only (v0.2):** abstenção=True · contém termos=False
  > O trecho fornecido nao contem a informacao sobre o atributo no relacionamento entre funcionario e projeto no exercicio de DDL. O contexto menciona exercicios de criacao e alteracao de tabelas, mas nao descreve o modelo ao lado nem o atributo especifico. NAO_SEI

- **texto+imagem (v0.3):** abstenção=False · contém termos=True · tipos de chunk=['imagem', 'texto']
  > No contexto fornecido, o modelo registra o atributo qtdhoras no relacionamento entre funcionario e projeto, conforme indicado nas figuras das páginas 8 e 18. Entretanto, a pergunta parece se referir a um atributo de relacionamento, e o texto menciona que atributos de relacionamento representam informações extras que variam com a combinação das instâncias, como número de horas trabalhadas por empregado em cada projeto. Assim, o atributo no relacionamento é qtdhoras, que representa a quantidade de horas. As fontes não fornecem detalhes adicionais sobre esse atributo além de sua presença no modelo. [1] tdp_aula03_sql_ddl.pdf, [2] tdp_aula03_sql_ddl.pdf, [3] tdp_aula01_modelagem_relacional.pdf

### #3

- **texto-only (v0.2):** abstenção=True · contém termos=False
  > NAO_SEI

- **texto+imagem (v0.3):** abstenção=False · contém termos=True · tipos de chunk=['imagem', 'texto']
  > Os cinco fatores apresentados na figura sobre a nova era da IA são: dados em maior quantidade e livres, computadores com mais memória e maior capacidade de armazenamento e processamento, redes neurais com modelos maiores e mais complexos, comunidade muito participativa e engajada em soluções open-source, e empresas que já perceberam a necessidade de utilizar IA nos seus processos. [1] dpia_aula01_intro_ia.pdf

<!-- ===== SECOES MANUAIS (nao geradas pelos avaliadores) ===== -->


## Seções manuais — dataset e processo (v0.3)

### Dataset

`data/golden_set/imagem/perguntas.json` (3 perguntas, entra no Git):

- cada pergunta aponta para um **doc + página** cuja resposta está numa figura
  (modelo "ao lado", diagrama ER ou figura explicativa) e traz os
  `termos_esperados` — tokens **ausentes do layer de texto** de todo o corpus
  (ex.: `nomecliente`, `dtcadastro`, `qtdhoras`), verificados antes da medição;
- métrica: o conteúdo da figura (termos) chega ao top-5 do RAG? Comparação
  contrafactual corpus texto-only (v0.2, sem imagens) vs corpus v0.3.

### Processo de construção do corpus v0.3

1. `src/projeto_final/rag/imagem.py::registro_imagens` — PyMuPDF extrai as
   imagens dos PDFs e deduplica por sha1 dos pixels (921 ocorrências → 275
   únicas, min. 40 px);
2. `ocr_imagens` — RapidOCR local (ONNX, CPU) por imagem (~7 min no total,
   cacheado em `data/processed/rag_v3/imagens_ocr.json`);
3. `chunks_de_imagem` — limpeza das linhas OCR, novidade vs página
   (`novo_ratio ≥ 0.45`), sinal de domínio vs documento (`≥ 0.40`) e
   ancoragem com o título do slide → **22 figuras úteis**
   (`imagens_chunks.json`);
4. `pipeline::carregar_corpus_v3` — merge texto (272) + imagem (22) = **294
   chunks** com ids renumerados (`rag_v3/chunks.json`), índice separado em
   `data/processed/rag_v3/` (a v0.2 em `data/processed/rag/` fica intacta);
5. `recuperar(..., base=rag_v3)` — RRF ponderado com peso 1.15 para chunks de
   imagem; contexto cita a origem: `[N] doc, página (imagem/figura OCR)`.

### Reproduzir

```bash
python scripts/avaliadores/avaliar_v3.py          # retrieval (sem LLM)
python scripts/avaliadores/avaliar_v3.py --e2e    # + respostas LLM (caso)
pytest                                           # suíte completa (18 testes)
```

### Custo / ambiente

OCR e embeddings são 100% locais (CPU/ONNX) — **US$ 0.00**. LLM remoto
(DeepSeek) usado apenas nas respostas finais das 3 perguntas do caso
documentado (configuração idêntica à v0.2: `deepseek-chat` + juiz não usado
nesta evidência; resposta avaliada por conteúdo esperado e citação).


