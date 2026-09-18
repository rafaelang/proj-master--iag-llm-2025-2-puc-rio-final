# Resumo para orientação — Assistente Generativo sobre Materiais do Master IAG & LLM

> Documento executivo para a conversa de orientação. Todas as evidências citadas
> estão em `docs/v0X.md` + `docs/v0X_evidencia.md` (uma dupla por release) e nos
> commits/tags do repositório (v0.1 → v1.0).

## 1. Identificação

| Campo | Valor |
|---|---|
| Projeto | Assistente generativo sobre o conteúdo do Master IAG & LLM (PUC-Rio) |
| Aluno | Rafael Garcia |
| Disciplina | PROJ · Master IAG & LLM 2025-2 |
| Repositório | `github.com/rafaelang/proj-master--iag-llm-2025-2-puc-rio-final` |
| **Status** | **v1.0 · Produção — concluída (2026-09-14)** · 79 commits · tags v0.1…v1.0 |
| URL pública | `https://rafaelang-assistente-master-iag.hf.space/` (Space `rafaelang/assistente-master-iag`) |
| Corpus | dataset **privado** `rafaelang/assistente-master-iag-dados` (fora do repo, por LGPD/direitos autorais) |

## 2. Proposta (uma frase)

Assistente **multimodal** que responde dúvidas sobre o conteúdo do curso
(voz → texto → RAG → resposta **citando a fonte** → áudio), roteia entre um
**SLM local** (rota simples, custo zero) e um **LLM remoto** (rota complexa),
e **se abstém** (`NAO_SEI` + motivo) quando a pergunta está fora do corpus.

## 3. Arquitetura final (v1.0)

```
voz (faster-whisper) ─┐
imagem (RapidOCR + visão multimodal) ─┤→ /chat → roteador cascade ─→ SIMPLES: SLM local Qwen 1.5B
texto ─────────────────┘                        │                   └─ (produção: deepseek-flash)
                                                └─ INDETERMINADO → R1 SLM few-shot   COMPLEXA: deepseek-v4-pro
                                      RAG: BM25 pt-BR + fastembed (MiniLM-L12-v2) + RRF ponderado
                                            corpus 193 docs · 3.401 chunks · 22 figuras
                                      fallback cruzado; ambas falham → abstenção com motivo
```

- **Roteador `cascade`** (default, θ=0.60): R2 TF-IDF+XGB local decide ~85% das
  perguntas a **<2 ms / US$ 0**; `INDETERMINADO` → R1 SLM few-shot.
- **Retrieval**: BM25 próprio pt-BR + embeddings locais (fastembed) + fusão RRF
  ponderada; índice texto+imagem (22 figuras via OCR local RapidOCR/ONNX).
- **Geração**: SLM local Qwen2.5-1.5B GGUF (100% CPU) na simples · `deepseek-v4-pro`
  na complexa; resposta com citações `[N] doc_id, página X`.
- **Deploy**: Space público (docker, cpu-upgrade) **sem corpus** + dataset privado
  baixado no boot (`snapshot_download` com `HF_TOKEN_READ`); só modelos remotos.

## 4. Trajetória v0.1 → v1.0 (marcos e evidência medida)

| Release | Entrega | Evidência medida |
|---|---|---|
| **v0.1 · Voz** | chat por voz ASR→LLM→TTS, página HTML | WER médio **0.2290 → 0.0863** com vocabulário de domínio (10 áudios próprios) |
| **v0.2 · RAG** | responde **citando fonte** + abstenção; ingestão anti-garbage, BM25+embeddings+RRF | recall@5 **1.000** / MRR **0.969**; fim-a-fim acurácia **0.700** (14/20); abstenção indevida 0.250 |
| **v0.3 · Imagem** | OCR local (RapidOCR/ONNX) nas figuras + visão multimodal no chat | 921→275→**22 figuras úteis**; conteúdo da figura no top-5 **0.000 → 1.000**; **3/3** casos em que a visão corrigiu o texto |
| **v0.4 · Agentes** | roteador SIMPLES/COMPLEXA, **SLM local de verdade** (Qwen 1.5B, CPU, US$ 0) + frontier, **fallback cruzado** | abstenção correta R1 **0.950**; default `cascade` **0.850** no golden; ~85% das rotas a custo zero |
| **v0.5 · Adaptação** | comparativo prompt/RAG/fine-tuning/pré-treino + decisão justificada | decisão **"não treinar"** (LoRA destilado piora acurácia por super-cautela); TCO em 3 cenários |
| **v0.5b · Expansão** | corpus **10 → 193 docs** (3.401 chunks; PDF/MD/IPYNB/PPTX), golden **44Q** | decisão RAG+frontier **mantida** no corpus 19× maior |
| **v0.6 · Avaliação** | golden **132Q** congelado, avaliação **por estrato**, plano P0–P4 do orientador | ver §5 |
| **v1.0 · Produção** | Space público + dataset privado + **custo por mil consultas** | `/chat` **HTTP 200 em 14,7 s** com citação; **US$ 0,27 / mil** (cascade); 86 testes |

## 5. Avaliação (v0.6 — golden 132Q congelado, régua oficial)

| Estrato | n | fim-a-fim cascade | com correção (projetado) |
|---|---|---|---|
| rotineira | 69 | 0.609 (39/64) | — |
| composta | 29 | 0.586 (17/29) | — |
| negativa | 16 | **0.125 (2/16)** ← elo fraco | **0.812** com regra V5 |
| adversarial | 18 | 0.833 (15/18) | — |
| **total** | 132 | **0.575 (73/127)** | **0.661 (84/127)** |

- **Retrieval isolado**: recall@5 DOC **0.659** (87/132) · MRR 0.592; **PÁGINA 0.605**
  (46/76) — **a página é o limitante real**, não o documento (13 casos doc-ok/página-✗).
- **Falha encontrada pelo aluno** (requisito da aula): o estrato **negativa** —
  o cascade abstinha demais (0.125); auditoria do juiz (P0) mostrou 0 leniência e a
  causa raiz em recall@5=0. Correção: **regra determinística V5** (literal ∪ janela
  14 tokens ∪ mapa pt→SQL), bypass do gerador, **US$ 0,00** → 11/14 no tipo A, 0 erros.
- **Custo**: cascade **US$ 0,00027/pergunta** (alvo v1.0 ≤ 0,001 ✓); `pro` forçado
  US$ 0,0015/pergunta (referência de teto).
- **Testes**: `pytest tests/` = **86** (fechamento v0.6 verde; v1.0 = 83 passed + 3
  deselected de integração que chamam API remota, fora do ar no fechamento).

### Alvos oficiais da v1.0 (definidos no fechamento v0.6)

| Régua | Baseline v0.6 | Alvo v1.0 |
|---|---|---|
| recall@5 DOC | 0.659 | ≥ 0.75 |
| recall@5 PÁGINA | 0.605 | ≥ 0.70 |
| fim-a-fim (cascade) | 0.575 → 0.661 projetado | ≥ 0.70 |
| negativa tipo A | 11/14 (0.786) | ≥ 0.80 |
| TCO por pergunta | 0.00027/Q | ≤ 0.001/Q ✓ |

> A v1.0 **não re-mede** (não altera o pipeline): ela entrega produção + custo e
> fecha o roadmap. O re-teste fim-a-fim foi feito no Space (HTTP 200, 14,7 s, resposta
> citada com fallback `flash→pro` — `docs/v10_evidencia.md` §1.1).

## 6. Decisões técnicas-chave (com justificativa medida)

1. **Não treinar** (v0.5): a falha do caso é de **fato** (grounding), não de forma;
   o destilado LoRA rebalanceado **piora** (baseline 0.758 não é batido; LoRA 0.529 com
   12 alucinações). RAG+frontier cumpre com custo e latência menores.
2. **Rerank cross-encoder desativado** (v0.2): piorava MRR no corpus rarefeito;
   reavaliado na v0.5b no corpus denso (pool 60 + rerank + OCR na correção P1).
3. **Roteador `cascade` θ=0.60 mantido** (v0.6 P2): subir o limiar não captura as 11
   compostas que o SLM erra (R2 p_sim 0.57–0.98); forçar `pro` resgata 6/11 mas a
   US$ 0,0015/Q não se paga na régua de custo.
4. **Regra determinística V5** para a negativa (v0.6 P1): US$ 0,00, sem LLM, bypass do
   gerador; 11/14, 0 erros, 0 disparo em adversarial.
5. **Índice de produção mantido** (v0.6 P3): título no índice ganho 0, remoção de
   boilerplate −1; dedup Jaccard 0.85 mantido.
6. **Juiz DeepSeek aceito como está** (v0.6 P0): auditoria com sondas (consistência,
   viés de comprimento, rubrica, grounding) — 0 leniência, adversarial 0.833.
7. **OCR local em vez de VLM remoto** para figuras (v0.3): custo R$ 0, offline; o VLM
   vision fica só para o chat por imagem.
8. **Corpus privado no deploy** (v1.0): só código no Space público; raw + índices em
   dataset privado baixado no boot (LGPD/direitos autorais).

## 7. Achados de produção (v1.0 — o que o smoke test do Space revelou)

1. **Nomes de modelo obsoletos penduram, não erram**: o endpoint DeepSeek expõe só
   `deepseek-flash` e `deepseek-v4-pro` (`GET /models`); chamadas a `deepseek-chat`
   inexistente ficavam **mudas** (sem timeout). Correção: defaults → `deepseek-flash`.
2. **SIGPIPE em container headless**: `set -o pipefail` + barra `tqdm` (2,5 GB) e
   `ls | head` derrubavam o boot (exit 141). Correção: `disable_progress_bars` + retry
   com backoff + `ls | wc -l`.
3. **Sem degradação graciosa**: `OpenAI()` sem timeout pendurava ~10 min em API fora
   do ar. Correção: `timeout=90 s` + `max_retries=1` → falha rápida e abstenção.
4. **Fallback cruzado resgatou na prática**: no smoke test o `gerador-flash` retornou
   vazio e o `gerador-pro` respondeu com citação — comportamento projetado na v0.4.
5. **Segurança**: token HF hardcoded num notebook do corpus — **redigido** (5 arquivos
   de `data/`) + recomendação de **rotacionar** o token que circulou.

## 8. Entregáveis

- Repositório com **79 commits** e **tags v0.1…v1.0** (uma release por aula, nunca
  quebrada); protótipo `projeto2/` como fonte de consulta (código reimplementado).
- **Evidência por release** em `docs/` (máx. 2 arquivos: `v0X.md` + `v0X_evidencia.md`);
  golden congelados por release em `data/golden_set/` (entram no Git).
- **86 testes** `pytest`; avaliadores padronizados em `scripts/avaliadores/`.
- **Produto vivo**: URL pública respondendo `/chat` com RAG+citação+abstenção e
  custo por mil consultas documentado.

## 9. Reprodução

```bash
cd projeto_final && uv sync && source .venv/bin/activate
pytest tests/ -q                            # 86 testes (v1.0: 83 + 3 e2e de API)
bash deploy/deploy.sh --dry-run             # bundle sem publicar
curl -s https://rafaelang-assistente-master-iag.hf.space/saude | python3 -m json.tool
curl -s -X POST https://rafaelang-assistente-master-iag.hf.space/chat -F "texto=O que é uma GAN?"

# Avaliação v0.6 (golden 132Q)
RAG_GOLDEN_SET_FILE=perguntas_v06.json python scripts/avaliadores/avaliar_retrieval_v05b.py
python scripts/avaliadores/avaliar_cascade_v05b.py
python scripts/avaliadores/avaliar_regra_negativa.py
python scripts/avaliadores/avaliar_retrieval_v06_experimentos.py
```