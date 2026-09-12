# Colab — Como rodar experimentos via opencode (MCP `colab`)

> Workflow para o opencode executar experimentos no Google Colab usando o
> servidor MCP `deploy/mcp_colab` (expõe o `google-colab-cli` como tools).
> **Instruções para agentes de IA.** Pré-requisito: MCP registrado em
> `opencode.json` da raiz do workspace (reiniciar o opencode após criar/editar).
> Caminhos de arquivos **locais** abaixo são relativos a `projeto_final/`
> (cwd de trabalho do opencode = raiz do workspace).

## Visão geral

Dois tipos de experimento rodam no Colab (GPU T4), porque exigem GPU/pacotes
pesados que não ficam na venv local:

| Experimento | Scripts | Onde ficam |
|---|---|---|
| Lab SLM (destilação LoRA) | `gen_dataset.py`, `train_slm.py`, `avaliar_slm_rag.py` | `projeto_final/scripts/colab/` |
| P1 retrieval (pool × rerank) | `avaliar_retrieval_pool.py`, `auditor_pool_v05b*.py` | `projeto_final/scripts/avaliadores/` |

## Pré-requisitos (uma vez)

1. Autenticar o CLI **fora do MCP** (fluxo interativo, não é tool):
   `cd projeto_final/deploy/mcp_colab && uv run colab auth && uv run colab whoami`.
   A sessão fica em `~/.config/colab-cli/`.
2. Confirmar que o MCP responde: tool `colab_whoami` deve listar a conta.
3. GPU disponível: `T4` (padrão dos experimentos), `L4`, `G4`, `H100`, `A100`.
4. **Toda execução remota** pode demorar — passar `timeout` explícito (o CLI
   corta em 30 s por padrão). Uso comum: `timeout=900` (modelos baixam na 1ª vez).

## Fluxo padrão (sequência de tools)

1. `colab_new_session(session="slm", gpu="T4")` — provisiona a VM (nome reutilizável).
2. `colab_install_packages(packages=[...])` — deps do experimento (ver abaixo).
3. `colab_upload_file(local, remote)` — dados do corpus (layout abaixo).
4. `colab_exec_code(...)` — se precisar setar env/`sys.path` (abaixo).
5. `colab_exec_file(file=<script local>, timeout=...)` — roda o experimento.
   Não precisa dar upload no script: o `exec -f` transmite o arquivo local.
6. `colab_download_file(remote, local)` — baixa resultados para `data/processed/`.
7. `colab_stop_session(session="slm")` — derruba a VM **ao terminar** (custo!).

Para inspecionar a execução: `colab_export_log(lines=...)`. Para anexar a uma
VM viva: repassar o mesmo `session=` nas demais calls.

## Layout de upload no Colab

Os scripts derivam os caminhos de `/content` (padrão dos scripts/colab e do
`config.py`, pois `RAIZ = parent×3 do config.py`):

```
/content/src/projeto_final/…   ← upload de projeto_final/src/projeto_final/…
/content/data/…                ← upload de projeto_final/data/…
/content/prompts/…             ← upload de projeto_final/prompts/…
/content/.env                  ← upload de projeto_final/.env (DEEPSEEK_API_KEY)
```

Antes de importar `projeto_final` (só os avaliadores P1 fazem isso), rodar:
`colab_exec_code(code='import sys; sys.path.insert(0, "/content/src")')`.

## Experimento — Lab SLM (`scripts/colab/`)

Deps: `["torch","transformers","peft","bitsandbytes","datasets","accelerate","openai","python-dotenv"]`.

1. **`gen_dataset.py`** — gera o dataset sintético curado a partir dos chunks.
   - Espera `/content/data/processed/rag/chunks.json` e `/content/.env`.
   - Env (default): `CHUNKS_PATH=/content/data/processed/rag/chunks.json`,
     `GEN_SAIDA=/content/data/golden_set/adaptacao/dataset_sintetico.json`,
     `GEN_MODELO`/`JUIZ_MODELO` (deepseek-v4-pro).
   - Roda com `colab_exec_file(file="scripts/colab/gen_dataset.py", timeout=1200)`.
   - Saída: `dataset_sintetico.json` (baixar para `data/golden_set/adaptacao/`).
2. **`train_slm.py`** — LoRA 4-bit no Qwen2.5-1.5B (T4, ~30 s + download).
   - Env: `TRAIN_DATASET` (default dataset_sintetico.json),
     `TRAIN_SAIDA=/content/data/processed/slm_adapter`.
   - Roda com `colab_exec_file(file="scripts/colab/train_slm.py", timeout=1800)`.
   - Saída: adapter (baixar para `data/processed/slm_adapter/`, fora do Git).
3. **`avaliar_slm_rag.py`** — matriz 2×2 (base × destilado × ±RAG) no golden.
   - Args posicionais: `<adapter>` (destilado) e/ou `--no-rag`.
   - Env: `PERGUNTAS_PATH=/content/data/golden_set/rag/perguntas_v05b.json`
     (default é `perguntas.json` — **para o golden expandido, sobrescrever**),
     `NOMEROLE` (sufixo do arquivo de saída).
   - Ex.: 4 execuções: sem arg / `--no-rag` / `<adapter>` / `<adapter> --no-rag`.
   - Saída: `/content/resultado_{base|destilado}[_no_rag].json`
     (baixar para `data/processed/v05b_ab/`).
   - Como passar env: `colab_exec_code(code='import os; os.environ["PERGUNTAS_PATH"]="…"')` **antes** de cada `colab_exec_file` (kernel da sessão persiste).

## Experimento — P1 retrieval (`scripts/avaliadores/`)

Deps: `["fastembed","onnxruntime","loguru","python-dotenv","numpy"]`.
Precisa de `/content/data/processed/rag/{chunks.json,bm25.pkl,embeddings.npy,chunk_ids.json}`
+ `/content/data/golden_set/rag/perguntas_v05b.json` + `sys.path.insert(0,"/content/src")`.
O reranker `jinaai/jina-reranker-v2-base-multilingual` baixa na 1ª execução.

1. **`avaliar_retrieval_pool.py`** — recall@5/MRR do retriever real variando o
   pool de over-fetch (`POOL_RERANK`), com rerank ligado.
   - Env: `RAG_GOLDEN_SET_FILE=perguntas_v05b.json` (o script usa `config.RAG_GOLDEN_SET`).
   - Uso: primeiro `colab_exec_code(code='import os,sys; sys.path.insert(0,"/content/src"); os.environ["RAG_GOLDEN_SET_FILE"]="perguntas_v05b.json"')`,
     depois `colab_exec_code` com `runpy` para passar o arg do pool — **`colab_exec_file`
     não aceita args** (só `colab_run_script` tem `script_args`):
     `code='import sys, runpy; sys.argv=["avaliar_retrieval_pool.py","--pool","60"]; runpy.run_path("/content/scripts/avaliadores/avaliar_retrieval_pool.py", run_name="__main__")'`.
     Sem `--pool` o script varre 30/60/100/200/400. (`avaliar_retrieval_pool.py`
     precisa estar em `/content/scripts/avaliadores/` — ou apontar o caminho certo.)
   - Saída: `data/processed/rag/retrieval_pool_{pool}.json` (no Colab).
2. **`auditor_pool_v05b.py`** / **`auditor_pool_v05b_def.py`** — posição do doc
   esperado das 12 perguntas difíceis por pool (30–400), pré/pós-dedup.
   - Sem args; exigem `sys.path` com `/content/src` (env não precisa).
   - Rodar com `colab_exec_file(file="scripts/avaliadores/auditor_pool_v05b_def.py", timeout=1800)`.

## Baixar resultados e fechar

- Baixar sempre com `colab_download_file(remote, local)`, preservando a pasta:
  - resultados → `data/processed/…` (fora do Git);
  - golden novo → `data/golden_set/…` (**entra** no Git, se for dataset oficial);
  - adapter → `data/processed/…` (fora do Git).
- Evidência oficial de release é **Markdown** em `docs/` (máx. 2 arquivos/release)
  — JSON de Colab é dado intermediário, não evidência.
- `colab_stop_session` no fim (GPU ativa custa). Se o experimento falhar no meio,
  `colab_export_log` primeiro, corrija o script, e **reuse a mesma sessão** (`session=`) se ainda viva.

## Gotchas

- `colab exec/run` cortam em **30 s** sem `timeout` — sempre passar (900–1800 s).
- Datasets/golden **congelados não se editam** para melhorar métrica (ver
  `melhoria_v0.5.md` e `data/golden_set/adaptacao/criterio_rotulo.md`).
- Scripts novos de experimento ficam **fora de `src/`** (`scripts/…`).
- Não subir `data/raw`, `.env` ou caches; não deixar `.env` no Colab público.