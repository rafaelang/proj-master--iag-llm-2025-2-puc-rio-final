# Deploy — Hugging Face Space (`assistente-master-iag`)

Publica o assistente (v0.3: voz + RAG + imagem) em um **Space privado** da
Hugging Face com SDK **docker**, hardware **CPU Upgrade** e **sleep de 1 h**.

## Configuração

Space privado **`assistente-master-iag`** (o dono é o usuário do token):

| Item | Valor |
|---|---|
| SDK | `docker` |
| Hardware | `cpu-upgrade` (2 vCPU · 8 GB · 50 GB) |
| Sleep | `3600` s (1 h de inatividade) |
| Visibilidade | privado (corpus fora de repositório público) |

## 1. Pré-requisitos (uma vez)

- `python` (o script usa `.venv/bin/python` do projeto — tem `huggingface_hub`);
- `rsync` e `git`;
- Token da Hugging Face com escopo de **write** em Spaces.

## 2. Segredos

Nunca commitar segredos. Eles ficam em `deploy/.env` (gitignored):

```bash
cp deploy/.env.example deploy/.env
# preencha:
#   HF_TOKEN=hf_...
#   DEEPSEEK_API_KEY=...   (vazio = herda do .env da raiz do projeto)
chmod 600 deploy/.env
```

> ⚠️ O script define `DEEPSEEK_API_KEY` como **secret do Space** (não fica no
> código). O `HF_TOKEN` é usado apenas localmente para criar o Space e fazer o
> push.

## 3. Deploy

```bash
bash deploy/deploy.sh          # publica (cria Space, define secret, envia)
bash deploy/deploy.sh --dry-run  # apenas monta o bundle (sem publicar)
```

O que o script faz:

1. carrega `HF_TOKEN`/`DEEPSEEK_API_KEY` de `deploy/.env` (fallback: `.env` da raiz);
2. cria/atualiza o Space privado (docker, `cpu-upgrade`, `space_sleep_time=3600`);
3. define o secret `DEEPSEEK_API_KEY`;
4. monta o **bundle** em `deploy/build/`:
   - código (`src/`, `static/`, `prompts/`), `Dockerfile`, `deploy/requirements.txt`;
   - **corpus** `data/raw/` e **índices processados** (`data/processed/rag*`,
     sem caches de modelo `*_models` — baixados em runtime no Space);
5. faz `git init` + push da branch `main` para o Space.

Depois: acompanhe o build em `https://huggingface.co/spaces/<usuario>/assistente-master-iag`.

## 4. Configurações no Space

- **Hardware/sleep**: já aplicados por `create_repo(space_hardware="cpu-upgrade",
  space_sleep_time=3600)`. Também dá para ajustar em *Settings → Hardware*.
- **Secret**: `DEEPSEEK_API_KEY` (definido pelo script; edite em *Settings →
  Variables and secrets*).
- **Público/privado**: o script cria privado; para mudar, use *Settings → Who
  can see this Space*.

## 5. Notas

- O app roda `uvicorn src.projeto_final.main:app` na porta **7860**
  (obrigatória nos Spaces) — ver `Dockerfile` na raiz.
- A 1ª execução baixa os modelos locais (faster-whisper, fastembed, Piper,
  OCR) e os cacheia em `data/processed/*_models`; as chamadas seguintes usam o
  cache do container.
- O corpus (`data/raw`) e os índices entram no repositório **privado** do Space
  via bundle (opção 1). Não tornar o Space público sem revisar direitos
  autorais/LGPD dos PDFs.
- Para um novo deploy após mudanças: rodar `bash deploy/deploy.sh` de novo
  (faz push incremental).
