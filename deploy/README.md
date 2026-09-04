# Deploy — Hugging Face (API pública + corpus privado)

Publica o assistente (v0.3: voz + RAG + imagem) em um **Space público** com a
**API acessível** na internet, mantendo o **corpus privado** fora do repositório
público (direitos autorais/LGPD).

## Arquitetura

```
Internet ──► Space PÚBLICO rafaelang/assistente-master-iag  (código, sem data/)
                │  boot: snapshot_download do dataset privado
                ▼            (secret HF_TOKEN_READ)
            /app/data  ← dataset PRIVADO rafaelang/assistente-master-iag-dados
                         (data/raw + índices processed/rag*)
```

| Item | Valor |
|---|---|
| Space (API) | `assistente-master-iag` — **público** · SDK `docker` |
| Dataset (corpus) | `assistente-master-iag-dados` — **privado** |
| Hardware / Sleep | `cpu-upgrade` · `3600` s (1 h) |

## Segredos (`deploy/.env`, gitignored)

```bash
cp deploy/.env.example deploy/.env
# HF_TOKEN=...            (seu token com permissao de write em Spaces/datasets)
# HF_TOKEN_READ=...       (ideal: token read-only p/ o container baixar o dataset)
# DEEPSEEK_API_KEY=...    (vazio = herda do .env da raiz)
chmod 600 deploy/.env
```

Secrets do Space definidos pelo script: `DEEPSEEK_API_KEY`, `HF_TOKEN_READ`,
`HF_DATA_REPO`. **Nunca commitar tokens.**

## Deploy

```bash
bash deploy/deploy.sh --dry-run    # revisa o bundle (código sem data/)
bash deploy/deploy.sh              # publica
```

O script: 1) cria/atualiza o **dataset privado** e envia `data/raw` + índices
(`processed/rag`, `processed/rag_v3`, sem `*_models`); 2) cria o **Space
público** (docker, `cpu-upgrade`, sleep 3600s); 3) define os secrets; 4) monta
`deploy/build/` **sem `data/`** e faz push.

No boot, `deploy/entrypoint.sh` baixa o dataset para `/app/data` e então sobe
`uvicorn` na porta 7860.

## Notas

- O corpus **não fica** no repositório público nem no histórico do Space;
- Modelos locais (whisper/fastembed/piper/OCR) são baixados em runtime; o GGUF
  do SLM (v0.4, `processed/slm_models/*`) vem do **dataset privado** no boot
  junto com o corpus;
- ⚠️ A API pública expõe `/chat` e `/chat/imagem` a qualquer pessoa (consome o
  `DEEPSEEK_API_KEY`). Recomenda-se criar um token **read-only** para o
  `HF_TOKEN_READ` e revisar uso/custo. Para restringir acesso no futuro, adicione
  uma chave de aplicação (ex.: header `X-API-Key`).
