# Space público + corpus em dataset privado

## Status

accepted

## Contexto e decisão

O corpus são PDFs/slides do curso (direitos autorais + possível PII). Decidimos que o
**Space público** (`assistente-master-iag`) contém **apenas o código**, e o corpus
(`data/raw`) + índices ficam num **dataset Hugging Face privado**
(`assistente-master-iag-dados`), baixado em runtime no boot do container
(`snapshot_download` com `HF_TOKEN_READ`).

## Consequences

- Nenhum PDF bruto vai ao repositório público nem ao Space (LGPD/direitos autorais).
- O boot do container depende do dataset privado; falha de download → serviço degradado.
- O bundle de deploy é **restrito** ao necessário para rodar (código, `deploy/`,
  `prompts/`, `static/`); `docs/`, `tests/` e `scripts/` **não** são publicados no Space.
