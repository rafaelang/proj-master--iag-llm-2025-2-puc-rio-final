# deploy-mcp — Servidor MCP para o deploy.sh (Hugging Face)

Servidor [Model Context Protocol](https://modelcontextprotocol.io/) que expõe o
script de publicação **`deploy/deploy.sh`** (Space público + dataset privado na
Hugging Face) como tools para agentes de IA (opencode, Claude, etc.). Usa
**FastMCP** com transporte **stdio** — nenhum servidor HTTP é aberto.

> Ferramenta de desenvolvimento local. Não faz parte do bundle do HF Space
> (o `deploy.sh` exclui `deploy/mcp_deploy/`).

## Estrutura

```
deploy/mcp_deploy/
├── pyproject.toml      # projeto uv isolado (fastmcp)
├── server.py           # servidor FastMCP (stdio)
├── smoke_stdio.py      # handshake MCP manual (não executa deploy)
└── tests/              # pytest (subprocess mockado)
```

## Instalação

```bash
cd deploy/mcp_deploy
uv sync                  # cria .venv/ local (fastmcp, pytest)
```

Pré-requisito para **publicar**: `deploy/.env` com `HF_TOKEN` (e opcionalmente
`DEEPSEEK_API_KEY`/`HF_TOKEN_READ`) — ver `deploy/README.md`. O dry-run e o
status funcionam sem token real.

## Tools

| Tool | Efeito | Confirmação |
|---|---|---|
| `deploy_status()` | Inspeção **read-only** local (script, `.env` presente — só nomes de variáveis —, bundle) | — |
| `deploy_dry_run()` | Monta `deploy/build/` **sem publicar** (`deploy.sh --dry-run`) | — |
| `deploy_publish(confirm)` | Publica Space público + dataset privado + secrets | `confirm="publicar"` explícito |

## Registro no opencode (workspace raiz `opencode.json`)

```jsonc
"deploy": {
  "type": "local",
  "command": ["uv", "run", "--project", "<caminho>/deploy/mcp_deploy", "deploy-mcp"],
  "enabled": true
}
```

## Executar / testar

```bash
cd deploy/mcp_deploy
uv run python server.py        # sobe o servidor MCP (stdio)
uv run python smoke_stdio.py   # handshake MCP + lista as tools
uv run python -m pytest tests/ -q   # testes com subprocess mockado
```

## Segurança

- **Nenhum token é aceito como argumento de tool** — o script lê `deploy/.env`
  e mascara o token na saída (`mask()`); o servidor nunca loga valores.
- `deploy_publish` é a única tool com efeito externo e exige `confirm="publicar"`:
  recomendado também bloqueá-la por permissão no opencode (aprovação humana).
- `deploy_status` reporta apenas os **nomes** das variáveis do `.env`, nunca os valores.
- O corpus (LGPD/direitos autorais) nunca vai para o Space: o `deploy.sh` sobe
  `data/raw` + índices para o **dataset privado** e exclui `data/` do bundle.