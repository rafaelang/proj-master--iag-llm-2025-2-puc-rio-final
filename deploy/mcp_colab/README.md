# colab-mcp — Servidor MCP para o google-colab-cli

Servidor [Model Context Protocol](https://modelcontextprotocol.io/) que expõe o
[google-colab-cli](https://github.com/googlecolab/google-colab-cli) como tools
para agentes de IA (Claude, Cursor, VS Code, etc.). Usa **FastMCP** com o
transporte **stdio** — nenhum servidor HTTP é aberto.

Provisionar VMs (CPU/GPU/TPU), executar código/scripts em kernels remotos,
transferir arquivos, montar Google Drive e instalar pacotes — tudo pela mesma
interface MCP.

> Ferramenta de desenvolvimento local. Não faz parte do bundle do HF Space
> (o `deploy.sh` exclui `deploy/mcp_colab/`).

## Estrutura

```
deploy/mcp_colab/
├── pyproject.toml      # projeto uv isolado (fastmcp + google-colab-cli)
├── server.py           # servidor FastMCP (stdio)
├── smoke_stdio.py      # handshake MCP manual (sem provisionar VM)
└── tests/              # pytest (subprocess mockado)
```

## Instalação

```bash
cd deploy/mcp_colab
uv sync                  # cria .venv/ local (fastmcp, google-colab-cli, pytest)
```

## Autenticação (uma vez)

O fluxo de login do CLI é interativo e não é exposto como tool. Faça antes de
usar o servidor:

```bash
cd deploy/mcp_colab
uv run colab auth        # abre o navegador / cola o código de autorização
uv run colab whoami      # confirma e-mail/escopos
```

A sessão fica salva em `~/.config/colab-cli/` e é compartilhada com o CLI.

## Executar / testar

```bash
cd deploy/mcp_colab
uv run python server.py     # sobe o servidor MCP (stdio)
uv run python smoke_stdio.py  # handshake MCP + lista as tools
uv run pytest tests -q      # testes unitários (sem tocar no Colab)
```

## Registrar num cliente MCP

Aponte o cliente para o comando abaixo (substitua o caminho absoluto). Usar
`uv run` garante que o `colab` (binário do CLI) esteja no PATH do servidor:

```json
{
  "mcpServers": {
    "colab": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/CAMINHO_ABSOLUTO/deploy/mcp_colab",
        "python",
        "server.py"
      ],
      "cwd": "/CAMINHO_ABSOLUTO/deploy/mcp_colab"
    }
  }
}
```

## Tools expostas

| Tool | Comando `colab` |
|---|---|
| `colab_new_session` | `new [-s] [--gpu] [--tpu]` |
| `colab_list_sessions` | `sessions` |
| `colab_session_status` | `status [-s]` |
| `colab_restart_kernel` | `restart-kernel [-s]` |
| `colab_stop_session` | `stop [-s]` |
| `colab_session_url` | `url [-s] [--open]` |
| `colab_exec_code` | `exec` (código via stdin) |
| `colab_exec_file` | `exec -f <arquivo>` (.py/.ipynb local) |
| `colab_run_script` | `run` (VM efêmera, `--gpu/--tpu/--keep`) |
| `colab_list_files` | `ls [caminho]` |
| `colab_upload_file` | `upload <local> <remoto>` |
| `colab_download_file` | `download <remoto> <local>` |
| `colab_remove_file` | `rm <remoto>` |
| `colab_mount_drive` | `drivemount` |
| `colab_install_packages` | `install [pacotes] [-r req]` |
| `colab_export_log` | `log [-o saída] [-n linhas]` |
| `colab_whoami` | `whoami` (conta autenticada) |
| `colab_version` | `version` |

## Limitações

- Comandos interativos/TTY não são tools: `repl`, `console`, `ssh` e `edit`
  (abrem terminal/editor local) e o login interativo `colab auth`.
- `colab exec`/`colab run` têm timeout remoto padrão de 30 s do próprio CLI;
  passe `timeout` quando o código demorar mais.
- Suporte oficial do CLI: Linux e macOS (Windows não suportado).
- GPUs aceitas: `T4`, `L4`, `G4`, `H100`, `A100`; TPUs: `v5e1`, `v6e1`
  (disponibilidade varia com o plano Colab).
