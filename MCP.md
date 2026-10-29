# MCP: assistente pessoal Fduplex

Esta branch incorpora um **servidor MCP local stdio** e um **agente MCP** ao assistente de voz.

## Instalação

No Ubuntu, no diretório do Fduplex:

```bash
git checkout feat/mcp-personal-assistant
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
mkdir -p "$HOME/Fduplex-workspace" "$HOME/Documents"
export FDUPLEX_WORKSPACE="$HOME/Fduplex-workspace"
export FDUPLEX_READ_ROOTS="$HOME/Documents:$HOME/Fduplex-workspace"
export FDUPLEX_WRITE_ROOTS="$HOME/Fduplex-workspace"
python assistant.py --mcp
```

As configurações de modelo, ROCm, ASR e TTS continuam descritas no README principal.

O modelo usado no servidor OpenAI-compatible **precisa suportar function/tool calling**, inclusive `tools` e `tool_calls`. Nem todos os modelos locais ou servidores OpenAI-compatible oferecem suporte funcional. Se seu modelo não suportar, rode `python assistant.py --no-mcp` até trocar o modelo/backend.

## Componentes

- `fduplex_mcp.py` servidor FastMCP em stdio. Ferramentas: `list_files`, `read_file`, `write_file`, `apply_patch`, `run_terminal`, `index_files`, `search_files`, `list_skills`, `read_skill`, `browse_page`, `web_search`, `system_info`.
- `fduplex_agent.py` cliente MCP, roteamento de chamadas de ferramenta, execução sequencial, histórico e compactação.
- `skills/*/SKILL.md` instruções especializadas que o modelo pode carregar conforme necessidade.

## Memória e contexto

Ao estimar 80.000 tokens no histórico, o agente resume as mensagens anteriores preservando instruções de sistema e interações recentes. `tiktoken` usa a codificação cl100k_base, portanto a contagem **é aproximada** e pode divergir do tokenizer do LLM. O resumo é uma compressão com perda; não substitui uma memória vetorial persistente. Resumos anteriores permanecem apenas na sessão em andamento.

## Segurança

O servidor tem escopo de leitura e escrita explícitos. `FDUPLEX_READ_ROOTS` e `FDUPLEX_WRITE_ROOTS` aceitam listas separadas por dois-pontos no Linux. Por padrão, escreve somente no workspace. As operações de arquivo recusam caminhos fora dos roots, inclusive symlinks resolvidos fora deles. O terminal **não é shell irrestrito**: só permite consultas limitadas e arquivos dentro do workspace. Para execução avançada de comandos, implemente um executor com autorização humana por ação, isolamento por container e auditoria; não amplie a allowlist às cegas.

O navegador é somente leitura, usa Chromium isolado e bloqueia destinos privados por resolução DNS; isso **não é isolamento forte** contra um adversário controlando DNS/rede. Para ambientes sensíveis, use namespace de rede/container sem acesso a serviços internos. Não carregue perfis autenticados por padrão.

`index_files` cria índice SQLite local, por nome/conteúdo, atualizável sob demanda; `search_files` usa ranking lexical e recência, **não embeddings semânticos**. Arquivos excluídos no índice são limpos somente após reconstrução futura; remova o banco manualmente se revogar permissão de leitura.

## Teste rápido

```bash
python -m compileall -q assistant.py fduplex_mcp.py fduplex_agent.py
python -m unittest discover -s tests -v
python fduplex_mcp.py  # servidor stdio, é normal não imprimir texto
```

Exemplos de solicitações por voz: “Crie um arquivo de anotações no meu workspace”, “Procure documentos sobre faturamento na pasta Documentos”, “Pesquise a documentação atual do Playwright”, “Qual é o espaço livre em disco?”.

A captura de áudio e a ASR permanecem como no projeto original. A saída das ferramentas pode introduzir latência na resposta; não é um agente com acesso irrestrito nem com automações em segundo plano.
