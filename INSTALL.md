# Instalação e execução pelo terminal

No Ubuntu, execute uma vez:

```bash
curl -fsSL https://raw.githubusercontent.com/wanbnn/Fduplex/main/install.sh | bash
```

**Atenção:** o comando acima só funcionará depois que o PR do instalador for incorporado à `main`. Para testar o PR antes do merge:

```bash
curl -fsSL https://raw.githubusercontent.com/wanbnn/Fduplex/feat/cli-installer/install.sh | bash
```

Ou, de dentro do repositório:

```bash
bash install.sh
```

O script cria `~/.local/share/fduplex` (quando baixado via curl), um ambiente virtual, instala `requirements.txt`, tenta instalar Chromium do Playwright e disponibiliza o executável `~/.local/bin/fduplex` em novos terminais. Não usa sudo. Se faltar suporte de sistema para áudio, instale os pacotes de sistema descritos no README principal.

Depois da instalação, abra um novo terminal e execute:

```bash
fduplex
```

Edite `~/.config/fduplex/env` para configurar `OPENAI_MODEL`, `OPENAI_BASE_URL`, workspace e raízes permitidas. O launcher carrega esse arquivo a cada início. Execute `fduplex --no-mcp` para desabilitar MCP ou `fduplex --help` para ver parâmetros.

**Limitações:** o instalador não baixa/compila automaticamente o `transcribe.cpp`, não baixa o GGUF do Parakeet, não inicia o LM Studio nem instala dependências de áudio do sistema; essas etapas continuam necessárias conforme o README. Ele cria o comando global e instala as bibliotecas Python. Caso o diretório de destino já seja um checkout Git, o instalador não executa `git pull` para não alterar mudanças locais. Para uso sem navegador defina `FDUPLEX_INSTALL_BROWSER=0`. Para instalar uma branch diferente defina `FDUPLEX_REF=nome-da-branch`.

Para desfazer, remova `~/.local/bin/fduplex`, `~/.config/fduplex` e, se criado pelo instalador, `~/.local/share/fduplex`. Seu workspace não é apagado automaticamente.
