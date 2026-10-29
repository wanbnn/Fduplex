<div align="center">

# 🎙️ Fduplex
### Seu assistente de voz local. Agora com ferramentas de verdade.

**Converse, pesquise, organize arquivos e execute tarefas usando a voz.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/Plataforma-Linux%20%7C%20Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://ubuntu.com)
[![MCP](https://img.shields.io/badge/MCP-Tools%20%26%20Skills-6D28D9?style=for-the-badge)](MCP.md)
[![Playwright](https://img.shields.io/badge/Browser-Playwright-2EAD33?style=for-the-badge&logo=playwright&logoColor=white)](https://playwright.dev/python/)
[![Local LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-111827?style=for-the-badge)](#-requisitos)
[![GitHub stars](https://img.shields.io/github/stars/wanbnn/Fduplex?style=for-the-badge&logo=github)](https://github.com/wanbnn/Fduplex/stargazers)

**[Instalar](#-instalação-rápida)** · **[Executar](#-abrir-o-assistente-completo)** · **[Funcionalidades](#-o-que-ele-faz)** · **[MCP e segurança](MCP.md)** · **[Guia detalhado](INSTALL.md)**

</div>

---

> [!IMPORTANT]
> O Fduplex é um projeto em desenvolvimento. O instalador configura o **app, o ambiente Python e o comando global**; o mecanismo ASR (binário + modelo GGUF), as dependências de áudio do sistema e o servidor LLM local **ainda exigem configuração**. Não se trata de uma instalação 100% autônoma.

## ✨ O que ele faz

| Recurso | Descrição |
| :--- | :--- |
| 🎤 **Conversa por voz** | Captura contínua, VAD e interrupção por voz (*barge-in*) |
| 🧠 **LLM local ou remoto** | Conecta-se a endpoints compatíveis com a API OpenAI |
| 🛠️ **MCP** | Ferramentas para arquivos, patches, consultas limitadas ao terminal e informações do sistema |
| 🌐 **Navegação** | Pesquisa e leitura de páginas usando Chromium + Playwright |
| 📂 **Busca de arquivos** | Índice SQLite com busca lexical por nome e conteúdo |
| 🧩 **Skills** | Instruções especializadas carregadas sob demanda |
| 🗜️ **Contexto** | Compactação por resumo quando o histórico estimado chega a 80.000 tokens |
| 🔈 **Resposta falada** | Síntese local com `espeak-ng` |
| 🔐 **Escopo de acesso** | Diretórios de leitura/escrita configuráveis, terminal restrito |

<details>
<summary><strong>🧱 Como funciona (arquitetura)</strong></summary>

```text
 Microfone
    │
    ▼
 VAD ────────────────► detecção de interrupção
    │
    ▼
 Parakeet Ultra (ASR)
    │
    ▼
 Agent + LLM OpenAI-compatible
    │       │
    │       └── MCP (stdio)
    │             ├── Arquivos / patch
    │             ├── Indexação e busca
    │             ├── Terminal restrito
    │             ├── Playwright / web
    │             └── Biblioteca de skills
    ▼
 Texto da resposta
    │
    ▼
 espeak-ng (TTS) ───► alto-falantes
```

A captura do microfone continua durante o processamento. **Parakeet Ultra não faz reconhecimento nativamente streaming**: resultados parciais são gerados processando trechos acumulados do áudio.

</details>

## 🚀 Instalação rápida

Para instalar o aplicativo no Ubuntu/Linux, execute **uma única vez**:

```bash
curl -fsSL https://raw.githubusercontent.com/wanbnn/Fduplex/main/install.sh | bash
```

> [!TIP]
> Antes de executar um instalador remoto, você pode inspecionar o conteúdo do [install.sh](install.sh). O script não precisa de `sudo`, cria um ambiente virtual e adiciona `fduplex` em `~/.local/bin`.

**Depois de instalar**, abra um terminal novo.

## ▶️ Abrir o assistente completo

Com o reconhecimento de voz e o servidor LLM já configurados, inicie a experiência com as ferramentas MCP habilitadas:

```bash
fduplex --mcp
```

Esse é o **comando para abrir o Fduplex completo**. O atalho `fduplex` sozinho também inicia o assistente com MCP habilitado por padrão.

Comandos úteis:

```bash
fduplex                 # Abrir assistente (MCP ativado por padrão)
fduplex --mcp           # Abrir explicitamente com as ferramentas MCP
fduplex --no-mcp        # Modo de conversa sem ferramentas
fduplex --help          # Opções de execução
```

**Importante:** o launcher inicia o `assistant.py`, mas **não inicializa automaticamente o LM Studio/servidor LLM**, nem instala por conta própria o modelo e o binário ASR. Configure esses pré-requisitos antes do primeiro uso.

## ⚙️ Requisitos

- **Ubuntu/Linux**, Git e Python **3.10+**; `python3-venv` para criar o ambiente.
- Dependências de áudio e compilação: `espeak-ng`, `cmake`, `build-essential`, `libopenblas-dev`, `libsndfile1`, `portaudio19-dev`.
- `transcribe.cpp` compilado e o arquivo `parakeet-ultra-Q8_0.gguf` disponível.
- Servidor **OpenAI-compatible** em execução. Para uso com MCP, o modelo e o servidor precisam suportar **tool/function calling**.
- Para GPU AMD com ROCm, suporte HIP e target de arquitetura adequados à sua placa; consulte a documentação do `transcribe.cpp`.
- Fones de ouvido recomendados para reduzir realimentação acústica.

### Preparar o reconhecimento de voz (Ubuntu + ROCm)

```bash
sudo apt update
sudo apt install -y git cmake build-essential libopenblas-dev libsndfile1 portaudio19-dev espeak-ng python3-venv

# Execute no diretório do aplicativo:
cd ~/.local/share/fduplex

git clone https://github.com/handy-computer/transcribe.cpp.git
# Ajuste gfx1200 para o alvo realmente suportado pela sua GPU.
cmake -S transcribe.cpp -B transcribe.cpp/build -DTRANSCRIBE_HIP=ON -DAMDGPU_TARGETS=gfx1200
cmake --build transcribe.cpp/build -j "$(nproc)"

./.venv/bin/python -m huggingface_hub download handy-computer/parakeet-ultra-gguf parakeet-ultra-Q8_0.gguf --local-dir models
```

> [!NOTE]
> Confirme o comando de download disponível na versão instalada de `huggingface_hub` (por exemplo, `hf download`). O alvo `gfx1200` é ilustrativo, **não uma garantia de compatibilidade**. Se instalou a partir de um checkout local, execute a preparação no diretório desse checkout, não necessariamente em `~/.local/share/fduplex`.

### Configurar o modelo de linguagem

O instalador cria o arquivo:

```text
~/.config/fduplex/env
```

Exemplo de valores (ajuste o identificador do modelo para o servidor que está executando):

```bash
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=local
OPENAI_MODEL=nome-exato-do-modelo
```

Para iniciar com MCP, habilite o suporte a chamadas de ferramentas no backend. Caso não esteja disponível, use `fduplex --no-mcp`.

## 🧩 MCP: ferramentas e skills

O servidor `fduplex_mcp.py` expõe ferramentas de **leitura/criação/edição de arquivos**, aplicação de *unified diff*, **busca local**, consultas restritas ao terminal, **pesquisa e leitura de páginas** com Playwright e **informações do sistema**.

As skills ficam em `skills/<nome>/SKILL.md`. O modelo pode descobrir e carregar instruções de pesquisa, desenvolvimento, análise de dados e organização pessoal quando necessário.

**Segurança:** por padrão, a escrita é limitada ao workspace. O terminal **não executa shell arbitrário**; a navegação é voltada à leitura. Resultados da web devem ser tratados como conteúdo não confiável. Consulte [MCP.md](MCP.md) para permissões e limitações.

## 📦 Estrutura

```text
Fduplex/
├── assistant.py             # Voz, ASR, TTS e integração com o agente
├── fduplex_agent.py         # Loop de ferramentas e compactação de contexto
├── fduplex_mcp.py           # Servidor MCP local (stdio)
├── skills/                  # Biblioteca de skills
├── tests/                   # Testes automatizados
├── install.sh               # Instalador inline / comando global
├── INSTALL.md               # Instalação e configuração
├── MCP.md                   # Arquitetura de ferramentas e segurança
└── requirements.txt
```

## 🧪 Desenvolvimento e validação

```bash
python3 -m compileall -q assistant.py fduplex_agent.py fduplex_mcp.py
python3 -m unittest discover -s tests -v
```

Os testes cobrem partes do MCP e regras de escopo; **não substituem** testes de microfone, ROCm, modelo de transcrição, Playwright e servidor LLM no hardware real.

## ⚠️ Limitações conhecidas

- Reconhecimento Parakeet Ultra **não é streaming nativo**; transcrições parciais dependem de novas passagens pelo ASR.
- Interrupção por VAD pode ser ativada por eco sem cancelamento acústico dedicado.
- O TTS `espeak-ng` prioriza simplicidade, não naturalidade.
- O histórico é compactado por **resumo com perda**, com contagem aproximada de tokens; não há memória durável entre sessões.
- Busca local é **lexical**, não semântica por embeddings.
- Ferramentas dependem de suporte real a `tool_calls` no LLM e no servidor escolhido.
- O launcher não instala/abre automaticamente o motor LLM ou o ASR.

---

<div align="center">

**Fduplex** · Um assistente de voz aberto à evolução.

Criado por [@wanbnn](https://github.com/wanbnn) · [Issues](https://github.com/wanbnn/Fduplex/issues) · [Código-fonte](https://github.com/wanbnn/Fduplex)

</div>
