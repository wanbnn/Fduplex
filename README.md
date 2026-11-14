# Fduplex — conversa por voz no terminal

Captura contínua, transcrição incremental, resposta de um servidor compatível com
OpenAI e síntese local interrompível. A interface mostra o estado do microfone,
a probabilidade de fala, a transcrição ao vivo e a conversa.

O modo padrão procura **Nemotron 3.5 ASR Streaming 0.6B**, com suporte a
português brasileiro. O áudio novo é enviado para a API nativa
`transcribe_stream_feed`; o modelo e seus caches permanecem carregados.
O Parakeet Ultra continua disponível para reconhecimento **por frase** e a
interface identifica esse modo como offline. Ele não oferece streaming nativo.

## Instalação no Ubuntu

```bash
sudo apt install git cmake build-essential libopenblas-dev libsndfile1 portaudio19-dev espeak-ng libspeexdsp1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/handy-computer/transcribe.cpp.git
# Ajuste o alvo para a GPU instalada. ROCm/HIP precisa estar instalado.
cmake -S transcribe.cpp -B transcribe.cpp/build \
  -DTRANSCRIBE_HIP=ON -DAMDGPU_TARGETS=gfx1200 -DBUILD_SHARED_LIBS=OFF
cmake --build transcribe.cpp/build --target transcribe-cli -j "$(nproc)"
python build_asr.py
python setup_audio.py
```

O worker usa a API C de transcribe.cpp 0.4.1 e o manifesto de link da compilação.
`build_asr.py --source /outro/checkout/transcribe.cpp` permite reutilizar um
checkout já compilado, sem recompilar os kernels HIP. O resultado fica em
`bin/fduplex-asr`; não é necessário iniciar um CLI para cada parcial.

`setup_audio.py` baixa cerca de 751 MB do modelo de streaming e o Silero VAD
ONNX. O Silero tem revisão e SHA-256 fixados. Os downloads ficam por padrão em
`$XDG_DATA_HOME/fduplex/models` (ou `~/.local/share/fduplex/models`).
Use `--models-dir ./models` para manter os arquivos no projeto.

Para instalar o comando `fduplex`, execute `bash install.sh`. O instalador
preserva configurações existentes. `FDUPLEX_TRANSCRIBE_DIR` indica o checkout
nativo; `FDUPLEX_INSTALL_AUDIO=0` evita baixar modelos na instalação.

## Executar

```bash
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=local
export OPENAI_MODEL='nome-exato-do-modelo-no-servidor'
python assistant.py
# Ou, após instalar:
fduplex
```

Fique em silêncio durante os primeiros 0,8 s de calibração e use fones.

| Tecla | Ação |
| --- | --- |
| M | Pausar ou retomar o microfone, preservando os processamentos em andamento |
| Esc | Interromper a resposta e o áudio da IA |
| Q / Ctrl+C | Encerrar e liberar áudio, MCP e o processo ASR |

`--plain` oferece saída sem tela interativa, também usada automaticamente em
pipes. `--list-devices` lista os dispositivos; `--mic N` seleciona a entrada.
Diagnósticos nativos e do MCP vão para
`$XDG_STATE_HOME/fduplex/diagnostics.log` (padrão: `~/.local/state/fduplex/diagnostics.log`).

## Ruído e sensibilidade

O frontend aplica supressão SpeexDSP de -25 dB e detecção neural Silero na CPU.
O início exige cerca de 192 ms de fala confirmada, rejeita impulsos isolados e
usa um limiar de energia adaptado ao piso de ruído. Uma frase precisa conter
pelo menos 256 ms de voz; o término aguarda 640 ms de pausa. O processamento
não roda no callback de captura.

Se o teclado ainda disparar a captura, experimente:

```bash
fduplex --speech-probability 0.8 --start-ms 256 --min-speech-ms 320
```

Para voz baixa, reduza `--speech-probability` (por exemplo, 0.55) ou `--threshold`.
Ajuste `--noise-suppress-db` entre -60 e 0; `--no-denoise` permite comparar o
áudio sem supressão. Isso reduz falsos disparos, mas não promete remover todo
teclado, fala de outras pessoas ou eco dos alto-falantes. Não há AEC.

## Streaming e confirmação final

O streaming usa blocos de cerca de 320 ms e lookahead configurável por
`--stream-right` (padrão 3, aproximadamente 240 ms; não é a latência total).
As parciais são hipóteses e podem divergir da frase final.

A conferência final está ligada por padrão: após finalizar o stream, uma única
passagem pela frase completa, no mesmo processo e modelo já carregado, confirma
o texto enviado ao LLM. Isso recupera omissões observadas no início de algumas
frases no teste local de streaming. `--no-verify-final` reduz o trabalho final,
com possível perda de precisão. A transcrição ao vivo continua usando streaming
nativo, sem retranscrever snapshots a cada atualização.

`--asr-mode streaming` exige um modelo com suporte nativo e falha claramente se
receber o Ultra. Para usar o modelo antigo:

```bash
fduplex --asr-mode offline --gguf ./models/parakeet-ultra-Q8_0.gguf
```

## Voz em streaming

A saída padrão usa **Pocket TTS 4.0.1**, modelo português de aproximadamente
109,5 milhões de parâmetros, com a voz `rafael`. A síntese roda na CPU, deixando
a GPU para o Nemotron. O modelo permanece carregado e fornece PCM incremental
por `generate_audio_stream`; a reprodução começa nos primeiros blocos, antes
de terminar a geração da frase. Não é um WAV completo repartido depois.

A resposta do LLM é encaminhada à voz por frases ou trechos de até cerca de
180 caracteres. Isso funciona também com MCP: a voz pode falar um preâmbulo
antes das ferramentas e continuar com a resposta. `Esc` e nova fala cancelam o áudio pendente e a geração em andamento.
`M` apenas pausa ou retoma a entrada do microfone; a fala já capturada é
finalizada, e transcrição, ferramentas, resposta e reprodução continuam.

```bash
# A dependência já faz parte de requirements.txt. Para instalar separadamente:
pip install -r requirements-tts.txt
# Baixar/cachear e preparar a voz, sem reproduzir áudio:
python setup_tts.py
fduplex --tts pocket --tts-language portuguese --tts-voice rafael
```

O PyTorch usado no Linux x86_64 é a versão CPU, sem bibliotecas NVIDIA. Modelos
e voz ficam no cache padrão do Hugging Face (`HF_HOME` pode mudar esse local).
A primeira inicialização baixa os arquivos; as seguintes reutilizam o cache.
`TTS_BACKEND`, `TTS_LANGUAGE` e `TTS_VOICE` também configuram essas opções.
`--tts espeak` mantém a voz antiga e `--tts none` desliga a síntese. A opção
`--tts-quantize` habilita int8, mas fica desligada por padrão para preservar a
configuração de qualidade testada.

No Ryzen 5 5600G local, duas gerações em português entregaram o primeiro PCM em
136–246 ms e produziram 6,72–7,28 s de áudio em 2,06–2,13 s. Esses números medem
somente a síntese, sem carregamento, LLM, espera por frase ou dispositivo de
saída. A naturalidade e o sotaque devem ser avaliados ouvindo a
[amostra gerada](samples/pocket-tts-portugues.wav).

Referência do modelo e suporte a português:
[Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts).

## Caminhos portáveis

Opções explícitas têm prioridade: `--asr-worker` / `FDUPLEX_ASR_WORKER`,
`--gguf` / `PARAKEET_GGUF`, `--vad-model` / `FDUPLEX_VAD_MODEL`.
Caminhos relativos são resolvidos a partir dos diretórios de descoberta e `~`
é expandido. Um override inválido gera erro, sem substituição silenciosa.

A descoberta procura no diretório de onde o comando foi chamado, na pasta do
aplicativo, em `FDUPLEX_WORKSPACE` (padrão `~/Fduplex-workspace`) e em
`$XDG_DATA_HOME/fduplex`. O worker também pode estar no `PATH`. Em modo auto,
um GGUF de streaming encontrado nesses locais tem preferência sobre o Ultra.
`--cli` / `TRANSCRIBE_CLI` é mantido para localizar um worker construído junto
a um checkout legado.

## Verificação

```bash
pip install pytest
python -m pytest -q tests
# Teste opcional com modelos reais, ROCm e fala sintetizada em português:
FDUPLEX_TEST_NATIVE=1 python -m pytest -q tests
# Teste opcional de geração e cancelamento com a voz real na CPU:
FDUPLEX_TEST_TTS=1 python -m pytest -q tests/test_pocket_native.py
```

Os testes cobrem impulsos, ruído contínuo, confirmação de voz, pausa, mute,
caminhos portáveis, layout do terminal e streaming incremental. Os testes
sintéticos não substituem a avaliação com seu microfone e teclado reais.

## Conversa e diretório das ferramentas

Cada fala é enviada como uma mensagem `user` separada. O histórico mantém o
contexto, inclusive texto já gerado de respostas interrompidas. As instruções
padrão pedem respostas curtas em português, sem reiniciar saudações, sem listas,
Markdown ou emojis. O TTS também remove esses elementos antes de sintetizar,
preservando acentos e pontuação normal.

O workspace do MCP é sempre o diretório onde o usuário chamou `fduplex`:
`cd meu-projeto && fduplex` permite criar, ler, editar e aplicar patches dentro
de `meu-projeto`. Caminhos relativos são resolvidos nesse diretório, inclusive
quando o launcher executa o Python da instalação em outra pasta. Configurações
antigas de `FDUPLEX_WORKSPACE`, `FDUPLEX_READ_ROOTS` e `FDUPLEX_WRITE_ROOTS` não
redirecionam as ferramentas para um workspace fixo. O índice fica em `.fduplex/`
dentro do projeto. Ao executar `python /caminho/assistant.py` diretamente, vale
o diretório atual do terminal.

## Aplicativos e latência de rede

O MCP oferece `launch_application` para abrir aplicativos instalados, como
Firefox, Chromium, VS Code, editores de texto e calculadora. O processo preserva
a sessão gráfica e o HOME real, usa o diretório do projeto e não espera o
aplicativo fechar. O retorno confirma o envio do pedido de abertura, sem
prometer que uma janela ficou visível.

`ping_host` mede ICMP localmente com até cinco pacotes e retorna a latência
medida ou o erro real. `run_terminal` também aceita `firefox`, `ping google.com`
e `ping -c 3 google.com`. O terminal usa Bash real, com pipes, redirecionamentos, argumentos e scripts.
As permissões são as do usuário Linux. A TUI mostra a ferramenta em execução. Se o usuário
interromper a ação, o histórico registra que o resultado não foi confirmado.

A cada chamada ao LLM, o aplicativo reforça respostas orais de uma ou duas
frases, normalmente até 60 palavras. A interface e o TTS removem listas,
Markdown e emojis mesmo quando o modelo não segue essas instruções.

## Números na voz

Antes da síntese em português, números são convertidos por extenso: `42 ms`
vira “quarenta e dois milissegundos”, `2,5%` vira “dois vírgula cinco por cento”
e `R$ 12,50` vira “doze reais e cinquenta centavos”. A conversão também trata
inteiros, anos, milhares no formato brasileiro, decimais com vírgula ou ponto,
valores negativos, unidades comuns e endereços IPv4. Os números na tela e no
histórico permanecem no formato original.

## Terminal real do usuário

`run_terminal` executa Bash no diretório de lançamento com HOME, PATH e ambiente
reais do usuário. Não há lista de comandos permitidos. Ele aceita scripts,
pipes, redirecionamentos, argumentos e CLIs de controle de aplicativos.
`background=true` inicia programas persistentes e retorna PID e caminho do log
em `.fduplex/`, sem esperar o programa fechar. O timeout de comandos em primeiro
plano pode ser ajustado até 300 segundos; interrupções encerram o grupo do
processo para não deixar pipelines presos.

`launch_application` também aceita o nome/caminho de um executável instalado e
uma lista opcional de argumentos. O modelo recebe instruções para investigar
as ferramentas instaladas e executar o pedido antes de alegar impossibilidade.
A disponibilidade de controle gráfico continua dependendo da sessão Linux e
das ferramentas do aplicativo; erros concretos são retornados pelo comando.
As ferramentas dedicadas de arquivos continuam relativas ao projeto.
