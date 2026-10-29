# Parakeet Ultra + ROCm + OpenAI-compatible full duplex

Arquitetura full duplex **no aplicativo**, não streaming nativo do modelo. Entrada continua aberta enquanto a IA responde. Headphones são fortemente recomendados para evitar feedback e auto-interrupções.

## Instalação no Ubuntu

```bash
sudo apt update
sudo apt install -y git cmake build-essential libopenblas-dev libsndfile1 portaudio19-dev espeak-ng
# opcional: verifique o alvo real da sua placa
rocminfo | grep -m 3 gfx
# requer ROCm HIP compiler já instalado
which hipcc

git clone https://github.com/handy-computer/transcribe.cpp.git
cmake -S transcribe.cpp -B transcribe.cpp/build -DTRANSCRIBE_HIP=ON -DAMDGPU_TARGETS=gfx1200
cmake --build transcribe.cpp/build -j "$(nproc)"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models
hf download handy-computer/parakeet-ultra-gguf parakeet-ultra-Q8_0.gguf --local-dir models
```

Verifique a ID do repositório no Hugging Face se houver alteração upstream: o modelo é o `parakeet-ultra-Q8_0.gguf` distribuído por handy-computer/parakeet-ultra-gguf. Compilação HIP não equivale a suporte já comprovado na RX 9060 XT; pode ser necessário ajustar biblioteca ou kernel.

## Teste independente do ASR

```bash
# Arquivo de entrada mono 16 kHz
transcribe.cpp/build/bin/transcribe-cli --backend rocm -m models/parakeet-ultra-Q8_0.gguf audio.wav
```

## Execução

```bash
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=local
export OPENAI_MODEL='nome-exato-do-modelo-no-servidor'
python assistant.py
```

Caso `rocm` não funcione, compile em diretório separado com `-DTRANSCRIBE_VULKAN=ON` e execute com `--backend vulkan`. É um fallback de GPU AMD, mas não roda via ROCm.

## Limitações

- Parakeet Ultra não é um ASR streaming nativo; a transcrição de cada frase ocorre após silêncio.
- Barge-in é por VAD, com risco de falso positivo por eco. Use fones de ouvido.
- O motor de fala `espeak-ng` é simples e soa robótico; pode ser substituído por Piper posteriormente.
- A interrupção corta áudio de saída e cancela geração LLM, mas não interrompe um processo ASR já em execução.
- Sem supressão acústica de eco (AEC), diarização ou isolamento por locutor.
- CLI relançado para cada frase: implica custo de carregamento do modelo; para latência de produção, substituir por bindings nativos persistentes.
- Este pacote não foi testado em hardware ROCm 7.2 / gfx1200 real.
