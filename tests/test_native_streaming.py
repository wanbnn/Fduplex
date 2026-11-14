"""Opt in with FDUPLEX_TEST_NATIVE=1; uses the real GPU and downloaded models."""
import asyncio
import io
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from asr_config import discover, STREAM_MODEL
from asr_engine import ASREngine
from audio_frontend import Denoiser, Silero, SpeechGate, FRAME

pytestmark = pytest.mark.skipif(os.getenv('FDUPLEX_TEST_NATIVE') != '1', reason='native ASR integration is opt-in')


def test_real_stream_updates_before_finalize_and_final_verification_preserves_sentence(tmp_path):
    if not shutil.which('espeak-ng'):
        pytest.skip('espeak-ng required')
    raw = subprocess.run(['espeak-ng', '-v', 'pt-br', '-s', '155', '--stdout',
                          'Bom dia. Eu gostaria de testar o reconhecimento de voz em português. '
                          'Você consegue entender esta frase enquanto eu ainda estou falando?'],
                         capture_output=True, check=True).stdout
    pcm, rate = sf.read(io.BytesIO(raw), dtype='float32')
    pcm = np.interp(np.arange(int(len(pcm) * 16000 / rate)) * rate / 16000, np.arange(len(pcm)), pcm).astype(np.float32)
    pcm = np.concatenate((np.zeros(16000, np.float32), pcm, np.zeros(16000, np.float32)))
    vad = Silero(discover(None, ['models/silero_vad.onnx'], 'vad'))
    denoiser = Denoiser()
    gate = SpeechGate(vad, denoiser)
    events = []
    audio = []
    try:
        for i in range(0, len(pcm) - FRAME, FRAME):
            clean, event = gate.process(pcm[i:i + FRAME])
            if event:
                audio.append(clean)
                events.append(event)
        assert events.count('start') == 1
        assert events.count('end') == 1
        assert 'discard' not in events
    finally:
        denoiser.close()
    utterance = np.concatenate(audio)

    async def scenario():
        args = SimpleNamespace(asr_worker=discover(None, ['bin/fduplex-asr'], 'worker', True),
                               gguf=discover(None, ['models/' + STREAM_MODEL], 'model'),
                               backend=os.getenv('ASR_BACKEND', 'rocm'), language='pt-BR', stream_right=3,
                               asr_mode='streaming')
        with (tmp_path / 'native.log').open('w') as log:
            engine = ASREngine(args, log)
            try:
                await engine.start()
                assert engine.streaming
                await engine.request('B')
                changes = []
                for i in range(0, len(utterance), 5120):
                    result = await engine.request('A', utterance[i:i + 5120])
                    if result['text'] and (not changes or changes[-1] != result['text']):
                        changes.append(result['text'])
                assert len(changes) >= 3
                await engine.request('F')
                final = await engine.request('R', utterance)
                text = final['text'].lower()
                assert 'bom dia' in text and 'português' in text and 'falando' in text
                # Back-to-back requests reuse one loaded process without retaining old text.
                process_id = engine.proc.pid
                await engine.request('B')
                silence = await engine.request('A', np.zeros(32000, dtype=np.float32))
                silence_final = await engine.request('F')
                assert not silence['text'] and not silence_final['text']
                assert engine.proc.pid == process_id
            finally:
                await engine.close()
    asyncio.run(scenario())
