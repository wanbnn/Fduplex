"""Opt-in CPU speech integration; downloads/cache are handled by Pocket TTS."""
import os
import threading
import time

import numpy as np
import pytest

from tts_engine import PocketVoice


@pytest.mark.skipif(os.getenv('FDUPLEX_TEST_TTS') != '1', reason='real CPU TTS integration is opt-in')
def test_real_portuguese_audio_stream_and_cooperative_stop():
    voice = PocketVoice()
    voice.load()
    stop = threading.Event()
    started = time.monotonic()
    chunks = voice.chunks('Olá! Esta frase precisa começar a ser falada enquanto o áudio ainda está sendo gerado.', stop)
    first = next(chunks)
    first_s = time.monotonic() - started
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert 0 < first.size < voice.rate  # An incremental packet, not a completed sentence.
    assert voice.rate == 24000
    stopped = time.monotonic()
    stop.set()
    assert not list(chunks)
    assert time.monotonic() - stopped < 2
    print(f'Pocket first PCM chunk: {first_s:.3f}s; samples: {first.size}')
