import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import sounddevice as sd

from tts_engine import PCMBuffer, SpeechOut, TextSegments


def test_pcm_callback_preserves_order_and_completes_last_block():
    buffer = PCMBuffer(threading.Event())
    buffer.put(np.arange(4, dtype=np.float32))
    buffer.put(np.arange(4, 8, dtype=np.float32))
    first = np.empty((6, 1), np.float32)
    buffer.callback(first, 6, None, None)
    np.testing.assert_array_equal(first[:, 0], np.arange(6))
    buffer.source_done.set()
    last = np.empty((6, 1), np.float32)
    with pytest.raises(sd.CallbackStop):
        buffer.callback(last, 6, None, None)
    np.testing.assert_array_equal(last[:, 0], [6, 7, 0, 0, 0, 0])


def test_interrupt_aborts_buffered_audio_immediately():
    stop = threading.Event()
    buffer = PCMBuffer(stop)
    buffer.put(np.ones(300, dtype=np.float32))
    stop.set()
    output = np.ones((100, 1), np.float32)
    with pytest.raises(sd.CallbackAbort):
        buffer.callback(output, 100, None, None)
    assert not output.any()
    assert not buffer.put(np.ones(10, np.float32))


def test_full_pcm_queue_is_cancellable_without_audio_thread():
    stop = threading.Event()
    buffer = PCMBuffer(stop, capacity=1)
    buffer.put(np.ones(4, np.float32))
    result = []
    thread = threading.Thread(target=lambda: result.append(buffer.put(np.ones(4, np.float32))))
    thread.start()
    stop.set()
    thread.join(timeout=0.2)
    assert not thread.is_alive() and result == [False]


def test_text_segments_emit_sentences_without_repeating_final_text():
    segments = TextSegments(max_chars=40)
    assert not segments.feed('Olá!')
    assert segments.feed(' Como vai? ') == ['Olá!', 'Como vai?']
    assert not segments.feed('A taxa é 12.')
    assert not segments.feed('5 por cento')
    assert segments.flush() == ['A taxa é 12.5 por cento']
    assert not segments.flush()


def test_text_segments_bound_wait_for_long_unpunctuated_answer():
    segments = TextSegments(max_chars=30)
    text = 'Uma resposta sem pontuação que precisa começar a ser falada cedo'
    ready = segments.feed(text) + segments.flush()
    assert len(ready) > 1
    assert ' '.join(ready) == text


def test_interrupt_cancels_generation_and_old_queued_sentences():
    async def scenario():
        ui = SimpleNamespace(fail=lambda message: pytest.fail(message))
        speech = SpeechOut(None, ui, SimpleNamespace(tts='pocket'))
        started, stopped = threading.Event(), threading.Event()
        def generate(text, stop):
            started.set()
            stop.wait(2)
            stopped.set()
        speech._stream = generate
        task = asyncio.create_task(speech.run())
        try:
            await speech.say('Primeira frase', speech.epoch)
            await speech.say('Segunda frase', speech.epoch)
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(.002)
            assert started.is_set()
            speech.interrupt()
            await asyncio.to_thread(stopped.wait, .2)
            assert stopped.is_set()
            assert speech.queue.empty()
            assert not speech.playing
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


def test_playback_begins_before_synthesis_finishes():
    audible = threading.Event()
    generated = threading.Event()
    observed = []

    class Backend:
        rate = 24000
        def chunks(self, text, stop):
            yield np.ones(480, dtype=np.float32)
            assert audible.wait(.5), 'Playback waited for the completed sentence'
            yield np.ones(480, dtype=np.float32) * 2
            yield np.ones(480, dtype=np.float32) * 3
            generated.set()

    class Stream:
        def __init__(self, callback, finished_callback, **kwargs):
            self.callback, self.finished = callback, finished_callback
            self.closed = threading.Event()
        def start(self):
            def pump():
                while not self.closed.is_set():
                    output = np.empty((480, 1), np.float32)
                    try:
                        self.callback(output, 480, None, None)
                    except (sd.CallbackStop, sd.CallbackAbort):
                        self.finished()
                        return
                    if output.any():
                        observed.append((output.copy(), generated.is_set()))
                        audible.set()
                    self.closed.wait(.001)
            self.thread = threading.Thread(target=pump)
            self.thread.start()
        def abort(self):
            self.closed.set()
            self.finished()
        def close(self):
            self.closed.set()
            self.thread.join(timeout=.5)

    speech = SpeechOut(None, SimpleNamespace(fail=lambda message: None), SimpleNamespace(tts='pocket'))
    speech.backend = Backend()
    with patch('tts_engine.sd.OutputStream', Stream):
        speech._stream('Uma frase', threading.Event())
    assert len(observed) == 3
    assert observed[0][1] is False
    assert [float(packet[0, 0]) for packet, _ in observed] == [1, 2, 3]


def test_spoken_text_removes_markdown_emoji_and_preserves_portuguese():
    from tts_engine import spoken_text
    text = '😊\n1. **Qual é o objetivo do projeto?**\n2. *O que você já fez?*\nVeja [os detalhes](https://example.com).'
    assert spoken_text(text) == 'Qual é o objetivo do projeto? O que você já fez? Veja os detalhes.'
