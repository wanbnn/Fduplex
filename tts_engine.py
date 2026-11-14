"""Persistent CPU TTS with incremental PCM playback and cooperative cancellation."""
import asyncio
import io
import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from speech_numbers import portuguese_numbers


class PCMBuffer:
    """The audio callback only copies PCM; inference and waits stay off its thread."""
    def __init__(self, stop, capacity=24):
        self.stop = stop
        self.queue = queue.Queue(maxsize=capacity)
        self.source_done = threading.Event()
        self.drained = threading.Event()
        self.current = np.empty(0, dtype=np.float32)
        self.offset = 0

    def put(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        while not self.stop.is_set():
            try:
                self.queue.put(samples, timeout=0.04)
                return True
            except queue.Full:
                pass
        return False

    def callback(self, outdata, frames, timing, status):
        outdata.fill(0)
        if self.stop.is_set():
            raise sd.CallbackAbort
        written = 0
        while written < frames:
            if self.offset >= self.current.size:
                try:
                    self.current = self.queue.get_nowait()
                    self.offset = 0
                except queue.Empty:
                    if self.source_done.is_set():
                        raise sd.CallbackStop
                    return
            take = min(frames - written, self.current.size - self.offset)
            outdata[written:written + take, 0] = self.current[self.offset:self.offset + take]
            self.offset += take
            written += take


class PocketVoice:
    name = 'Pocket TTS • português • CPU'

    def __init__(self, language='portuguese', voice='rafael', quantize=False):
        self.language, self.voice_name, self.quantize = language, voice, quantize
        self.model = self.voice = None

    def load(self):
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        from pocket_tts import TTSModel
        self.model = TTSModel.load_model(language=self.language, quantize=self.quantize)
        self.voice = self.model.get_state_for_audio_prompt(self.voice_name)
        # Exercise the decoder once, avoiding cold-start work during the first answer.
        for _ in self.chunks('Olá.', threading.Event()):
            pass

    @property
    def rate(self):
        return self.model.sample_rate

    def chunks(self, text, stop):
        # Each call copies the cached voice state. Keep the loaded model between turns.
        for chunk in self.model.generate_audio_stream(self.voice, text, stop=stop):
            if not stop.is_set():
                yield chunk.detach().cpu().numpy().astype(np.float32, copy=False)


def spoken_text(text):
    """Remove presentation markup, retaining Portuguese letters and normal punctuation."""
    import re
    import unicodedata
    text = re.sub(r'```.*?```', ' Trecho de código disponível na tela. ', text, flags=re.S)
    text = re.sub(r'!\[([^]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)', '', text)
    text = re.sub(r'(?<!\w)[*_~`]+|[*_~`]+(?!\w)', '', text)
    text = unicodedata.normalize('NFC', text).replace('\ufe0f', '').replace('\ufe0e', '')
    text = ''.join(char for char in text if not unicodedata.category(char).startswith(('S', 'C')) or char in '\n\t')
    return ' '.join(text.split())


class SpeechOut:
    def __init__(self, loop, ui, args):
        self.loop, self.ui, self.args = loop, ui, args
        self.queue = asyncio.Queue(maxsize=32)
        self.epoch = 0
        self.playing = False
        self.stop = threading.Event()
        self.backend = None
        self.name = 'desligada' if args.tts == 'none' else 'espeak-ng'
        self.metrics = {}

    async def start(self, log):
        if self.args.tts != 'pocket':
            return
        logger = logging.getLogger('pocket_tts')
        handler = logging.StreamHandler(log)
        logger.addHandler(handler)
        logger.propagate = False
        self._log_handler = handler
        self.backend = PocketVoice(self.args.tts_language, self.args.tts_voice, self.args.tts_quantize)
        loading = asyncio.create_task(asyncio.to_thread(self.backend.load))
        try:
            await asyncio.shield(loading)
        except asyncio.CancelledError:
            await loading
            raise
        except Exception as exc:
            raise RuntimeError(f'Pocket TTS: {exc}. Instale requirements-tts.txt ou use --tts espeak.') from exc
        self.name = self.backend.name

    def interrupt(self):
        self.epoch += 1
        self.stop.set()
        # Pocket playback aborts at the next 20 ms callback, including buffered PCM.
        if self.args.tts == 'espeak':
            sd.stop()
        self.playing = False
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def say(self, text, epoch):
        if getattr(self.args, 'tts_language', 'portuguese').startswith('portuguese'):
            text = portuguese_numbers(text)
        text = spoken_text(text)
        if self.args.tts != 'none' and text and epoch == self.epoch:
            await self.queue.put((text, epoch))

    def _stream(self, text, stop):
        buffer = PCMBuffer(stop)
        started = time.monotonic()
        first_chunk = None
        samples = 0
        stream = None
        try:
            for chunk in self.backend.chunks(text, stop):
                if stop.is_set():
                    continue
                if first_chunk is None:
                    first_chunk = time.monotonic() - started
                samples += len(chunk)
                if not buffer.put(chunk):
                    continue
                if stream is None:
                    stream = sd.OutputStream(samplerate=self.backend.rate, blocksize=self.backend.rate // 50,
                                             channels=1, dtype='float32', latency='low',
                                             callback=buffer.callback, finished_callback=buffer.drained.set)
                    stream.start()
            generated = time.monotonic() - started
            buffer.source_done.set()
            if stream:
                while not buffer.drained.wait(0.04):
                    if stop.is_set():
                        stream.abort()
                        break
            self.metrics = {'first_chunk_s': first_chunk, 'elapsed_s': generated,
                            'audio_s': samples / self.backend.rate}
        finally:
            stop.set()
            buffer.source_done.set()
            if stream:
                stream.close()

    async def _espeak(self, text, epoch):
        proc = await asyncio.create_subprocess_exec(
            'espeak-ng', '-v', 'pt-br', '-s', '165', '--stdout', text,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            raw, errors = await proc.communicate()
            if proc.returncode:
                raise RuntimeError(errors.decode(errors='replace'))
            import soundfile as sf
            samples, rate = sf.read(io.BytesIO(raw), dtype='float32')
            if epoch != self.epoch:
                return
            sd.play(samples, rate, blocking=False)
            end = time.monotonic() + len(samples) / rate
            while epoch == self.epoch and time.monotonic() < end:
                await asyncio.sleep(0.02)
        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            sd.stop()

    async def run(self):
        while True:
            text, epoch = await self.queue.get()
            if epoch != self.epoch:
                continue
            self.stop = threading.Event()
            self.playing = True
            try:
                if self.args.tts == 'pocket':
                    worker = asyncio.create_task(asyncio.to_thread(self._stream, text, self.stop))
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        self.stop.set()
                        await worker
                        raise
                elif self.args.tts == 'espeak':
                    await self._espeak(text, epoch)
            except Exception as exc:
                self.ui.fail('TTS: ' + str(exc))
            finally:
                self.playing = False

    def close(self):
        self.interrupt()
        if getattr(self, '_log_handler', None):
            logging.getLogger('pocket_tts').removeHandler(self._log_handler)
            self._log_handler = None
        self.backend = None


class TextSegments:
    """Turn LLM deltas into short phrases without splitting a word or replaying text."""
    def __init__(self, max_chars=180):
        self.pending = ''
        self.max_chars = max_chars

    def feed(self, delta):
        import re
        self.pending += delta
        ready = []
        while True:
            boundary = re.search(r'[.!?;]\s', self.pending)
            cut = boundary.end() if boundary else 0
            if not cut and len(self.pending) >= self.max_chars:
                cut = self.pending.rfind(' ', 0, self.max_chars + 1)
            if cut <= 0:
                break
            text, self.pending = self.pending[:cut].strip(), self.pending[cut:].lstrip()
            if text:
                ready.append(text)
        return ready

    def flush(self):
        text, self.pending = self.pending.strip(), ''
        return [text] if text else []
