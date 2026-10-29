#!/usr/bin/env python3
"""Parakeet Ultra + ROCm ASR + OpenAI-compatible LLM + interruptible local TTS.
Full duplex at the application layer: capture continues during ASR, LLM and TTS.
Parakeet Ultra itself is an offline, non-streaming recognizer.
"""
import argparse
import asyncio
import collections
import io
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd
import webrtcvad
from openai import AsyncOpenAI
from fduplex_agent import ToolAgent

RATE = 16000
FRAME_MS = 20
FRAME = RATE * FRAME_MS // 1000


def wav_write(path, samples):
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())


class Microphone:
    def __init__(self, loop, on_utterance, on_partial, interrupt, vad_mode=2, threshold=180, pause_ms=520, partial_ms=1100):
        self.loop = loop
        self.on_utterance = on_utterance
        self.on_partial = on_partial
        self.partial_ms = partial_ms
        self.last_partial = 0.0
        self.utterance_id = 0
        self.interrupt = interrupt
        self.vad = webrtcvad.Vad(vad_mode)
        self.threshold = threshold
        self.pause_frames = max(1, pause_ms // FRAME_MS)
        self.prefix = collections.deque(maxlen=12)
        self.frames = []
        self.active = False
        self.silent = 0
        self.consecutive = 0
        self.pre = []
        self.mic_queue = asyncio.Queue(maxsize=250)

    def callback(self, indata, frames, timing, status):
        # Audio callback must never block or run inference.
        if status:
            pass
        packet = indata[:, 0].copy()
        try:
            self.loop.call_soon_threadsafe(self._enqueue, packet)
        except RuntimeError:
            pass

    def _enqueue(self, packet):
        if self.mic_queue.full():
            try: self.mic_queue.get_nowait()
            except asyncio.QueueEmpty: pass
        self.mic_queue.put_nowait(packet)

    async def run(self):
        buffer = np.empty(0, dtype=np.float32)
        while True:
            packet = await self.mic_queue.get()
            buffer = np.concatenate((buffer, packet))
            while buffer.size >= FRAME:
                frame, buffer = buffer[:FRAME], buffer[FRAME:]
                pcm = (np.clip(frame, -1, 1) * 32767).astype('<i2')
                voiced = self.vad.is_speech(pcm.tobytes(), RATE)
                loud = np.sqrt(float(np.mean(frame ** 2))) * 32767 >= self.threshold
                voice = voiced and loud
                if not self.active:
                    self.prefix.append(frame.copy())
                    self.consecutive = self.consecutive + 1 if voice else 0
                    if self.consecutive >= 3:
                        self.active = True
                        self.frames = list(self.prefix)
                        self.utterance_id += 1
                        self.last_partial = time.monotonic()
                        self.silent = 0
                        self.interrupt()
                else:
                    self.frames.append(frame.copy())
                    now = time.monotonic()
                    if (now - self.last_partial) * 1000 >= self.partial_ms and len(self.frames) >= 35:
                        self.last_partial = now
                        # Partial ASR runs on a rolling snapshot of the current utterance.
                        self.on_partial(self.utterance_id, np.concatenate(self.frames[-RATE * 14 // FRAME:]))
                    self.silent = 0 if voice else self.silent + 1
                    if self.silent >= self.pause_frames or len(self.frames) >= 1400:
                        chunk = np.concatenate(self.frames)
                        self.active = False
                        self.frames = []
                        self.prefix.clear()
                        self.consecutive = 0
                        if len(chunk) >= RATE * 0.32:
                            self.on_utterance(self.utterance_id, chunk)


class SpeechOut:
    def __init__(self, loop):
        self.loop = loop
        self.queue = asyncio.Queue()
        self.epoch = 0
        self.task = None
        self.playing = False

    def interrupt(self):
        self.epoch += 1
        sd.stop()
        self.playing = False
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except asyncio.QueueEmpty: break

    async def say(self, text, epoch):
        if text.strip():
            await self.queue.put((text.strip(), epoch))

    async def run(self):
        while True:
            text, epoch = await self.queue.get()
            if epoch != self.epoch:
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    'espeak-ng', '-v', 'pt-br', '-s', '165', '--stdout', text,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                raw, errors = await proc.communicate()
                if proc.returncode:
                    print('TTS:', errors.decode(errors='replace'))
                    continue
                import soundfile as sf
                samples, rate = sf.read(io.BytesIO(raw), dtype='float32')
                if epoch != self.epoch:
                    continue
                self.playing = True
                await asyncio.to_thread(self._play, samples, rate, epoch)
            except Exception as exc:
                print('Falha TTS:', exc)
            finally:
                self.playing = False

    def _play(self, samples, rate, epoch):
        # Blocking=False leaves a callback active; sd.stop() interrupts immediately.
        try:
            sd.play(samples, rate, blocking=False)
            total = len(samples) / rate
            start = time.monotonic()
            while time.monotonic() - start < total:
                if epoch != self.epoch:
                    sd.stop()
                    break
                time.sleep(0.025)
            sd.stop()
        except Exception as exc:
            print('Áudio de saída:', exc)


class Assistant:
    def __init__(self, args):
        self.args = args
        self.loop = None
        self.client = AsyncOpenAI(base_url=args.url, api_key=args.api_key)
        self.speech = SpeechOut(self.loop)
        self.asr_queue = asyncio.Queue(maxsize=8)
        self.partial_queue = asyncio.Queue(maxsize=1)
        self.asr_lock = asyncio.Lock()
        self.live_id = -1
        self.chat_task = None
        self.epoch = 0
        self.history = [{'role': 'system', 'content': args.system}]
        self.agent = ToolAgent(self.client, args.model, args.system, threshold=args.compact_at) if args.mcp else None
        self.mic = None

    def queue_partial(self, utterance_id, samples):
        # Keep newest snapshot only; older partial results are obsolete.
        if self.partial_queue.full():
            try: self.partial_queue.get_nowait()
            except asyncio.QueueEmpty: pass
        self.partial_queue.put_nowait((utterance_id, samples))

    def queue_utterance(self, utterance_id, samples):
        if self.asr_queue.full():
            try: self.asr_queue.get_nowait()
            except asyncio.QueueEmpty: pass
        self.asr_queue.put_nowait((utterance_id, samples))

    def barge_in(self):
        self.epoch += 1
        self.live_id = self.mic.utterance_id if self.mic else -1
        self.speech.interrupt()
        if self.chat_task is not None and not self.chat_task.done():
            self.chat_task.cancel()
        print('\n🎤 ouvindo...', flush=True)

    def transcribe_sync(self, samples):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'speech.wav'
            wav_write(file, samples)
            cmd = [self.args.cli, '--backend', self.args.backend,
                   '-m', self.args.gguf, '--timestamps', 'none', str(file)]
            done = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=90, check=False)
            if done.returncode:
                raise RuntimeError((done.stderr or done.stdout)[-2500:])
            # transcribe.cpp emits diagnostics and alignment information on stdout.
            # Only the single 'text:' record is an actual transcript.
            raw = re.sub(r'\x1b\[[0-9;]*m', '', done.stdout)
            match = re.search(r'\btext:\s*(.*?)(?=\s+(?:segments|words|tokens|realtime):|$)', raw, re.S)
            if not match:
                raise RuntimeError('Saída ASR sem campo text:. Trecho: ' + raw[-500:])
            return ' '.join(match.group(1).split())

    async def partial_worker(self):
        while True:
            utterance_id, samples = await self.partial_queue.get()
            if utterance_id != self.live_id:
                continue
            # Never compete with final ASR on the same ROCm device.
            async with self.asr_lock:
                if utterance_id != self.live_id or not self.asr_queue.empty():
                    continue
                try:
                    partial = await asyncio.to_thread(self.transcribe_sync, samples)
                    if partial and utterance_id == self.live_id:
                        print('\r📝 parcial: ' + partial[:200] + '          ', flush=True)
                except Exception as exc:
                    print('\nASR parcial:', exc)

    async def asr_worker(self):
        while True:
            utterance_id, samples = await self.asr_queue.get()
            self.live_id = -1  # invalidate any running partial results
            try:
                async with self.asr_lock:
                    text = await asyncio.to_thread(self.transcribe_sync, samples)
                if text:
                    print('\nVocê:', text, flush=True)
                    if self.chat_task and not self.chat_task.done():
                        self.chat_task.cancel()
                    self.epoch += 1
                    self.speech.interrupt()
                    current = self.epoch
                    self.chat_task = asyncio.create_task(self.reply(text, current))
            except Exception as exc:
                print('Erro ASR:', exc)

    async def reply(self, user_text, epoch):
        print('IA: ', end='', flush=True)
        try:
            if self.agent:
                async def emit(text):
                    if epoch != self.epoch:
                        return
                    print(text, flush=True)
                    # Sentence-level TTS; use bounded segments rather than speaking markup.
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    for sentence in sentences:
                        if epoch != self.epoch:
                            return
                        if sentence.strip():
                            await self.speech.say(sentence, self.speech.epoch)
                await self.agent.ask(user_text, emit, interrupt=lambda: epoch != self.epoch)
                return
            # Legacy chat mode without tools
            self.history.append({'role':'user', 'content':user_text})
            stream = await self.client.chat.completions.create(
                model=self.args.model, messages=self.history[-self.args.history:],
                stream=True, temperature=0.6)
            response_parts = []
            sentence = ''
            async for part in stream:
                if epoch != self.epoch:
                    return
                token = part.choices[0].delta.content or ''
                if not token:
                    continue
                print(token, end='', flush=True)
                response_parts.append(token)
                sentence += token
                if re.search(r'[.!?](?:\s|$)', sentence) and len(sentence) > 30:
                    await self.speech.say(sentence, self.speech.epoch)
                    sentence = ''
            print()
            if sentence.strip():
                await self.speech.say(sentence, self.speech.epoch)
            if epoch == self.epoch:
                self.history.append({'role':'assistant','content':''.join(response_parts)})
        except asyncio.CancelledError:
            print('\n[interrompido]')
            raise
        except Exception as exc:
            print('\nErro LLM:', exc)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.speech.loop = self.loop
        if self.agent:
            names = await self.agent.start()
            print('MCP ferramentas:', ', '.join(names))
        self.mic = Microphone(self.loop, self.queue_utterance, self.queue_partial, self.barge_in,
                              self.args.vad_mode, self.args.threshold, self.args.pause_ms,
                              self.args.partial_ms)
        print('Parakeet Ultra full-duplex | fones de ouvido recomendados | Ctrl+C sai')
        print('ASR:', self.args.gguf, '| GPU:', self.args.backend)
        with sd.InputStream(samplerate=RATE, blocksize=FRAME, channels=1,
                            dtype='float32', device=self.args.mic,
                            callback=self.mic.callback):
            await asyncio.gather(self.mic.run(), self.asr_worker(),
                                 self.partial_worker(), self.speech.run())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', default=os.getenv('TRANSCRIBE_CLI', './transcribe.cpp/build/bin/transcribe-cli'))
    p.add_argument('--gguf', default=os.getenv('PARAKEET_GGUF', './models/parakeet-ultra-Q8_0.gguf'))
    p.add_argument('--backend', default=os.getenv('ASR_BACKEND', 'rocm'))
    p.add_argument('--url', default=os.getenv('OPENAI_BASE_URL', 'http://localhost:1234/v1'))
    p.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY', 'local'))
    p.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'local-model'))
    p.add_argument('--mic', type=int, default=None)
    p.add_argument('--vad-mode', type=int, default=2, choices=[0,1,2,3])
    p.add_argument('--threshold', type=int, default=180, help='Minimum RMS for VAD start')
    p.add_argument('--pause-ms', type=int, default=520)
    p.add_argument('--partial-ms', type=int, default=1100, help='Intervalo em ms para transcrição parcial (>=700 recomendado)')
    p.add_argument('--history', type=int, default=24)
    p.add_argument('--mcp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--compact-at', type=int, default=80000)
    p.add_argument('--system', default='Você está conversando diretamente com uma pessoa por voz, em português brasileiro. Responda ao que ela disse, de forma natural, breve e sem markdown. Não descreva processos de reconhecimento de fala, arquivos de áudio ou transcrições.')
    return p.parse_args()


if __name__ == '__main__':
    try:
        asyncio.run(Assistant(parse_args()).run())
    except KeyboardInterrupt:
        print('\nEncerrado.')
