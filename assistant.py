#!/usr/bin/env python3
"""Full-duplex voice assistant with persistent ASR, neural VAD and a terminal UI."""
import argparse
import asyncio
import os
from pathlib import Path
import re

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI

from asr_config import resolve_asr_paths
from asr_engine import ASREngine
from audio_frontend import Denoiser, Silero, SpeechGate, RATE, FRAME, FRAME_MS
from conversation import VOICE_SYSTEM, TURN_STYLE
from workspace import session_workspace
from terminal_ui import TerminalUI
from tts_engine import SpeechOut, TextSegments

try:
    from fduplex_agent import ToolAgent
except ModuleNotFoundError as exc:
    if exc.name != 'fduplex_agent':
        raise
    ToolAgent = None


class Microphone:
    def __init__(self, loop, gate, on_audio, on_start, ui):
        self.loop, self.gate, self.on_audio, self.on_start, self.ui = loop, gate, on_audio, on_start, ui
        self.queue = asyncio.Queue(maxsize=64)
        self.utterance_id = 0
        self.muted = False
        self.finalize_on_mute = False
        self.pending = []
        self.resync = False
        self.closed = False
        self.warning = None

    def callback(self, indata, frames, timing, status):
        if self.closed:
            return
        try:
            self.loop.call_soon_threadsafe(self._enqueue, indata[:, 0].copy(), str(status) if status else '')
        except RuntimeError:
            pass

    def _enqueue(self, packet, warning=''):
        if self.closed:
            return
        if warning:
            self.warning = 'Captura de áudio: ' + warning
            self.resync = True
        if self.queue.full():
            # Never splice across a dropped packet and present it as continuous speech.
            self.queue.get_nowait()
            self.resync = True
            self.warning = 'Captura atrasada; segmento incompleto descartado.'
        self.queue.put_nowait(packet)

    def set_muted(self, value):
        self.muted = value
        self.finalize_on_mute |= value
        self.resync = True
        self.ui.muted = value

    async def run(self):
        buffer = np.empty(0, dtype=np.float32)
        while True:
            packet = await self.queue.get()
            if self.warning:
                self.ui.fail(self.warning)
                self.warning = None
            if self.resync:
                if self.gate.active:
                    if self.finalize_on_mute:
                        # Finish already captured speech; muting only stops future input.
                        tail = np.concatenate(self.pending) if self.pending else np.zeros(FRAME, dtype=np.float32)
                        self.on_audio(self.utterance_id, 'end', tail)
                    else:
                        self.on_audio(self.utterance_id, 'discard', None)
                self.finalize_on_mute = False
                self.gate.reset()
                self.pending.clear()
                buffer = np.empty(0, dtype=np.float32)
                self.resync = False
            if self.muted:
                continue
            buffer = np.concatenate((buffer, packet))
            while buffer.size >= FRAME:
                frame, buffer = buffer[:FRAME], buffer[FRAME:]
                inference = asyncio.create_task(asyncio.to_thread(self.gate.process, frame))
                try:
                    clean, event = await asyncio.shield(inference)
                except asyncio.CancelledError:
                    await inference
                    raise
                if self.muted or self.resync:
                    continue
                if event == 'start':
                    self.utterance_id += 1
                    self.on_start(self.utterance_id)
                    self.on_audio(self.utterance_id, 'start', clean)
                    self.pending.clear()
                elif event in ('audio', 'end', 'discard'):
                    self.pending.append(clean)
                    if len(self.pending) >= 10 or event != 'audio':
                        self.on_audio(self.utterance_id, event, np.concatenate(self.pending))
                        self.pending.clear()


class Assistant:
    def __init__(self, args):
        self.args = args
        self.ui = TerminalUI(args.plain)
        self.client = AsyncOpenAI(base_url=args.url, api_key=args.api_key)
        self.speech = SpeechOut(None, self.ui, args)
        self.audio_queue = asyncio.Queue(maxsize=96)
        self.live_id = -1
        self.epoch = 0
        self.chat_task = None
        args.system += f'\nO diretório de trabalho desta sessão é {session_workspace()}. Caminhos relativos das ferramentas são relativos a esse diretório.'
        self.history = [{'role': 'system', 'content': args.system}]
        self.agent = ToolAgent(self.client, args.model, args.system, threshold=args.compact_at) if args.mcp else None
        self.mic = self.engine = None
        self.quit = asyncio.Event()
        self.failed_ids = set()

    def queue_audio(self, utterance_id, event, samples):
        if self.audio_queue.full():
            self.failed_ids.add(utterance_id)
            self.ui.fail('ASR não acompanhou a captura. Segmento descartado; tente uma frase mais curta.')
            return
        self.audio_queue.put_nowait((utterance_id, event, samples))

    def barge_in(self, utterance_id=None):
        self.epoch += 1
        if utterance_id is not None:
            self.live_id = utterance_id
            self.failed_ids.intersection_update({utterance_id})
            self.ui.partial = self.ui.committed = ''
        self.speech.interrupt()
        self.ui.answer = ''
        if self.chat_task and not self.chat_task.done():
            self.chat_task.cancel()
        self.ui.status = 'Fala confirmada • ouvindo…' if utterance_id is not None else 'Resposta interrompida'

    def key(self, char):
        if char.lower() == 'q':
            self.quit.set()
        elif char.lower() == 'm' and self.mic:
            self.mic.set_muted(not self.mic.muted)
        elif char == '\x1b':
            self.barge_in()

    @staticmethod
    def clean_text(text):
        text = re.sub(r'<lang-[^>]+>', '', text or '').strip()
        return '' if text.lower() in ('(empty)', '[blank_audio]', '[silence]') else text

    async def asr_worker(self):
        current = None
        collected = []
        while True:
            utterance_id, event, samples = await self.audio_queue.get()
            self.ui.backlog = self.audio_queue.qsize()
            if utterance_id != self.live_id:
                continue
            if event == 'start':
                current = utterance_id
                collected = []
            if utterance_id != current:
                continue
            if utterance_id in self.failed_ids or event == 'discard':
                self.ui.status = 'Segmento descartado • aguardando fala'
                if event in ('end', 'discard'):
                    self.failed_ids.discard(utterance_id)
                    current = None
                continue
            try:
                if self.engine.streaming:
                    if self.args.verify_final:
                        collected.append(samples)
                    if event == 'start':
                        await self.engine.request('B')
                    result = await self.engine.request('A', samples)
                    if utterance_id == self.live_id:
                        self.ui.committed = self.clean_text(result.get('committed'))
                        self.ui.partial = result.get('tentative', '')
                    if event != 'end':
                        continue
                    result = await self.engine.request('F')
                    text = self.clean_text(result.get('text'))
                    if self.args.verify_final and utterance_id == self.live_id:
                        self.ui.status = 'Conferindo transcrição final…'
                        result = await self.engine.request('R', np.concatenate(collected))
                        text = self.clean_text(result.get('text'))
                else:
                    collected.append(samples)
                    if event != 'end':
                        continue
                    self.ui.status = 'Transcrevendo frase…'
                    result = await self.engine.request('R', np.concatenate(collected))
                    text = self.clean_text(result.get('text'))
                current = None
                collected = []
                if utterance_id != self.live_id:
                    continue
                self.ui.committed, self.ui.partial = text, ''
                if text:
                    self.ui.message('Você', text)
                    if self.chat_task and not self.chat_task.done():
                        self.chat_task.cancel()
                    self.epoch += 1
                    self.chat_task = asyncio.create_task(self.reply(text, self.epoch))
                else:
                    self.ui.status = 'Aguardando fala'
            except Exception as exc:
                self.ui.fail(str(exc))
                current = None
                collected = []
                # A broken protocol must not consume a delayed reply as the next result.
                raise

    async def reply(self, text, epoch):
        self.ui.status = 'IA respondendo • pode falar para interromper'
        self.ui.answer = ''
        answer = []
        segments = TextSegments()

        async def speak_delta(delta):
            if epoch != self.epoch:
                return
            ready = segments.flush() if delta is None else segments.feed(delta)
            for part in ready:
                await self.speech.say(part, self.speech.epoch)

        try:
            if self.agent:
                async def emit(body):
                    if epoch != self.epoch:
                        return
                    answer.append(body)
                    self.ui.answer = body
                def preview(body):
                    if epoch == self.epoch:
                        self.ui.answer = body
                def on_tool(name, arguments):
                    self.ui.status = f'Executando {name}…'
                await self.agent.ask(text, emit, on_tool=on_tool, interrupt=lambda: epoch != self.epoch, preview=preview, stream_emit=speak_delta)
            else:
                self.history.append({'role': 'user', 'content': text})
                messages = [self.history[0]] + self.history[1:][-self.args.history:]
                stream = await self.client.chat.completions.create(
                    model=self.args.model, messages=messages + [{'role': 'system', 'content': TURN_STYLE}], stream=True, temperature=0.6)
                try:
                    async for part in stream:
                        if epoch != self.epoch:
                            return
                        if not part.choices:
                            continue
                        token = part.choices[0].delta.content or ''
                        answer.append(token)
                        self.ui.answer += token
                        await speak_delta(token)
                finally:
                    await stream.close()
                await speak_delta(None)
                if epoch == self.epoch:
                    self.history.append({'role': 'assistant', 'content': ''.join(answer)})
            if epoch == self.epoch:
                self.ui.message('IA', ''.join(answer))
                self.ui.answer = ''
                self.ui.status = 'Aguardando fala • microfone aberto'
        except asyncio.CancelledError:
            if answer and not self.agent:
                self.history.append({'role': 'assistant', 'content': ''.join(answer)})
            if answer:
                self.ui.message('IA', ''.join(answer) + ' [interrompida]')
            raise
        except Exception as exc:
            self.ui.fail('LLM: ' + str(exc))
        finally:
            if epoch == self.epoch:
                self.ui.answer = ''

    async def run(self):
        loop = asyncio.get_running_loop()
        self.speech.loop = loop
        log_dir = Path(os.getenv('XDG_STATE_HOME', str(Path.home() / '.local/state'))) / 'fduplex'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.ui.log_path = str(log_dir / 'diagnostics.log')
        tasks = []
        denoiser = None
        log_path = Path(self.ui.log_path)
        if log_path.exists() and log_path.stat().st_size > 5_000_000:
            log_path.replace(log_path.with_suffix('.log.1'))
        log = open(log_path, 'a', buffering=1)
        try:
            self.ui.start_keys(self.key)
            tasks.append(asyncio.create_task(self.ui.run()))
            denoiser = Denoiser(self.args.denoise, self.args.noise_suppress_db)
            vad = await asyncio.to_thread(Silero, self.args.vad_model)
            gate = SpeechGate(vad, denoiser, self.args.threshold, self.args.speech_probability,
                              self.args.start_ms, self.args.min_speech_ms, self.args.pause_ms)
            self.ui.gate, self.ui.denoiser = gate, denoiser.name
            self.engine = ASREngine(self.args, log)
            await self.engine.start()
            self.ui.status = 'Carregando voz e preparando reprodução…'
            await self.speech.start(log)
            self.ui.tts = self.speech.name
            self.ui.mode = 'streaming nativo' if self.engine.streaming else 'offline • por frase'
            self.ui.backend = self.engine.backend
            self.ui.model = Path(self.args.gguf).stem
            if self.agent:
                await self.agent.start(errlog=log)
            self.mic = Microphone(loop, gate, self.queue_audio, self.barge_in, self.ui)
            self.ui.status = 'Calibrando ruído por 0,8 s • fique em silêncio'
            if self.ui.plain:
                self.ui.console.print(f'Fduplex | {self.ui.mode} | {self.engine.backend} | {denoiser.name} | Silero VAD')
            with sd.InputStream(samplerate=RATE, blocksize=FRAME, channels=1,
                                dtype='float32', device=self.args.mic, callback=self.mic.callback):
                tasks += [asyncio.create_task(self.mic.run()), asyncio.create_task(self.asr_worker()),
                          asyncio.create_task(self.speech.run()), asyncio.create_task(self.quit.wait()),
                          asyncio.create_task(self.monitor())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        finally:
            if self.mic:
                self.mic.closed = True
            if self.chat_task:
                tasks.append(self.chat_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.ui.close()
            self.speech.interrupt()
            try:
                if self.agent:
                    await self.agent.close()
            finally:
                if self.engine:
                    await self.engine.close()
                if denoiser:
                    denoiser.close()
                self.speech.close()
                await self.client.close()
                log.close()

    async def monitor(self):
        while True:
            if self.mic and self.mic.gate.frames_seen >= self.mic.gate.calibration_frames and self.ui.status.startswith('Calibrando'):
                self.ui.status = 'Aguardando fala • microfone aberto'
            self.ui.backlog = self.audio_queue.qsize()
            await asyncio.sleep(0.1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', default=os.getenv('TRANSCRIBE_CLI'), help='CLI legado: usado para localizar o checkout do worker')
    p.add_argument('--asr-worker', default=os.getenv('FDUPLEX_ASR_WORKER'))
    p.add_argument('--gguf', default=os.getenv('PARAKEET_GGUF'))
    p.add_argument('--asr-mode', choices=['auto', 'streaming', 'offline'], default=os.getenv('ASR_MODE', 'auto'))
    p.add_argument('--backend', choices=['auto', 'rocm', 'cpu', 'vulkan', 'cuda', 'metal'], default=os.getenv('ASR_BACKEND', 'rocm'))
    p.add_argument('--language', default=os.getenv('ASR_LANGUAGE', 'pt-BR'))
    p.add_argument('--stream-right', type=int, choices=[-1, 0, 3, 6, 13], default=3, help='Lookahead Nemotron em frames de 80 ms')
    p.add_argument('--verify-final', action=argparse.BooleanOptionalAction, default=True, help='Conferir a frase completa após finalizar o stream, reutilizando o modelo carregado')
    p.add_argument('--vad-model', default=os.getenv('FDUPLEX_VAD_MODEL'))
    p.add_argument('--speech-probability', type=float, default=0.65)
    p.add_argument('--start-ms', type=int, default=192)
    p.add_argument('--min-speech-ms', type=int, default=256)
    p.add_argument('--threshold', type=float, default=100, help='Piso RMS mínimo; limiar adaptativo também é aplicado')
    p.add_argument('--pause-ms', type=int, default=640)
    p.add_argument('--denoise', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--noise-suppress-db', type=int, default=-25)
    p.add_argument('--tts', choices=['pocket', 'espeak', 'none'], default=os.getenv('TTS_BACKEND', 'pocket'))
    p.add_argument('--tts-language', default=os.getenv('TTS_LANGUAGE', 'portuguese'))
    p.add_argument('--tts-voice', default=os.getenv('TTS_VOICE', 'rafael'))
    p.add_argument('--tts-quantize', action=argparse.BooleanOptionalAction, default=False, help='Quantização int8 do Pocket na CPU')
    p.add_argument('--plain', action='store_true', help='Saída simples, sem tela interativa')
    p.add_argument('--list-devices', action='store_true')
    p.add_argument('--mic', type=int)
    p.add_argument('--url', default=os.getenv('OPENAI_BASE_URL', 'http://localhost:1234/v1'))
    p.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY', 'local'))
    p.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'local-model'))
    p.add_argument('--history', type=int, default=24)
    p.add_argument('--mcp', action=argparse.BooleanOptionalAction, default=ToolAgent is not None)
    p.add_argument('--compact-at', type=int, default=80000)
    p.add_argument('--system', default=VOICE_SYSTEM)
    args = p.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        p.exit()
    if args.mcp and ToolAgent is None:
        p.error('MCP não instalado nesta cópia. Use a instalação completa ou --no-mcp.')
    if not 0 < args.speech_probability < 1 or args.start_ms < 64 or args.min_speech_ms < args.start_ms or args.pause_ms < 64:
        p.error('VAD: probabilidade deve estar entre 0 e 1; início >=64 ms, duração mínima >= início, pausa >=64 ms.')
    if args.threshold < 0 or not -60 <= args.noise_suppress_db <= 0:
        p.error('RMS deve ser >=0 e supressão deve estar entre -60 e 0 dB.')
    try:
        resolve_asr_paths(args)
    except FileNotFoundError as exc:
        p.error(str(exc))
    return args


if __name__ == '__main__':
    try:
        asyncio.run(Assistant(parse_args()).run())
    except KeyboardInterrupt:
        print('Encerrado.')
    except Exception as exc:
        print(f'Falha: {exc}')
        raise SystemExit(1)
