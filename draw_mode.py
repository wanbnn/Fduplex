"""ASR-to-LLM-to-MCP drawing session; no speech synthesis or spoken replies."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
import webbrowser

import numpy as np
import sounddevice as sd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from asr_engine import ASREngine
from audio_frontend import Denoiser, Silero, SpeechGate, RATE, FRAME
from draw_canvas import atomic_json, initial_state, launch_viewer, read_json
from terminal_ui import TerminalUI


DRAW_SYSTEM = """Você é o MOTOR DE DIAGRAMAS do Fduplex, NÃO um chatbot.
Recebe transcrições ASR incrementais de uma pessoa descrevendo um fluxo.
Sua ÚNICA saída operacional deve ser uma chamada à ferramenta update_canvas.
Nunca responda à pessoa com texto ou voz. Não execute comandos, arquivos ou rede.
Desenhe imediatamente etapas, decisões e ligações realmente expressas na fala.
A transcrição é cumulativa e pode ser corrigida a cada atualização.
O estado atual do canvas é fornecido em JSON a cada chamada; reutilize os mesmos
IDs estáveis ao editar. Acrescente só elementos novos, altere rótulos e conexões
existentes quando houver correção. Nunca duplique nós a cada atualização.
Tipos: start, process, decision, input, note, end. Para ramificações em decisões,
rotule as conexões 'Sim' e 'Não' quando o usuário disser isso.
Não crie etapas que a pessoa não disse. Se ainda não houver informação suficiente,
não chame a ferramenta. Preserve o diagrama anterior ao receber pausas."""


class DrawAgent:
    """Dedicated stdio MCP client with only the safe drawing tool exposed to the LLM."""
    def __init__(self, client: AsyncOpenAI, model: str, state_path: Path):
        self.client, self.model, self.state_path = client, model, state_path
        self.stack = AsyncExitStack()
        self.session = None
        self.tools = []

    async def start(self, errlog=None):
        env = dict(os.environ, FDUPLEX_DRAW_STATE=str(self.state_path))
        params = StdioServerParameters(
            command=sys.executable, args=[str(Path(__file__).resolve().parent / "draw_canvas.py")],
            env=env, cwd=str(self.state_path.parent))
        streams = await self.stack.enter_async_context(stdio_client(params, errlog=errlog or sys.stderr))
        self.session = await self.stack.enter_async_context(ClientSession(*streams))
        await self.session.initialize()
        definitions = await self.session.list_tools()
        self.tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description or "",
             "parameters": t.inputSchema}} for t in definitions.tools if t.name == "update_canvas"
        ]
        if len(self.tools) != 1:
            raise RuntimeError("Servidor MCP do desenho não oferece update_canvas")

    async def close(self):
        await self.stack.aclose()

    async def draw(self, transcript: str) -> int:
        current = read_json(self.state_path, initial_state())
        prompt = json.dumps({"transcricao_cumulativa": transcript[-15000:],
                             "canvas_atual": current}, ensure_ascii=False)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": DRAW_SYSTEM},
                      {"role": "user", "content": prompt}],
            tools=self.tools, tool_choice="auto", temperature=0.1)
        calls = response.choices[0].message.tool_calls or []
        if not calls:
            return 0
        applied = 0
        for call in calls:
            if call.function.name != "update_canvas":
                raise ValueError("Ferramenta não autorizada: " + call.function.name)
            arguments = json.loads(call.function.arguments or "{}")
            result = await self.session.call_tool("update_canvas", arguments)
            if result.isError:
                detail = " ".join(c.text for c in result.content if c.type == "text")
                raise RuntimeError("MCP canvas: " + detail[:600])
            applied += 1
        return applied


class DrawAssistant:
    def __init__(self, args, microphone_cls):
        self.args = args
        self.microphone_cls = microphone_cls
        self.ui = TerminalUI(args.plain)
        self.client = AsyncOpenAI(base_url=args.url, api_key=args.api_key, timeout=90)
        folder = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "fduplex" / "canvases"
        folder.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.state_path = folder / (name + ".json")
        self.transcript_path = folder / (name + ".transcript.json")
        atomic_json(self.state_path, initial_state())
        self.agent = DrawAgent(self.client, args.model, self.state_path)
        self.queue = asyncio.Queue(maxsize=96)
        self.events = asyncio.Event()
        self.final_texts = []
        self.active_id = -1
        self.current_committed = ""
        self.current_partial = ""
        self.latest = ""
        self.status = "Preparando ASR…"
        self.quit = asyncio.Event()
        self.mic = None
        self.engine = None
        self.server = None
        self.failed_ids = set()

    @staticmethod
    def clean(text):
        text = re.sub(r"<lang-[^>]+>", "", text or "").strip()
        return "" if text.lower() in ("(empty)", "[blank_audio]", "[silence]") else text

    def key(self, char):
        if char.lower() == "q":
            self.quit.set()
        elif char.lower() == "m" and self.mic:
            self.mic.set_muted(not self.mic.muted)

    def barge_in(self, utterance_id=None):
        # Unlike voice mode, a new utterance must not cancel an in-flight drawing.
        if utterance_id is not None:
            self.active_id = utterance_id
            self.current_committed = self.current_partial = ""
            self.ui.committed = self.ui.partial = ""
            self.failed_ids.intersection_update({utterance_id})
            self.publish("Ouvindo e construindo…")

    def queue_audio(self, uid, event, samples):
        if self.queue.full():
            self.failed_ids.add(uid)
            self.ui.fail("ASR atrasado: segmento descartado")
            return
        self.queue.put_nowait((uid, event, samples))

    def publish(self, status: str | None = None):
        if status:
            self.status = status
        self.ui.status = self.status
        preview = " ".join(self.final_texts + [self.current_committed, self.current_partial]).strip()
        atomic_json(self.transcript_path, {"text": preview[-16000:], "status": self.status})
        confirmed = " ".join(self.final_texts + [self.current_committed]).strip()
        if confirmed and confirmed != self.latest:
            self.latest = confirmed
            self.events.set()

    def finalize(self, text: str):
        self.current_committed = self.current_partial = ""
        if text:
            self.final_texts.append(text)
            self.ui.message("Você", text)
            self.ui.committed = text
            self.ui.partial = ""
        self.publish("Fala registrada • desenho em atualização")

    async def asr_worker(self):
        current = None
        collected = []
        while True:
            uid, event, samples = await self.queue.get()
            self.ui.backlog = self.queue.qsize()
            if uid != self.active_id:
                continue
            if event == "start":
                current, collected = uid, []
            if uid != current:
                continue
            if uid in self.failed_ids or event == "discard":
                if event in ("end", "discard"):
                    self.failed_ids.discard(uid)
                    current = None
                    self.current_committed = self.current_partial = ""
                    self.publish("Trecho descartado • ouvindo…")
                continue
            if self.engine.streaming:
                if self.args.verify_final:
                    collected.append(samples)
                if event == "start":
                    await self.engine.request("B")
                result = await self.engine.request("A", samples)
                self.current_committed = self.clean(result.get("committed"))
                self.current_partial = self.clean(result.get("tentative"))
                self.ui.committed, self.ui.partial = self.current_committed, self.current_partial
                self.publish("ASR ao vivo • LLM desenhando…")
                if event != "end":
                    continue
                result = await self.engine.request("F")
                text = self.clean(result.get("text"))
                if self.args.verify_final:
                    self.publish("Conferindo transcrição…")
                    result = await self.engine.request("R", np.concatenate(collected))
                    text = self.clean(result.get("text"))
            else:
                collected.append(samples)
                if event != "end":
                    continue
                self.publish("Transcrevendo frase…")
                result = await self.engine.request("R", np.concatenate(collected))
                text = self.clean(result.get("text"))
            current, collected = None, []
            if uid == self.active_id:
                self.finalize(text)

    async def draw_worker(self):
        last_sent, last_time = "", 0.0
        while True:
            await self.events.wait()
            remaining = self.args.draw_interval - (time.monotonic() - last_time)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self.events.clear()
            transcript = self.latest
            if not transcript or transcript == last_sent:
                continue
            last_sent, last_time = transcript, time.monotonic()
            try:
                self.publish("LLM atualizando o canvas…")
                operations = await self.agent.draw(transcript)
                if operations:
                    self.publish("Fluxograma atualizado • ouvindo…")
                else:
                    self.publish("LLM aguardando detalhes do fluxo…")
            except Exception as exc:
                self.ui.fail("Desenho / LLM: " + str(exc))
                self.publish("Erro na atualização; próxima fala tentará novamente")

    async def run(self):
        loop = asyncio.get_running_loop()
        log_dir = self.state_path.parent.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        self.ui.log_path = str(log_dir / "diagnostics.log")
        log = open(self.ui.log_path, "a", buffering=1)
        tasks = []
        denoiser = None
        try:
            self.ui.start_keys(self.key)
            tasks.append(asyncio.create_task(self.ui.run()))
            self.server, url = launch_viewer(self.state_path, self.transcript_path, self.args.draw_port)
            self.ui.status = "Canvas: " + url
            if not self.args.no_draw_browser:
                webbrowser.open(url, new=2)
            self.ui.mode = "DRAW • sem TTS"
            denoiser = Denoiser(self.args.denoise, self.args.noise_suppress_db)
            vad = await asyncio.to_thread(Silero, self.args.vad_model)
            gate = SpeechGate(vad, denoiser, self.args.threshold, self.args.speech_probability,
                              self.args.start_ms, self.args.min_speech_ms, self.args.pause_ms)
            self.ui.gate, self.ui.denoiser = gate, denoiser.name
            self.engine = ASREngine(self.args, log)
            await self.engine.start()
            self.ui.mode = "DRAW • streaming" if self.engine.streaming else "DRAW • por frase"
            self.ui.backend = self.engine.backend
            self.ui.model = Path(self.args.gguf).stem
            await self.agent.start(errlog=log)
            self.publish("Canvas em " + url + " • fale para desenhar")
            print("Fduplex Draw | " + url + " | salvo em " + str(self.state_path), flush=True)
            self.mic = self.microphone_cls(loop, gate, self.queue_audio, self.barge_in, self.ui)
            with sd.InputStream(samplerate=RATE, blocksize=FRAME, channels=1,
                                dtype="float32", device=self.args.mic, callback=self.mic.callback):
                tasks += [asyncio.create_task(self.mic.run()), asyncio.create_task(self.asr_worker()),
                          asyncio.create_task(self.draw_worker()), asyncio.create_task(self.quit.wait())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        finally:
            if self.mic:
                self.mic.closed = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.ui.close()
            try:
                await self.agent.close()
            finally:
                if self.engine:
                    await self.engine.close()
                if denoiser:
                    denoiser.close()
                await self.client.close()
                if self.server:
                    await asyncio.to_thread(self.server.shutdown)
                    self.server.server_close()
                log.close()
