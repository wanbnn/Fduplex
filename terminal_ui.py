"""One terminal surface; partial text never becomes conversation history."""
import asyncio
from collections import deque
import os
import sys
import termios
import tty

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from tts_engine import spoken_text


class TerminalUI:
    def __init__(self, plain=False):
        self.console = Console()
        self.plain = plain or not self.console.is_terminal or os.getenv('TERM') == 'dumb'
        self.messages = deque(maxlen=12)
        self.status = 'Preparando áudio e modelo…'
        self.mode = 'carregando'
        self.backend = ''
        self.model = ''
        self.partial = self.committed = self.answer = ''
        self.error = ''
        self.gate = None
        self.muted = False
        self.denoiser = ''
        self.tts = ''
        self.backlog = 0
        self.key_handler = None
        self.log_path = ''
        self._terminal = None

    def message(self, role, text):
        if not text.strip():
            return
        if role == 'IA':
            text = spoken_text(text)
        self.messages.append((role, text))
        if self.plain:
            self.console.print(Text(f'{role}: {text}'))

    def fail(self, message):
        self.error = str(message)
        if self.plain:
            self.console.print(Text(f'Erro: {message}', style='red'))

    def render(self):
        header = Text('FDUPLEX  ', style='bold cyan')
        header.append(f'{self.mode} • {self.backend} • {self.model}', style='dim')
        header.no_wrap, header.overflow = True, 'ellipsis'
        state = Text('MIC PAUSADO' if self.muted else self.status, style='yellow' if self.muted else 'green')
        if self.gate:
            level = min(20, int(self.gate.level / max(self.gate.gate_level, 1) * 5))
            state.append('\nVolume ', style='dim')
            state.append('▰' * level + '▱' * (20 - level), style='cyan')
            state.append(f'  voz {self.gate.p:.0%}  piso {self.gate.noise_floor:.0f}  limiar {self.gate.gate_level:.0f}', style='dim')
        state.append(f'\n{self.denoiser} • Silero VAD • {self.tts or 'voz carregando'}', style='dim')
        transcript = Text(self.committed, style='white')
        transcript.append(self.partial, style='italic dim')
        if not transcript.plain:
            transcript = Text('Sua fala aparecerá aqui…', style='dim')
        chat = Text()
        # Fit conversation to the terminal; keep the most recent messages visible.
        compact = self.console.size.height < 22
        chat_height = max(3, self.console.size.height - (10 if compact else 12) - bool(self.error))
        budget = max(1, chat_height - 3 - (3 if self.answer else 0))
        width = max(20, self.console.size.width - 6)
        recent = Text()
        for role, body in reversed(self.messages):
            item = Text(f'{role}  ', style='bold cyan' if role == 'Você' else 'bold magenta')
            item.append(body + '\n', style='white')
            lines = item.wrap(self.console, width)
            if len(lines) > budget:
                lines = lines[-budget:]
            recent = Text('\n').join(lines) + Text('\n') + recent
            budget -= len(lines)
            if budget <= 0:
                break
        chat.append_text(recent)
        if self.answer:
            chat.append('IA  ', style='bold magenta')
            chat.append(spoken_text(self.answer)[-max(80, width * 3):])
        if not chat.plain:
            chat.append('Pronto para conversar. Use fones de ouvido.', style='dim')
        if compact:
            state = Text(('MIC PAUSADO' if self.muted else self.status) + f' • {self.mode}', overflow='ellipsis', no_wrap=True)
        transcript_lines = transcript.wrap(self.console, max(10, self.console.size.width - 4))
        transcript = Text('\n').join(transcript_lines[-(2 if compact else 3):])
        items = [header, state if compact else Panel(state, title='Áudio', border_style='green', height=5),
                 Panel(transcript, title='Transcrição ao vivo' if self.mode == 'streaming nativo' else 'Transcrição por frase', border_style='cyan', height=4 if compact else 5),
                 Panel(chat, title='Conversa', border_style='blue', height=chat_height)]
        if self.error:
            items.append(Text(self.error[:300], style='bold red', no_wrap=True, overflow='ellipsis'))
        controls = 'M mic • Esc parar • Q sair' if self.console.size.width < 80 else 'M pausar microfone  •  Esc interromper resposta  •  Q sair  •  Ctrl+C sair'
        items.append(Text(controls, style='dim', no_wrap=True, overflow='ellipsis'))
        return Group(*items)

    async def run(self):
        if self.plain:
            while True:
                await asyncio.sleep(1)
        with Live(self.render(), console=self.console, screen=True, auto_refresh=False,
                  redirect_stdout=False, redirect_stderr=False) as live:
            while True:
                live.update(self.render(), refresh=True)
                await asyncio.sleep(0.12)

    def start_keys(self, handler):
        if self.plain or not sys.stdin.isatty():
            return
        self.key_handler = handler
        self._terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        asyncio.get_running_loop().add_reader(sys.stdin, self._read_key)

    def _read_key(self):
        chars = os.read(sys.stdin.fileno(), 32).decode(errors='ignore')
        for char in chars:
            self.key_handler(char)

    def close(self):
        if self._terminal is not None:
            asyncio.get_running_loop().remove_reader(sys.stdin)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal)
            self._terminal = None
