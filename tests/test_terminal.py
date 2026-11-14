from io import StringIO
from rich.console import Console

from terminal_ui import TerminalUI


def test_live_partial_stays_out_of_conversation_and_fits_normal_terminal():
    ui = TerminalUI()
    ui.mode = 'streaming nativo'
    ui.committed, ui.partial = 'Bom dia, ', 'como vai?'
    ui.message('Você', 'Mensagem anterior')
    ui.answer = 'Resposta atual'
    out = StringIO()
    console = Console(file=out, width=100, height=36, force_terminal=False)
    ui.console = console
    console.print(ui.render())
    rendered = out.getvalue()
    assert 'Bom dia, como vai?' in rendered
    assert 'Resposta atual' in rendered
    assert 'M pausar microfone' in rendered
    assert list(ui.messages) == [('Você', 'Mensagem anterior')]
    assert len(rendered.splitlines()) <= 36


def test_plain_mode_does_not_emit_ansi():
    ui = TerminalUI(plain=True)
    output = StringIO()
    ui.console = Console(file=output, force_terminal=False)
    ui.message('Você', '[texto literal]')
    assert '\x1b' not in output.getvalue()
    assert '[texto literal]' in output.getvalue()


def test_small_terminal_keeps_controls_visible_with_long_messages():
    for width, height in [(80, 24), (60, 18), (40, 15)]:
        ui = TerminalUI()
        ui.messages.extend([('Você', 'Mensagem longa. ' * 80), ('IA', 'Resposta longa. ' * 80)] * 5)
        ui.partial = 'Transcrição em andamento. ' * 30
        ui.answer = 'Resposta atual. ' * 80
        output = StringIO()
        ui.console = Console(file=output, width=width, height=height)
        ui.console.print(ui.render())
        lines = output.getvalue().splitlines()
        assert len(lines) <= height
        assert 'Q sair' in lines[-1]


def test_ai_messages_and_live_preview_remove_speech_unfriendly_formatting():
    ui = TerminalUI(plain=True)
    output = StringIO()
    ui.console = Console(file=output, width=100, height=36)
    ui.message('IA', '**Resposta:**\n- Primeiro item.\n- Segundo item. 😊')
    assert ui.messages[-1] == ('IA', 'Resposta: Primeiro item. Segundo item.')
    ui.answer = '1. **Outra resposta.** 😊'
    ui.console.print(ui.render())
    text = output.getvalue()
    assert '**' not in text and '😊' not in text and '1.' not in text
