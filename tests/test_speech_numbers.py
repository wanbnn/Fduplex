import asyncio
from types import SimpleNamespace

import pytest

from speech_numbers import portuguese_numbers
from tts_engine import SpeechOut


@pytest.mark.parametrize(('text', 'expected'), [
    ('O ping foi 42 ms.', 'O ping foi quarenta e dois milissegundos.'),
    ('Demorou 1 ms.', 'Demorou um milissegundo.'),
    ('Perda de 2,5%.', 'Perda de dois vírgula cinco por cento.'),
    ('Latência de 12.05 ms.', 'Latência de doze vírgula zero cinco milissegundos.'),
    ('Em 2026 temos 1.234 itens.', 'Em dois mil e vinte e seis temos mil, duzentos e trinta e quatro itens.'),
    ('Custa R$ 12,50.', 'Custa doze reais e cinquenta centavos.'),
    ('Custa R$ 1.234,56.', 'Custa mil, duzentos e trinta e quatro reais e cinquenta e seis centavos.'),
    ('R$ 1,00', 'um real'),
    ('Saldo -5,2.', 'Saldo menos cinco vírgula dois.'),
    ('IP 8.8.8.8.', 'IP oito ponto oito ponto oito ponto oito.'),
    ('Código 007.', 'Código zero zero sete.'),
    ('Memória 16 GB.', 'Memória dezesseis gigabytes.'),
    ('1. **Resultado:** 42 ms.', '**Resultado:** quarenta e dois milissegundos.'),
    ('Boa tarde, tudo bem?', 'Boa tarde, tudo bem?'),
])
def test_portuguese_numbers(text, expected):
    assert portuguese_numbers(text) == expected


def test_only_synthesis_queue_receives_words_and_original_text_is_unchanged():
    async def scenario():
        original = 'O ping é **42 ms**, com perda de 0%.'
        speech = SpeechOut(None, SimpleNamespace(), SimpleNamespace(tts='pocket', tts_language='portuguese'))
        await speech.say(original, speech.epoch)
        text, _ = speech.queue.get_nowait()
        assert text == 'O ping é quarenta e dois milissegundos, com perda de zero por cento.'
        assert original == 'O ping é **42 ms**, com perda de 0%.'
    asyncio.run(scenario())
