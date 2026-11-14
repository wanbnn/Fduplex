import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fduplex_agent import ToolAgent


def test_response_preview_and_stream_cleanup():
    class Stream:
        closed = False
        def __aiter__(self):
            return self.parts()
        async def parts(self):
            for token in ['Olá, ', 'tudo bem?']:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token, tool_calls=None))])
        async def close(self):
            self.closed = True

    async def scenario():
        stream = Stream()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))))
        agent = ToolAgent(client, 'test', 'system')
        agent.maybe_compact = AsyncMock()
        final, previews, speech = [], [], []
        async def stream_emit(delta):
            speech.append(delta)
        async def emit(text):
            final.append(text)
        await agent.ask('Oi', emit, preview=previews.append, stream_emit=stream_emit)
        assert previews == ['Olá, ', 'Olá, tudo bem?']
        assert final == ['Olá, tudo bem?']
        assert speech == ['Olá, ', 'tudo bem?', None]
        assert stream.closed
    asyncio.run(scenario())


def test_consecutive_utterances_remain_separate_and_interrupted_reply_is_retained():
    import copy
    import pytest

    class Stream:
        def __init__(self, text):
            self.text = text
        def __aiter__(self):
            return self.parts()
        async def parts(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=self.text, tool_calls=None))])
        async def close(self):
            pass

    async def scenario():
        sent = []
        async def create(**kwargs):
            sent.append(copy.deepcopy(kwargs['messages']))
            return Stream('Oi! Tudo bem por aqui.' if len(sent) == 1 else 'Me conte sobre o projeto.')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        agent = ToolAgent(client, 'test', 'system')
        agent.maybe_compact = AsyncMock()
        async def emit(text):
            pass
        async def interrupted(delta):
            if delta:
                raise asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await agent.ask('Oi, tudo bem?', emit, stream_emit=interrupted)
        await agent.ask('Tenho uma dúvida sobre um projeto.', emit)
        assert [(m['role'], m['content']) for m in sent[1][:-1]] == [
            ('system', 'system'),
            ('user', 'Oi, tudo bem?'),
            ('assistant', 'Oi! Tudo bem por aqui.'),
            ('user', 'Tenho uma dúvida sobre um projeto.'),
        ]
    asyncio.run(scenario())


def test_cancelled_tool_call_gets_a_result_marker_in_history():
    class Stream:
        def __aiter__(self):
            return self.parts()
        async def parts(self):
            call = SimpleNamespace(index=0, id='call_ping', function=SimpleNamespace(name='ping_host', arguments='{"host":"google.com"}'))
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[call]))])
        async def close(self):
            pass

    async def scenario():
        entered = asyncio.Event()
        async def tool(*args):
            entered.set()
            await asyncio.Event().wait()
        create = AsyncMock(return_value=Stream())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        agent = ToolAgent(client, 'test', 'system')
        agent.maybe_compact = AsyncMock()
        agent.tools = [{'function': {'name': 'ping_host'}}]
        agent.session = SimpleNamespace(call_tool=tool)
        async def emit(text):
            pass
        task = asyncio.create_task(agent.ask('Qual é o ping?', emit))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert agent.history[-1]['role'] == 'tool'
        assert agent.history[-1]['tool_call_id'] == 'call_ping'
        assert 'não confirmado' in agent.history[-1]['content']
    asyncio.run(scenario())
