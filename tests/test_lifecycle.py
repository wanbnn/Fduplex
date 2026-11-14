"""Exercise the real ASR/MCP lifecycle without opening the physical microphone."""
import asyncio
import os
import sys
from unittest.mock import patch

import pytest

import assistant


@pytest.mark.skipif(os.getenv('FDUPLEX_TEST_NATIVE') != '1', reason='native lifecycle is opt-in')
def test_cancel_closes_resources_from_the_task_that_opened_them(monkeypatch):
    class InputStream:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(sys, 'argv', ['assistant', '--plain', '--tts', 'none'])

    async def scenario():
        app = assistant.Assistant(assistant.parse_args())
        with patch.object(assistant.sd, 'InputStream', InputStream):
            task = asyncio.create_task(app.run())
            try:
                async with asyncio.timeout(20):
                    while app.mic is None:
                        if task.done():
                            task.result()
                        await asyncio.sleep(0.02)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert app.engine.proc is None
                assert app.client.is_closed()
                if app.agent:
                    assert not app.agent.stack._exit_callbacks
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())
