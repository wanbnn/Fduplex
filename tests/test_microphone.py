import asyncio
from types import SimpleNamespace

import numpy as np

from assistant import Microphone
from audio_frontend import FRAME


class Gate:
    active = False
    def reset(self):
        self.active = False
    def process(self, frame):
        if not self.active:
            self.active = True
            return frame, 'start'
        return frame, 'audio'


def test_mute_finalizes_captured_audio_and_keeps_capture_responsive():
    async def scenario():
        events = []
        ui = SimpleNamespace(muted=False, fail=lambda message: None)
        mic = Microphone(asyncio.get_running_loop(), Gate(), lambda *event: events.append(event), lambda _: None, ui)
        task = asyncio.create_task(mic.run())
        try:
            mic._enqueue(np.ones(FRAME, dtype=np.float32))
            for _ in range(30):
                await asyncio.sleep(0.002)
                if events:
                    break
            assert events[0][1] == 'start'
            mic.set_muted(True)
            mic._enqueue(np.ones(FRAME, dtype=np.float32))
            await asyncio.sleep(0.02)
            assert events[-1][1] == 'end'
            before = len(events)
            mic._enqueue(np.ones(FRAME, dtype=np.float32))
            await asyncio.sleep(0.02)
            assert len(events) == before
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


def test_mute_and_unmute_preserve_running_reply_and_speech():
    from assistant import Assistant
    from unittest.mock import Mock

    async def scenario():
        app = Assistant.__new__(Assistant)
        app.mic = SimpleNamespace(muted=False)
        def set_muted(value):
            app.mic.muted = value
        app.mic.set_muted = set_muted
        app.epoch, app.live_id = 7, 12
        app.ui = SimpleNamespace(answer='Resposta em andamento')
        app.speech = SimpleNamespace(interrupt=Mock())
        app.chat_task = asyncio.create_task(asyncio.Event().wait())
        try:
            app.key('m')
            await asyncio.sleep(0)
            assert app.mic.muted
            assert not app.chat_task.done()
            app.key('m')
            assert not app.mic.muted
            assert app.epoch == 7 and app.live_id == 12
            assert app.ui.answer == 'Resposta em andamento'
            app.speech.interrupt.assert_not_called()
        finally:
            app.chat_task.cancel()
            await asyncio.gather(app.chat_task, return_exceptions=True)
    asyncio.run(scenario())


def test_asr_finishes_and_replies_while_microphone_is_muted():
    from assistant import Assistant
    from unittest.mock import AsyncMock

    async def scenario():
        app = Assistant.__new__(Assistant)
        app.args = SimpleNamespace(verify_final=False)
        app.engine = SimpleNamespace(streaming=False, request=AsyncMock(return_value={'text': 'Uma pergunta pronta'}))
        app.mic = SimpleNamespace(muted=True)
        app.ui = SimpleNamespace(backlog=0, status='', committed='', partial='', message=lambda *args: None)
        app.live_id, app.epoch = 1, 0
        app.failed_ids = set()
        app.chat_task = None
        app.reply = AsyncMock()
        app.audio_queue = asyncio.Queue()
        await app.audio_queue.put((1, 'start', np.ones(FRAME, np.float32)))
        await app.audio_queue.put((1, 'end', np.zeros(FRAME, np.float32)))
        worker = asyncio.create_task(app.asr_worker())
        try:
            for _ in range(20):
                await asyncio.sleep(0)
                if app.reply.await_count:
                    break
            app.reply.assert_awaited_once_with('Uma pergunta pronta', 1)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            if app.chat_task:
                await asyncio.gather(app.chat_task, return_exceptions=True)
    asyncio.run(scenario())
