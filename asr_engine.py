"""Async client for the persistent native ASR worker."""
import asyncio
import json
from pathlib import Path
import struct

import numpy as np


class ASREngine:
    def __init__(self, args, log):
        self.args, self.log = args, log
        self.proc = None
        self.streaming = False
        self.backend = ''

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            self.args.asr_worker, self.args.gguf, self.args.backend,
            self.args.language, str(self.args.stream_right),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log)
        reply = await self.read(timeout=120)
        self.streaming = bool(reply.get('streaming'))
        self.backend = reply.get('backend', self.args.backend)
        if self.args.asr_mode == 'streaming' and not self.streaming:
            raise RuntimeError('Este GGUF não suporta streaming nativo. Use Nemotron 3.5 ASR Streaming ou --asr-mode auto.')
        if self.args.asr_mode == 'offline':
            self.streaming = False

    async def read(self, timeout=90):
        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
        if not line:
            raise RuntimeError('Motor ASR encerrou. Veja o log de diagnóstico.')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError('ASR: ' + reply['error'])
        return reply

    async def request(self, command, samples=None):
        pcm = np.asarray(samples, dtype='<f4').tobytes() if samples is not None else b''
        self.proc.stdin.write(command.encode('ascii') + struct.pack('<I', len(pcm) // 4) + pcm)
        await self.proc.stdin.drain()
        return await self.read()

    async def close(self):
        if not self.proc:
            return
        if self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 3)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        self.proc = None
