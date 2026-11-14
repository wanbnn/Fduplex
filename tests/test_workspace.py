import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from workspace import session_workspace


def test_launcher_directory_overrides_old_fixed_workspace(tmp_path, monkeypatch):
    launch = tmp_path / 'um projeto com espaços'
    launch.mkdir()
    monkeypatch.setenv('FDUPLEX_LAUNCH_DIR', str(launch))
    monkeypatch.setenv('FDUPLEX_WORKSPACE', str(tmp_path / 'obsolete'))
    assert session_workspace() == launch


def test_direct_python_uses_callers_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv('FDUPLEX_LAUNCH_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    assert session_workspace() == tmp_path


def test_real_mcp_creates_reads_and_patches_relative_to_launch_directory(tmp_path):
    import os
    import sys
    launch = tmp_path / 'project with spaces'
    launch.mkdir()
    environment = dict(os.environ, FDUPLEX_LAUNCH_DIR=str(launch),
                       FDUPLEX_WORKSPACE=str(tmp_path / 'obsolete'),
                       FDUPLEX_WRITE_ROOTS=str(tmp_path / 'obsolete'),
                       FDUPLEX_READ_ROOTS=str(tmp_path / 'obsolete'))
    server = Path(__file__).resolve().parent.parent / 'fduplex_mcp.py'

    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=[str(server)], env=environment,
                                       cwd=str(server.parent))
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                created = await client.call_tool('write_file', {'path': 'src/note.txt', 'content': 'before\n'})
                assert not created.isError
                assert (launch / 'src/note.txt').read_text() == 'before\n'
                patched = await client.call_tool('apply_patch', {'path': 'src/note.txt',
                    'patch': '--- a/note.txt\n+++ b/note.txt\n@@ -1 +1 @@\n-before\n+after\n'})
                assert not patched.isError
                assert (launch / 'src/note.txt').read_text() == 'after\n'
                read = await client.call_tool('read_file', {'path': 'src/note.txt'})
                assert not read.isError and 'after' in read.content[0].text
                denied = await client.call_tool('write_file', {'path': '../outside.txt', 'content': 'no'})
                assert denied.isError and not (tmp_path / 'outside.txt').exists()
    asyncio.run(scenario())
