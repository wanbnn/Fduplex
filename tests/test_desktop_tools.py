import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
import fduplex_mcp as server


def test_firefox_launch_keeps_desktop_environment_and_does_not_wait_for_window(tmp_path):
    proc = Mock(pid=12345)
    proc.poll.return_value = None
    with patch.dict(os.environ, {'HOME': str(tmp_path), 'DISPLAY': ':0',
                                 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=test', 'OPENAI_API_KEY': 'private'}), \
         patch.object(server, 'WORKSPACE', tmp_path), \
         patch('shutil.which', return_value='/usr/bin/firefox'), \
         patch.object(server.subprocess, 'Popen', return_value=proc) as spawn, \
         patch.object(server.asyncio, 'sleep', new=AsyncMock()):
        result = asyncio.run(server.run_terminal('firefox'))
        assert result['status'] == 'process_started' and result['pid'] == 12345
        options = spawn.call_args.kwargs
        assert spawn.call_args.args[0] == ['/usr/bin/firefox']
        assert options['env']['HOME'] == str(tmp_path)
        assert options['env']['DISPLAY'] == ':0'
        assert options['env']['DBUS_SESSION_BUS_ADDRESS'] == 'unix:path=test'
        assert 'OPENAI_API_KEY' not in options['env']
        assert options['cwd'] == str(tmp_path) and options['start_new_session']


def test_missing_application_is_reported():
    with patch('shutil.which', return_value=None):
        result = asyncio.run(server.launch_application('firefox'))
        assert 'não instalado' in result['error']


def test_ping_dispatches_to_real_measurement_tool():
    with patch.object(server, 'ping_host', new=AsyncMock(return_value={'avg_ms': 5.2})) as ping:
        assert asyncio.run(server.run_terminal('ping -c 3 google.com')) == {'avg_ms': 5.2}
        ping.assert_awaited_once_with('google.com', 3)


def test_empty_command_is_rejected():
    with pytest.raises(ValueError):
        asyncio.run(server.run_terminal('   '))


def test_real_bash_supports_pipes_redirects_and_workspace_paths(tmp_path):
    with patch.object(server, 'WORKSPACE', tmp_path):
        result = asyncio.run(server.run_terminal("printf 'funciona\\n' | tr a-z A-Z > result.txt && cat ./result.txt"))
    assert result['returncode'] == 0 and result['stdout'].strip() == 'FUNCIONA'
    assert (tmp_path / 'result.txt').read_text() == 'FUNCIONA\n'


def test_terminal_preserves_real_home_and_shell_expansion(tmp_path):
    with patch.object(server, 'WORKSPACE', tmp_path):
        result = asyncio.run(server.run_terminal('printf "%s" "$HOME"'))
    assert result['returncode'] == 0 and result['stdout'] == os.environ['HOME']


def test_background_process_reports_pid_and_writes_log(tmp_path):
    with patch.object(server, 'WORKSPACE', tmp_path), patch.object(server, 'STATE_DIR', tmp_path):
        result = asyncio.run(server.run_terminal("printf 'background works'", background=True))
    assert result['pid'] > 0
    proc = next(proc for proc in server.LAUNCHED_APPS if proc.pid == result['pid'])
    assert proc.wait(timeout=2) == 0
    from pathlib import Path
    assert Path(result['log']).read_text() == 'background works'


def test_timeout_kills_foreground_pipeline(tmp_path):
    with patch.object(server, 'WORKSPACE', tmp_path):
        result = asyncio.run(server.run_terminal('sleep 30 | cat', timeout=1))
    assert result['error'] == 'timeout'


def test_real_ping_localhost():
    import shutil
    if not shutil.which('ping'):
        pytest.skip('ping not installed')
    result = asyncio.run(server.ping_host('127.0.0.1', 1))
    assert result['returncode'] == 0
    assert result['avg_ms'] >= 0
