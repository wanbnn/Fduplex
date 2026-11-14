from types import SimpleNamespace

import pytest

import asr_config


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(0o755)


def test_portable_assets_prefer_streaming_across_roots(tmp_path, monkeypatch):
    checkout, data = tmp_path / 'checkout', tmp_path / 'data'
    executable(checkout / 'bin/fduplex-asr')
    (checkout / 'models').mkdir()
    (checkout / 'models' / asr_config.ULTRA_MODEL).touch()
    (data / 'models').mkdir(parents=True)
    (data / 'models' / asr_config.STREAM_MODEL).touch()
    (data / 'models/silero_vad.onnx').touch()
    monkeypatch.setattr(asr_config, 'asset_roots', lambda: [checkout, data])
    args = SimpleNamespace(cli=None, asr_worker=None, gguf=None, vad_model=None, asr_mode='auto')
    asr_config.resolve_asr_paths(args)
    assert args.gguf == str(data / 'models' / asr_config.STREAM_MODEL)
    assert args.asr_worker == str(checkout / 'bin/fduplex-asr')


def test_explicit_invalid_override_is_not_silently_replaced(tmp_path, monkeypatch):
    executable(tmp_path / 'bin/fduplex-asr')
    monkeypatch.setattr(asr_config, 'asset_roots', lambda: [tmp_path])
    with pytest.raises(FileNotFoundError):
        asr_config.discover('/missing/worker', ['bin/fduplex-asr'], 'worker', True)
