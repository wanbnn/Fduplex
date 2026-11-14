#!/usr/bin/env python3
"""Download the streaming ASR and pinned Silero VAD to the user data directory."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request

from huggingface_hub import hf_hub_download

from asr_config import STREAM_MODEL

ASR_REV = '53ce288f69326b415aeef030cfcb02a08a8096dd'
SILERO_REV = '066409f3d9db45a30d21dfcbf9147826ffdd7052'
SILERO_SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'


def setup(directory, vad_only=False):
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'silero_vad.onnx'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != SILERO_SHA256:
        url = f'https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_REV}/src/silero_vad/data/silero_vad.onnx'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SILERO_SHA256:
            raise RuntimeError('Checksum Silero inválido.')
        temporary = path.with_suffix('.download')
        temporary.write_bytes(data)
        temporary.replace(path)
    print(f'Silero VAD: {path}')
    if not vad_only:
        result = hf_hub_download('handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf', STREAM_MODEL, revision=ASR_REV, local_dir=directory)
        print(f'ASR streaming: {result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path,
                        default=Path(os.getenv('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'fduplex/models')
    parser.add_argument('--vad-only', action='store_true')
    args = parser.parse_args()
    setup(args.models_dir, args.vad_only)
