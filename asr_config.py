"""Portable asset discovery, with explicit options taking precedence."""
import os
from pathlib import Path
import shutil

STREAM_MODEL = 'nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf'
ULTRA_MODEL = 'parakeet-ultra-Q8_0.gguf'


def asset_roots():
    roots = [Path(os.getenv('FDUPLEX_LAUNCH_DIR', str(Path.cwd()))),
             Path(__file__).resolve().parent,
             Path(os.getenv('FDUPLEX_WORKSPACE', str(Path.home() / 'Fduplex-workspace'))),
             Path(os.getenv('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'fduplex']
    return list(dict.fromkeys(path.expanduser().resolve() for path in roots))


def discover(value, relatives, option, executable=False, extra_roots=()):
    roots = list(extra_roots) + asset_roots()
    if value:
        value = os.path.expandvars(os.path.expanduser(value))
        path = Path(value)
        if executable and path.name == value:
            found = shutil.which(value)
            if found:
                return str(Path(found).resolve())
        candidates = [path] if path.is_absolute() else [root / path for root in roots]
    else:
        candidates = [root / relative for relative in relatives for root in roots]
    for path in candidates:
        if path.is_file() and (not executable or os.access(path, os.X_OK)):
            return str(path.resolve())
    if not value and executable:
        found = shutil.which(Path(relatives[0]).name)
        if found:
            return str(Path(found).resolve())
    raise FileNotFoundError(f'{option}: arquivo ausente. Execute setup_audio.py / build_asr.py ou configure o caminho. Verificados: '
                            + ', '.join(str(path) for path in candidates))


def resolve_asr_paths(args):
    extra = []
    if args.cli:
        # Legacy --cli remains useful to locate a worker built beside that checkout.
        cli = discover(args.cli, [], '--cli', executable=True)
        extra.append(Path(cli).parents[3])
    args.asr_worker = discover(args.asr_worker, ['bin/fduplex-asr'], '--asr-worker', True, extra)
    names = [ULTRA_MODEL] if args.asr_mode == 'offline' else [STREAM_MODEL, ULTRA_MODEL]
    args.gguf = discover(args.gguf, ['models/' + name for name in names], '--gguf / PARAKEET_GGUF')
    args.vad_model = discover(args.vad_model, ['models/silero_vad.onnx'], '--vad-model / FDUPLEX_VAD_MODEL')
