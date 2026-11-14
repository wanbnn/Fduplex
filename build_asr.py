#!/usr/bin/env python3
"""Link the persistent worker to an existing transcribe.cpp build, without rebuilding HIP."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def build(source, build_dir, output):
    source, build_dir, output = map(lambda p: Path(p).resolve(), (source, build_dir, output))
    manifest = json.loads((build_dir / 'transcribe-link.json').read_text())
    command = ['c++', '-std=c++17', '-O2', '-I', str(source / 'include'),
               str(Path(__file__).parent / 'native/asr_worker.cpp'), '-o', str(output)]
    for name in manifest['libraries']:
        matches = sorted(build_dir.rglob(f'lib{name}.a')) or sorted(build_dir.rglob(f'lib{name}.so'))
        if not matches:
            raise SystemExit(f'Biblioteca ausente: {name}. Compile transcribe.cpp primeiro.')
        command.append(str(matches[0]))
        if matches[0].suffix == '.so':
            command.append(f'-Wl,-rpath,{matches[0].parent}')
    command += manifest.get('library_paths', [])
    command += ['-l' + name for name in manifest.get('system_libs', [])]
    command += manifest.get('link_flags', [])
    for name in manifest.get('frameworks', []):
        command += ['-framework', name]
    output.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(command))
    subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).parent / 'transcribe.cpp')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'bin/fduplex-asr')
    args = parser.parse_args()
    build(args.source, args.build or args.source / 'build', args.output)
