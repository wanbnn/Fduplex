#!/usr/bin/env python3
"""Cache and warm up the Portuguese Pocket TTS voice without playing audio."""
import argparse
import logging

from tts_engine import PocketVoice


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', default='portuguese')
    parser.add_argument('--voice', default='rafael')
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    voice = PocketVoice(args.language, args.voice, args.quantize)
    voice.load()
    print(f'Pocket TTS pronto: {args.language}, voz {args.voice}, {voice.rate} Hz, CPU.')
