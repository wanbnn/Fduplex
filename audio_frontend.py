"""16 kHz speech frontend: Speex suppression + Silero + temporal confirmation."""
import collections
import ctypes
import ctypes.util
import math
from pathlib import Path

import numpy as np

RATE = 16000
FRAME = 512
FRAME_MS = 32


class Denoiser:
    def __init__(self, enabled=True, suppress_db=-25):
        self.state = None
        if not enabled:
            self.name = 'desligada'
            return
        library = ctypes.util.find_library('speexdsp')
        if not library:
            raise RuntimeError('Instale libspeexdsp1 para supressão de ruído, ou use --no-denoise.')
        self.lib = ctypes.CDLL(library)
        self.lib.speex_preprocess_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.speex_preprocess_state_init.restype = ctypes.c_void_p
        self.lib.speex_preprocess_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.speex_preprocess_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16)]
        self.lib.speex_preprocess_state_destroy.argtypes = [ctypes.c_void_p]
        self.state = self.lib.speex_preprocess_state_init(FRAME, RATE)
        if not self.state:
            raise RuntimeError('Não foi possível inicializar SpeexDSP.')
        for option, value in [(0, 1), (2, 0), (18, suppress_db)]:
            setting = ctypes.c_int(value)
            self.lib.speex_preprocess_ctl(self.state, option, ctypes.byref(setting))
        self.name = f'Speex {suppress_db} dB'

    def process(self, frame):
        # Remove DC before denoising; do not apply automatic gain to keyboard clicks.
        frame = np.asarray(frame, dtype=np.float32)
        frame = frame - frame.mean()
        if self.state is None:
            return frame
        pcm = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        self.lib.speex_preprocess_run(self.state, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))
        return pcm.astype(np.float32) / 32768

    def close(self):
        if self.state:
            self.lib.speex_preprocess_state_destroy(self.state)
            self.state = None


class Silero:
    def __init__(self, path):
        import onnxruntime as ort
        if not Path(path).is_file():
            raise RuntimeError('Modelo Silero ausente. Execute setup_audio.py ou configure --vad-model.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def probability(self, frame):
        pcm = np.concatenate((self.context, frame.reshape(1, FRAME)), axis=1)
        out, self.state = self.session.run(None, {'input': pcm, 'state': self.state,
                                                 'sr': np.array(RATE, dtype=np.int64)})
        self.context = pcm[:, -64:].copy()
        return float(out[0, 0])


class SpeechGate:
    """Onset and endpoint use audio duration, never wall-clock inference timing."""
    def __init__(self, vad, denoiser, threshold=100, probability=0.65,
                 start_ms=192, min_speech_ms=256, pause_ms=640, calibration_ms=768):
        self.vad, self.denoiser = vad, denoiser
        self.threshold, self.probability = threshold, probability
        self.start_frames = max(2, math.ceil(start_ms / FRAME_MS))
        self.min_frames = max(self.start_frames, math.ceil(min_speech_ms / FRAME_MS))
        self.pause_frames = max(2, math.ceil(pause_ms / FRAME_MS))
        self.calibration_frames = math.ceil(calibration_ms / FRAME_MS)
        self.noise = collections.deque(maxlen=160)
        self.prefix = collections.deque(maxlen=max(12, self.start_frames + 3))
        self.onset = collections.deque(maxlen=self.start_frames + 2)
        self.frames_seen = 0
        self.active = False
        self.voiced_frames = 0
        self.silent = 0
        self.duration = 0
        self.level = self.p = self.noise_floor = 0.0
        self.gate_level = threshold

    def reset(self):
        self.active = False
        self.prefix.clear()
        self.onset.clear()
        self.voiced_frames = self.silent = self.duration = 0
        self.vad.reset()

    def process(self, frame):
        raw_rms = float(np.sqrt(np.mean(frame ** 2))) * 32768
        clean = self.denoiser.process(frame)
        self.p = self.vad.probability(clean)
        self.level = raw_rms
        self.frames_seen += 1
        if self.p < 0.15 or self.frames_seen <= self.calibration_frames:
            self.noise.append(raw_rms)
        # A low percentile is robust to isolated key presses and doesn't chase speech.
        self.noise_floor = float(np.percentile(self.noise, 25)) if self.noise else 0
        self.gate_level = max(self.threshold, self.noise_floor * 2.5)
        peak_ratio = float(np.max(np.abs(frame))) * 32768 / max(raw_rms, 1)
        voiced = (self.p >= (self.probability - 0.18 if self.active else self.probability)
                  and raw_rms >= self.gate_level * (0.65 if self.active else 1)
                  and (self.active or peak_ratio < 10))
        if self.frames_seen <= self.calibration_frames:
            return clean, None
        if not self.active:
            self.prefix.append(clean.copy())
            self.onset.append(voiced)
            # Allow two gaps, but require the latest two frames to contain speech.
            if (sum(self.onset) >= self.start_frames and len(self.onset) >= 2
                    and all(list(self.onset)[-2:])):
                self.active = True
                self.voiced_frames = sum(self.onset)
                self.duration = len(self.prefix)
                self.silent = 0
                return np.concatenate(self.prefix), 'start'
            return clean, None
        self.duration += 1
        self.voiced_frames += int(voiced)
        self.silent = 0 if voiced else self.silent + 1
        if self.silent >= self.pause_frames or self.duration >= math.ceil(28_000 / FRAME_MS):
            event = 'end' if self.voiced_frames >= self.min_frames else 'discard'
            self.reset()
            return clean, event
        return clean, 'audio'
