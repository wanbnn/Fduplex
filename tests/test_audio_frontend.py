import numpy as np

from audio_frontend import FRAME, SpeechGate, Denoiser, Silero


class Identity:
    def process(self, frame):
        return frame


class Vad:
    p = 0.0
    def probability(self, frame):
        return self.p
    def reset(self):
        pass


def frame():
    return (np.sin(np.arange(FRAME) * 0.11) * 0.05).astype(np.float32)


def make_gate(**kwargs):
    vad = Vad()
    return vad, SpeechGate(vad, Identity(), calibration_ms=0, **kwargs)


def test_isolated_clicks_cannot_interrupt_even_with_false_vad_positive():
    vad, gate = make_gate()
    vad.p = 1
    click = np.zeros(FRAME, np.float32)
    click[100] = 0.9
    for _ in range(120):
        _, event = gate.process(click)
        assert event != 'start'


def test_brief_speech_like_bursts_do_not_open_gate():
    vad, gate = make_gate()
    for i in range(100):
        vad.p = 0.99 if i % 12 < 3 else 0
        assert gate.process(frame())[1] != 'start'


def test_onset_preserves_prefix_and_endpoint_waits_for_silence():
    vad, gate = make_gate(pause_ms=128)
    vad.p = 0.98
    events = [gate.process(frame()) for _ in range(12)]
    starts = [pcm for pcm, event in events if event == 'start']
    assert len(starts) == 1
    assert starts[0].size >= FRAME * 6
    vad.p = 0
    silence = np.zeros(FRAME, np.float32)
    assert [gate.process(silence)[1] for _ in range(4)] == ['audio', 'audio', 'audio', 'end']
    assert not gate.active


def test_short_false_onset_is_discarded():
    vad, gate = make_gate(start_ms=64, min_speech_ms=256, pause_ms=64)
    vad.p = 0.99
    assert gate.process(frame())[1] is None
    assert gate.process(frame())[1] == 'start'
    vad.p = 0
    assert gate.process(np.zeros(FRAME, np.float32))[1] == 'audio'
    assert gate.process(np.zeros(FRAME, np.float32))[1] == 'discard'


def test_noise_floor_tracks_background_without_chasing_speech():
    vad, gate = make_gate()
    for _ in range(30):
        gate.process(frame() * 0.1)
    floor = gate.noise_floor
    vad.p = 0.99
    for _ in range(20):
        gate.process(frame())
    assert gate.noise_floor == floor
    assert gate.active


def test_real_silero_and_speex_reject_synthetic_typing_and_fan():
    from asr_config import discover
    path = discover(None, ['models/silero_vad.onnx'], 'vad')
    vad, denoiser = Silero(path), Denoiser()
    gate = SpeechGate(vad, denoiser)
    random = np.random.default_rng(42)
    try:
        for i in range(200):
            packet = random.normal(0, 0.001, FRAME).astype(np.float32)
            if i > 30 and i % 4 == 0:
                # A short decaying broadband transient, repeated like typing.
                packet[:80] += random.normal(0, 0.14, 80) * np.exp(-np.arange(80) / 12)
            assert gate.process(packet)[1] != 'start'
    finally:
        denoiser.close()
