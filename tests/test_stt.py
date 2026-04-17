import importlib
import struct
import sys
from unittest.mock import MagicMock


def _make_pcm_chunk(num_samples: int = 480, channels: int = 2) -> bytes:
    """Return silent 16-bit PCM bytes (48 kHz stereo by default)."""
    return struct.pack(f"<{num_samples * channels}h", *([0] * num_samples * channels))


def _make_stt():
    """Reload STT with fresh mocks so WhisperModel is captured correctly."""
    whisper_cls = MagicMock()
    faster_whisper_mod = MagicMock()
    faster_whisper_mod.WhisperModel = whisper_cls
    sys.modules["faster_whisper"] = faster_whisper_mod

    pydub_mod = MagicMock()
    sys.modules["pydub"] = pydub_mod

    import STT as _mod
    importlib.reload(_mod)

    return _mod.STT(), whisper_cls


class TestSTTInit:
    def test_model_created_with_turbo(self):
        _, whisper_cls = _make_stt()
        whisper_cls.assert_called_once()
        args, _ = whisper_cls.call_args
        assert args[0] == "turbo"

    def test_model_uses_cuda(self):
        _, whisper_cls = _make_stt()
        _, kwargs = whisper_cls.call_args
        assert kwargs.get("device") == "cuda"


class TestTranscribeAudio:
    def test_empty_chunks_returns_empty_string(self):
        stt, _ = _make_stt()
        assert stt.transcribe_audio([]) == ""

    def test_none_chunks_returns_empty_string(self):
        stt, _ = _make_stt()
        assert stt.transcribe_audio(None) == ""

    def test_valid_chunks_calls_transcribe(self):
        stt, _ = _make_stt()
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text, seg2.text = "Olá", " mundo"
        stt.model.transcribe.return_value = ([seg1, seg2], None)

        assert stt.transcribe_audio([_make_pcm_chunk()]) == "Olá mundo"

    def test_transcribe_called_with_portuguese(self):
        stt, _ = _make_stt()
        stt.model.transcribe.return_value = ([], None)
        stt.transcribe_audio([_make_pcm_chunk()])
        _, kwargs = stt.model.transcribe.call_args
        assert kwargs.get("language") == "pt"

    def test_vad_filter_enabled(self):
        stt, _ = _make_stt()
        stt.model.transcribe.return_value = ([], None)
        stt.transcribe_audio([_make_pcm_chunk()])
        _, kwargs = stt.model.transcribe.call_args
        assert kwargs.get("vad_filter") is True

    def test_empty_segments_returns_empty_string(self):
        stt, _ = _make_stt()
        stt.model.transcribe.return_value = ([], None)
        assert stt.transcribe_audio([_make_pcm_chunk()]) == ""

    def test_result_is_stripped(self):
        stt, _ = _make_stt()
        seg = MagicMock()
        seg.text = "  texto com espaços  "
        stt.model.transcribe.return_value = ([seg], None)
        assert stt.transcribe_audio([_make_pcm_chunk()]) == "texto com espaços"

    def test_multiple_chunks_are_joined(self):
        stt, _ = _make_stt()
        seg = MagicMock()
        seg.text = "ok"
        stt.model.transcribe.return_value = ([seg], None)
        assert stt.transcribe_audio([_make_pcm_chunk(), _make_pcm_chunk()]) == "ok"