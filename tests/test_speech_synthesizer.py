import asyncio
import importlib
import sys
from unittest.mock import MagicMock


def _make_synthesizer():
    tts_cls = MagicMock()
    tts_instance = MagicMock()
    tts_cls.return_value.to.return_value = tts_instance

    tts_api_mod = MagicMock()
    tts_api_mod.TTS = tts_cls
    sys.modules["TTS.api"] = tts_api_mod

    import iara.tts as _mod
    importlib.reload(_mod)

    return _mod.SpeechSynthesizer(), tts_instance


class TestSpeechSynthesizerInit:
    def test_tts_loaded_with_xtts_v2(self):
        _, _ = _make_synthesizer()
        tts_cls = sys.modules["TTS.api"].TTS
        _, kwargs = tts_cls.call_args
        assert kwargs.get("model_name") == "xtts_v2"

    def test_model_moved_to_cuda(self):
        _, _ = _make_synthesizer()
        tts_cls = sys.modules["TTS.api"].TTS
        tts_cls.return_value.to.assert_called_with("cuda")


class TestGenerateTTSFile:
    def test_empty_text_does_not_call_tts(self):
        synth, tts_instance = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("", "/tmp/out.wav"))
        tts_instance.tts_to_file.assert_not_called()

    def test_generates_file_for_valid_text(self):
        synth, tts_instance = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("Olá mundo", "/tmp/out.wav"))
        tts_instance.tts_to_file.assert_called_once()

    def test_output_path_passed_correctly(self):
        synth, tts_instance = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("teste", "/tmp/resultado.wav"))
        _, kwargs = tts_instance.tts_to_file.call_args
        assert kwargs.get("file_path") == "/tmp/resultado.wav"

    def test_language_is_portuguese(self):
        synth, tts_instance = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("texto", "/tmp/out.wav"))
        _, kwargs = tts_instance.tts_to_file.call_args
        assert kwargs.get("language") == "pt"

    def test_punctuation_stripped_before_tts(self):
        synth, tts_instance = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("Olá! (mundo).", "/tmp/out.wav"))
        _, kwargs = tts_instance.tts_to_file.call_args
        text_sent = kwargs.get("text")
        assert "!" not in text_sent
        assert "(" not in text_sent
        assert ")" not in text_sent
        assert "." not in text_sent

    def test_none_like_empty_string_does_not_call_tts(self):
        synth, tts_instance = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("", "/tmp/out.wav"))
        tts_instance.tts_to_file.assert_not_called()
