import asyncio
import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_synthesizer():
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = iter([])

    mock_kokoro = MagicMock()
    mock_kokoro.KPipeline.return_value = mock_pipeline_instance
    sys.modules["kokoro"] = mock_kokoro

    mock_sf = MagicMock()
    sys.modules["soundfile"] = mock_sf

    import iara.tts as _mod
    importlib.reload(_mod)

    synth = _mod.SpeechSynthesizer()
    return synth, mock_pipeline_instance, mock_sf


class TestSpeechSynthesizerInit:
    def test_pipeline_created_with_portuguese(self):
        _, _, _ = _make_synthesizer()
        mock_kokoro = sys.modules["kokoro"]
        _, kwargs = mock_kokoro.KPipeline.call_args
        assert kwargs.get("lang_code") == 'p'

    def test_default_voice_is_mixed_tensor(self):
        synth, mock_pipeline, _ = _make_synthesizer()
        mock_pipeline.load_single_voice.assert_called()
        assert synth.voice is not None


class TestGenerateTTSFile:
    def test_empty_text_does_not_call_pipeline(self):
        synth, mock_pipeline, _ = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("", "/tmp/out.wav"))
        mock_pipeline.assert_not_called()

    def test_whitespace_only_does_not_call_pipeline(self):
        synth, mock_pipeline, _ = _make_synthesizer()
        asyncio.run(synth.generate_tts_file("   ", "/tmp/out.wav"))
        mock_pipeline.assert_not_called()

    def test_generates_file_for_valid_text(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("Olá mundo", "/tmp/out.wav"))
        mock_pipeline.assert_called_once()

    def test_output_path_passed_to_soundfile(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("texto", "/tmp/resultado.wav"))
        args, _ = mock_sf.write.call_args
        assert args[0] == "/tmp/resultado.wav"

    def test_punctuation_stripped_before_synthesis(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("Olá! (mundo).", "/tmp/out.wav"))
        _, kwargs = mock_pipeline.call_args
        text_sent = mock_pipeline.call_args[0][0]
        assert "!" not in text_sent
        assert "(" not in text_sent
        assert ")" not in text_sent
        assert "." not in text_sent

    def test_voice_passed_to_pipeline(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("texto", "/tmp/out.wav"))
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("voice") is synth.voice

    def test_no_audio_chunks_skips_write(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        mock_pipeline.return_value = iter([("seg", "ph", None)])
        asyncio.run(synth.generate_tts_file("texto", "/tmp/out.wav"))
        mock_sf.write.assert_not_called()
