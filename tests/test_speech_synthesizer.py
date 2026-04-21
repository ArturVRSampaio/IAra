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

    def test_default_voice_is_pf_dora(self):
        synth, _, _ = _make_synthesizer()
        assert synth.voice == 'pf_dora'


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
        assert kwargs.get("voice") == "pf_dora"

    def test_no_audio_chunks_skips_write(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        mock_pipeline.return_value = iter([("seg", "ph", None)])
        asyncio.run(synth.generate_tts_file("texto", "/tmp/out.wav"))
        mock_sf.write.assert_not_called()

    def test_newlines_replaced_with_space(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("linha um\nlinha dois", "/tmp/out.wav"))
        text_sent = mock_pipeline.call_args[0][0]
        assert "\n" not in text_sent
        assert "\r" not in text_sent

    def test_emojis_stripped_before_synthesis(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("que legal \U0001F600", "/tmp/out.wav"))
        text_sent = mock_pipeline.call_args[0][0]
        assert "\U0001F600" not in text_sent

    def test_question_mark_stripped_before_synthesis(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("tudo bem?", "/tmp/out.wav"))
        text_sent = mock_pipeline.call_args[0][0]
        assert "?" not in text_sent

    def test_sample_rate_is_24000(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("texto", "/tmp/out.wav"))
        args, _ = mock_sf.write.call_args
        assert args[2] == 24000

    def test_executor_is_single_threaded(self):
        synth, _, _ = _make_synthesizer()
        assert synth._executor._max_workers == 1

    def test_special_tokens_stripped_before_synthesis(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("olá <|eom_id|> mundo", "/tmp/out.wav"))
        text_sent = mock_pipeline.call_args[0][0]
        assert "<|eom_id|>" not in text_sent

    def test_role_markers_stripped_before_synthesis(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("olá [user diz algo", "/tmp/out.wav"))
        text_sent = mock_pipeline.call_args[0][0]
        assert "[user" not in text_sent

    def test_code_blocks_stripped_before_synthesis(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("olá ```python\nprint('hi')\n``` mundo", "/tmp/out.wav"))
        text_sent = mock_pipeline.call_args[0][0]
        assert "```" not in text_sent
        assert "print" not in text_sent

    def test_inline_code_stripped_before_synthesis(self):
        synth, mock_pipeline, mock_sf = _make_synthesizer()
        import numpy as np
        mock_pipeline.return_value = iter([("seg", "ph", np.zeros(100, dtype="float32"))])
        asyncio.run(synth.generate_tts_file("use `foo()` para isso", "/tmp/out.wav"))
        text_sent = mock_pipeline.call_args[0][0]
        assert "`" not in text_sent
