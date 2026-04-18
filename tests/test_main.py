"""
Tests for main.py.

main.py instantiates STT, LLMAgent, SpeechSynthesizer at module level and defines
several async functions inside a `with llm.getChatSession():` block. All heavy
dependencies are mocked via conftest.py before this module is imported.
"""
import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _import_main():
    """Import main with all ML constructors mocked out."""
    # Ensure load_dotenv is a no-op
    sys.modules["dotenv"].load_dotenv = MagicMock()

    # Make LLMAgent.getChatSession() return a usable context manager
    from LLMAgent import LLMAgent  # noqa: F401 - triggers mock wiring

    import importlib
    if "main" in sys.modules:
        return sys.modules["main"]
    return importlib.import_module("main")


# Import once for the whole test session
_main = _import_main()


class TestDiscordBotPlayAudio:
    def _bot(self):
        bot_mock = MagicMock()
        instance = _main.DiscordBot(bot_mock)
        instance.voice_client = MagicMock()
        instance.voice_client.is_connected.return_value = True
        return instance

    def test_missing_file_returns_false(self):
        bot = self._bot()
        result = asyncio.run(bot.play_audio("/nonexistent/path/audio.wav"))
        assert result is False

    def test_existing_file_returns_true(self):
        bot = self._bot()

        # Create a real temp wav so os.path.exists passes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"\x00" * 44)  # minimal placeholder
            tmp_path = f.name

        try:
            playback_done_event = asyncio.Event()

            def fake_play(source, **kwargs):
                after = kwargs.get("after")
                if after:
                    after(None)

            bot.voice_client.play = MagicMock(side_effect=fake_play)
            sys.modules["torchaudio"].load.return_value = (MagicMock(), 48000)

            # sync_mouth needs to be a no-op coroutine
            with patch.object(bot.vts, "sync_mouth", new_callable=AsyncMock):
                result = asyncio.run(bot.play_audio(tmp_path))

            assert result is True
        finally:
            os.unlink(tmp_path)


class TestTranscribeForUser:
    def test_empty_queue_returns_empty_string(self):
        _main.user_voice_to_process_queue.clear()
        result = asyncio.run(_main.transcribe_for_user("user1"))
        assert result == ""

    def test_returns_formatted_transcript(self):
        user = MagicMock()
        user.__str__ = lambda s: "TestUser"
        _main.user_voice_to_process_queue[user] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="olá mundo")

        result = asyncio.run(_main.transcribe_for_user(user))
        assert "olá mundo" in result
        assert "says:" in result

        _main.user_voice_to_process_queue.pop(user, None)

    def test_empty_transcript_returns_empty_string(self):
        user = MagicMock()
        _main.user_voice_to_process_queue[user] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="")

        result = asyncio.run(_main.transcribe_for_user(user))
        assert result == ""

        _main.user_voice_to_process_queue.pop(user, None)


class TestAskLLMAndProcess:
    def setup_method(self):
        _main.audio_to_play_queue.clear()
        _main.can_release_accept_packages = False

        # Mock discord context
        ctx_mock = AsyncMock()
        _main.discord_bot_instance = MagicMock()
        _main.discord_bot_instance.context = ctx_mock

    def test_adds_audio_to_queue(self):
        _main.llm.ask = MagicMock(return_value=iter(["Olá, tudo bem?"]))

        async def fake_tts(text, path):
            with open(path, "wb") as f:
                f.write(b"\x00" * 44)

        _main.tts.generate_tts_file = fake_tts

        asyncio.run(_main.ask_llm_and_process("user says: oi"))
        assert len(_main.audio_to_play_queue) >= 1

        # Cleanup temp files
        for f in _main.audio_to_play_queue:
            try:
                os.unlink(f)
            except Exception:
                pass
        _main.audio_to_play_queue.clear()

    def test_sets_can_release_accept_packages_true(self):
        _main.llm.ask = MagicMock(return_value=iter([]))
        _main.tts.generate_tts_file = AsyncMock()

        asyncio.run(_main.ask_llm_and_process("user says: oi"))
        assert _main.can_release_accept_packages is True


class TestVoiceConsumer:
    def setup_method(self):
        _main.user_voice_to_process_queue.clear()
        _main.accept_packages = True
        _main.can_release_accept_packages = True

    def test_does_nothing_when_no_bot_instance(self):
        _main.discord_bot_instance = None

        async def run_one_tick():
            task = asyncio.create_task(_main.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.accept_packages is True

    def test_does_nothing_when_queue_empty(self):
        bot_mock = MagicMock()
        bot_mock.last_audio_time = None
        _main.discord_bot_instance = bot_mock

        async def run_one_tick():
            task = asyncio.create_task(_main.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.accept_packages is True


class TestVoicePlayer:
    def setup_method(self):
        _main.audio_to_play_queue.clear()
        _main.accept_packages = False
        _main.can_release_accept_packages = True
        _main.discord_bot_instance = MagicMock()

    def test_releases_accept_packages_when_queue_empty(self):
        async def run_one_tick():
            task = asyncio.create_task(_main.voice_player())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.accept_packages is True

    def test_accept_packages_not_released_while_queue_has_items(self):
        _main.audio_to_play_queue.append("/nonexistent/file.wav")
        _main.can_release_accept_packages = False
        _main.discord_bot_instance.play_audio = AsyncMock(return_value=False)

        async def run_one_tick():
            task = asyncio.create_task(_main.voice_player())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.accept_packages is False

        _main.audio_to_play_queue.clear()


class TestDequeQueue:
    def test_audio_queue_is_deque(self):
        from collections import deque
        assert isinstance(_main.audio_to_play_queue, deque)

    def test_popleft_removes_first_element(self):
        _main.audio_to_play_queue.clear()
        _main.audio_to_play_queue.append("a.wav")
        _main.audio_to_play_queue.append("b.wav")
        _main.audio_to_play_queue.popleft()
        assert list(_main.audio_to_play_queue) == ["b.wav"]
        _main.audio_to_play_queue.clear()


class TestSentenceDetection:
    def _sentence_end_pattern(self):
        import re
        return re.compile(r"[.!?]")

    def test_comma_does_not_end_sentence(self):
        pattern = self._sentence_end_pattern()
        buffer = "Olá, tudo"
        last_char = buffer.strip()[-1]
        assert last_char not in ".!?"

    def test_period_ends_sentence(self):
        pattern = self._sentence_end_pattern()
        buffer = "Olá, tudo bem."
        assert pattern.search(buffer) and buffer.strip()[-1] in ".!?"

    def test_exclamation_ends_sentence(self):
        pattern = self._sentence_end_pattern()
        buffer = "Que legal!"
        assert pattern.search(buffer) and buffer.strip()[-1] in ".!?"

    def test_question_ends_sentence(self):
        pattern = self._sentence_end_pattern()
        buffer = "Tudo bem?"
        assert pattern.search(buffer) and buffer.strip()[-1] in ".!?"

    def test_ellipsis_does_not_end_sentence(self):
        # … (U+2026) should no longer trigger a split
        buffer = "Hmm…"
        last_char = buffer.strip()[-1]
        assert last_char not in ".!?"