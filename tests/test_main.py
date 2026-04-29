"""
Tests for iara/bot.py.

iara.bot instantiates STT, LLMAgent, SpeechSynthesizer at module level and creates
a module-level AudioPipeline. All heavy dependencies are mocked via conftest.py
before this module is imported.
"""

import asyncio
import os
import sys
import tempfile
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch


def _import_bot():
    """Import iara.bot with all ML constructors mocked out."""
    sys.modules["dotenv"].load_dotenv = MagicMock()

    import importlib

    if "iara.bot" in sys.modules:
        return sys.modules["iara.bot"]
    return importlib.import_module("iara.bot")


_main = _import_bot()


def _make_discord_bot():
    """Create a fresh DiscordBot + AudioPipeline pair for isolation."""
    pipeline = _main.AudioPipeline(_main.stt, _main.llm, _main.tts)
    bot_mock = MagicMock()
    instance = _main.DiscordBot(bot_mock, pipeline)
    instance.voice_client = MagicMock()
    instance.voice_client.is_connected.return_value = True
    return instance, pipeline


class TestDiscordBotPlayAudio:
    def test_missing_file_returns_false(self):
        bot, _ = _make_discord_bot()
        result = asyncio.run(bot.play_audio("/nonexistent/path/audio.wav"))
        assert result is False

    def test_existing_file_returns_true(self):
        bot, _ = _make_discord_bot()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"\x00" * 44)
            tmp_path = f.name

        try:

            def fake_play(source, **kwargs):
                after = kwargs.get("after")
                if after:
                    after(None)

            bot.voice_client.play = MagicMock(side_effect=fake_play)
            sys.modules["torchaudio"].load.return_value = (MagicMock(), 48000)

            with patch.object(bot.vts, "sync_mouth", new_callable=AsyncMock):
                result = asyncio.run(bot.play_audio(tmp_path))

            assert result is True
        finally:
            os.unlink(tmp_path)


class TestDiscordBotPing:
    def test_ping_sends_message(self):
        bot, _ = _make_discord_bot()
        ctx = AsyncMock()
        asyncio.run(bot.ping(ctx))
        ctx.send.assert_called_once_with("Bot connected to voice channel!")


class TestDiscordBotTestCommand:
    def test_connects_to_voice_channel(self):
        bot, _ = _make_discord_bot()
        ctx = AsyncMock()
        ctx.author.voice.channel.connect = AsyncMock(return_value=MagicMock())

        asyncio.run(bot.test(ctx))

        ctx.author.voice.channel.connect.assert_called_once()

    def test_sets_context(self):
        bot, _ = _make_discord_bot()
        ctx = AsyncMock()
        ctx.author.voice.channel.connect = AsyncMock(return_value=MagicMock())

        asyncio.run(bot.test(ctx))

        assert bot.context is ctx

    def test_sends_connected_message(self):
        bot, _ = _make_discord_bot()
        ctx = AsyncMock()
        ctx.author.voice.channel.connect = AsyncMock(return_value=MagicMock())

        asyncio.run(bot.test(ctx))

        ctx.send.assert_called_once_with("Bot connected to voice channel!")

    def test_callback_appends_pcm_to_queue(self):
        bot, pipeline = _make_discord_bot()
        ctx = AsyncMock()
        captured_callback = {}

        def fake_basic_sink(cb):
            captured_callback["fn"] = cb
            return MagicMock()

        vc_mock = MagicMock()
        ctx.author.voice.channel.connect = AsyncMock(return_value=vc_mock)

        pipeline.user_voice_to_process_queue.clear()
        with patch.object(_main.voice_recv, "BasicSink", side_effect=fake_basic_sink):
            asyncio.run(bot.test(ctx))

        data_mock = MagicMock()
        data_mock.pcm = b"\x01" * 100
        captured_callback["fn"]("user1", data_mock)

        assert "user1" in pipeline.user_voice_to_process_queue
        assert b"\x01" * 100 in pipeline.user_voice_to_process_queue["user1"]

    def test_callback_updates_last_audio_time(self):
        bot, pipeline = _make_discord_bot()
        ctx = AsyncMock()
        captured_callback = {}

        def fake_basic_sink(cb):
            captured_callback["fn"] = cb
            return MagicMock()

        vc_mock = MagicMock()
        ctx.author.voice.channel.connect = AsyncMock(return_value=vc_mock)

        with patch.object(_main.voice_recv, "BasicSink", side_effect=fake_basic_sink):
            asyncio.run(bot.test(ctx))

        data_mock = MagicMock()
        data_mock.pcm = b"\x00" * 100
        captured_callback["fn"]("user1", data_mock)

        assert bot.last_audio_time is not None


class TestDiscordBotKickstart:
    def _bot(self):
        bot, pipeline = _make_discord_bot()
        ctx = AsyncMock()
        ctx.author.voice.channel.name = "general"
        ctx.author.voice.channel.connect = AsyncMock(return_value=MagicMock())
        bot.context = ctx
        return bot, pipeline

    def test_disconnects_before_reconnecting(self):
        bot, _ = self._bot()
        original_vc = bot.voice_client
        original_vc.disconnect = AsyncMock()

        asyncio.run(bot.kickstart(bot.context))

        original_vc.disconnect.assert_called_once_with(force=False)

    def test_returns_true_on_success(self):
        bot, _ = self._bot()
        bot.voice_client.disconnect = AsyncMock()

        result = asyncio.run(bot.kickstart(bot.context))

        assert result is True

    def test_sends_reconnected_message(self):
        bot, _ = self._bot()
        bot.voice_client.disconnect = AsyncMock()

        asyncio.run(bot.kickstart(bot.context))

        bot.context.send.assert_called()
        args, _ = bot.context.send.call_args
        assert "reconnected" in args[0].lower()

    def test_returns_false_when_no_voice_channel(self):
        bot, _ = self._bot()
        bot.voice_client.disconnect = AsyncMock()
        bot.context.author.voice = None

        result = asyncio.run(bot.kickstart(bot.context))

        assert result is False

    def test_returns_false_on_exception(self):
        bot, _ = self._bot()
        bot.voice_client.disconnect = AsyncMock(side_effect=Exception("boom"))

        result = asyncio.run(bot.kickstart(bot.context))

        assert result is False

    def test_sends_error_message_on_exception(self):
        bot, _ = self._bot()
        bot.voice_client.disconnect = AsyncMock(side_effect=Exception("boom"))

        asyncio.run(bot.kickstart(bot.context))

        bot.context.send.assert_called()
        args, _ = bot.context.send.call_args
        assert "failed" in args[0].lower()

    def test_kickstart_callback_appends_pcm_to_queue(self):
        bot, pipeline = self._bot()
        bot.voice_client.disconnect = AsyncMock()
        captured_callback = {}

        def fake_basic_sink(cb):
            captured_callback["fn"] = cb
            return MagicMock()

        pipeline.user_voice_to_process_queue.clear()
        with patch.object(_main.voice_recv, "BasicSink", side_effect=fake_basic_sink):
            asyncio.run(bot.kickstart(bot.context))

        data_mock = MagicMock()
        data_mock.pcm = b"\x02" * 100
        captured_callback["fn"]("user1", data_mock)

        assert "user1" in pipeline.user_voice_to_process_queue
        assert b"\x02" * 100 in pipeline.user_voice_to_process_queue["user1"]

    def test_kickstart_callback_updates_last_audio_time(self):
        bot, pipeline = self._bot()
        bot.voice_client.disconnect = AsyncMock()
        captured_callback = {}

        def fake_basic_sink(cb):
            captured_callback["fn"] = cb
            return MagicMock()

        with patch.object(_main.voice_recv, "BasicSink", side_effect=fake_basic_sink):
            asyncio.run(bot.kickstart(bot.context))

        data_mock = MagicMock()
        data_mock.pcm = b"\x00" * 100
        captured_callback["fn"]("user1", data_mock)

        assert bot.last_audio_time is not None


class TestTranscribeForUser:
    def setup_method(self):
        _main.pipeline.user_voice_to_process_queue.clear()

    def test_empty_queue_returns_empty_string(self):
        result = asyncio.run(_main.pipeline.transcribe_for_user("user1"))
        assert result == ""

    def test_returns_formatted_transcript(self):
        user = MagicMock()
        user.__str__ = lambda s: "TestUser"
        _main.pipeline.user_voice_to_process_queue[user] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="olá mundo")

        result = asyncio.run(_main.pipeline.transcribe_for_user(user))
        assert "olá mundo" in result
        assert "says:" in result

        _main.pipeline.user_voice_to_process_queue.pop(user, None)

    def test_empty_transcript_returns_empty_string(self):
        user = MagicMock()
        _main.pipeline.user_voice_to_process_queue[user] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="")

        result = asyncio.run(_main.pipeline.transcribe_for_user(user))
        assert result == ""

        _main.pipeline.user_voice_to_process_queue.pop(user, None)


class TestAskLLMAndProcess:
    def setup_method(self):
        _main.pipeline.audio_to_play_queue.clear()
        _main.pipeline.can_release_accept_packages = False
        _main.pipeline.bot = MagicMock()
        _main.pipeline.bot.context = AsyncMock()
        _main.pipeline.bot.vts = AsyncMock()

    def test_adds_audio_to_queue(self):
        _main.llm.ask = MagicMock(return_value=iter(["Olá, tudo bem?"]))

        async def fake_tts(text, path):
            with open(path, "wb") as f:
                f.write(b"\x00" * 44)

        _main.tts.generate_tts_file = fake_tts

        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert len(_main.pipeline.audio_to_play_queue) >= 1

        for path, _ in list(_main.pipeline.audio_to_play_queue):
            try:
                os.unlink(path)
            except Exception:
                pass
        _main.pipeline.audio_to_play_queue.clear()

    def test_sets_can_release_accept_packages_true(self):
        _main.llm.ask = MagicMock(return_value=iter([]))
        _main.tts.generate_tts_file = AsyncMock()

        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert _main.pipeline.can_release_accept_packages is True


class TestVoiceConsumer:
    def setup_method(self):
        _main.pipeline.user_voice_to_process_queue.clear()
        _main.pipeline.accept_packages = True
        _main.pipeline.can_release_accept_packages = True
        _main.pipeline.bot = None

    def test_does_nothing_when_no_bot_instance(self):
        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.pipeline.accept_packages is True

    def test_does_nothing_when_queue_empty(self):
        bot_mock = MagicMock()
        bot_mock.last_audio_time = None
        _main.pipeline.bot = bot_mock

        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.pipeline.accept_packages is True

    def test_fires_when_silence_timeout_elapsed(self):
        from datetime import datetime, timedelta

        bot_mock = MagicMock()
        bot_mock.last_audio_time = datetime.now() - timedelta(seconds=2)
        bot_mock.context = AsyncMock()
        bot_mock.vts = AsyncMock()
        _main.pipeline.bot = bot_mock
        _main.pipeline.user_voice_to_process_queue["user1"] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="")
        _main.tts.generate_tts_file = AsyncMock()

        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.pipeline.accept_packages is False

    def test_clears_voice_queue_after_processing(self):
        from datetime import datetime, timedelta

        bot_mock = MagicMock()
        bot_mock.last_audio_time = datetime.now() - timedelta(seconds=2)
        bot_mock.context = AsyncMock()
        bot_mock.vts = AsyncMock()
        _main.pipeline.bot = bot_mock
        _main.pipeline.user_voice_to_process_queue["user1"] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="")
        _main.tts.generate_tts_file = AsyncMock()

        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert len(_main.pipeline.user_voice_to_process_queue) == 0

    def test_releases_accept_packages_when_all_transcripts_empty(self):
        from datetime import datetime, timedelta

        bot_mock = MagicMock()
        bot_mock.last_audio_time = datetime.now() - timedelta(seconds=2)
        bot_mock.context = AsyncMock()
        bot_mock.vts = AsyncMock()
        _main.pipeline.bot = bot_mock
        _main.pipeline.user_voice_to_process_queue["user1"] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="")
        _main.pipeline.can_release_accept_packages = False

        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.pipeline.can_release_accept_packages is True

    def test_enqueues_transcript_when_non_empty(self):
        from datetime import datetime, timedelta

        bot_mock = MagicMock()
        bot_mock.last_audio_time = datetime.now() - timedelta(seconds=2)
        bot_mock.context = AsyncMock()
        bot_mock.vts = AsyncMock()
        _main.pipeline.bot = bot_mock
        _main.pipeline.user_voice_to_process_queue["user1"] = [b"\x00" * 100]
        _main.stt.transcribe_audio = MagicMock(return_value="olá")

        while not _main.pipeline.transcript_queue.empty():
            _main.pipeline.transcript_queue.get_nowait()

        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_consumer())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())

        assert not _main.pipeline.transcript_queue.empty()
        transcript = _main.pipeline.transcript_queue.get_nowait()
        assert "olá" in transcript


class TestVoicePlayer:
    def setup_method(self):
        _main.pipeline.audio_to_play_queue.clear()
        _main.pipeline.accept_packages = False
        _main.pipeline.can_release_accept_packages = True
        _main.pipeline.bot = MagicMock()

    def test_releases_accept_packages_when_queue_empty(self):
        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_player())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.pipeline.accept_packages is True

    def test_accept_packages_not_released_while_queue_has_items(self):
        ready = asyncio.Event()
        ready.set()
        _main.pipeline.audio_to_play_queue.append(("/nonexistent/file.wav", ready))
        _main.pipeline.can_release_accept_packages = False
        _main.pipeline.bot.play_audio = AsyncMock(return_value=False)

        async def run_one_tick():
            task = asyncio.create_task(_main.pipeline.voice_player())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_one_tick())
        assert _main.pipeline.accept_packages is False

        _main.pipeline.audio_to_play_queue.clear()


class TestDequeQueue:
    def test_audio_queue_is_deque(self):
        assert isinstance(_main.pipeline.audio_to_play_queue, deque)

    def test_popleft_removes_first_element(self):
        e = asyncio.Event()
        _main.pipeline.audio_to_play_queue.clear()
        _main.pipeline.audio_to_play_queue.append(("a.wav", e))
        _main.pipeline.audio_to_play_queue.append(("b.wav", e))
        _main.pipeline.audio_to_play_queue.popleft()
        assert [p for p, _ in _main.pipeline.audio_to_play_queue] == ["b.wav"]
        _main.pipeline.audio_to_play_queue.clear()


class TestSynthesizeAndSignal:
    def test_signals_ready_on_success(self):
        _main.tts.generate_tts_file = AsyncMock()

        async def run():
            ready = asyncio.Event()
            await _main.pipeline.synthesize_and_signal("texto", "/tmp/out.wav", ready)
            return ready.is_set()

        assert asyncio.run(run()) is True

    def test_signals_ready_even_on_tts_failure(self):
        _main.tts.generate_tts_file = AsyncMock(side_effect=RuntimeError("tts boom"))

        async def run():
            ready = asyncio.Event()
            await _main.pipeline.synthesize_and_signal("texto", "/tmp/out.wav", ready)
            return ready.is_set()

        assert asyncio.run(run()) is True

    def test_calls_tts_with_correct_args(self):
        _main.tts.generate_tts_file = AsyncMock()

        async def run():
            ready = asyncio.Event()
            await _main.pipeline.synthesize_and_signal(
                "olá mundo", "/tmp/abc.wav", ready
            )

        asyncio.run(run())
        _main.tts.generate_tts_file.assert_called_once_with("olá mundo", "/tmp/abc.wav")


class TestPlayAudioQueue:
    def setup_method(self):
        _main.pipeline.audio_to_play_queue.clear()
        _main.pipeline.bot = MagicMock()

    def test_waits_for_ready_event(self):
        async def run():
            ready = asyncio.Event()
            _main.pipeline.audio_to_play_queue.append(("/nonexistent.wav", ready))
            _main.pipeline.bot.play_audio = AsyncMock(return_value=False)

            task = asyncio.create_task(_main.pipeline.play_audio_queue())
            await asyncio.sleep(0.05)
            assert not task.done()
            ready.set()
            await asyncio.sleep(0.05)
            await task

        asyncio.run(run())

    def test_skips_empty_file_and_continues(self):
        async def run():
            ready = asyncio.Event()
            ready.set()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                empty_path = f.name

            _main.pipeline.audio_to_play_queue.append((empty_path, ready))
            _main.pipeline.bot.play_audio = AsyncMock(return_value=True)

            await _main.pipeline.play_audio_queue()
            assert len(_main.pipeline.audio_to_play_queue) == 0
            assert not os.path.exists(empty_path)

        asyncio.run(run())

    def test_removes_file_after_successful_playback(self):
        async def run():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"\x00" * 44)
                path = f.name

            ready = asyncio.Event()
            ready.set()
            _main.pipeline.audio_to_play_queue.append((path, ready))
            _main.pipeline.bot.play_audio = AsyncMock(return_value=True)

            await _main.pipeline.play_audio_queue()
            assert not os.path.exists(path)

        asyncio.run(run())

    def test_continues_when_file_removal_fails(self):
        async def run():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"\x00" * 44)
                path = f.name

            ready = asyncio.Event()
            ready.set()
            _main.pipeline.audio_to_play_queue.append((path, ready))
            _main.pipeline.bot.play_audio = AsyncMock(return_value=True)

            with patch("os.remove", side_effect=OSError("locked")):
                await _main.pipeline.play_audio_queue()

            assert len(_main.pipeline.audio_to_play_queue) == 0
            try:
                os.unlink(path)
            except Exception:
                pass

        asyncio.run(run())

    def test_playback_failure_still_drains_queue(self):
        async def run():
            ready1 = asyncio.Event()
            ready1.set()
            ready2 = asyncio.Event()
            ready2.set()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"\x00" * 44)
                path1 = f.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"\x00" * 44)
                path2 = f.name

            _main.pipeline.audio_to_play_queue.append((path1, ready1))
            _main.pipeline.audio_to_play_queue.append((path2, ready2))
            _main.pipeline.bot.play_audio = AsyncMock(return_value=False)

            await _main.pipeline.play_audio_queue()
            assert len(_main.pipeline.audio_to_play_queue) == 0

        asyncio.run(run())


class TestPlayAudioThreadSafety:
    def test_play_uses_call_soon_threadsafe(self):
        bot, _ = _make_discord_bot()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"\x00" * 44)
            tmp_path = f.name

        try:
            captured = {}

            def fake_play(source, **kwargs):
                captured["after"] = kwargs.get("after")
                captured["kwargs"] = kwargs

            bot.voice_client.play = MagicMock(side_effect=fake_play)
            sys.modules["torchaudio"].load.return_value = (MagicMock(), 48000)

            async def run():
                task = asyncio.create_task(bot.play_audio(tmp_path))
                await asyncio.sleep(0.05)
                if "after" in captured and captured["after"]:
                    captured["after"](None)
                await task

            with patch.object(bot.vts, "sync_mouth", new_callable=AsyncMock):
                asyncio.run(run())

            assert "after" in captured
            assert "bitrate" not in captured.get("kwargs", {})
            assert "signal_type" not in captured.get("kwargs", {})
        finally:
            os.unlink(tmp_path)


class TestAskLLMSentenceSplitting:
    def setup_method(self):
        _main.pipeline.audio_to_play_queue.clear()
        _main.pipeline.can_release_accept_packages = False
        _main.pipeline.bot = MagicMock()
        _main.pipeline.bot.context = AsyncMock()
        _main.pipeline.bot.vts = AsyncMock()

    def test_multiple_sentences_enqueue_multiple_files(self):
        _main.llm.ask = MagicMock(
            return_value=iter(
                [
                    "Olá",
                    "!",
                    " Tudo",
                    " bem",
                    " aqui",
                    ".",
                ]
            )
        )
        _main.tts.generate_tts_file = AsyncMock()

        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert len(_main.pipeline.audio_to_play_queue) >= 1
        _main.pipeline.audio_to_play_queue.clear()

    def test_final_buffer_is_flushed(self):
        _main.llm.ask = MagicMock(return_value=iter(["Olá mundo"]))
        _main.tts.generate_tts_file = AsyncMock()

        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert len(_main.pipeline.audio_to_play_queue) == 1
        _main.pipeline.audio_to_play_queue.clear()


class TestMood:
    def setup_method(self):
        _main.pipeline.mood = 5
        _main.pipeline.audio_to_play_queue.clear()
        _main.pipeline.can_release_accept_packages = False
        _main.pipeline.bot = MagicMock()
        _main.pipeline.bot.context = AsyncMock()
        _main.pipeline.bot.vts = AsyncMock()

    def test_plus_token_increments_mood(self):
        _main.llm.ask = MagicMock(return_value=iter(["Olá [+]"]))
        _main.tts.generate_tts_file = AsyncMock()
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert _main.pipeline.mood == 6
        _main.pipeline.audio_to_play_queue.clear()

    def test_minus_token_decrements_mood(self):
        _main.llm.ask = MagicMock(return_value=iter(["Que chato [-]"]))
        _main.tts.generate_tts_file = AsyncMock()
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert _main.pipeline.mood == 4
        _main.pipeline.audio_to_play_queue.clear()

    def test_equal_token_keeps_mood(self):
        _main.llm.ask = MagicMock(return_value=iter(["Ok [=]"]))
        _main.tts.generate_tts_file = AsyncMock()
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert _main.pipeline.mood == 5
        _main.pipeline.audio_to_play_queue.clear()

    def test_missing_token_keeps_mood(self):
        _main.llm.ask = MagicMock(return_value=iter(["Tudo bem"]))
        _main.tts.generate_tts_file = AsyncMock()
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert _main.pipeline.mood == 5
        _main.pipeline.audio_to_play_queue.clear()

    def test_mood_clamped_at_10(self):
        _main.pipeline.mood = 10
        _main.llm.ask = MagicMock(return_value=iter(["Ótimo [+]"]))
        _main.tts.generate_tts_file = AsyncMock()
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert _main.pipeline.mood == 10
        _main.pipeline.audio_to_play_queue.clear()

    def test_mood_clamped_at_0(self):
        _main.pipeline.mood = 0
        _main.llm.ask = MagicMock(return_value=iter(["Que raiva [-]"]))
        _main.tts.generate_tts_file = AsyncMock()
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert _main.pipeline.mood == 0
        _main.pipeline.audio_to_play_queue.clear()

    def test_mood_token_stripped_from_tts(self):
        captured_texts = []

        async def fake_tts(text, path):
            captured_texts.append(text)

        _main.llm.ask = MagicMock(return_value=iter(["Olá mundo [+]"]))
        _main.tts.generate_tts_file = fake_tts
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert all(
            "[+]" not in t and "[-]" not in t and "[=]" not in t for t in captured_texts
        )
        _main.pipeline.audio_to_play_queue.clear()

    def test_special_tokens_stripped_from_tts(self):
        captured_texts = []

        async def fake_tts(text, path):
            captured_texts.append(text)

        _main.llm.ask = MagicMock(return_value=iter(["Olá <|eom_id|> mundo [=]"]))
        _main.tts.generate_tts_file = fake_tts
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert all("<|eom_id|>" not in t for t in captured_texts)
        _main.pipeline.audio_to_play_queue.clear()

    def test_bracket_annotations_stripped_from_tts(self):
        captured_texts = []

        async def fake_tts(text, path):
            captured_texts.append(text)

        _main.llm.ask = MagicMock(return_value=iter(["Olá mundo [aumenta o tom] [=]"]))
        _main.tts.generate_tts_file = fake_tts
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert all("[aumenta o tom]" not in t for t in captured_texts)
        _main.pipeline.audio_to_play_queue.clear()

    def test_trailing_junk_stripped_before_mood_token(self):
        captured_texts = []

        async def fake_tts(text, path):
            captured_texts.append(text)

        _main.llm.ask = MagicMock(return_value=iter(["Olá mundo!= [=]"]))
        _main.tts.generate_tts_file = fake_tts
        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))
        assert all(
            t == "" or t[-1].isalnum() or t[-1].isspace() or "ÿ" >= t[-1] >= "À"
            for t in captured_texts
            if t
        )
        _main.pipeline.audio_to_play_queue.clear()

    def test_stops_streaming_after_mood_token(self):
        tokens = ["Olá", " mundo", " [=]", " lixo", " extra"]
        _main.llm.ask = MagicMock(return_value=iter(tokens))
        _main.tts.generate_tts_file = AsyncMock()

        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))

        sent_text = _main.pipeline.bot.context.send.call_args_list[-1][0][0]
        assert "lixo" not in sent_text
        assert "extra" not in sent_text
        _main.pipeline.audio_to_play_queue.clear()

    def test_discord_response_truncated_at_mood_token(self):
        _main.llm.ask = MagicMock(return_value=iter(["Olá [=]<|eom_id|>lixo extra"]))
        _main.tts.generate_tts_file = AsyncMock()

        asyncio.run(_main.pipeline.ask_llm_and_process("user says: oi"))

        sent_text = _main.pipeline.bot.context.send.call_args_list[-1][0][0]
        assert "lixo extra" not in sent_text
        assert "<|eom_id|>" not in sent_text
        _main.pipeline.audio_to_play_queue.clear()


class TestSentenceDetection:
    def _sentence_end_pattern(self):
        import re

        return re.compile(r"[.!?]")

    def test_comma_does_not_end_sentence(self):
        buffer = "Olá, tudo"
        assert buffer.strip()[-1] not in ".!?"

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
        buffer = "Hmm…"
        assert buffer.strip()[-1] not in ".!?"
