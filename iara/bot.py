from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import discord
import torchaudio
from discord.ext import commands, voice_recv
from dotenv import load_dotenv

from iara.llm import LLMAgent
from iara.stt import STT
from iara.tts import SpeechSynthesizer
from iara.vtube import VTubeStudioTalk

load_dotenv()

stt = STT()
llm = LLMAgent()
tts = SpeechSynthesizer()

_MOOD_RE = re.compile(r'\[([+\-=])\]')
_MOOD_DELTA: dict[str, int] = {'+': 1, '-': -1, '=': 0}
_TRAILING_JUNK_RE = re.compile(r'[^\w\sÀ-ÿ]+$')


class AudioPipeline:
    def __init__(self, stt: STT, llm: LLMAgent, tts: SpeechSynthesizer) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.bot: DiscordBot | None = None
        self.mood: int = 5
        self.user_voice_to_process_queue: dict[Any, list[bytes]] = {}
        self.audio_to_play_queue: deque[tuple[str, asyncio.Event]] = deque()
        self.transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self.accept_packages: bool = True
        self.can_release_accept_packages: bool = True

    async def transcribe_for_user(self, user: Any) -> str:
        pcm_chunks = self.user_voice_to_process_queue.get(user, [])
        transcript = self.stt.transcribe_audio(pcm_chunks)
        return f"{user} says: {transcript}\n" if transcript else ""

    async def synthesize_and_signal(self, text: str, path: str, ready: asyncio.Event) -> None:
        try:
            await self.tts.generate_tts_file(text, path)
        except Exception as e:
            print(f"TTS synthesis failed: {e}")
        finally:
            ready.set()

    async def ask_llm_and_process(self, transcript: str) -> None:
        asyncio.create_task(self.bot.context.send(  # type: ignore[union-attr]
            f"===============================================\n"
            f"**Full Transcript**:\n{transcript}\n"
            f"==============================================="
        ))

        full_response = ""
        buffer = ""
        sentence_end = re.compile(r"[.!?]")

        _special_token_re = re.compile(r'<\|[^|]*\|>')
        _bracket_re = re.compile(r'\[[^\]]*\]')

        def enqueue_synthesis(text: str) -> None:
            text = _special_token_re.sub('', text)
            text = _bracket_re.sub('', text)
            text = _TRAILING_JUNK_RE.sub('', text).strip()
            if not text:
                return
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.close()
            ready = asyncio.Event()
            self.audio_to_play_queue.append((tmp_file.name, ready))
            asyncio.create_task(self.synthesize_and_signal(text, tmp_file.name, ready))

        for token in self.llm.ask(transcript):
            full_response += token
            print(token, end="", flush=True)
            buffer += token

            if _MOOD_RE.search(full_response):
                pre_mood = _MOOD_RE.split(buffer)[0]
                if pre_mood.strip():
                    enqueue_synthesis(pre_mood.strip())
                buffer = ""
                break

            if (
                sentence_end.search(buffer)
                and len(buffer.strip().split()) >= 3
                and buffer.strip()[-1] in ".!?"
            ):
                enqueue_synthesis(buffer.strip())
                buffer = ""

        if buffer.strip():
            enqueue_synthesis(buffer.strip())

        mood_match = _MOOD_RE.search(full_response)
        if mood_match:
            self.mood = max(0, min(10, self.mood + _MOOD_DELTA[mood_match.group(1)]))
            asyncio.create_task(self.bot.vts.trigger_mood_expression(self.mood))  # type: ignore[union-attr]
            clean_response = _TRAILING_JUNK_RE.sub('', full_response[:mood_match.start()].strip())
        else:
            clean_response = _special_token_re.sub('', full_response).strip()

        print(f"\n[MOOD: {self.mood}/10]")
        asyncio.create_task(self.bot.context.send(  # type: ignore[union-attr]
            f"===============================================\n"
            f"**Full Response**:\n**IAra says:** {clean_response}\n"
            f"**Mood:** {self.mood}/10\n"
            f"==============================================="
        ))
        self.can_release_accept_packages = True

    async def play_audio_queue(self) -> None:
        while len(self.audio_to_play_queue) > 0:
            audio_file, ready = self.audio_to_play_queue[0]
            await ready.wait()

            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                print(f"Skipping empty/failed audio file: {audio_file}")
                try:
                    os.remove(audio_file)
                except Exception:
                    pass
                self.audio_to_play_queue.popleft()
                continue

            success = await self.bot.play_audio(audio_file)  # type: ignore[union-attr]

            if success:
                try:
                    os.remove(audio_file)
                    print(f"Removed file: {audio_file}")
                except Exception as e:
                    print(f"Error removing file {audio_file}: {e}")
                self.audio_to_play_queue.popleft()
            else:
                print("Playback failed, stopping to avoid infinite loop")
                break

    async def voice_consumer(self) -> None:
        while True:
            current_time = datetime.now()
            if (self.bot and self.bot.last_audio_time and
                    (current_time - self.bot.last_audio_time >= timedelta(seconds=1) and
                     self.accept_packages and self.user_voice_to_process_queue)):
                self.accept_packages = False
                self.can_release_accept_packages = False
                print("Stopped accepting packages.")
                asyncio.create_task(self.bot.vts.execute_animation("IAra_Thinking"))

                tasks = [self.transcribe_for_user(user) for user in self.user_voice_to_process_queue]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                full_transcript = ""
                for result in results:
                    if isinstance(result, str) and result:
                        full_transcript += result
                        print(f"[TRANSCRIPT] {result}")
                        asyncio.create_task(self.bot.context.send(f"**{result.strip()}**"))  # type: ignore[union-attr]

                if not full_transcript:
                    self.can_release_accept_packages = True

                self.user_voice_to_process_queue.clear()

                if full_transcript:
                    await self.transcript_queue.put(full_transcript)
            await asyncio.sleep(0.1)

    async def session_worker(self) -> None:
        max_turns = int(os.getenv("IARA_MAX_SESSION_TURNS", "15"))
        while True:
            turns = 0
            print(f"[SESSION] Opening fresh LLM session (mood {self.mood}/10)")
            with self.llm.getChatSession(self.mood):
                while turns < max_turns:
                    transcript = await self.transcript_queue.get()
                    try:
                        await self.ask_llm_and_process(transcript)
                        turns += 1
                    except Exception as e:
                        print(f"Error processing transcript: {e}")
                        self.can_release_accept_packages = True
            print(f"[SESSION] {turns} turns reached — resetting to preserve system prompt")

    async def voice_player(self) -> None:
        while True:
            if self.audio_to_play_queue:
                await self.play_audio_queue()
            elif self.can_release_accept_packages and not self.accept_packages:
                self.accept_packages = True
                self.user_voice_to_process_queue.clear()
            await asyncio.sleep(0.1)


class DiscordBot(commands.Cog):
    def __init__(self, bot: commands.Bot, pipeline: AudioPipeline) -> None:
        self.vts: VTubeStudioTalk = VTubeStudioTalk()
        self.last_audio_time: datetime | None = None
        self.bot = bot
        self.voice_client: voice_recv.VoiceRecvClient | None = None
        self.context: commands.Context | None = None
        self.pipeline = pipeline
        pipeline.bot = self

    def _voice_callback(self, user: Any, data: voice_recv.VoiceData) -> None:
        self.last_audio_time = datetime.now()
        print(f"Data received from user: {user} at {self.last_audio_time}")
        if user not in self.pipeline.user_voice_to_process_queue:
            self.pipeline.user_voice_to_process_queue[user] = []
        self.pipeline.user_voice_to_process_queue[user].append(data.pcm)

    async def play_audio(self, audio_file: str) -> bool:
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return False

        audio_source = discord.FFmpegPCMAudio(audio_file, executable="ffmpeg")
        loop = asyncio.get_running_loop()
        playback_done = asyncio.Event()

        waveform, sample_rate = torchaudio.load(audio_file)

        self.voice_client.play(  # type: ignore[union-attr]
            audio_source,
            after=lambda e: loop.call_soon_threadsafe(playback_done.set)
        )
        mouth_task = asyncio.create_task(self.vts.sync_mouth(waveform, sample_rate))

        await playback_done.wait()

        mouth_task.cancel()
        return True

    @commands.command()
    async def ping(self, ctx: commands.Context) -> None:
        await ctx.send("Bot connected to voice channel!")

    @commands.command()
    async def test(self, ctx: commands.Context) -> None:
        self.context = ctx
        self.voice_client = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)  # type: ignore[union-attr]
        self.voice_client.listen(voice_recv.BasicSink(self._voice_callback))
        await ctx.send("Bot connected to voice channel!")

    @commands.command()
    async def kickstart(self, ctx: commands.Context) -> bool:
        self.context = ctx
        try:
            if self.voice_client and self.voice_client.is_connected():
                print("Disconnecting from voice channel to reconnect...")
                await self.voice_client.disconnect(force=False)

            if not self.context or not self.context.author.voice or not self.context.author.voice.channel:  # type: ignore[union-attr]
                print("Cannot reconnect: No valid voice channel or context found.")
                return False

            print(f"Reconnecting to voice channel: {self.context.author.voice.channel.name}")  # type: ignore[union-attr]
            self.voice_client = await self.context.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)  # type: ignore[union-attr]
            self.voice_client.listen(voice_recv.BasicSink(self._voice_callback))
            print("Voice listener reinitialized successfully.")
            await self.context.send("Bot reconnected to voice channel!")
            return True

        except Exception as e:
            print(f"Error during reconnection: {e}")
            await self.context.send(f"Failed to reconnect to voice channel: {str(e)}")
            return False


pipeline = AudioPipeline(stt, llm, tts)


async def main() -> None:
    bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

    @bot.event
    async def on_ready() -> None:
        assert bot.user is not None
        print(f"Logged in as {bot.user} ({bot.user.id})")

    @bot.event
    async def setup_hook() -> None:
        await bot.add_cog(DiscordBot(bot, pipeline))

    asyncio.create_task(pipeline.voice_consumer())
    asyncio.create_task(pipeline.voice_player())
    asyncio.create_task(pipeline.session_worker())

    token = os.getenv("DISCORD_TOKEN")
    assert token is not None, "DISCORD_TOKEN environment variable is not set"
    await bot.start(token)
