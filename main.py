import asyncio
import os
import re
import tempfile
from datetime import datetime, timedelta
from io import BytesIO

import discord
import torch
import torchaudio
from discord.ext import commands, voice_recv
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydub import AudioSegment

from LLMAgent import LLMAgent
from SpeechSynthesizer import SpeechSynthesizer
from VTubeStudioTalk import VTubeStudioTalk

load_dotenv()

torch.set_num_threads(8)

class DiscordBot(commands.Cog):
    def __init__(self, bot: commands.Bot, llm: LLMAgent, speech_synth: SpeechSynthesizer):
        """Initializes the bot with dependencies."""
        self.can_release_processing = True
        self.vts_talk = None
        self.bot = bot
        self.llm = llm
        self.speech_synth = speech_synth
        self.audio_queue = []
        self.pcm_buffers = {}  # Maps user_id to list of PCM audio chunks
        self.last_audio_time = None  # Timestamp of the last audio packet
        self.processing_task = None  # Task for processing audio
        self.loop = None
        self.context = None  # Stores Discord context for sending messages
        self.voice_client = None  # Stores the voice client for playback
        self.synthesis_task_queue = asyncio.Queue()  # Queue for audio synthesis tasks
        self.model = WhisperModel(
            "turbo",
            cpu_threads=4,
            num_workers=5,
            device="auto",
        )

    def transcribe_audio(self, user: discord.User) -> str:
        """Transcribes audio chunks for a user into text."""
        pcm_chunks = self.pcm_buffers.get(user, [])
        if not pcm_chunks:
            return ""

        # Combine PCM chunks and convert to mono WAV
        raw_pcm = b"".join(pcm_chunks)
        audio = AudioSegment(
            data=raw_pcm,
            sample_width=2,
            frame_rate=48000,
            channels=2,
        ).set_channels(1)  # Convert to mono for Whisper

        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # Transcribe audio using Whisper model
        segments, _ = self.model.transcribe(
            wav_buffer,
            language="pt",
            vad_filter=True,
            hotwords="IAra, Vtuber"
        )
        return "".join(seg.text for seg in segments).strip()

    async def play_audio(self, audio_file: str) -> bool:
        """Plays an audio file in the voice channel and syncs with VTube Studio."""
        if not self.voice_client or not self.voice_client.is_connected():
            self.voice_client = await self.context.author.voice.channel.connect()

        if not os.path.exists(audio_file):
            await self.context.send(f"Audio file not found: {audio_file}")
            return False

        try:
            audio_source = discord.FFmpegPCMAudio(audio_file, executable="ffmpeg")
            playback_done = asyncio.Event()

            def after_playback(error):
                if error:
                    print(f"Playback error: {error}")
                self.loop.call_soon_threadsafe(playback_done.set)

            if not self.voice_client.is_playing():
                waveform, sample_rate = torchaudio.load(audio_file)
                self.vts_talk.run_sync_mouth(waveform, sample_rate)
                self.voice_client.play(audio_source, after=after_playback)
                await playback_done.wait()
                return True
            else:
                await self.context.send("Bot is already playing audio. Adding to queue.")
                return False
        except Exception as e:
            print(f"Error playing {audio_file}: {e}")
            return False

    async def process_audio(self) -> None:
        """Processes buffered audio and triggers LLM responses."""
        while self.last_audio_time is not None:
            current_time = datetime.now()
            if (
                self.last_audio_time
                and self.pcm_buffers
                and current_time - self.last_audio_time >= timedelta(seconds=1)
                and not self.llm.is_processing
            ):
                self.llm.is_processing = True
                self.can_release_processing = False

                # Transcribe audio for all users in parallel
                async def transcribe_for_user(user):
                    transcript = self.transcribe_audio(user)
                    return f"{user} says: {transcript}\n" if transcript else ""

                tasks = [transcribe_for_user(user) for user in self.pcm_buffers]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                transcript = ""
                for result in results:
                    if isinstance(result, str) and result:
                        transcript += result
                        print(f"[TRANSCRIPT] {result}")
                        await self.context.send(f"**{result.strip()}**")

                if not transcript:
                    self.can_release_processing = True

                self.pcm_buffers.clear()

                if transcript:
                    await self.ask_llm_and_process(transcript)

            await asyncio.sleep(0.1)  # Avoid busy-waiting

    async def ask_llm_and_process(self, transcript: str) -> None:
        """Sends transcript to LLM and processes the response into audio."""
        await self.context.send(
            f"===============================================\n"
            f"**Full Transcript**:\n{transcript}\n"
            f"==============================================="
        )

        full_response = ""
        buffer = ""
        sentence_end = re.compile(r"[.!?,…]")  # Sentence-ending punctuation

        for token in self.llm.ask(transcript):
            full_response += token
            print(token, end="", flush=True)
            buffer += token

            # Process complete sentences
            if (
                sentence_end.search(buffer)
                and len(buffer.strip().split()) >= 3
                and buffer.strip()[-1] in ".!?,…"
            ):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    await self.speech_synth.generate_tts_file(buffer.strip(), tmp_file.name)
                    self.audio_queue.append(tmp_file.name)
                buffer = ""

        # Process remaining text
        if buffer.strip():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                await self.speech_synth.generate_tts_file(buffer.strip(), tmp_file.name)
                self.audio_queue.append(tmp_file.name)
                print(f"Added to queue (final buffer): {tmp_file.name}")

        print(full_response)
        await self.context.send(
            f"===============================================\n"
            f"**Full Response**:\n**IAra says:** {full_response}\n"
            f"==============================================="
        )
        self.can_release_processing = True

    async def process_voice(self, data: voice_recv.VoiceData, user: discord.User) -> None:
        """Handles incoming voice data from a user."""
        print(f"Voice data received from user: {user}")
        if user not in self.pcm_buffers:
            self.pcm_buffers[user] = []
        self.pcm_buffers[user].append(data.pcm)
        self.last_audio_time = datetime.now()

        if self.processing_task is None or self.processing_task.done():
            self.processing_task = self.loop.create_task(self.process_audio())

    @commands.command()
    async def test(self, ctx: commands.Context) -> None:
        """Connects the bot to the user's voice channel for testing."""
        def callback(user, data: voice_recv.VoiceData):
            if not self.llm.is_processing:
                self.loop.create_task(self.process_voice(data, user))

        self.context = ctx
        self.voice_client = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        self.voice_client.listen(voice_recv.BasicSink(callback))
        await ctx.send("Bot connected to voice channel!")

    async def play_audio_queue(self) -> bool:
        """Plays all audio files in the queue."""
        while self.audio_queue:
            audio_file = self.audio_queue[0]
            success = await self.play_audio(audio_file)
            if success:
                try:
                    os.remove(audio_file)
                    print(f"Removed file: {audio_file}")
                except Exception as e:
                    print(f"Error removing file {audio_file}: {e}")
                self.audio_queue.pop(0)
            else:
                print("Playback failed, stopping to avoid infinite loop")
                break
        return len(self.audio_queue) == 0

    async def play_audio_loop(self) -> None:
        """Continuously checks and plays audio from the queue."""
        while True:
            if len(self.audio_queue) >= 2:
                await self.play_audio_queue()
            elif (self.audio_queue
                  and self.synthesis_task_queue.empty()
                  and self.can_release_processing):
                await self.play_audio_queue()
            elif (self.synthesis_task_queue.empty()
                    and self.can_release_processing):
                    self.llm.is_processing = False
            await asyncio.sleep(0.1)

    async def cog_load(self) -> None:
        """Initializes the cog with the event loop and audio queue."""
        self.loop = asyncio.get_running_loop()
        self.vts_talk = VTubeStudioTalk(self.loop)
        await self.vts_talk.connect()
        self.audio_queue = []
        self.audio_task = self.loop.create_task(self.play_audio_loop())


# Initialize bot and dependencies
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())


@bot.event
async def on_ready() -> None:
    """Logs when the bot is ready."""
    print(f"Logged in as {bot.user} ({bot.user.id})")


@bot.event
async def setup_hook() -> None:
    """Sets up the bot by adding the DiscordBot cog."""
    await bot.add_cog(DiscordBot(bot, llm, synth))


# Instantiate dependencies
synth = SpeechSynthesizer()
llm = LLMAgent()

# Run the bot
with llm.getChatSession():
    bot.run(os.getenv("DISCORD_TOKEN"))