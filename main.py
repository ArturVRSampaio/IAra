import asyncio
import os
import tempfile
from datetime import datetime, timedelta
import re

import discord
import torchaudio
from discord.ext import commands, voice_recv
from dotenv import load_dotenv

from LLMAgent import LLMAgent
from STT import STT
from SpeechSynthesizer import SpeechSynthesizer
from VTubeStudioTalk import VTubeStudioTalk

load_dotenv()

discord_bot_instance = None
user_voice_to_process_queue={}
audio_to_play_queue = []
accept_packages = True
can_release_accept_packages=True
stt = STT()
llm = LLMAgent()
tts = SpeechSynthesizer()

class DiscordBot(commands.Cog):
    def __init__(self, bot):
        self.vts = VTubeStudioTalk()
        self.last_audio_time = None
        self.bot = bot
        self.voice_client = None
        self.context = None
        global discord_bot_instance
        discord_bot_instance = self  # Store the instance globally

    async def play_audio(self, audio_file: str) -> bool:
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return False

        audio_source = discord.FFmpegPCMAudio(audio_file, executable="ffmpeg")
        playback_done = asyncio.Event()

        waveform, sample_rate = torchaudio.load(audio_file)

        mouth_task = asyncio.create_task(self.vts.sync_mouth(waveform, sample_rate))
        asyncio.create_task(
            asyncio.to_thread(
                self.voice_client.play,
                source=audio_source,
                signal_type="voice",
                application="lowdelay",
                after=lambda e: playback_done.set())
        )

        await playback_done.wait()

        mouth_task.cancel()
        return True

    @commands.command()
    async def ping(self, ctx: commands.Context) -> None:
        await ctx.send("Bot connected to voice channel!")

    @commands.command()
    async def test(self, ctx: commands.Context) -> None:
        """Connects the bot to the user's voice channel for testing."""
        def callback(user, data: voice_recv.VoiceData):
            self.last_audio_time = datetime.now()
            print(f"Data received from user: {user} at {self.last_audio_time}")
            if user not in user_voice_to_process_queue:
                user_voice_to_process_queue[user] = []
            user_voice_to_process_queue[user].append(data.pcm)


        self.context = ctx
        self.voice_client = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        self.voice_client.listen(voice_recv.BasicSink(callback))
        await ctx.send("Bot connected to voice channel!")

with llm.getChatSession():
    async def ask_llm_and_process(transcript: str) -> None:
        global can_release_accept_packages
        """Sends transcript to LLM and processes the response into audio."""
        asyncio.create_task(discord_bot_instance.context.send(
            f"===============================================\n"
            f"**Full Transcript**:\n{transcript}\n"
            f"==============================================="
        ))

        full_response = ""
        buffer = ""
        sentence_end = re.compile(r"[.!?,…]")  # Sentence-ending punctuation

        for token in llm.ask(transcript):
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
                    await tts.generate_tts_file(buffer.strip(), tmp_file.name)
                    audio_to_play_queue.append(tmp_file.name)
                buffer = ""

        # Process remaining text
        if buffer.strip():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                await tts.generate_tts_file(buffer.strip(), tmp_file.name)
                audio_to_play_queue.append(tmp_file.name)
                print(f"Added to queue (final buffer): {tmp_file.name}")

        print(full_response)
        asyncio.create_task(discord_bot_instance.context.send(
            f"===============================================\n"
            f"**Full Response**:\n**IAra says:** {full_response}\n"
            f"==============================================="
        ))
        can_release_accept_packages = True



    async def transcribe_for_user(user):
        pcm_chunks = user_voice_to_process_queue.get(user, [])
        transcript = stt.transcribe_audio(pcm_chunks)
        return f"{user} says: {transcript}\n" if transcript else ""

    async def voice_consumer():
        global accept_packages
        global can_release_accept_packages
        while True:
            current_time = datetime.now()
            if (discord_bot_instance and discord_bot_instance.last_audio_time and
                    (current_time - discord_bot_instance.last_audio_time >= timedelta(seconds=1) and
                     accept_packages and user_voice_to_process_queue)):
                accept_packages = False
                can_release_accept_packages = False
                print("Stopped accepting packages.")

                tasks = [transcribe_for_user(user) for user in user_voice_to_process_queue]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                print("tasks release")
                full_transcript = ""
                for result in results:
                    if isinstance(result, str) and result:
                        full_transcript += result
                        print(f"[TRANSCRIPT] {result}")
                        asyncio.create_task(discord_bot_instance.context.send(f"**{result.strip()}**"))

                if not full_transcript:
                    can_release_accept_packages = True

                user_voice_to_process_queue.clear()

                if full_transcript:
                    await ask_llm_and_process(full_transcript)
            await asyncio.sleep(0.1)

    async def play_audio_queue():
        """Plays all audio files in the queue."""
        while len(audio_to_play_queue) > 0:
            audio_file = audio_to_play_queue[0]
            success = await discord_bot_instance.play_audio(audio_file)

            if success:
                try:
                    os.remove(audio_file)
                    print(f"Removed file: {audio_file}")
                except Exception as e:
                    print(f"Error removing file {audio_file}: {e}")
                audio_to_play_queue.pop(0)
            else:
                print("Playback failed, stopping to avoid infinite loop")
                break

    async def voice_player():
        global accept_packages
        global can_release_accept_packages
        while True:
            if audio_to_play_queue:
                await play_audio_queue()
            elif can_release_accept_packages and not accept_packages:
                accept_packages = True
                user_voice_to_process_queue.clear()
            await asyncio.sleep(0.1)

    async def main():
        bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

        @bot.event
        async def on_ready():
            """Logs when the bot is ready."""
            print(f"Logged in as {bot.user} ({bot.user.id})")

        @bot.event
        async def setup_hook():
            """Sets up the bot by adding the DiscordBot cog."""
            await bot.add_cog(DiscordBot(bot))

        # Start the voice consumer and player as background tasks
        asyncio.create_task(voice_consumer())
        asyncio.create_task(voice_player())

        # Run the bot
        await bot.start(os.getenv("DISCORD_TOKEN"))

    if __name__ == "__main__":
        asyncio.run(main())