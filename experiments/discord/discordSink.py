import asyncio
import os
from io import BytesIO
from dotenv import load_dotenv
import discord
from discord.ext import commands, voice_recv
from discord.ext.commands.context import Context
from faster_whisper import WhisperModel
from pydub import AudioSegment

load_dotenv()


class Testing(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.audio_buffers = {}       # user_id -> asyncio.Queue
        self.processing_tasks = {}    # user_id -> Task
        self.pcm_buffers = {}         # user_id -> list of PCM bytes
        self.loop = None
        self.model = WhisperModel("base")
        self.last_audio_time = {}     # user_id -> last audio packet timestamp
        self.ctx = None               # Store context for sending messages
        self.voice_client = None      # Store voice client for playback

    async def process_audio(self, user):
        queue = self.audio_buffers[user]
        self.pcm_buffers[user] = []
        self.last_audio_time[user] = asyncio.get_event_loop().time()

        while True:
            try:
                # Wait for audio data with a 2-second timeout
                data = await asyncio.wait_for(queue.get(), timeout=2.0)
                if data is None:
                    break
                self.pcm_buffers[user].append(data.pcm)
                self.last_audio_time[user] = asyncio.get_event_loop().time()
            except asyncio.TimeoutError:
                # If 2 seconds have passed without new audio, transcribe
                if self.pcm_buffers[user]:
                    await self.transcribe_audio(user)
                    self.pcm_buffers[user] = []  # Clear buffer after transcription
                continue

    async def transcribe_audio(self, user):
        pcm_chunks = self.pcm_buffers.get(user, [])
        if not pcm_chunks:
            return

        raw_pcm = b''.join(pcm_chunks)

        audio = AudioSegment(
            data=raw_pcm,
            sample_width=2,
            frame_rate=48000,
            channels=2
        ).set_channels(1)  # Whisper works better with mono

        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        segments, _ = self.model.transcribe(wav_buffer)
        transcript = ''.join([seg.text for seg in segments])
        if transcript.strip():  # Only send non-empty transcripts
            print(f"[TRANSCRIÇÃO] {user}:\n{transcript}\n")
            if self.ctx:
                await self.ctx.send(f"**{user}**: {transcript}")

    @commands.command()
    async def test(self, ctx: Context):
        self.ctx = ctx  # Store the context
        def callback(user, data: voice_recv.VoiceData):
            if user not in self.audio_buffers:
                self.audio_buffers[user] = asyncio.Queue()
                self.processing_tasks[user] = self.loop.create_task(
                    self.process_audio(user)
                )
            self.audio_buffers[user].put_nowait(data)

        self.voice_client = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        self.voice_client.listen(voice_recv.BasicSink(callback))
        await ctx.send("Bot conectado ao canal de voz! Transcrevendo fala após pausas de 2 segundos.")

    @commands.command()
    async def playtest(self, ctx: Context):
        if not ctx.author.voice:
            await ctx.send("Você precisa estar em um canal de voz para usar este comando!")
            return

        # Connect to voice channel if not already connected
        if not self.voice_client or not self.voice_client.is_connected():
            self.voice_client = await ctx.author.voice.channel.connect()
            self.ctx = ctx  # Store context for sending messages
            await ctx.send("Bot conectado ao canal de voz!")

        # Ensure FFmpeg is available
        try:
            audio_file = "lastVoice.wav"
            if not os.path.exists(audio_file):
                await ctx.send("Arquivo lastVoice.wav não encontrado!")
                return

            # Create audio source
            audio_source = discord.FFmpegPCMAudio(
                audio_file,
                executable="ffmpeg"  # Ensure FFmpeg is in PATH or specify full path
            )

            # Play the audio
            if not self.voice_client.is_playing():
                self.voice_client.play(audio_source)
                await ctx.send("Tocando test.wav no canal de voz!")
            else:
                await ctx.send("O bot já está tocando um áudio. Aguarde até terminar!")
        except Exception as e:
            await ctx.send(f"Erro ao tocar o áudio: {str(e)}")
            print(f"Erro ao tocar test.wav: {e}")

    @commands.command()
    async def stop(self, ctx: Context):
        # Signal all audio processing to stop
        for queue in self.audio_buffers.values():
            queue.put_nowait(None)

        # Wait for processing tasks to complete
        for task in self.processing_tasks.values():
            await task

        # Stop any playing audio
        if self.voice_client and self.voice_client.is_playing():
            self.voice_client.stop()

        # Clean up
        self.audio_buffers.clear()
        self.processing_tasks.clear()
        self.pcm_buffers.clear()
        self.last_audio_time.clear()
        self.ctx = None

        if self.voice_client and self.voice_client.is_connected():
            await self.voice_client.disconnect()
            self.voice_client = None
            await ctx.send("Bot desconectado do canal de voz.")
        else:
            await ctx.send("Bot não está em um canal de voz.")

    async def cog_load(self):
        self.loop = asyncio.get_running_loop()

bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} ({bot.user.id})')

@bot.event
async def setup_hook():
    await bot.add_cog(Testing(bot))

bot.run(os.getenv('DISCORD_TOKEN'))