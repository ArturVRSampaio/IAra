import asyncio
import os
import threading
from datetime import datetime
from io import BytesIO

import torchaudio
from TTS.api import TTS
from dotenv import load_dotenv
import discord
from discord.ext import commands, voice_recv
from discord.ext.commands.context import Context
from faster_whisper import WhisperModel
from gpt4all import GPT4All
from pydub import AudioSegment

from Bcolors import Bcolors

load_dotenv()

class SpeechSynthesizer:
    def __init__(self):
        self.tts = TTS(model_name="tts_models/en/ljspeech/vits")

    def generateTtsFile(self, text: str):
        if not text:
            return
        output_file = "lastVoice.wav"
        self.tts.tts_to_file(text=text, file_path=output_file)
        print("Audio file saved")


class LLMAgent:
    def __init__(self):
        self.model = GPT4All(
            n_threads=8,
            model_name="Llama-3.2-3B-Instruct-Q4_0.gguf",
            model_path="/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"
        )
        self.is_processing = False
        self.lock = threading.Lock()
        self.last_response_time = datetime.now()

    def ask(self, text: str) -> str:
        if not text:
            return ""
        with self.lock:
            self.is_processing = True
            print(Bcolors.WARNING + "Processing started")
        try:
            response = self.model.generate(text,
                                           max_tokens=200,
                                           n_batch=16,
                                           temp=0.8,
                                           top_p=0.6,
                                           top_k=1)
            print(Bcolors.OKBLUE + response)
            self.is_processing = False
            return str(response)
        except:
            self.last_response_time = datetime.now()
            self.is_processing = False
            print(Bcolors.WARNING + "Processing finished")
            return ""



class DiscordBot(commands.Cog):
    def __init__(self,
                 bot,
                 llm: LLMAgent,
                 speech_synth: SpeechSynthesizer):
        self.llm = llm
        self.synth = speech_synth
        self.bot = bot
        self.audio_buffers = {}       # user_id -> asyncio.Queue
        self.processing_tasks = {}    # user_id -> Task
        self.pcm_buffers = {}         # user_id -> list of PCM bytes
        self.loop = None
        self.model = WhisperModel("tiny",
                                        cpu_threads=8,
                                        local_files_only=False,
                                        compute_type="int8_float32")
        self.last_audio_time = {}     # user_id -> last audio packet timestamp
        self.ctx = None               # Store context for sending messages
        self.voice_client = None      # Store voice client for playback


    async def playAudio(self):
        if not self.voice_client or not self.voice_client.is_connected():
            self.voice_client = await self.ctx.author.voice.channel.connect()

        try:
            audio_file = "lastVoice.wav"
            if not os.path.exists(audio_file):
                await self.ctx.send("Arquivo test.wav não encontrado!")
                return

            # Create audio source
            audio_source = discord.FFmpegPCMAudio(
                audio_file,
                executable="ffmpeg"  # Ensure FFmpeg is in PATH or specify full path
            )

            # Play the audio
            if not self.voice_client.is_playing():
                self.voice_client.play(audio_source)
                await self.ctx.send("Tocando no canal de voz!")
            else:
                await self.ctx.send("O bot já está tocando um áudio. Aguarde até terminar!")
        except Exception as e:
            await self.ctx.send(f"Erro ao tocar o áudio: {str(e)}")
            print(f"Erro ao tocar test.wav: {e}")

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
                if self.pcm_buffers[user] and self.llm.is_processing == False:
                    transcript = await self.transcribe_audio(user)

                    if transcript:
                        print(f"[TRANSCRIÇÃO] {user}:\n{transcript}\n")
                        if self.ctx:
                            await self.ctx.send(f"**{user}**: {transcript}")
                            llm_response = self.llm.ask(transcript)
                            if llm_response:
                                await self.ctx.send(f"**IAra**: {llm_response}")
                                self.synth.generateTtsFile(llm_response)
                                await self.playAudio()

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
        return transcript.strip()



    @commands.command()
    async def test(self, ctx: Context):
        self.ctx = ctx  # Store the context
        def callback(user, data: voice_recv.VoiceData):
            print('voice_data received, user ' + str(user))
            if user not in self.audio_buffers:
                self.audio_buffers[user] = asyncio.Queue()
                self.processing_tasks[user] = self.loop.create_task(
                    self.process_audio(user)
                )
            self.audio_buffers[user].put_nowait(data)

        self.voice_client = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        self.voice_client.listen(voice_recv.BasicSink(callback))
        await ctx.send("Bot conectado ao canal de voz!")

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

llm = LLMAgent()
synth = SpeechSynthesizer()


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} ({bot.user.id})')

@bot.event
async def setup_hook():
    await bot.add_cog(DiscordBot(bot, llm, synth))

bot.run(os.getenv('DISCORD_TOKEN'))