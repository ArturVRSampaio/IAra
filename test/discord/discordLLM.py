import asyncio
import os
import threading
from datetime import datetime, timedelta
from io import BytesIO
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
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to('cpu')

    def generateTtsFile(self, text: str):
        if not text:
            return
        output_file = "lastVoice.wav"

        # Generate the TTS audio
        self.tts.tts_to_file(
            text=text,
            language="pt-br", #pt-br or en
            speaker=self.tts.speakers[2],
            file_path=output_file
        )

        print("Audio file saved")


class LLMAgent:
    def __init__(self):
        self.model = GPT4All(
            n_threads=8,
            model_name="Llama-3.2-1B-Instruct-Q4_0.gguf",
            model_path="/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/",
            device='kompute'
        )
        self.is_processing = False
        self.lock = threading.Lock()
        self.last_response_time = datetime.now()

    async def ask(self, text: str) -> str:
        text = ("your name is iara. you are in a discord conversation with other people. "
                "here it is the description of their speech. "
                "respond as if you were someone listening to all this \n ") + text
        if not text:
            return ""
        with self.lock:
            self.is_processing = True
            print(Bcolors.WARNING + "Processing started")
        try:
            response = await asyncio.to_thread(self.model.generate,
                                               text,
                                               max_tokens=100,
                                               n_batch=16,
                                               temp=0.8,
                                               top_p=0.6,
                                               top_k=1)
            print(Bcolors.OKBLUE + response)
            return str(response)
        except:
            self.last_response_time = datetime.now()
            print(Bcolors.WARNING + "Processing finished")
            return ""


class DiscordBot(commands.Cog):
    def __init__(self, bot, llm: LLMAgent, speech_synth: SpeechSynthesizer):
        self.llm = llm
        self.synth = speech_synth
        self.bot = bot
        self.pcm_buffers = {}  # user_id -> list of PCM bytes
        self.last_audio_time = None  # Timestamp of last audio packet from any user
        self.processing_task = None  # Single task for processing audio
        self.loop = None
        self.model = WhisperModel("turbo",
                                  cpu_threads=8,
                                  local_files_only=False,
                                  compute_type="int8_float32")
        self.ctx = None  # Store context for sending messages
        self.voice_client = None  # Store voice client for playback

    async def transcribe_audio(self, user) -> str:
        print("transcribe audio")
        pcm_chunks = self.pcm_buffers.get(user, [])
        if not pcm_chunks:
            return ""

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

        segments, _ = self.model.transcribe(wav_buffer, language='pt') # pt or en
        transcript = ''.join([seg.text for seg in segments])
        return transcript.strip()

    async def playAudio(self):
        if not self.voice_client or not self.voice_client.is_connected():
            self.voice_client = await self.ctx.author.voice.channel.connect()

        try:
            audio_file = "lastVoice.wav"
            if not os.path.exists(audio_file):
                asyncio.create_task(self.ctx.send("Arquivo test.wav não encontrado!"))
                return

            # Create audio source
            audio_source = discord.FFmpegPCMAudio(
                audio_file,
                executable="ffmpeg"  # Ensure FFmpeg is in PATH or specify full path
            )

            # Create an event to signal when playback is complete
            playback_done = asyncio.Event()

            def after_playback(error):
                if error:
                    print(f"Erro durante a reprodução: {error}")
                asyncio.run_coroutine_threadsafe(playback_done.set(), self.loop)

            # Play the audio
            if not self.voice_client.is_playing():
                self.voice_client.play(audio_source, after=after_playback)
                asyncio.create_task(self.ctx.send("Tocando no canal de voz!"))
                await playback_done.wait()  # Wait until playback is complete
            else:
                await self.ctx.send("O bot já está tocando um áudio. Aguarde até terminar!")
        except Exception as e:
            await self.ctx.send(f"Erro ao tocar o áudio: {str(e)}")
            print(f"Erro ao tocar test.wav: {e}")

    async def process_audio(self):
        while self.last_audio_time is not None:  # Continue until stopped
            current_time = datetime.now()
            if (self.last_audio_time is not None and
                    self.pcm_buffers and
                    current_time - self.last_audio_time >= timedelta(seconds=1) and
                    not self.llm.is_processing):

                transcript=""
                for user in list(self.pcm_buffers.keys()):
                    user_transcript = await self.transcribe_audio(user)
                    if user_transcript:
                        transcript += "\n user: " + str(user) + " says: " + user_transcript
                        print(f"[TRANSCRIÇÃO] {user}:\n{transcript}\n")
                    if self.ctx:
                        await self.ctx.send(f"**{user}**: {transcript}")
                    self.pcm_buffers[user] = []

                if self.ctx and transcript:
                    asyncio.create_task(self.ctx.send(f"===============================================\n "
                                                      f"**full transcript**: \n"
                                                      f"{transcript}\n"
                                                      f"==============================================="))
                    llm_response = await self.llm.ask(transcript)
                    if llm_response:
                        asyncio.create_task(self.ctx.send(f"===============================================\n"
                                                          f"**IAra**: {llm_response}\n"
                                                          f"==============================================="))
                        self.synth.generateTtsFile(llm_response)
                        await self.playAudio()

                    self.llm.is_processing = False
            await asyncio.sleep(0.1)  # Sleep briefly to avoid busy-waiting

    @commands.command()
    async def test(self, ctx: Context):
        self.ctx = ctx  # Store the context

        def callback(user, data: voice_recv.VoiceData):
            print('voice_data received, user ' + str(user))
            if user not in self.pcm_buffers:
                self.pcm_buffers[user] = []
            self.pcm_buffers[user].append(data.pcm)
            self.last_audio_time = datetime.now()
            # Start processing task if not already running
            if self.processing_task is None or self.processing_task.done():
                self.processing_task = self.loop.create_task(self.process_audio())

        self.voice_client = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        self.voice_client.listen(voice_recv.BasicSink(callback))
        await ctx.send("Bot conectado ao canal de voz!")

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