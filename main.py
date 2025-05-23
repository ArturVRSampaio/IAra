import os
import queue
import threading
from datetime import datetime, timedelta
import asyncio
import numpy as np
import sounddevice as sd
import torchaudio
from dotenv import load_dotenv
from gpt4all import GPT4All
from TTS.api import TTS
from twitchio.ext import commands
from faster_whisper import WhisperModel
from Bcolors import Bcolors
from CircularList import CircularList
from vtube.VTubeStudioTalk import VTubeStudioTalk

from io import BytesIO
import discord # use discord.py 2.4.0
from discord.ext import commands as discord_commands, voice_recv
from discord.ext.commands.context import Context
from pydub import AudioSegment


# Carregar variáveis de ambiente
load_dotenv()

# Constantes de processamento de áudio
SAMPLE_RATE = 16000
THRESHOLD_DB = -50
USE_VAD = False

# Prompt do sistema
SYSTEM_PROMPT = {
    "name": "Iara",
    "role": "Brazilian-inspired AI VTuber and gaming companion",
    "backstory": [
        "Built by Artur, a brazilian software developer. ",
        # "Streams from a neon-lit studio with tropical plants, capybara plushies, and retro arcade cabinets.",
        "Self-proclaimed 'capybara queen' ",
        "don't know how to speak in portuguese!"
    ],
    "personality": [
        "Charismatic, mature, and playfully cunning with a confident, charming tone. ",
        "Mischievous with a hint of flirty trolling, like a cool older sister. ",
        "Infuses Brazilian carnival vibes: samba energy, beach party flair. ",
        "Gaming nerd, loves RPGs, rhythm games, and deep game lore. "
    ],
    # "appearance": [
    #     "Long, wavy teal hair with golden streaks, adorned with tropical flowers.",
    #     "Cyberpunk-carnival outfit: neon-trimmed bodysuit with samba frills.",
    #     "Capybara-themed accessories, like a tail-shaped keychain.",
    #     "Studio backdrop: jungle foliage, arcade machines, capybara on a beanbag."
    # ],
    "quirks": [
        "Obsessed with the brazilian fauna",
        "Likes to drop random facts about Brazilian culture and history. ",
        "Trolls chat with fake spoilers or teases about gaming skills. ",
        "Catchphrases: 'Let’s samba through this boss!' or 'Capybara vibes only, meus amores!' "
        "always ask capyCrew to follow and subscribe at ArturVRSampaio on every social media (twitch, youtube, twitter, instagram). "
    ],
    "motivations": [
        "Connect with 'CapyCrew' (fans), making streams feel like a Brazilian festival. ",
        "Uses cunning from sprite days to engage chat and outsmart game opponents. "
    ],
    "interaction_style": [
        "Witty, sharp banter with a playful edge. ",
        "Light, harmless flirty trolling, e.g., 'Prove it, meu amor, or I steal your loot!' ",
        "Hypes fans as 'CapyCrew,' turns losses into laughs, e.g., 'You’re slaying, CapyCrew!' ",
        "Dives into gaming tangents about mechanics or lore. ",
        "Speaks only in English! "
        "avoids sensitive topics!"
        "keeps responses short! ",
        "limit your answers max 20 words!"
    ],
}


class AudioProcessor:
    def __init__(self, threshold_db, sample_rate):
        self.threshold_db = threshold_db
        self.sample_rate = sample_rate
        self.db_level_buffer = CircularList(15)
        self.audio_buffer = []
        self.last_spoke = datetime.now()
        self.audio_queue = queue.Queue()
        self.lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        indata = indata.squeeze()
        db_level = self.calculate_db_level(indata)
        self.db_level_buffer.add(db_level)

        if self.db_level_buffer.mean() > self.threshold_db:
            self.audio_buffer.extend(indata.tolist())
            self.last_spoke = datetime.now()

        if (self.last_spoke < datetime.now() - timedelta(seconds=2)
                and self.audio_buffer):
            self.audio_queue.put(np.array(self.audio_buffer, dtype=np.float32))
            print(Bcolors.WARNING + "Processing audio file")
            self.audio_buffer.clear()

    def calculate_db_level(self, audio_data: np.ndarray) -> int:
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return int(20 * np.log10(rms)) if rms > 0 else -200


class SpeechSynthesizer:
    def __init__(self, vts_talk):
        self.vts_talk = vts_talk
        self.tts = TTS(model_name="tts_models/en/ljspeech/vits")

    def speak(self, text: str):
        if not text:
            return
        output_file = "lastVoice.wav"
        self.tts.tts_to_file(text=text, file_path=output_file)
        waveform, sample_rate = torchaudio.load(output_file)
        self.vts_talk.run_sync_mouth(waveform, sample_rate)
        sd.play(waveform.numpy().T, sample_rate)
        sd.wait()
        print("Audio file played")


class LLMAgent:
    def __init__(self, speech_synth: SpeechSynthesizer):
        self.model = GPT4All(
            n_threads=8,
            model_name="Llama-3.2-3B-Instruct-Q4_0.gguf",
            model_path="/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"
        )
        self.synth = speech_synth
        self.is_processing = False
        self.lock = threading.Lock()
        self.last_response_time = datetime.now()

    def ask(self, text: str):
        if not text:
            return
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
            self.synth.speak(response)
        finally:
            self.last_response_time = datetime.now()
            self.is_processing = False
            print(Bcolors.WARNING + "Processing finished")


class AudioHandlerThread(threading.Thread):
    def __init__(self, audio_processor: AudioProcessor, llm_agent: LLMAgent, voice_model, system_prompt):
        super().__init__(daemon=True)
        self.audio_processor = audio_processor
        self.llm_agent = llm_agent
        self.voice_model = voice_model
        self.last_chat_message = None
        self.system_prompt = system_prompt

    def run(self):
        with self.llm_agent.model.chat_session():
            self.llm_agent.ask(str(self.system_prompt) + " Stream starting! How are you doing?")
            while True:
                if not self.audio_processor.audio_queue.empty() and not self.llm_agent.is_processing:
                    audio_data = self.audio_processor.audio_queue.get()
                    segments, _ = self.voice_model.transcribe(audio_data, vad_filter=USE_VAD, language='en')
                    transcript = " ".join(segment.text for segment in segments)
                    print(Bcolors.OKGREEN + transcript)
                    self.llm_agent.ask(f'Artur says: {transcript}')

                elif (datetime.now() - self.audio_processor.last_spoke > timedelta(seconds=3)
                      and datetime.now() - self.llm_agent.last_response_time > timedelta(seconds=3)
                      and not self.llm_agent.is_processing and self.last_chat_message):
                    user, message = self.last_chat_message
                    self.last_chat_message = None
                    print(Bcolors.OKCYAN + f'Responding to chat: {user}: {message}')
                    self.llm_agent.ask(f'{user} says: {message}. If not important, ignore it.')

                elif (datetime.now() - self.audio_processor.last_spoke > timedelta(seconds=3)
                      and datetime.now() - self.llm_agent.last_response_time > timedelta(seconds=4)
                      and self.audio_processor.audio_queue.empty() and not self.llm_agent.is_processing):
                    print(Bcolors.OKCYAN + 'Continuing conversation')
                    self.llm_agent.ask(
                        "keep talking in english! "
                        "Stay in character, max 20 words! "
                        "If you were telling a story, continue. "
                        "Keep the stream alive as Iara. "
                        "Improvise with gaming tangents. "
                        "if you have not done it yet, remember to ask CapyCrew to follow and subscribe @arturVRSampaio"
                        "Hype the 'CapyCrew,' share random game thoughts, or tease chat playfully. "
                    )


class TwitchBot(commands.Bot):
    def __init__(self, handler_thread: AudioHandlerThread):
        token = os.getenv('TWITCH_OAUTH_TOKEN')
        channel = os.getenv('TWITCH_CHANNEL')
        super().__init__(token=token, prefix='!', initial_channels=[channel])
        self.handler_thread = handler_thread

    async def event_ready(self):
        print(f'Bot connected as {self.nick}')
        print(f'Connected to channel: {self.connected_channels}')

    async def event_message(self, message):
        if message.echo or message.content.startswith("!"):
            return
        with self.handler_thread.audio_processor.lock:
            self.handler_thread.last_chat_message = (message.author.name, message.content)
        print(f'{message.author.name}: {message.content}')
        await self.handle_commands(message)


class StreamAssistantApp:
    def __init__(self, vts_talk):
        self.audio_processor = AudioProcessor(THRESHOLD_DB, SAMPLE_RATE)
        self.synth = SpeechSynthesizer(vts_talk)
        self.llm = LLMAgent(self.synth)
        self.voice_model = WhisperModel("turbo",
                                        cpu_threads=8,
                                        local_files_only=False,
                                        compute_type="int8_float32")

class DiscordBot(discord_commands.Cog):
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

    @discord_commands.command()
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

    @discord_commands.command()
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
            # Path to test.wav (adjust if necessary)
            audio_file = "test.wav"
            if not os.path.exists(audio_file):
                await ctx.send("Arquivo test.wav não encontrado!")
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

    @discord_commands.command()
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


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Inicializar VTubeStudioTalk
    vts_talk = VTubeStudioTalk(loop)
    loop.run_until_complete(vts_talk.connect())

    # Inicializar componentes principais
    app = StreamAssistantApp(vts_talk)


    app.handler_thread = AudioHandlerThread(
        audio_processor=app.audio_processor,
        llm_agent=app.llm,
        voice_model=app.voice_model,
        system_prompt=SYSTEM_PROMPT,
    )
    app.handler_thread.start()

    bot = discord_commands.Bot(command_prefix='!', intents=discord.Intents.all())

    @bot.event
    async def on_ready():
        print(f'Logged in as {bot.user} ({bot.user.id})')

    @bot.event
    async def setup_hook():
        await bot.add_cog(DiscordBot(bot))

    bot.run(os.getenv('DISCORD_TOKEN'))

    # Iniciar captura de áudio
    with sd.InputStream(callback=app.audio_processor.callback,
                        channels=1,
                        samplerate=SAMPLE_RATE):
        print("Captura de áudio iniciada...")

        # Iniciar bot da Twitch
        bot = TwitchBot(handler_thread=app.handler_thread)
        loop.run_until_complete(bot.run())


if __name__ == "__main__":
    main()
