import asyncio
import os
import tempfile

from datetime import datetime, timedelta
from io import BytesIO
import re

from TTS.api import TTS
from dotenv import load_dotenv
import discord
from discord.ext import commands, voice_recv
from discord.ext.commands.context import Context
from faster_whisper import WhisperModel
from pydub import AudioSegment

from LLMAgent import LLMAgent

load_dotenv()



class SpeechSynthesizer:
    def __init__(self):
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to('cuda')

    async def generateTtsFile(self, text: str, audio_file_name):
        if not text:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.tts.tts_to_file(
            text=text,
            speaker_wav='../TTS/coquitts/agatha_voice.wav',
            # speaker=self.tts.speakers[0],
            file_path=audio_file_name,
            split_sentences=False,
            language='pt'
        ))


class DiscordBot(commands.Cog):
    def __init__(self, bot, llm: LLMAgent, speech_synth: SpeechSynthesizer):
        self.canReleaseIsProcessing = True
        self.audio_task = None
        self.llm = llm
        self.synth = speech_synth
        self.bot = bot
        self.pcm_buffers = {}  # user_id -> list of PCM bytes
        self.last_audio_time = None  # Timestamp of last audio packet from any user
        self.processing_task = None  # Single task for processing audio
        self.loop = None
        self.context = None  # Store context for sending messages
        self.voice_client = None  # Store voice client for playback
        self.audioQueue=[]
        self.model = WhisperModel("turbo",
                                  cpu_threads = 4,
                                  num_workers = 5,
                                  device='auto')

    def transcribe_audio(self, user) -> str:
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

        segments, _ = self.model.transcribe(wav_buffer,
                                            language='pt', # pt or en
                                            vad_filter=True,
                                            hotwords="IAra, Vtuber")
        transcript = ''.join([seg.text for seg in segments])
        return transcript.strip()

    async def playAudio(self, audio_file_name: str):
        if not self.voice_client or not self.voice_client.is_connected():
            self.voice_client = await self.context.author.voice.channel.connect()

        try:
            if not os.path.exists(audio_file_name):
                self.loop.create_task(self.context.send(f"Arquivo de áudio não encontrado: {audio_file_name}"))
                return False  # Return False to indicate failure

            # Create audio source
            audio_source = discord.FFmpegPCMAudio(
                audio_file_name,
                executable="ffmpeg"
            )

            # Create an event to signal when playback is complete
            playback_done = asyncio.Event()

            def after_playback(error):
                if error:
                    print(f"Erro durante a reprodução: {error}")
                self.loop.call_soon_threadsafe(playback_done.set)

            # Play the audio
            if not self.voice_client.is_playing():
                self.voice_client.play(audio_source, after=after_playback)
                await playback_done.wait()  # Wait until playback is complete
                return True  # Playback successful
            else:
                self.loop.create_task(self.context.send("O bot já está tocando um áudio. Adicionando à fila."))
                return False  # Playback not started (already playing)
        except Exception as e:
            print(f"Erro ao tocar {audio_file_name}: {e}")
            return False  # Playback failed

    async def process_audio(self):
        while self.last_audio_time is not None:  # Continue until stopped
            current_time = datetime.now()
            if (self.last_audio_time is not None and
                    self.pcm_buffers and
                    current_time - self.last_audio_time >= timedelta(seconds=1) and
                    not self.llm.is_processing):
                self.llm.is_processing = True
                self.canReleaseIsProcessing = False

                transcript = ""

                # Create a list of transcription tasks for all users
                async def transcribe_for_user(user):
                    user_transcript = self.transcribe_audio(user)
                    if user_transcript:
                        return f"{user} says: {user_transcript}\n"
                    return ""

                # Run transcription tasks in parallel
                tasks = [transcribe_for_user(user) for user in list(self.pcm_buffers.keys())]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Combine results into transcript
                for result in results:
                    if isinstance(result, str) and result:
                        transcript += result
                        print(f"[TRANSCRIÇÃO] {result}")
                        if self.context:
                            self.loop.create_task(self.context.send(f"**{result.strip()}**"))
                    else:
                        self.canReleaseIsProcessing = True
                self.pcm_buffers.clear()

                if self.context and transcript:
                    self.loop.create_task(self.askLLMAndProcessIt(transcript))
            await asyncio.sleep(0.1)  # Sleep briefly to avoid busy-waiting

    async def askLLMAndProcessIt(self, transcript):
        self.loop.create_task(self.context.send(f"===============================================\n "
                                f"**full transcript**: \n"
                                f"{transcript}\n"
                                f"==============================================="))
        full_response = ""
        llm_response_stream = self.llm.ask(transcript)
        buffer = ""
        sentence_end = re.compile(r"[.!?,…]")  # pontuação que finaliza a frase
        for token in llm_response_stream:
            full_response += token
            print(token, end="", flush=True)
            buffer += token

            # Processa quando detecta final de frase
            if sentence_end.search(buffer) and len(buffer.strip().split()) >= 3 and buffer.strip()[-1] in " .!?,…":
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                    await self.synth.generateTtsFile(buffer.strip(), tmp_audio_file.name)
                    self.audioQueue.append(tmp_audio_file.name)
                buffer = ""
        # Processa o resto da string
        if buffer.strip():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                await self.synth.generateTtsFile(buffer.strip(), tmp_audio_file.name)
                self.audioQueue.append(tmp_audio_file.name)
                print(f"Adicionado à fila (buffer final): {tmp_audio_file.name}")
        print(full_response)
        self.loop.create_task(self.context.send(f"===============================================\n "
                                f"**full Response**: \n"
                                f"**IAra says:** {full_response}\n"
                                f"==============================================="))
        self.canReleaseIsProcessing = True

    async def processVoice(self, data, user):
        print('voice_data received, user ' + str(user))
        if user not in self.pcm_buffers:
            self.pcm_buffers[user] = []
        self.pcm_buffers[user].append(data.pcm)
        self.last_audio_time = datetime.now()
        # Start processing task if not already running
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = self.loop.create_task(self.process_audio())

    @commands.command()
    async def test(self, context: Context):
        def callback(user, data: voice_recv.VoiceData):
            if not self.llm.is_processing:
                self.loop.create_task(self.processVoice(data, user))

        self.context = context  # Store the context

        self.voice_client = await self.context.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        self.voice_client.listen(voice_recv.BasicSink(callback))
        self.loop.create_task(self.context.send("Bot conectado ao canal de voz!"))

    async def playAudioQueue(self):
        while self.audioQueue:  # Process all files in the queue
            audio_file = self.audioQueue[0]  # Get the first file
            success = await self.playAudio(audio_file)
            if success:
                try:
                    os.remove(audio_file)  # Remove file only if playback was successful
                    print(f"Arquivo removido: {audio_file}")
                except Exception as e:
                    print(f"Erro ao remover arquivo {audio_file}: {e}")
                self.audioQueue.pop(0)  # Remove the file from the queueaudioQueue
            else:
                # If playback failed, break to avoid infinite loop
                print("playback failed, break to avoid infinite loop")
                break
        return len(self.audioQueue) == 0  # Return True if queue is empty

    async def play_audio_loop(self):
        while True:
            if self.audioQueue:
                await self.playAudioQueue()
            else:
                if self.canReleaseIsProcessing:
                    self.llm.is_processing = False
            await asyncio.sleep(0.1)

    async def cog_load(self):
        self.loop = asyncio.get_running_loop()
        self.audioQueue = []
        self.audio_task = self.loop.create_task(self.play_audio_loop())


bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} ({bot.user.id})')

@bot.event
async def setup_hook():
    await bot.add_cog(DiscordBot(bot, llm, synth))



synth = SpeechSynthesizer()
llm = LLMAgent()

with llm.getChatSession():
    bot.run(os.getenv('DISCORD_TOKEN'))