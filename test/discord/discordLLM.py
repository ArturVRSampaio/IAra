import asyncio
import os

from datetime import datetime, timedelta
from io import BytesIO

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
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to('cuda')

    def generateTtsFile(self, text: str):
        if not text:
            return
        output_file = "lastVoice.wav"

        # Generate the TTS audio
        self.tts.tts_to_file(
            text=text,
            language="pt-br", #pt-br or en
            speaker=self.tts.speakers[0],
            file_path=output_file,
            split_sentences=True,
        )

        print("Audio file saved")




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
                                  cpu_threads = 4,
                                  num_workers = 5,
                                  device='cuda')
        self.ctx = None  # Store context for sending messages
        self.voice_client = None  # Store voice client for playback

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
                                            hotwords="IAra")
        transcript = ''.join([seg.text for seg in segments])
        return transcript.strip()

    async def playAudio(self):
        if not self.voice_client or not self.voice_client.is_connected():
            self.voice_client = await self.ctx.author.voice.channel.connect()

        try:
            audio_file = "lastVoice.wav"
            if not os.path.exists(audio_file):
                await self.ctx.send("Arquivo lastVoice.wav não encontrado!")
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
                self.loop.call_soon_threadsafe(playback_done.set)

            # Play the audio
            if not self.voice_client.is_playing():
                self.voice_client.play(audio_source, after=after_playback)
                await self.ctx.send("Tocando no canal de voz!")
                await playback_done.wait()  # Wait until playback is complete
            else:
                await self.ctx.send("O bot já está tocando um áudio. Aguarde até terminar!")
        except Exception as e:
            await self.ctx.send(f"Erro ao tocar o áudio: {str(e)}")
            print(f"Erro ao tocar lastVoice.wav: {e}")

    async def process_audio(self):
        while self.last_audio_time is not None:  # Continue until stopped
            current_time = datetime.now()
            if (self.last_audio_time is not None and
                    self.pcm_buffers and
                    current_time - self.last_audio_time >= timedelta(seconds=1) and
                    not self.llm.is_processing):
                self.llm.is_processing = True

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
                        if self.ctx:
                            await self.ctx.send(f"**{result.strip()}**")

                self.pcm_buffers.clear()

                if self.ctx and transcript:
                    await self.ctx.send(f"===============================================\n "
                                        f"**full transcript**: \n"
                                        f"{transcript}\n"
                                        f"===============================================")
                    llm_response = self.llm.ask(transcript)
                    if llm_response:
                        await self.ctx.send(f"===============================================\n"
                                            f"**IAra**: {llm_response}\n"
                                            f"===============================================")
                        self.synth.generateTtsFile(llm_response)
                        await self.playAudio()
                self.llm.is_processing = False

            await asyncio.sleep(0.1)  # Sleep briefly to avoid busy-waiting

    @commands.command()
    async def test(self, ctx: Context):
        self.ctx = ctx  # Store the context

        def callback(user, data: voice_recv.VoiceData):
            if not self.llm.is_processing :
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