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
        "Speaks onl+y in English! "
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
        self.tts = TTS(model_name="tts_models/en/ljspeech/vits")
        self.vts_talk = vts_talk

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
