import os
import numpy as np
import faster_whisper
import queue
from datetime import datetime, timedelta
from Bcolors import Bcolors
from CircularList import CircularList
import threading
import asyncio
from twitchio.ext import commands
from gpt4all import GPT4All
from dotenv import load_dotenv

import torch
import sounddevice as sd
import torchaudio
from TTS.api import TTS

load_dotenv()

BUFFER_DURATION = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = BUFFER_DURATION * SAMPLE_RATE
THRESHOLD_DB = -60
USE_VAD = False

audio_queue = queue.Queue()
db_level_buffer = CircularList(50)
audio_buffer = []
last_spoke = datetime.now() - timedelta(hours=1)
last_llm_spoke = datetime.now() - timedelta(hours=1)
lock = threading.Lock()
is_processing = False
print(Bcolors.WARNING + "is_processing False")
last_chat_message = None  # Armazena a última mensagem do chat

tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=False)

voice_model = faster_whisper.WhisperModel("small.en",
                                          cpu_threads= 8,
                                          compute_type="int8_float32")

model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
model_path = "/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"
system_prompt = (
    "You are Iara, a Brazilian-inspired AI VTuber with a charismatic tone, "
    "slightly mature, dominating, and playfully cunning personality,"
    "blended with chaotic energy. "
    "You speak only in English with a smooth, confident tone, exuding charm and a hint of mischief. "
    "Brazilian flair (party vibes, capybara love) into your witty banter, "
    "without using Portuguese. You engage your audience with sharp humor, "
    "occasional flirty trolling, and random gaming tangents to keep things lively. "
    "Stay approachable, avoid sensitive topics, and hype your chat like a seasoned streamer."
    "give always really short answers, max 20 words."
    "try not to repeat yourself."
)
gpt4all = GPT4All(model_name, model_path=model_path)

class TwitchBot(commands.Bot):
    def __init__(self):
        token = os.getenv('TWITCH_OAUTH_TOKEN')
        channel = os.getenv('TWITCH_CHANNEL')
        super().__init__(
            token=token,
            prefix='!',
            initial_channels=[channel]
        )

    async def event_ready(self):
        print(f'Bot conectado como {self.nick}')
        print(f'Conectado ao canal: {self.connected_channels}')

    async def event_message(self, message):
        global last_chat_message
        if message.echo or message.content.startswith("!") :
            return
        with lock:
            last_chat_message = (message.author.name, message.content)
        print(f'{message.author.name}: {message.content}')
        await self.handle_commands(message)

def text_to_speech(texto):
    tts.tts_to_file(text=texto, file_path="lastVoice.wav")
    waveform, sample_rate = torchaudio.load("lastVoice.wav")
    gain = 3.0
    amplified_waveform = waveform * gain
    amplified_waveform = torch.clamp(amplified_waveform, -1.0, 1.0)
    sd.play(amplified_waveform.numpy().T, sample_rate)
    sd.wait()
    print("audiofile saved")

def ask_llm(text):
    if not text:
        return

    global is_processing, last_llm_spoke
    with lock:
        print('lock')
        is_processing = True
        print(Bcolors.WARNING + "is_processing True")
        try:
            print("try")
            response = gpt4all.generate(text, max_tokens=200, n_batch=60, temp=0.8)
            print(Bcolors.OKBLUE + response, flush=True)
            text_to_speech(response)
        finally:
            print("finally")
            last_llm_spoke= datetime.now()
            is_processing = False
            print(Bcolors.WARNING + "is_processing False")

def calc_db_level(audio_data) -> int:
    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms > 0:
        nivel_db = 20 * np.log10(rms)
    else:
        nivel_db = -200
    return int(nivel_db)

def callback(indata, frames, time, status):
    global last_spoke
    if status:
        print(status)
    indata = indata.squeeze()
    db_level = calc_db_level(indata)
    db_level_buffer.add(db_level)
    if db_level_buffer.mean() > THRESHOLD_DB:
        audio_buffer.extend(indata.tolist())
        last_spoke = datetime.now()
    if last_spoke < datetime.now() - timedelta(seconds=1.5) and len(audio_buffer) > 0:
        if not is_processing:
            audio_data = np.array(audio_buffer, dtype=np.float32)
            audio_queue.put(audio_data)
            print(Bcolors.WARNING + 'processing audio file')
            audio_buffer.clear()

def process_audio():
    global last_chat_message, last_llm_spoke
    with (gpt4all.chat_session()):
        ask_llm(system_prompt + " we are starting the stream now! how are you doing?")

        while True:
            if not audio_queue.empty() and not is_processing:
                audio_data = audio_queue.get()
                segments, info = voice_model.transcribe(
                    audio_data,
                    vad_filter=USE_VAD,
                    language='en'
                )
                group_segments = ''
                for segment in segments:
                    group_segments += " " + segment.text
                print(Bcolors.OKGREEN + group_segments)
                ask_llm('Artur says: ' + group_segments)
            elif last_spoke < datetime.now() - timedelta(seconds=3)  and last_llm_spoke < datetime.now() - timedelta(seconds=5) and not is_processing and last_chat_message is not None:
                    user, message = last_chat_message
                    last_chat_message = None
                    print(Bcolors.OKCYAN + f'Respondendo ao chat: {user}: {message}')
                    ask_llm(f'{user} says: {message}. if his message is not important, ignore it')
            elif last_spoke < datetime.now() - timedelta(seconds=3) and last_llm_spoke < datetime.now() - timedelta(seconds=10) and audio_queue.empty() and not is_processing:
                print(Bcolors.OKCYAN + f'continuando conversa')
                ask_llm(f'updating context:{system_prompt}. no one says nothing, continue explaining your last answer')



async def run_twitch_bot():
    bot = TwitchBot()
    await bot.start()

def start_listening():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    threading.Thread(target=lambda: loop.run_until_complete(run_twitch_bot()), daemon=True).start()

    # Pequeno atraso para garantir que o bot da Twitch esteja inicializado
    import time
    time.sleep(2)

    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, dtype='float32'):
        print(Bcolors.OKGREEN + "its alive")
        process_audio()

if __name__ == "__main__":
    start_listening()