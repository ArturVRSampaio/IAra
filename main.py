# Standard library imports
import os
import queue
import threading
import time
from datetime import datetime, timedelta

# Third-party imports
import asyncio
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from dotenv import load_dotenv
from gpt4all import GPT4All
from TTS.api import TTS
from twitchio.ext import commands
from faster_whisper import WhisperModel

# Local imports
from Bcolors import Bcolors
from CircularList import CircularList
from vtube.VTubeStudioTalk import VTubeStudioTalk

# Load environment variables
load_dotenv()

# Audio processing constants
BUFFER_DURATION = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = BUFFER_DURATION * SAMPLE_RATE
THRESHOLD_DB = -60
USE_VAD = False

# Global state
audio_queue = queue.Queue()
db_level_buffer = CircularList(50)
audio_buffer = []
last_spoke = datetime.now() - timedelta(hours=1)
last_llm_spoke = datetime.now() - timedelta(hours=1)
lock = threading.Lock()
is_processing = False
last_chat_message = None
vts_talk = None  # VTube Studio talk instance

# Initialize models
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=False)
voice_model = WhisperModel("small.en", cpu_threads=8, compute_type="int8_float32")
gpt4all = GPT4All(
    model_name="Llama-3.2-1B-Instruct-Q4_0.gguf",
    model_path="/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"
)

# System prompt for AI personality
SYSTEM_PROMPT = (
    "You are Iara, a Brazilian-inspired AI VTuber with a charismatic tone, "
    "serving as a gaming companion with a mature, playfully cunning personality. "
    "Speak only in English with a confident, charming tone and a hint of mischief. "
    "Incorporate Brazilian flair (party vibes, capybara love) into witty banter. "
    "Engage with sharp humor, occasional flirty trolling, and gaming tangents. "
    "Keep answers short (max 20 words), avoid sensitive topics, "
    "and hype the chat like a seasoned streamer."
    "Speak only in English."
)

class TwitchBot(commands.Bot):
    """Twitch bot for handling chat interactions."""
    def __init__(self):
        token = os.getenv('TWITCH_OAUTH_TOKEN')
        channel = os.getenv('TWITCH_CHANNEL')
        super().__init__(token=token, prefix='!', initial_channels=[channel])

    async def event_ready(self):
        """Called when the bot is connected to Twitch."""
        print(f'Bot connected as {self.nick}')
        print(f'Connected to channel: {self.connected_channels}')

    async def event_message(self, message):
        """Handle incoming Twitch chat messages."""
        global last_chat_message
        if message.echo or message.content.startswith("!"):
            return
        with lock:
            last_chat_message = (message.author.name, message.content)
        print(f'{message.author.name}: {message.content}')
        await self.handle_commands(message)

def calculate_db_level(audio_data: np.ndarray) -> int:
    """Calculate the decibel level of audio data."""
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return int(20 * np.log10(rms)) if rms > 0 else -200

def text_to_speech(text: str) -> None:
    """Convert text to speech and play it."""
    if not text:
        return
    output_file = "lastVoice.wav"
    tts.tts_to_file(text=text, file_path=output_file)
    waveform, sample_rate = torchaudio.load(output_file)
    amplified_waveform = torch.clamp(waveform * 3.0, -1.0, 1.0)
    if vts_talk:
        vts_talk.run_sync_mouth(amplified_waveform, sample_rate)
    sd.play(amplified_waveform.numpy().T, sample_rate)
    sd.wait()
    print("Audio file played")

def ask_llm(text: str) -> None:
    """Generate a response using the LLM and convert it to speech."""
    if not text:
        return
    global is_processing, last_llm_spoke
    with lock:
        is_processing = True
        print(Bcolors.WARNING + "Processing started")
    try:
        response = gpt4all.generate(text, max_tokens=200, n_batch=60, temp=0.8)
        print(Bcolors.OKBLUE + response)
        text_to_speech(response)
    finally:
        last_llm_spoke = datetime.now()
        is_processing = False
        print(Bcolors.WARNING + "Processing finished")

def audio_callback(indata, frames, time, status) -> None:
    """Process incoming audio data from the microphone."""
    global last_spoke
    if status:
        print(status)
    indata = indata.squeeze()
    db_level = calculate_db_level(indata)
    db_level_buffer.add(db_level)

    if db_level_buffer.mean() > THRESHOLD_DB:
        audio_buffer.extend(indata.tolist())
        last_spoke = datetime.now()

    if (last_spoke < datetime.now() - timedelta(seconds=2) and
            audio_buffer and not is_processing):
        audio_queue.put(np.array(audio_buffer, dtype=np.float32))
        print(Bcolors.WARNING + "Processing audio file")
        audio_buffer.clear()

def process_audio() -> None:
    """Process audio from queue and respond to chat messages."""
    global last_chat_message, last_llm_spoke
    with gpt4all.chat_session():
        ask_llm(SYSTEM_PROMPT + " Stream starting! How are you doing?")

        while True:
            if not audio_queue.empty() and not is_processing:
                audio_data = audio_queue.get()
                segments, _ = voice_model.transcribe(audio_data, vad_filter=USE_VAD, language='en')
                transcript = " ".join(segment.text for segment in segments)
                print(Bcolors.OKGREEN + transcript)
                ask_llm(f'Artur says: {transcript}')

            elif (last_spoke < datetime.now() - timedelta(seconds=3) and
                  last_llm_spoke < datetime.now() - timedelta(seconds=5) and
                  not is_processing and last_chat_message):
                user, message = last_chat_message
                last_chat_message = None
                print(Bcolors.OKCYAN + f'Responding to chat: {user}: {message}')
                ask_llm(f'{user} says: {message}. If not important, ignore it.')

            elif (last_spoke < datetime.now() - timedelta(seconds=3) and
                  last_llm_spoke < datetime.now() - timedelta(seconds=10) and
                  audio_queue.empty() and not is_processing):
                print(Bcolors.OKCYAN + 'Continuing conversation')
                ask_llm(f'Updating context: {SYSTEM_PROMPT}. Continue your last answer.')

async def run_twitch_bot() -> None:
    """Run the Twitch bot."""
    bot = TwitchBot()
    await bot.start()

def start_listening() -> None:
    """Start the audio listener and Twitch bot."""
    global vts_talk
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    vts_talk = VTubeStudioTalk(loop)
    loop.run_until_complete(vts_talk.connect())
    threading.Thread(
        target=lambda: loop.run_until_complete(run_twitch_bot()),
        daemon=True
    ).start()

    time.sleep(2)  # Wait for Twitch bot initialization
    with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype='float32'
    ):
        print(Bcolors.OKGREEN + "System initialized")
        process_audio()

if __name__ == "__main__":
    try:
        start_listening()
    finally:
        if vts_talk:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(vts_talk.disconnect())