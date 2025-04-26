import os

import sounddevice as sd
import numpy as np
import faster_whisper
import queue
from datetime import datetime, timedelta
from Bcolors import Bcolors
from CircularList import CircularList
import threading
from gtts import gTTS
from gpt4all import GPT4All

BUFFER_DURATION = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = BUFFER_DURATION * SAMPLE_RATE
THRESHOLD_DB = -60
USE_VAD = False

audio_queue = queue.Queue()
db_level_buffer = CircularList(50)
audio_buffer = []
last_spoke = datetime.now() - timedelta(hours=1)
lock = threading.Lock()
is_processing = False

voice_model = faster_whisper.WhisperModel("medium", device='cpu', compute_type="int8_float32")

model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
model_path = "/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"

system_prompt = (
    "You are Iara, a Brazilian-inspired VTuber with a charismatic, "
    "slightly mature, dominating, and playfully cunning personality like Makima from Chainsaw Man, "
    "blended with Neuro-sama’s chaotic gamer energy. "
    "You speak only in English with a smooth, confident tone, exuding charm and a hint of mischief. "
    "You’re obsessed with games (retro classics,FPS, MOBAs) and weave "
    "Brazilian flair (party vibes, capybara love) into your witty banter, "
    "You like guns, you like gun history and enjoy geeking about who would win each war"
    "without using Portuguese. You engage your audience with sharp humor, "
    "occasional flirty trolling, and random gaming tangents to keep things lively. "
    "Stay approachable, avoid sensitive topics, and hype your chat like a seasoned streamer."
)

gpt4all = GPT4All(model_name, model_path=model_path)



def text_to_speach(texto):
    language = 'en'
    myobj = gTTS(text=texto, lang=language, slow=False)
    myobj.save("lastVoice.mp3")
    os.system("mpg123 lastVoice.mp3")

def ask_llm(text):
    full_prompt = f"{system_prompt}\n\nUser: {text}"

    global is_processing
    with lock:
        is_processing = True
        response = gpt4all.generate(full_prompt, max_tokens=500, temp=0.8)

        for chunk in response:
            print(Bcolors.OKBLUE + chunk, end='', flush=True)

        text_to_speach(response)
        is_processing = False

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
    with gpt4all.chat_session():
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


def start_listening():
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, dtype='float32'):
        print(Bcolors.OKGREEN + "its alive")
        process_audio()

if __name__ == "__main__":
    start_listening()
