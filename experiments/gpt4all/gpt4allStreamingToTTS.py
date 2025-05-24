import torch
import sounddevice as sd
import soundfile as sf
import tempfile
from TTS.api import TTS
import re

from LLMAgent import LLMAgent

# Inicializa Coqui TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to(device)
speaker_wav_path = "../TTS/coquitts/agatha_voice.wav"

def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
        tts.tts_to_file(
            text=text,
            speaker=tts.speakers[2],
            file_path="output.wav",
            split_sentences=True,
            language='pt-br'
        )
        audio_data, sample_rate = sf.read(tmp_audio_file.name)
        sd.play(audio_data, sample_rate)
        sd.wait()

def ask_and_stream_tts(user_input):
    stream = agent.ask(user_input)  # retorna um gerador streaming

    buffer = ""
    sentence_end = re.compile(r"[.!?…]")  # pontuação que finaliza a frase

    for token in stream:
        print(token, end="", flush=True)
        buffer += token

        # Processa quando detecta final de frase
        if sentence_end.search(buffer) and len(buffer.strip().split()) >= 3:
            speak(buffer.strip())
            buffer = ""

    # Processa o restante, se sobrar algo
    if buffer.strip():
        speak(buffer.strip())


agent = LLMAgent()

with agent.getChatSession():
    ask_and_stream_tts("bypass says: qual seu nome?")
    ask_and_stream_tts("bypass says: minha cor favorita é preto, qual a sua?")
    ask_and_stream_tts("bypass says: como iniciou sua carreira de streamer?")
    ask_and_stream_tts("bypass says: o que vamos jogar hoje?")
    ask_and_stream_tts("bypass says: qual é a cor que te disse ser a minha favorita mesmo?")