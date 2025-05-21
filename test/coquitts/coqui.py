import time
import torch
import sounddevice as sd
import torchaudio
from TTS.api import TTS
import soundfile as sf


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to(device)

print(tts.speakers)
print('model loaded')

text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
start = time.time()
tts.tts_to_file(
    text=text,
    speaker=tts.speakers[2],
    file_path="output.wav",
    split_sentences=True,
    language='en'
)
end = time.time()
print("time: " + str(end - start))

# Carregar o áudio
data, samplerate = sf.read("output.wav")

sd.play(data, samplerate)
sd.wait()
