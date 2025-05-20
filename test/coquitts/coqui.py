import time
import torch
import sounddevice as sd
import torchaudio
from TTS.api import TTS

print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to('cuda')

print(tts.speakers)
print('model loaded')

text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
start = time.time()
tts.tts_to_file(
    text=text,
    speaker=tts.speakers[0],
    file_path="output.wav",
    split_sentences=True,
    language='en'
)
end = time.time()
print("time: " + str(end - start))

# Carregar o áudio
waveform, sample_rate = torchaudio.load("output.wav")

# Garantir que o áudio esteja no formato correto para reprodução
if waveform.shape[0] > 2:
    raise ValueError("Áudio com mais de 2 canais não suportado.")
elif waveform.shape[0] == 2:
    waveform = waveform.mean(dim=0, keepdim=True)  # converter para mono
elif waveform.shape[0] == 1:
    waveform = waveform.squeeze(0)  # remover dimensão extra

# Reproduzir
sd.play(waveform.numpy(), sample_rate)
sd.wait()
