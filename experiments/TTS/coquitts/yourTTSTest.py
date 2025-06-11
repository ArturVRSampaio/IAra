import time
import torch
import sounddevice as sd
from TTS.api import TTS
import soundfile as sf

torch.set_num_threads(16)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/your_tts",
          progress_bar=True).to(device)

print(tts.speakers)
print('model loaded')

text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
text = "levei muito tempo para desenvolver uma voz, e agora que tenho uma não vou ficar calada"

output_file = "output.wav"
start = time.time()
tts.tts_to_file(
    text=text,
    # speaker_wav="./agatha_voice.wav",
    speaker=tts.speakers[2],
    file_path=output_file,
    split_sentences=True,
    language='pt-br'
)
end = time.time()
print("time: " + str(end - start))

# Carregar o áudio
data, samplerate = sf.read("output.wav")
print(f"Sample rate: {samplerate}, Data shape: {data.shape}")


sd.play(data, samplerate)
sd.wait()

