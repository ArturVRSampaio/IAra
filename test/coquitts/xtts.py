import time

import torch
from TTS.api import TTS
import sounddevice as sd
import soundfile as sf

text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
text = "levei muito tempo para desenvolver uma voz, e agora que tenho uma não vou ficar calada"
print(TTS().list_models())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print(tts.speakers)
print('model loaded')

start = time.time()

tts.tts_to_file(text=text,
                file_path="output.wav",
                speaker_wav='agatha_voice.wav',
                # speaker=tts.speakers[0],
                language="pt",
                split_sentences=True,
                )

end = time.time()

print("time: " + str(end - start))

data, samplerate = sf.read("output.wav")
print(f"Sample rate: {samplerate}, Data shape: {data.shape}")

sd.play(data, samplerate)
sd.wait()





start = time.time()

tts.tts_to_file(text=text,
                file_path="output.wav",
                speaker_wav='agatha_voice.wav',
                # speaker=tts.speakers[0],
                language="pt",
                split_sentences=True,
                )

end = time.time()

print("time: " + str(end - start))
