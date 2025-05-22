import time

import torch
from TTS.api import TTS
import sounddevice as sd
import soundfile as sf


text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
print(TTS().list_models())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print(tts.speakers)
print('model loaded')

start = time.time()

tts.tts_to_file(text=text,
                file_path="output.wav",
                speaker=tts.speakers[0],
                language="en",
                split_sentences=True
                )

end = time.time()
print("time: " + str(end - start))

data, samplerate = sf.read("output.wav")

sd.play(data, samplerate)
sd.wait()
