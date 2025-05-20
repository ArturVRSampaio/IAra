import time
import torchaudio
from TTS.api import TTS
import sounddevice as sd

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')

text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."


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

waveform, sample_rate = torchaudio.load("output.wav")

sd.play(waveform.numpy().T, sample_rate)
sd.wait()
