import torchaudio
from TTS.api import TTS
from safetensors import torch
import sounddevice as sd

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)


text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."


print(tts.speakers)
print('model loaded')


tts.tts_to_file(
                text=text,
                file_path="output.wav",
                speaker=tts.speakers[0],
                language="en",
                split_sentences=True
                )


waveform, sample_rate = torchaudio.load("output.wav")
gain = 3.0  # Adjust this value (e.g., 1.5, 2.0, 3.0) to increase volume
amplified_waveform = waveform * gain

sd.play(amplified_waveform.numpy().T, sample_rate)
sd.wait()
print("Amplified audio file 'output.wav' has been played successfully!")