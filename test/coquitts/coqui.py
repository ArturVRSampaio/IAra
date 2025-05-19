import torch
import sounddevice as sd
import torchaudio
from TTS.api import TTS


print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True,  )

print(tts.speakers)
print('model loaded')


text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

tts.tts_to_file(
    text=text,
    speaker=tts.speakers[0],
    file_path="output.wav",
    split_sentences=True
)


waveform, sample_rate = torchaudio.load("output.wav")
gain = 3.0  # Adjust this value (e.g., 1.5, 2.0, 3.0) to increase volume
amplified_waveform = waveform * gain

amplified_waveform = torch.clamp(amplified_waveform, -1.0, 1.0)

sd.play(amplified_waveform.numpy().T, sample_rate)
sd.wait()
print("Amplified audio file 'output.wav' has been played successfully!")