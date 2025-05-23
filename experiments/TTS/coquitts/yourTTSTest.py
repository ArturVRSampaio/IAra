import time
import torch
import sounddevice as sd
import torchaudio
from TTS.api import TTS
import soundfile as sf


def enhance_audio(audio_path, save_path, solver="midpoint",
                  nfe=64, tau=0.5, device="cuda", target_sr=16000):
    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9

    dwav, sr = torchaudio.load(audio_path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav2 = wav2.cpu().unsqueeze(0)

    if new_sr != target_sr:
        resampler = torchaudio.transforms.Resample(new_sr, target_sr)
        wav2 = resampler(wav2)

    torchaudio.save(save_path, wav2, target_sr)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to(device)

print(tts.speakers)
print('model loaded')

text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
text = "levei muito tempo para desenvolver uma voz, e agora que tenho uma não vou ficar calada"

start = time.time()
tts.tts_to_file(
    text=text,
    speaker_wav="./agatha_voice.wav",
    # speaker=tts.speakers[2],
    file_path="output.wav",
    split_sentences=True,
    language='pt-br'
)
end = time.time()
print("time: " + str(end - start))

# Carregar o áudio
data, samplerate = sf.read("output.wav")

sd.play(data, samplerate)
sd.wait()
