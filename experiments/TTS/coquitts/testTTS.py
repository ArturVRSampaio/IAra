import time
import torch
import sounddevice as sd
import torchaudio
from TTS.api import TTS
import soundfile as sf

# Verificar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")
if device == "cuda":
    print(f"Memória da GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")

# Listar modelos disponíveis
print(TTS().list_models())

# Inicializar o modelo
try:
    tts = TTS("tts_models/en/multi-dataset/tortoise-v2").to(device)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# Verificar speakers disponíveis
print(f"Speakers disponíveis: {tts.speakers}")
speaker = None  # Não especificar speaker, já que tts.speakers é None

# Texto para síntese
text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

# Medir tempo de execução
start = time.time()

# Gerar áudio
try:
    tts.tts_to_file(
        text=text,
        file_path="output.wav",
        speaker=speaker,
        num_autoregressive_samples=512,
        diffusion_iterations=200
    )
    print("Áudio gerado com sucesso!")
except Exception as e:
    print(f"Erro ao gerar áudio: {e}")
    exit()

end = time.time()
print(f"Tempo de execução: {end - start} segundos")

# Reproduzir áudio
try:
    data, samplerate = sf.read("output.wav")
    print(f"Samplerate: {samplerate}, Data shape: {data.shape}")
    sd.play(data, samplerate)
    sd.wait()
except Exception as e:
    print(f"Erro ao reproduzir áudio: {e}")