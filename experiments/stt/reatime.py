import webrtcvad
import numpy as np
from faster_whisper import WhisperModel
import librosa

# Configurações
FRAME_DURATION_MS = 30  # Frame duration in milliseconds (10, 20, or 30ms supported by webrtcvad)
RATE = 16000  # Target sample rate (16kHz)
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)  # Calculate chunk size (480 samples for 30ms at 16kHz)
WAV_FILE = "agatha_long_voice.wav"  # Path to the .wav file

# Inicializar VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # Sensitivity level (0 to 3, 3 is most aggressive)

# Inicializar Faster Whisper
model = WhisperModel('turbo', device="auto")

# Ler e reamostrar arquivo .wav
audio_data, sample_rate = librosa.load(WAV_FILE, sr=RATE, mono=True)  # Resample to 16kHz and convert to mono
audio_data = (audio_data * 32768).astype(np.int16)  # Convert to int16 for VAD

# Buffer de áudio
audio_buffer = np.array([], dtype=np.int16)

print("Processando arquivo WAV (simulando stream de microfone)...")
# Processar áudio em blocos, como se fosse uma stream de microfone
for i in range(0, len(audio_data), CHUNK):
    audio_chunk = audio_data[i:i + CHUNK]
    if len(audio_chunk) == CHUNK:  # Ensure chunk is the correct size
        try:
            audio_bytes = audio_chunk.tobytes()
            if vad.is_speech(audio_bytes, RATE):
                audio_buffer = np.append(audio_buffer, audio_chunk)
                if len(audio_buffer) >= RATE * 2:  # 2 seconds of audio
                    audio_float = audio_buffer.astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(audio_float, language="pt", beam_size=5)
                    for segment in segments:
                        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
                    audio_buffer = audio_buffer[-int(RATE * 0.5):]  # Keep last 0.5s
        except webrtcvad.Error as e:
            print(f"VAD error: {e}. Skipping chunk.")
            continue
    else:
        print(f"Skipping incomplete chunk of size {len(audio_chunk)} at position {i}")

print("Processamento concluído.")