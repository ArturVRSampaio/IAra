import torch
from faster_whisper import WhisperModel
from pathlib import Path

def transcribe_audio(audio_path):
    # Verifica se o arquivo de áudio existe
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    # Carrega o modelo Whisper (large-v3, o maior disponível)
    model = WhisperModel(
        model_size_or_path="large-v3",
        device="cpu",
        compute_type="int8_float32",  # Otimização para desempenho
        cpu_threads=6  # Número de threads para CPU
    )

    # Realiza a transcrição
    try:
        segments, info = model.transcribe(
            audio=audio_path,
            language="pt",  # Define o idioma como português
            task="transcribe",
            beam_size=5,  # Tamanho do beam para maior precisão
            vad_filter=True  # Filtro VAD para melhorar detecção de fala
        )
        # Concatena os segmentos de texto
        transcription = " ".join(segment.text for segment in segments)
        return transcription
    except Exception as e:
        raise Exception(f"Erro durante a transcrição: {str(e)}")

def save_transcription(text, output_path):
    # Salva a transcrição em um arquivo de texto
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcrição salva em: {output_path}")

if __name__ == "__main__":
    # Caminho do arquivo de áudio
    audio_file = "audio.ogg"
    output_file = "transcricao.txt"

    try:
        # Transcreve o áudio
        transcription = transcribe_audio(audio_file)
        print("\nTranscrição:\n", transcription)

        # Salva a transcrição em um arquivo
        save_transcription(transcription, output_file)
    except Exception as e:
        print(f"Erro: {str(e)}")