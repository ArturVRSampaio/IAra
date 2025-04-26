import pyvts
import asyncio
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import time
import threading

plugin_info = {
    "plugin_name": "start pyvts",
    "developer": "Genteki",
    "authentication_token_path": "./token.txt"
}


# Função para carregar o áudio
def load_audio(file_path):
    audio = AudioSegment.from_wav(file_path)
    sample_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalizar para [-1, 1]
    return samples, sample_rate, audio


# Função para calcular a intensidade do áudio em blocos
def get_audio_intensity(samples, sample_rate, block_duration=0.05):
    block_size = int(sample_rate * block_duration)  # Tamanho do bloco em amostras
    intensities = []
    for i in range(0, len(samples), block_size):
        block = samples[i:i + block_size]
        intensity = np.abs(block).mean()  # Média da amplitude absoluta
        intensities.append(intensity)
    return intensities


# Função para reproduzir o áudio em uma thread separada
def play_audio(audio):
    play(audio)


async def main():
    # Configurar VTube Studio
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()
    await vts.request_authenticate_token()
    await vts.request_authenticate()

    # Carregar o arquivo de áudio
    audio_file = "lastVoice.wav"
    samples, sample_rate, audio = load_audio(audio_file)

    # Calcular intensidades
    block_duration = 0.05  # 50ms por bloco
    intensities = get_audio_intensity(samples, sample_rate, block_duration)

    # Iniciar reprodução do áudio em uma thread separada
    audio_thread = threading.Thread(target=play_audio, args=(audio,))
    audio_thread.start()

    # Sincronizar com VTube Studio
    start_time = time.time()
    for intensity in intensities:
        # Normalizar a intensidade para o parâmetro MouthOpen (0 a 1)
        mouth_open_value = min(max(intensity * 10, 0), 1)  # Ajuste o fator 10 conforme necessário

        # Enviar comando para abrir/fechar a boca
        request_msg = vts.vts_request.requestSetParameterValue(
            parameter="MouthOpen",
            value=mouth_open_value,
            weight=1.0
        )
        response = await vts.request(request_msg)
        print(f"Boca ajustada: {mouth_open_value}, Resposta: {response}")

        # Aguardar o próximo bloco de áudio
        elapsed_time = time.time() - start_time
        expected_time = (intensities.index(intensity) + 1) * block_duration
        sleep_time = max(0, expected_time - elapsed_time)
        await asyncio.sleep(sleep_time)

    # Aguardar a thread de áudio terminar
    audio_thread.join()

    # Fechar conexão
    await vts.close()


# Executar
if __name__ == "__main__":
    asyncio.run(main())