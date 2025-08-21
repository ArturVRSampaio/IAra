import time

import torch
from TTS.api import TTS
# import sounddevice as sd
import soundfile as sf

long_text = """
O Rock in Rio é um festival de música idealizado pelo empresário brasileiro Roberto Medina pela primeira vez em 1985. É reconhecido como um dos maiores festivais musicais do planeta. Foi originalmente organizado no Rio de Janeiro, de onde vem o nome. Tornou-se um evento de repercussão em nível mundial e teve em 2004 sua primeira edição fora do Brasil, em Lisboa, Portugal.

Ao longo da sua história, o Rock in Rio teve 22 edições, nove no Brasil, nove em Portugal, três na Espanha e uma nos Estados Unidos. Em 2008, foi realizado pela primeira vez em dois locais diferentes, Lisboa e Madrid. Além destas, duas edições foram canceladas: Madrid e Buenos Aires, ambas programadas para 2014.[1][2]

O hino do festival é de autoria do compositor Nelson Wellington e do maestro Eduardo Souto Neto e foi gravado originalmente pelo grupo Roupa Nova.[3] O festival é considerado o oitavo melhor do mundo pelo site especializado Festival Fling.[4] Desde sua 4ª edição no Brasil, o festival costuma ocorrer bianualmente no início da primavera em seu país de origem (Brasil). A nona edição do festival ocorreu em 2022 no Parque Olímpico do Rio de Janeiro.

Em 2024, foi realizada a décima edição do festival, denominada Rock in Rio X, que contará com nomes como Ed Sheeran, Scorpions e Katy Perry.

Também neste edição o festival terá um espetáculo de teatro musical especial pelos 40 anos do festival chamado 'Sonhos, Lama e Rock and Roll'. A peça contará a história do Rock In Rio, com a direção de Charles Möeller, direção musical de Zé Ricardo e será protagonizada pelos atores Beto Sargentelli, Malu Rodrigues e Rodrigo Pandolfo.[5]
"""

text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
text = "levei muito tempo para desenvolver uma voz,"# e agora que tenho uma não vou ficar calada"
print(TTS().list_models())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

tts = TTS("xtts_v2", progress_bar=True, gpu=True).to(device)

print(tts.speakers)
print('model loaded')

start = time.time()

output_file = "output.wav"


tts.tts_to_file(text=text,
                file_path=output_file,
                speaker_wav='agatha_voice.wav',
                # speaker=tts.speakers[0],
                language="pt",
                split_sentences=False,
                )

end = time.time()

print("time: " + str(end - start))

data, samplerate = sf.read("output.wav")
print(f"Sample rate: {samplerate}, Data shape: {data.shape}")

# sd.play(data, samplerate)
# sd.wait()

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

