import asyncio

import torchaudio

from vtube.VTubeStudioTalk import VTubeStudioTalk


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Inicializar VTubeStudioTalk
vts_talk = VTubeStudioTalk(loop)


loop.run_until_complete(vts_talk.connect())

audio_file = "agatha_long_voice.wav"
waveform, sample_rate = torchaudio.load(audio_file)
loop.run_until_complete(vts_talk.sync_mouth(waveform, sample_rate))





