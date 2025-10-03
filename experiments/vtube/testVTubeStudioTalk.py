import asyncio
import torchaudio
from VTubeStudioTalk import VTubeStudioTalk


async def main():
    vts_talk = VTubeStudioTalk()

    # Connect to VTube Studio
    await vts_talk.connect()

    # Load audio file
    audio_file = "voice_example.wav"
    waveform, sample_rate = torchaudio.load(audio_file)

    # Sync mouth with audio
    await vts_talk.sync_mouth(waveform, sample_rate)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())