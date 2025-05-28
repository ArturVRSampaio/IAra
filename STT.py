from faster_whisper import WhisperModel
from pydub import AudioSegment
from io import BytesIO

class STT:
    def __init__(self):
        self.model = WhisperModel(
            "turbo",
            cpu_threads=4,
            num_workers=5,
            device="auto",
        )

    def transcribe_audio(self, pcm_chunks) -> str:
        """Transcribes audio chunks for a user into text."""
        if not pcm_chunks:
            return ""

        # Combine PCM chunks and convert to mono WAV
        raw_pcm = b"".join(pcm_chunks)
        audio = AudioSegment(
            data=raw_pcm,
            sample_width=2,
            frame_rate=48000,
            channels=2,
        ).set_channels(1)  # Convert to mono for Whisper

        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # Transcribe audio using Whisper model
        segments, _ = self.model.transcribe(
            wav_buffer,
            language="pt",
            vad_filter=True,
            hotwords="IAra, Vtuber"
        )
        return "".join(seg.text for seg in segments).strip()
