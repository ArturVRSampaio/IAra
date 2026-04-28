from io import BytesIO

from faster_whisper import WhisperModel
from pydub import AudioSegment


class STT:
    def __init__(self) -> None:
        self.model: WhisperModel = WhisperModel(
            "turbo",
            cpu_threads=4,
            num_workers=5,
            device="cuda",
        )

    def transcribe_audio(self, pcm_chunks: list[bytes]) -> str:
        if not pcm_chunks:
            return ""

        raw_pcm = b"".join(pcm_chunks)
        audio = AudioSegment(
            data=raw_pcm,
            sample_width=2,
            frame_rate=48000,
            channels=2,
        ).set_channels(1)

        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        segments, _ = self.model.transcribe(
            wav_buffer, language="pt", vad_filter=True, hotwords="IAra, Vtuber"
        )
        return "".join(seg.text for seg in segments).strip()
