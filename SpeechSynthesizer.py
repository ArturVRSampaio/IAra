import asyncio
import torch
from TTS.api import TTS


torch.set_num_threads(14)

class SpeechSynthesizer:
    """Handles text-to-speech conversion using the TTS model."""

    def __init__(self):
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/your_tts",
            progress_bar=False,
        ).to("cpu")

    async def generate_tts_file(self, text: str, output_path: str) -> None:
        """Generates a TTS audio file from the provided text."""
        if not text:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.tts.tts_to_file(
                text=text,
                speaker=self.tts.speakers[2],
                speaker_wav='./experiments/TTS/coquitts/agatha_voice.wav',
                file_path=output_path,
                language='pt-br'
            )
        )
