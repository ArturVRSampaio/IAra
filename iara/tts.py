import asyncio
import re

import torch
from TTS.api import TTS

torch.set_num_threads(14)


class SpeechSynthesizer:
    def __init__(self):
        self.tts = TTS(
            model_name="xtts_v2",
            progress_bar=False,
            gpu=True,
        ).to("cuda")

    async def generate_tts_file(self, text: str, output_path: str) -> None:
        if not text:
            return

        pattern = r'[!.()]+'
        text = re.sub(pattern, '', text)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.tts.tts_to_file(
                text=text,
                speaker=self.tts.speakers[2],
                file_path=output_path,
                language='pt'
            )
        )
