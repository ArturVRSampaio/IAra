import asyncio
import time
import numpy as np
import torch
import torchaudio
from pyvts import vts
from Bcolors import Bcolors

# VTube Studio plugin info
PLUGIN_INFO = {
    "plugin_name": "Iara VTuber",
    "developer": "Artur",
    "authentication_token_path": "./token.txt"
}


class VTubeStudioTalk:
    """Manages VTube Studio connection and mouth animation synchronization."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.vts = None
        self.loop = loop

    async def connect(self):
        """Initialize and authenticate VTube Studio connection."""
        self.vts = vts(plugin_info=PLUGIN_INFO)
        await self.vts.connect()
        await self.vts.request_authenticate_token()
        await self.vts.request_authenticate()
        print(Bcolors.OKGREEN + "VTube Studio connected")

    async def disconnect(self):
        """Close VTube Studio connection."""
        if self.vts:
            await self.vts.close()
            self.vts = None
            print(Bcolors.OKGREEN + "VTube Studio disconnected")

    def get_audio_intensity(self, waveform: torch.Tensor, sample_rate: int, block_duration: float = 0.05) -> list:
        """Calculate audio intensity for each block."""
        samples = waveform.numpy().mean(axis=0)  # Convert to mono if stereo
        block_size = int(sample_rate * block_duration)
        intensities = []
        for i in range(0, len(samples), block_size):
            block = samples[i:i + block_size]
            intensity = np.abs(block).mean()
            intensities.append(intensity)
        return intensities

    async def sync_mouth(self, waveform: torch.Tensor, sample_rate: int):
        """Sync VTube Studio mouth animation with audio."""
        if not self.vts:
            print(Bcolors.WARNING + "VTube Studio not connected")
            return

        intensities = self.get_audio_intensity(waveform, sample_rate, block_duration=0.05)
        start_time = time.time()

        for intensity in intensities:
            mouth_open_value = min(max(intensity * 10, 0), 1)  # Adjust scaling factor as needed
            request_msg = self.vts.vts_request.requestSetParameterValue(
                parameter="MouthOpen",
                value=mouth_open_value,
                weight=1.0
            )
            await self.vts.request(request_msg)
            elapsed_time = time.time() - start_time
            expected_time = (intensities.index(intensity) + 1) * 0.05
            sleep_time = max(0, expected_time - elapsed_time)
            await asyncio.sleep(sleep_time)

    def run_sync_mouth(self, waveform: torch.Tensor, sample_rate: int):
        """Schedule mouth sync in the event loop without blocking."""
        asyncio.run_coroutine_threadsafe(self.sync_mouth(waveform, sample_rate), self.loop)