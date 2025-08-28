import asyncio
import time
import numpy as np
import pyvts
import torch

from Bcolors import Bcolors

# VTube Studio plugin info
plugin_info = {
    "plugin_name": "Iara VTuber",
    "developer": "Artur",
    "authentication_token_path": "./token.txt"
}

class VTubeStudioTalk:
    """Manages VTube Studio connection and mouth animation synchronization."""

    def __init__(self):
        self.vts = pyvts.vts(plugin_info=plugin_info)
        self.is_connected = False
        self.lock = asyncio.Lock()

    async def connect(self):
        """Initialize and authenticate VTube Studio connection."""
        async with self.lock:
            await self.vts.connect()
            await self.vts.request_authenticate_token()
            is_auth = await self.vts.request_authenticate()
            print(Bcolors.OKGREEN + "VTube Studio connected : " + str(is_auth))


    def _get_audio_intensity(self, waveform: torch.Tensor, sample_rate: int, block_duration: float = 0.05) -> list:
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
        start_time = time.time()

        if not self.is_connected:
            await self.connect()

        intensities = self._get_audio_intensity(waveform, sample_rate, block_duration=0.05)

        for intensity in intensities:
            mouth_open_value = float(min(max(intensity * 10, 0), 1))
            request_msg = self.vts.vts_request.requestSetParameterValue(
                parameter="MouthOpen",
                value=mouth_open_value,
                weight=1.0
            )

            try:
                async with self.lock:
                    await self.vts.request(request_msg)
            except Exception as e:
                self.is_connected = False
                print(e)
            elapsed_time = time.time() - start_time
            expected_time = (intensities.index(intensity) + 1) * 0.05
            sleep_time = max(0, expected_time - elapsed_time)
            await asyncio.sleep(sleep_time)

    async def change_emotion(self, emotion_hotkey):
        if not self.is_connected:
            await self.connect()


        hotkey_response = await self.vts.request(self.vts.vts_request.requestHotKeyList())
        hotkeys = [hk['name'] for hk in hotkey_response['data']['availableHotkeys']]
        print("Available hotkeys:", hotkeys)


        if emotion_hotkey in hotkeys:
            await self.vts.request(self.vts.vts_request.requestTriggerHotKey(emotion_hotkey))
            print(f"Triggered {emotion_hotkey}")

