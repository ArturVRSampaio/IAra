import asyncio
import time
from typing import Any

import numpy as np
import pyvts
import torch

from iara.utils import Bcolors

plugin_info = {
    "plugin_name": "Iara VTuber",
    "developer": "Artur",
    "authentication_token_path": "./token.txt",
}


class VTubeStudioTalk:
    def __init__(self) -> None:
        self.vts: Any = pyvts.vts(plugin_info=plugin_info)
        self.is_connected: bool = False
        self.lock: asyncio.Lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self.lock:
            await self.vts.connect()
            await self.vts.request_authenticate_token()
            is_auth: bool = await self.vts.request_authenticate()
            self.is_connected = is_auth
            print(Bcolors.OKGREEN + "VTube Studio connected : " + str(is_auth))

    def _get_audio_intensity(
        self, waveform: torch.Tensor, sample_rate: int, block_duration: float = 0.05
    ) -> list[float]:
        samples = waveform.numpy().mean(axis=0)
        block_size = int(sample_rate * block_duration)
        intensities: list[float] = []
        for i in range(0, len(samples), block_size):
            block = samples[i : i + block_size]
            intensity = np.abs(block).mean()
            intensities.append(intensity)
        return intensities

    async def sync_mouth(self, waveform: torch.Tensor, sample_rate: int) -> None:
        start_time = time.time()

        if not self.is_connected:
            await self.connect()

        intensities = self._get_audio_intensity(
            waveform, sample_rate, block_duration=0.05
        )

        for i, intensity in enumerate(intensities):
            mouth_open_value = float(min(max(intensity * 10, 0), 1))
            request_msg = self.vts.vts_request.requestSetMultiParameterValue(
                parameters=["MouthOpen"], values=[mouth_open_value], weight=1
            )

            try:
                async with self.lock:
                    await self.vts.request(request_msg)
            except Exception:
                self.is_connected = False
            elapsed_time = time.time() - start_time
            expected_time = (i + 1) * 0.05
            sleep_time = max(0, expected_time - elapsed_time)
            await asyncio.sleep(sleep_time)

    async def execute_animation(self, animation_hotkey: str) -> None:
        if not self.is_connected:
            await self.connect()

        if not self.is_connected:
            print(f"VTube Studio not authenticated — cannot trigger {animation_hotkey}")
            return

        try:
            async with self.lock:
                hotkey_response = await self.vts.request(
                    self.vts.vts_request.requestHotKeyList()
                )
            hotkeys = [
                hk["name"]
                for hk in hotkey_response.get("data", {}).get("availableHotkeys", [])
            ]

            if animation_hotkey in hotkeys:
                async with self.lock:
                    await self.vts.request(
                        self.vts.vts_request.requestTriggerHotKey(animation_hotkey)
                    )
                print(f"Triggered {animation_hotkey}")
            else:
                print(f"Hotkey '{animation_hotkey}' not found in VTube Studio")
        except Exception as e:
            print(f"VTube Studio connection lost during animation: {e}")
            self.is_connected = False

    def _mood_to_hotkey(self, mood: int) -> str:
        if mood <= 2:
            return "IAra_Sad"
        elif mood <= 4:
            return "IAra_Neutral"
        elif mood == 5:
            return "IAra_Happy"
        else:
            return "IAra_Excited"

    async def trigger_mood_expression(self, mood: int) -> None:
        await self.execute_animation(self._mood_to_hotkey(mood))

    async def change_expression(self) -> None:
        if not self.is_connected:
            await self.connect()

        request_msg = self.vts.vts_request.requestSetMultiParameterValue(
            parameters=["MouthOpen", "MouthOpen"], values=[1, 1], weight=1
        )

        await self.vts.request(request_msg)
