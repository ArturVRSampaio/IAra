import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch


def _make_vts():
    vts_cls = MagicMock()
    vts_instance = AsyncMock()
    vts_instance.request_authenticate.return_value = True
    vts_cls.return_value = vts_instance
    sys.modules["pyvts"].vts = vts_cls
    from VTubeStudioTalk import VTubeStudioTalk
    return VTubeStudioTalk(), vts_instance


class TestVTubeStudioTalkInit:
    def test_is_connected_false_on_init(self):
        vts_talk, _ = _make_vts()
        assert vts_talk.is_connected is False

    def test_lock_created(self):
        vts_talk, _ = _make_vts()
        assert vts_talk.lock is not None


class TestConnect:
    def test_connect_calls_authenticate(self):
        vts_talk, vts_instance = _make_vts()
        asyncio.run(vts_talk.connect())
        vts_instance.connect.assert_called_once()
        vts_instance.request_authenticate_token.assert_called_once()
        vts_instance.request_authenticate.assert_called_once()

    def test_is_connected_set_to_auth_result(self):
        vts_talk, vts_instance = _make_vts()
        vts_instance.request_authenticate.return_value = True
        asyncio.run(vts_talk.connect())
        assert vts_talk.is_connected is True

    def test_is_connected_false_when_auth_fails(self):
        vts_talk, vts_instance = _make_vts()
        vts_instance.request_authenticate.return_value = False
        asyncio.run(vts_talk.connect())
        assert vts_talk.is_connected is False


class TestGetAudioIntensity:
    def _silent_waveform(self, num_samples=4800, channels=1):
        return torch.zeros(channels, num_samples)

    def _constant_waveform(self, value=0.5, num_samples=4800, channels=1):
        return torch.full((channels, num_samples), value)

    def test_returns_list(self):
        vts_talk, _ = _make_vts()
        result = vts_talk._get_audio_intensity(self._silent_waveform(), 48000)
        assert isinstance(result, list)

    def test_silent_waveform_returns_zero_intensities(self):
        vts_talk, _ = _make_vts()
        result = vts_talk._get_audio_intensity(self._silent_waveform(), 48000)
        assert all(i == 0.0 for i in result)

    def test_number_of_blocks_matches_duration(self):
        vts_talk, _ = _make_vts()
        sample_rate = 48000
        duration_s = 0.5
        num_samples = int(sample_rate * duration_s)
        block_duration = 0.05
        expected_blocks = int(duration_s / block_duration)

        waveform = torch.zeros(1, num_samples)
        result = vts_talk._get_audio_intensity(waveform, sample_rate, block_duration)
        assert len(result) == expected_blocks

    def test_loud_waveform_returns_nonzero_intensities(self):
        vts_talk, _ = _make_vts()
        result = vts_talk._get_audio_intensity(self._constant_waveform(0.8), 48000)
        assert all(i > 0 for i in result)

    def test_stereo_waveform_averaged_to_mono(self):
        vts_talk, _ = _make_vts()
        waveform = torch.ones(2, 4800)
        result = vts_talk._get_audio_intensity(waveform, 48000)
        assert len(result) > 0
        assert all(i > 0 for i in result)

    def test_custom_block_duration(self):
        vts_talk, _ = _make_vts()
        sample_rate = 48000
        num_samples = 4800
        block_duration = 0.1
        expected_blocks = num_samples // int(sample_rate * block_duration)

        waveform = torch.zeros(1, num_samples)
        result = vts_talk._get_audio_intensity(waveform, sample_rate, block_duration)
        assert len(result) == expected_blocks


class TestSyncMouth:
    def test_calls_connect_if_not_connected(self):
        vts_talk, vts_instance = _make_vts()
        vts_talk.is_connected = False
        waveform = torch.zeros(1, 480)

        async def run():
            with patch.object(vts_talk, "connect", new_callable=AsyncMock) as mock_connect:
                mock_connect.side_effect = lambda: setattr(vts_talk, "is_connected", True) or None
                await vts_talk.sync_mouth(waveform, 48000)
                mock_connect.assert_called_once()

        asyncio.run(run())

    def test_does_not_reconnect_if_already_connected(self):
        vts_talk, vts_instance = _make_vts()
        vts_talk.is_connected = True
        waveform = torch.zeros(1, 480)

        async def run():
            with patch.object(vts_talk, "connect", new_callable=AsyncMock) as mock_connect:
                await vts_talk.sync_mouth(waveform, 48000)
                mock_connect.assert_not_called()

        asyncio.run(run())

    def test_exception_in_request_sets_disconnected(self):
        vts_talk, vts_instance = _make_vts()
        vts_talk.is_connected = True
        vts_instance.request.side_effect = Exception("connection lost")
        waveform = torch.ones(1, 480)

        asyncio.run(vts_talk.sync_mouth(waveform, 48000))
        assert vts_talk.is_connected is False

    def test_timing_uses_enumerate_not_index(self):
        # Waveform with repeated identical intensity values — list.index() would
        # always return 0 for these, causing all sleep times to be negative.
        # enumerate() returns the correct position, so sleep_time stays >= 0.
        vts_talk, vts_instance = _make_vts()
        vts_talk.is_connected = True
        # Constant waveform → all intensity blocks identical → triggers the bug
        waveform = torch.full((1, 4800), 0.5)
        sleep_times = []

        original_sleep = asyncio.sleep
        async def capturing_sleep(t):
            sleep_times.append(t)
            # don't actually sleep

        async def run():
            with patch("asyncio.sleep", side_effect=capturing_sleep):
                await vts_talk.sync_mouth(waveform, 48000)

        asyncio.run(run())
        # With correct enumerate, sleep times should be non-negative and increasing
        assert all(t >= 0 for t in sleep_times)


class TestExecuteAnimation:
    def test_triggers_hotkey_when_present(self):
        vts_talk, vts_instance = _make_vts()
        vts_talk.is_connected = True
        hotkey_name = "Wave"
        vts_instance.request.return_value = {
            "data": {"availableHotkeys": [{"name": hotkey_name}]}
        }

        asyncio.run(vts_talk.execute_animation(hotkey_name))
        assert vts_instance.request.call_count == 2  # list + trigger

    def test_does_not_trigger_when_hotkey_missing(self):
        vts_talk, vts_instance = _make_vts()
        vts_talk.is_connected = True
        vts_instance.request.return_value = {
            "data": {"availableHotkeys": [{"name": "OtherHotkey"}]}
        }

        asyncio.run(vts_talk.execute_animation("Wave"))
        assert vts_instance.request.call_count == 1  # only list call


class TestChangeExpression:
    def test_sends_mouth_open_request(self):
        vts_talk, vts_instance = _make_vts()
        vts_talk.is_connected = True

        asyncio.run(vts_talk.change_expression())
        vts_instance.request.assert_called_once()