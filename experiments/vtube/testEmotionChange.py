"""
Manual expression tester for VTube Studio.

Usage:
    python testEmotionChange.py

Enter a mood value (0-10) to trigger the matching IAra expression hotkey,
or type the hotkey name directly. Ctrl+C to quit.

Expected hotkeys in VTube Studio:
  IAra_Angry   — mood 0-2
  IAra_Sad     — mood 3-4
  IAra_Neutral — mood 5
  IAra_Happy   — mood 6-7
  IAra_Excited — mood 8-10
"""
import asyncio
import os
import sys
sys.path.insert(0, "../../")

os.chdir("../../")  # token.txt lives in the project root

from iara.vtube import VTubeStudioTalk


async def main():
    vts = VTubeStudioTalk()
    await vts.connect()
    print("Connected to VTube Studio.")
    print("Enter a mood value (0-10) or a hotkey name directly. Ctrl+C to quit.\n")

    while True:
        try:
            raw = input("mood / hotkey > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not raw:
            continue

        if raw.isdigit():
            mood = max(0, min(10, int(raw)))
            hotkey = vts._mood_to_hotkey(mood)
            print(f"  mood {mood}/10 → {hotkey}")
        else:
            hotkey = raw
            print(f"  triggering → {hotkey}")

        await vts.execute_animation(hotkey)


if __name__ == "__main__":
    asyncio.run(main())
