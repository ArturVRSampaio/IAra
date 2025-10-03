import asyncio
from VTubeStudioTalk import VTubeStudioTalk


async def main():
    vts_talk = VTubeStudioTalk()

    await vts_talk.connect()
    while True:
        await vts_talk.change_expression()

if __name__ == "__main__":
    asyncio.run(main())