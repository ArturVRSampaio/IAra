import asyncio
from VTubeStudioTalk import VTubeStudioTalk


async def main():
    vts_talk = VTubeStudioTalk()

    # emotions = {
    #     'surprise' : 'O形嘴',
    #     'ultra_surprise' : '呆呆脸',
    #     'shy' : '害羞脸',
    #     'hands_up' : '挥挥手',
    #     'happy cl' : '眯眯眼脸',
    #     '瞳孔缩小',
    #     'blush' : '脸红',
    #     '脸黑',
    #     neutral : ''}


    # Connect to VTube Studio
    await vts_talk.connect()

    await vts_talk.change_emotion('')


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())