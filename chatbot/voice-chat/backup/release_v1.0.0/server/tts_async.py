# server/tts_async.py
import edge_tts
import asyncio
import os

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

def cleanup_old_audio_files(max_files=5):
    files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")]
    files.sort(key=os.path.getmtime)  # เรียงจากไฟล์ที่เก่าที่สุด
    while len(files) > max_files:
        os.remove(files.pop(0))  # ลบไฟล์ที่เก่าที่สุด

async def speak_and_stream(text: str, lang="th-TH", voice="th-TH-PremwadeeNeural"):
    filename = f"{int(asyncio.get_event_loop().time() * 1000)}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filepath)

    # ✨ เรียกใช้ฟังก์ชันลบไฟล์เก่า
    cleanup_old_audio_files(max_files=5)

    return filepath
