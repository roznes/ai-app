# main.py
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# โหลด API Key จาก .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# เชื่อมต่อ OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# กำหนด System Prompt (คงที่ตลอด session)
messages = [
    {
        "role": "system",
        "content": (
            "คุณคือผู้ช่วย AI ที่ตอบคำถามเป็นภาษาไทย "
            "โดยต้องตอบแบบสั้นและกระชับ ไม่เกิน 2 ประโยค "
            "และเสนอว่าจะให้ข้อมูลเพิ่มเติมหากผู้ใช้ต้องการ"
        )
    }
]

print("🟢 เริ่มใช้งาน Chatbot แล้ว พิมพ์ 'exit' เพื่อจบ\n")

# Loop ถาม-ตอบ
while True:
    user_input = input("👤 คุณ: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 จบการสนทนา")
        break

    # เพิ่มคำถามลงในประวัติข้อความ
    messages.append({"role": "user", "content": user_input})

    # เริ่มจับเวลา
    start_time = time.time()

    # ขอคำตอบจาก AI
    response = client.chat.completions.create(
        model="qwen/qwen3-8b:free",
        messages=messages
    )

    # จับเวลา
    end_time = time.time()
    elapsed = end_time - start_time

    # เพิ่มคำตอบ AI ลงในประวัติ
    assistant_reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": assistant_reply})

    # แสดงผล
    print(f"🤖 AI: {assistant_reply}")
    print(f"⏱️ {elapsed:.2f} วินาที\n")
