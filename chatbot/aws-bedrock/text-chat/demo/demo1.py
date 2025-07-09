# main.py
import os
import json
import boto3
from dotenv import load_dotenv
from botocore.config import Config

# โหลด .env ที่มี AWS Key และ Region
load_dotenv()

# สร้าง Bedrock client ด้วย timeout ที่เหมาะกับ model ขนาดใหญ่
config = Config(read_timeout=3600, connect_timeout=5)
client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    config=config
)

# เลือกโมเดล: Amazon Nova Micro (Text → Text)
model_id = "amazon.nova-micro-v1:0"

# ข้อความที่ต้องการให้ AI ตอบ (พร้อมระบบคำสั่งใน prompt)
messages = [
    {
        "role": "user",
        "content": [
            {
                "text": (
                    "คุณคือผู้ช่วยอัจฉริยะด้านอาหารไทย "
                    "กรุณาตอบกลับด้วย *ประโยคเดียวเท่านั้น* "
                    "ห้ามแสดงหัวข้อย่อย ห้ามเกินหนึ่งประโยค "
                    "คำถาม: ต้มยำกุ้งกับแกงส้มต่างกันยังไง?"
                )
            }
        ]
    }
]

# เรียกใช้งานโมเดลผ่าน converse API
response = client.converse(
    modelId=model_id,
    messages=messages,
    inferenceConfig={
        "temperature": 0.7,
        "topP": 1.0,
        "stopSequences": [],
        "maxTokens": 300
    }
)

# แสดงผลลัพธ์
ai_message = response["output"]["message"]["content"][0]["text"]
print("🥘 คำตอบแบบประโยคเดียว:\n")
print(ai_message)
