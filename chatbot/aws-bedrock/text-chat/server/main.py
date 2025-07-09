# server/main.py
import os, json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from botocore.config import Config
import boto3

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ตั้งค่า Bedrock Client
config = Config(read_timeout=60, connect_timeout=5)
client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    config=config
)

MODEL_ID = "amazon.nova-micro-v1:0"

async def ask_bedrock(message: str) -> str:
    # สร้าง prompt
    prompt = (
        "คุณคือผู้ช่วยอัจฉริยะด้านอาหารไทย กรุณาตอบกระชับ ไม่เกิน 1-2 ประโยค\n\n"
        f"คำถาม: {message}"
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "temperature": 0.7,
            "topP": 1.0,
            "stopSequences": [],
            "maxTokens": 300
        }
    }

    # เรียก Bedrock API
    response = client.converse(
        modelId=MODEL_ID,
        messages=payload["messages"],
        inferenceConfig=payload["inferenceConfig"]
    )

    return response["output"]["message"]["content"][0]["text"]

@app.post("/chat")
async def chat(body: dict):
    reply = await ask_bedrock(body.get("message", ""))
    return {"reply": reply}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")
            reply = await ask_bedrock(message)
            await websocket.send_text(reply)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()
