# server/main.py
import os, json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from botocore.config import Config
import boto3
from tts_async import speak_and_stream

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

config = Config(read_timeout=60, connect_timeout=5)
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    config=config
)

MODEL_ID = "amazon.nova-micro-v1:0"

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

async def ask_bedrock(message: str) -> str:
    prompt = f"คุณคือผู้ช่วยตอบคำถามเรื่องอาหารไทย ตอบสั้น กระชับ\nคำถาม: {message}"

    response = bedrock_client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.7, "maxTokens": 300}
    )

    return response["output"]["message"]["content"][0]["text"]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")
            reply = await ask_bedrock(message)
            mp3_path = await speak_and_stream(reply)
            await websocket.send_json({
                "text": reply,
                "audio": f"/audio/{os.path.basename(mp3_path)}"
            })
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()