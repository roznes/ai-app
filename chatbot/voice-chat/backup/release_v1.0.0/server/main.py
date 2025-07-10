# server/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx, json, os
from tts_async import speak_and_stream
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OLLAMA_URL = "http://127.0.0.1:11555"
MODEL_NAME = "scb10x/llama3.2-typhoon2-3b-instruct"

# Ensure audio directory exists
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

class ChatInput(BaseModel):
    message: str
    model: str = MODEL_NAME

async def generate_reply(message: str, model: str = MODEL_NAME) -> str:
    system_prompt = "คุณคือผู้ช่วยตอบคำถามเรื่องอาหารไทยเท่านั้น ตอบสั้น กระชับ และใช้ภาษาธรรมชาติ"
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{message}\n<|assistant|>\n"
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        res.raise_for_status()
        return res.json()["response"].strip()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")
            model = payload.get("model", MODEL_NAME)
            reply = await generate_reply(message, model)
            mp3_path = await speak_and_stream(reply)
            await websocket.send_json({
                "text": reply,
                "audio": f"/audio/{os.path.basename(mp3_path)}"
            })
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
