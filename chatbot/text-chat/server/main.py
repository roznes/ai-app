# server\main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, json

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OLLAMA_URL = "http://127.0.0.1:11555"
MODEL_NAME = "qwen2.5"

class ChatInput(BaseModel):
    message: str
    model: str = MODEL_NAME

async def generate_reply(message: str, model: str = MODEL_NAME) -> str:
    system_prompt = "คุณคือผู้ช่วยที่ตอบคำถามเกี่ยวกับอาหารไทยเท่านั้น กรุณาตอบเป็นภาษาไทยแบบกระชับ ไม่เกิน 2-3 บรรทัด"
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{message}\n<|assistant|>\n"
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        res.raise_for_status()
        return res.json()["response"].strip()

@app.post("/chat")
async def chat(input: ChatInput):
    reply = await generate_reply(input.message, input.model)
    return {"reply": reply}

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
            await websocket.send_text(reply)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        await websocket.close()