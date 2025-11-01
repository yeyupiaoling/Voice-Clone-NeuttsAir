import os
import time
import uuid
from contextlib import asynccontextmanager

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from utils.neutts import NeuTTSAir

model: NeuTTSAir | None = None
# 模型路径，可以修改不同模型以支持不同语言
model_dir = "./models/neutts-air-zh"
language = "cmn"


def init_model():
    global model
    # 要运行的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    logger.info("正在加载模型...")
    model = NeuTTSAir(
        backbone_repo=model_dir,
        backbone_device=device,
        codec_repo="./models/neucodec",
        codec_device=device,
        language=language
    )
    logger.info("模型加载完成")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield


os.makedirs("temp/", exist_ok=True)
os.makedirs("static/tts/", exist_ok=True)

app = FastAPI(title="夜雨飘零语音克隆合成服务", lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/infer")
async def infer(request: Request,
                text: str = Body(..., description="要合成的文本"),
                ref_audio: UploadFile = File(..., description="要克隆的音频"),
                ref_text: str = Body(..., description="要克隆的音频中对应的文本")):
    start_time = time.time()
    # 保存参考音频
    safe_name = ref_audio.filename.replace("..", "_").replace("/", "_").replace("\\", "_")
    ref_audio_path = f"temp/{safe_name}"
    with open(ref_audio_path, "wb") as f:
        f.write(await ref_audio.read())

    # 编码与合成
    ref_codes = model.encode_reference(ref_audio_path)
    wav = model.infer(text, ref_codes, ref_text)

    # 写入结果音频
    out_name = f"{str(uuid.uuid4())}.wav"
    save_path = f"static/tts/{out_name}"
    sf.write(save_path, wav, 24000)
    os.remove(ref_audio_path)
    logger.info(f"推理完成，耗时 {int((time.time() - start_time) * 1000)} ms")

    return JSONResponse({
        "success": True,
        "audio_url": f"/static/tts/{out_name}",
        "sample_rate": 24000
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
