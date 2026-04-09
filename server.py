"""
Qwen3-8B Medical QLoRA 推理 API 服务

启动方式:
    python server.py                                           # 默认 8000 端口
    python server.py --port 8080                               # 自定义端口
    python server.py --checkpoint ./output/Qwen3-8B/checkpoint-200  # 指定 checkpoint
"""

import argparse
import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# ===================== 配置 =====================
MODEL_LOCAL_PATH = "./Qwen/Qwen3-8B"
DEFAULT_CHECKPOINT = "./output/Qwen3-8B/checkpoint-best"
SYSTEM_PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_NEW_TOKENS = 2048

logger = logging.getLogger("medical-server")

# ===================== 全局状态 =====================
model = None
tokenizer = None
inference_lock = threading.Lock()


def load_model(checkpoint_path: str):
    """加载 4-bit 量化基座模型 + QLoRA adapter"""
    global model, tokenizer

    logger.info("Loading tokenizer from %s", MODEL_LOCAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_PATH, use_fast=False, trust_remote_code=True
    )

    logger.info("Loading base model in 4-bit from %s", MODEL_LOCAL_PATH)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    logger.info("Loading QLoRA adapter from %s", checkpoint_path)
    model = PeftModel.from_pretrained(model, model_id=checkpoint_path)
    model.eval()
    logger.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时释放显存"""
    checkpoint = getattr(app.state, "checkpoint_path", DEFAULT_CHECKPOINT)
    load_model(checkpoint)
    yield
    global model, tokenizer
    del model, tokenizer
    torch.cuda.empty_cache()


# ===================== FastAPI =====================
app = FastAPI(
    title="Qwen3-8B Medical API",
    description="QLoRA 微调医疗推理模型 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 请求/响应模型 =====================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096, description="医学问题")
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")
    max_tokens: Optional[int] = Field(default=MAX_NEW_TOKENS, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(default=False, description="是否流式输出")


class ChatResponse(BaseModel):
    question: str
    response: str
    thinking: Optional[str] = None
    answer: Optional[str] = None
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    gpu_total_memory_gb: float


# ===================== 推理逻辑 =====================
def generate_response(question: str, system_prompt: str, max_tokens: int,
                      temperature: float, top_p: float) -> str:
    """同步生成回答"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
        )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def generate_stream(question: str, system_prompt: str, max_tokens: int,
                    temperature: float, top_p: float):
    """流式生成，逐 token 返回"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        input_ids=model_inputs.input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        do_sample=temperature > 0,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text_chunk in streamer:
        yield f"data: {text_chunk}\n\n"

    thread.join()
    yield "data: [DONE]\n\n"


def parse_thinking_response(full_response: str):
    """解析 R1 风格的 <think>...</think> 响应，拆分为思考过程和最终回答"""
    thinking = None
    answer = full_response

    if "<think>" in full_response and "</think>" in full_response:
        think_start = full_response.index("<think>") + len("<think>")
        think_end = full_response.index("</think>")
        thinking = full_response[think_start:think_end].strip()
        answer = full_response[think_end + len("</think>"):].strip()

    return thinking, answer


# ===================== API 端点 =====================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """非流式问答接口"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    system_prompt = request.system_prompt or SYSTEM_PROMPT
    start_time = time.time()

    try:
        with inference_lock:
            response = await asyncio.to_thread(
                generate_response,
                request.question, system_prompt,
                request.max_tokens, request.temperature, request.top_p,
            )
    except Exception as e:
        logger.error("Generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

    elapsed = time.time() - start_time
    thinking, answer = parse_thinking_response(response)

    return ChatResponse(
        question=request.question,
        response=response,
        thinking=thinking,
        answer=answer,
        elapsed_seconds=round(elapsed, 2),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式问答接口（Server-Sent Events）"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    system_prompt = request.system_prompt or SYSTEM_PROMPT

    def locked_stream():
        with inference_lock:
            yield from generate_stream(
                request.question, system_prompt,
                request.max_tokens, request.temperature, request.top_p,
            )

    return StreamingResponse(locked_stream(), media_type="text/event-stream")


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查 + GPU 显存状态"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    else:
        allocated = reserved = total = 0.0

    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model_loaded=model is not None,
        gpu_memory_allocated_gb=round(allocated, 2),
        gpu_memory_reserved_gb=round(reserved, 2),
        gpu_total_memory_gb=round(total, 2),
    )


# ===================== 入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical QLoRA API Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    app.state.checkpoint_path = args.checkpoint

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
