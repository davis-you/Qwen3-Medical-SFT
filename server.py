"""
Qwen3 Medical 推理 API 服务（支持全参微调 + QLoRA）

启动方式:
    # QLoRA 模式（默认，Qwen3-8B + 4-bit 量化 + LoRA adapter，默认端口 6007）
    python server.py --mode lora --checkpoint ./output/Qwen3-8B/checkpoint-best

    # 全参微调模式（Qwen3-1.7B，直接加载 checkpoint）
    python server.py --mode full --model-path ./Qwen/Qwen3-1.7B --checkpoint ./output/Qwen3-1.7B/checkpoint-1084

    # 自定义端口
    python server.py --mode full --port 34567 --checkpoint ./output/Qwen3-1.7B/checkpoint-1084
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
SYSTEM_PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_NEW_TOKENS = 2048

logger = logging.getLogger("medical-server")

# ===================== 全局状态 =====================
model = None
tokenizer = None
inference_lock = threading.Lock()
current_mode = None


def load_model_lora(model_path: str, checkpoint_path: str):
    """QLoRA 模式：加载 4-bit 量化基座模型 + LoRA adapter"""
    global model, tokenizer

    logger.info("[lora] Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    logger.info("[lora] Loading base model in 4-bit from %s", model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    logger.info("[lora] Loading QLoRA adapter from %s", checkpoint_path)
    model = PeftModel.from_pretrained(model, model_id=checkpoint_path)
    model.eval()
    logger.info("[lora] Model loaded successfully")


def load_model_full(model_path: str, checkpoint_path: str):
    """全参微调模式：直接加载 checkpoint 完整权重"""
    global model, tokenizer

    logger.info("[full] Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    logger.info("[full] Loading model from %s", checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    logger.info("[full] Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时释放显存"""
    global current_mode
    mode = getattr(app.state, "mode", "lora")
    model_path = getattr(app.state, "model_path", "./Qwen/Qwen3-8B")
    checkpoint = getattr(app.state, "checkpoint_path", "./output/Qwen3-8B/checkpoint-best")
    current_mode = mode

    if mode == "lora":
        load_model_lora(model_path, checkpoint)
    else:
        load_model_full(model_path, checkpoint)
    yield
    global model, tokenizer
    del model, tokenizer
    torch.cuda.empty_cache()


# ===================== FastAPI =====================
app = FastAPI(
    title="Qwen3 Medical API",
    description="医疗推理模型 API（支持全参微调 + QLoRA）",
    version="1.1.0",
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
    mode: Optional[str] = None
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
        mode=current_mode,
        model_loaded=model is not None,
        gpu_memory_allocated_gb=round(allocated, 2),
        gpu_memory_reserved_gb=round(reserved, 2),
        gpu_total_memory_gb=round(total, 2),
    )


# ===================== 入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical API Server")
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--mode", type=str, default="lora", choices=["full", "lora"],
                        help="full: 全参微调模型, lora: QLoRA 微调模型")
    parser.add_argument("--model-path", type=str, default=None,
                        help="原始模型路径（默认: lora→./Qwen/Qwen3-8B, full→./Qwen/Qwen3-1.7B）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 路径（默认: lora→./output/Qwen3-8B/checkpoint-best, full→./output/Qwen3-1.7B/checkpoint-1084）")
    args = parser.parse_args()

    # 根据 mode 设置默认值
    if args.model_path is None:
        args.model_path = "./Qwen/Qwen3-8B" if args.mode == "lora" else "./Qwen/Qwen3-1.7B"
    if args.checkpoint is None:
        args.checkpoint = "./output/Qwen3-8B/checkpoint-best" if args.mode == "lora" else "./output/Qwen3-1.7B/checkpoint-1084"

    app.state.mode = args.mode
    app.state.model_path = args.model_path
    app.state.checkpoint_path = args.checkpoint

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting server: mode=%s, model=%s, checkpoint=%s", args.mode, args.model_path, args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)
