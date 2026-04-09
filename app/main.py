"""
FastAPI inference server — serves the quantised SFT+DPO model.
Endpoints:
  POST /chat      — generate a customer service response
  GET  /health    — health check
  GET  /metrics   — latency stats
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from bitsandbytes import BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "../outputs/quantised/merged"
MAX_NEW_TOKENS = 200

# ── Global model state ────────────────────────────────────────────────────────
model     = None
tokenizer = None
latency_log = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading quantised model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    yield
    del model, tokenizer


app = FastAPI(
    title="Customer Service Conversational AI",
    description="E-commerce customer service LLM — Qwen2.5-0.5B SFT+DPO+INT4",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = (
        "You are a helpful e-commerce customer service assistant. "
        "Handle customer queries about orders, refunds, shipping, and disputes accurately."
    )
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS


class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    tokens_generated: int


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = (
        f"<|im_start|>system\n{req.system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{req.message}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    latency_log.append(latency_ms)

    response_ids = outputs[0][input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    tokens_generated = len(response_ids)

    return ChatResponse(
        response=response_text,
        latency_ms=round(latency_ms, 2),
        tokens_generated=tokens_generated,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    if not latency_log:
        return MetricsResponse(total_requests=0, avg_latency_ms=0.0, p95_latency_ms=0.0)
    sorted_lat = sorted(latency_log)
    return MetricsResponse(
        total_requests=len(latency_log),
        avg_latency_ms=round(sum(latency_log) / len(latency_log), 2),
        p95_latency_ms=round(sorted_lat[int(0.95 * len(sorted_lat))], 2),
    )
