"""
Stage 3: INT4 Quantisation (model compression) for edge AI deployment.
Benchmarks latency and model size across all 4 configs:
  base → SFT → SFT+DPO → quantised (INT4)
"""

import json
import os
import time

import mlflow
import torch
from bitsandbytes import BitsAndBytesConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID     = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DIR      = "./outputs/sft/final"
DPO_DIR      = "./outputs/dpo/final"
QUANT_DIR    = "./outputs/quantised"
MLFLOW_EXP   = "cs-llm-finetuning"
BENCH_RUNS   = 50
MAX_NEW      = 100


def get_model_size_gb(model):
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return round(total_bytes / 1e9, 3)


def benchmark_latency(model, tokenizer, prompt, runs=BENCH_RUNS):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    latencies = []
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=MAX_NEW,
                           pad_token_id=tokenizer.eos_token_id)
        for _ in range(runs):
            start = time.perf_counter()
            model.generate(**inputs, max_new_tokens=MAX_NEW,
                           pad_token_id=tokenizer.eos_token_id)
            latencies.append(time.perf_counter() - start)
    avg_ms  = round(sum(latencies) / len(latencies) * 1000, 1)
    p95_ms  = round(sorted(latencies)[int(0.95 * runs)] * 1000, 1)
    return avg_ms, p95_ms


SAMPLE_PROMPT = (
    "<|im_start|>system\nYou are a helpful e-commerce customer service assistant.\n<|im_end|>\n"
    "<|im_start|>user\nI want to cancel my order and get a refund.\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def run_quantisation():
    mlflow.set_experiment(MLFLOW_EXP)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ── 1. Base ───────────────────────────────────────────────────────────────
    print("Benchmarking base model (FP16)...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    avg, p95 = benchmark_latency(base, tokenizer, SAMPLE_PROMPT)
    results["base"] = {"size_gb": get_model_size_gb(base), "avg_ms": avg, "p95_ms": p95}
    del base; torch.cuda.empty_cache()

    # ── 2. SFT ────────────────────────────────────────────────────────────────
    print("Benchmarking SFT model (FP16)...")
    sft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    sft = PeftModel.from_pretrained(sft_base, SFT_DIR)
    avg, p95 = benchmark_latency(sft, tokenizer, SAMPLE_PROMPT)
    results["sft"] = {"size_gb": get_model_size_gb(sft), "avg_ms": avg, "p95_ms": p95}
    del sft, sft_base; torch.cuda.empty_cache()

    # ── 3. SFT + DPO ──────────────────────────────────────────────────────────
    print("Benchmarking SFT+DPO model (FP16)...")
    dpo_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    dpo = PeftModel.from_pretrained(dpo_base, DPO_DIR)
    avg, p95 = benchmark_latency(dpo, tokenizer, SAMPLE_PROMPT)
    results["sft_dpo"] = {"size_gb": get_model_size_gb(dpo), "avg_ms": avg, "p95_ms": p95}

    # Merge DPO for quantisation
    dpo_merged = dpo.merge_and_unload()
    del dpo; torch.cuda.empty_cache()

    # ── 4. INT4 Quantised ─────────────────────────────────────────────────────
    print("Quantising SFT+DPO model to INT4 (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Save merged model first, then reload with quantisation config
    dpo_merged.save_pretrained(QUANT_DIR + "/merged")
    tokenizer.save_pretrained(QUANT_DIR + "/merged")
    del dpo_merged; torch.cuda.empty_cache()

    quant_model = AutoModelForCausalLM.from_pretrained(
        QUANT_DIR + "/merged",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    avg, p95 = benchmark_latency(quant_model, tokenizer, SAMPLE_PROMPT)
    results["quantised_int4"] = {
        "size_gb": get_model_size_gb(quant_model),
        "avg_ms":  avg,
        "p95_ms":  p95,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    base_size = results["base"]["size_gb"]
    quant_size = results["quantised_int4"]["size_gb"]
    base_lat  = results["base"]["avg_ms"]
    quant_lat = results["quantised_int4"]["avg_ms"]

    compression_pct = round((1 - quant_size / base_size) * 100, 1)
    speedup_pct     = round((1 - quant_lat / base_lat) * 100, 1)

    results["summary"] = {
        "size_reduction_pct":    compression_pct,
        "latency_improvement_pct": speedup_pct,
    }

    print("\n=== Quantisation Benchmark ===")
    for config, metrics in results.items():
        print(f"\n{config.upper()}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Log to MLflow
    with mlflow.start_run(run_name="quantisation-benchmark"):
        for config, metrics in results.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"{config}_{k}", v)
        mlflow.log_dict(results, "quantisation_results.json")

    with open("quantisation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSize reduction:    {compression_pct}%")
    print(f"Latency improvement: {speedup_pct}%")
    print("\nUpdate these in your CV!")

    return results


if __name__ == "__main__":
    run_quantisation()
