"""
Evaluation: Compare base vs SFT vs SFT+DPO
Metrics: ROUGE-L, BERTScore, Intent Accuracy
All results logged to MLflow.
"""

import json
import time

import mlflow
import torch
from bert_score import score as bert_score
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID      = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DIR       = "./outputs/sft/final"
DPO_DIR       = "./outputs/dpo/final"
MLFLOW_EXP    = "cs-llm-finetuning"
TEST_SAMPLES  = 200


def build_prompt(instruction):
    return (
        f"<|im_start|>system\n"
        f"You are a helpful e-commerce customer service assistant.\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_responses(model, tokenizer, prompts, max_new=150):
    responses = []
    latencies = []
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latencies.append(time.time() - start)
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        responses.append(response.strip())
    return responses, latencies


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def compute_bert_score(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return F1.mean().item()


def compute_intent_accuracy(predictions, intents):
    """Checks if the predicted response contains any keyword from the intent label."""
    intent_keywords = {
        "cancel_order": ["cancel", "cancellation"],
        "track_order": ["track", "tracking", "shipment", "delivery"],
        "refund": ["refund", "money back", "reimburs"],
        "complaint": ["sorry", "apologize", "inconvenience"],
        "payment_issue": ["payment", "charge", "billing"],
    }
    correct = 0
    for pred, intent in zip(predictions, intents):
        keywords = intent_keywords.get(intent.lower(), [intent.lower()])
        if any(kw in pred.lower() for kw in keywords):
            correct += 1
    return correct / len(predictions)


def evaluate_model(name, model, tokenizer, prompts, references, intents):
    print(f"\nEvaluating: {name}")
    predictions, latencies = generate_responses(model, tokenizer, prompts)

    rouge_l   = compute_rouge_l(predictions, references)
    bs_f1     = compute_bert_score(predictions, references)
    intent_acc = compute_intent_accuracy(predictions, intents)
    avg_lat   = sum(latencies) / len(latencies) * 1000  # ms

    results = {
        "rouge_l":         round(rouge_l, 4),
        "bertscore_f1":    round(bs_f1, 4),
        "intent_accuracy": round(intent_acc, 4),
        "avg_latency_ms":  round(avg_lat, 1),
    }
    print(json.dumps(results, indent=2))
    return results, predictions


def run_evaluation():
    mlflow.set_experiment(MLFLOW_EXP)

    # Load test data
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split=f"train[:{TEST_SAMPLES}]"
    )
    prompts    = [build_prompt(ex["instruction"]) for ex in ds]
    references = [ex["response"] for ex in ds]
    intents    = [ex["intent"] for ex in ds]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # ── Base model ────────────────────────────────────────────────────────────
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    results, _ = evaluate_model("base", base_model, tokenizer, prompts, references, intents)
    all_results["base"] = results
    del base_model
    torch.cuda.empty_cache()

    # ── SFT model ─────────────────────────────────────────────────────────────
    sft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    sft_model = PeftModel.from_pretrained(sft_base, SFT_DIR)
    results, _ = evaluate_model("sft", sft_model, tokenizer, prompts, references, intents)
    all_results["sft"] = results
    del sft_model, sft_base
    torch.cuda.empty_cache()

    # ── SFT + DPO model ───────────────────────────────────────────────────────
    dpo_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    dpo_model = PeftModel.from_pretrained(dpo_base, DPO_DIR)
    results, _ = evaluate_model("sft_dpo", dpo_model, tokenizer, prompts, references, intents)
    all_results["sft_dpo"] = results
    del dpo_model, dpo_base
    torch.cuda.empty_cache()

    # ── Log all to MLflow ─────────────────────────────────────────────────────
    with mlflow.start_run(run_name="evaluation-comparison"):
        for stage, metrics in all_results.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{stage}_{metric_name}", value)

        mlflow.log_dict(all_results, "evaluation_results.json")

    print("\n=== Final Comparison ===")
    for stage, metrics in all_results.items():
        print(f"\n{stage.upper()}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
