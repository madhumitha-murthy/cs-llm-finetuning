"""
Stage 2: DPO (Direct Preference Optimisation) — alternative to RLHF
Uses SFT model outputs as chosen, base model outputs as rejected.
"""

import mlflow
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MODEL_DIR  = "./outputs/sft/final"
DATASET_ID     = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR     = "./outputs/dpo"
MLFLOW_EXP     = "cs-llm-finetuning"
MAX_SEQ_LEN    = 512
MAX_SAMPLES    = 2000   # subset for DPO preference pairs
BATCH_SIZE     = 2
GRAD_ACCUM     = 8
EPOCHS         = 1
LR             = 5e-5
BETA           = 0.1    # DPO temperature


# ── Generate preference pairs ─────────────────────────────────────────────────
def generate_preference_pairs(tokenizer, base_model, sft_model, prompts, max_new=150):
    """
    chosen  = SFT model response (better, aligned)
    rejected = base model response (worse, unaligned)
    """
    pairs = []
    base_model.eval()
    sft_model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            base_out = base_model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=0.9, pad_token_id=tokenizer.eos_token_id
            )
            sft_out = sft_model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )

        base_response = tokenizer.decode(
            base_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        sft_response = tokenizer.decode(
            sft_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        pairs.append({
            "prompt": prompt,
            "chosen": sft_response,
            "rejected": base_response,
        })

    return pairs


def build_prompt(instruction):
    return (
        f"<|im_start|>system\n"
        f"You are a helpful e-commerce customer service assistant.\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    mlflow.set_experiment(MLFLOW_EXP)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load SFT model
    print("Loading SFT model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    sft_model = PeftModel.from_pretrained(sft_model, SFT_MODEL_DIR)
    sft_model = sft_model.merge_and_unload()

    # Load dataset — use a subset for DPO pairs
    print("Generating preference pairs...")
    raw_ds = load_dataset(DATASET_ID, split=f"train[:{MAX_SAMPLES}]")
    prompts = [build_prompt(ex["instruction"]) for ex in raw_ds]

    pairs = generate_preference_pairs(tokenizer, base_model, sft_model, prompts)
    dpo_dataset = Dataset.from_list(pairs)
    dpo_dataset = dpo_dataset.train_test_split(test_size=0.1, seed=42)

    # Free base model from GPU before DPO training
    del base_model
    torch.cuda.empty_cache()

    # DPO training on SFT model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        beta=BETA,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="mlflow",
        run_name="dpo-qwen2.5-0.5b",
        max_length=MAX_SEQ_LEN,
        max_prompt_length=256,
    )

    trainer = DPOTrainer(
        model=sft_model,
        ref_model=None,   # uses implicit reference via PEFT adapter
        args=dpo_config,
        train_dataset=dpo_dataset["train"],
        eval_dataset=dpo_dataset["test"],
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    with mlflow.start_run(run_name="dpo"):
        mlflow.log_params({
            "beta": BETA,
            "lr": LR,
            "max_samples": MAX_SAMPLES,
            "epochs": EPOCHS,
        })
        trainer.train()
        trainer.save_model(OUTPUT_DIR + "/final")
        tokenizer.save_pretrained(OUTPUT_DIR + "/final")
        print(f"DPO model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    train()
