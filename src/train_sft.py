"""
Stage 1: Supervised Fine-Tuning (SFT) with LoRA
Model  : Qwen/Qwen2.5-0.5B-Instruct
Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
"""

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID     = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR     = "./outputs/sft"
MLFLOW_EXP     = "cs-llm-finetuning"
MAX_SEQ_LEN    = 512
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
EPOCHS         = 3
LR             = 2e-4
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_and_format():
    ds = load_dataset(DATASET_ID, split="train")

    # 90/10 train/val split
    ds = ds.train_test_split(test_size=0.1, seed=42)

    def format_example(example):
        return {
            "text": (
                f"<|im_start|>system\n"
                f"You are a helpful e-commerce customer service assistant. "
                f"Answer customer queries accurately and concisely.\n<|im_end|>\n"
                f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
                f"<|im_start|>assistant\n{example['response']}\n<|im_end|>"
            )
        }

    ds = ds.map(format_example, remove_columns=ds["train"].column_names)
    return ds["train"], ds["test"]


# ── Model + LoRA ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    mlflow.set_experiment(MLFLOW_EXP)

    train_ds, eval_ds = load_and_format()
    model, tokenizer = load_model_and_tokenizer()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="mlflow",
        run_name="sft-qwen2.5-0.5b-lora",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    with mlflow.start_run(run_name="sft"):
        mlflow.log_params({
            "model": MODEL_ID,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        })
        trainer.train()
        trainer.save_model(OUTPUT_DIR + "/final")
        tokenizer.save_pretrained(OUTPUT_DIR + "/final")
        print(f"SFT model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    train()
