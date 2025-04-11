# config.py

from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch

# Base model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./jetpack-compose-finetuned"
DATASET_PATH = "./combined_cleaned.json"
MERGED_MODEL_DIR = "./merged-model"
# Tokenizer settings
PAD_TOKEN = "<|endoftext|>"  # We'll override this with tokenizer.eos_token

# Quantization (4-bit)
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA config
PEFT_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head"]
)

# Training hyperparameters
TRAINING_ARGS = {
    "output_dir": "./models/tinyllama-finetuned",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "save_total_limit": 2,
    "logging_steps": 10,
    "optim": "paged_adamw_8bit",
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "report_to": "none",
    "gradient_checkpointing": True
}


TRAINING_ARGS = {
    "output_dir":OUTPUT_DIR,
    "per_device_train_batch_size":4,
    "gradient_accumulation_steps":1,
    "num_train_epochs":5,
    "learning_rate":1e-5,
    "weight_decay":0.01,
    "max_grad_norm":0.5,
    "bf16":True,
    "optim":"adamw_torch_fused",
    "logging_steps":25,
    "save_strategy":"steps",
    "save_steps":100,
    "evaluation_strategy":"no",
    "group_by_length":True,
    "gradient_checkpointing":True,
    "report_to":"none",
    "max_steps":300
}