from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
import torch
from tqdm import tqdm
from config.config import MODEL_NAME, QUANT_CONFIG, LORA_CONFIG, TRAINING_ARGS, DATASET_PATH, MERGED_MODEL_DIR, OUTPUT_DIR
import json

# =====================
# MODEL & TOKENIZER SETUP
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=QUANT_CONFIG,
    device_map="auto",
    # attn_implementation="flash_attention_2"
)


model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LORA_CONFIG)

def format_instruction(example):
    return {
        "text": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]}
            ],
            tokenize=False,
            add_generation_prompt=False
        ) + tokenizer.eos_token
    }

def preprocess_data(example):
    return tokenizer(
        example["text"],
        max_length=1024,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt"
    )

# Load and process dataset
dataset = load_dataset("json", data_files=DATASET_PATH)
# dataset = dataset.map(validate_code_format)
dataset = dataset.map(format_instruction)
processed_data = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# =====================
# TRAINING SETUP
# =====================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(**TRAINING_ARGS)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data["train"],
    data_collator=data_collator,
    # callbacks=[CodeGenerationCallback()]
)

print("Starting training...")
trainer.train()
print("Training completed!")


model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# =====================
# MERGE & EXPORT
# =====================
print("Merging adapter weights...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=QUANT_CONFIG,
    device_map="auto"
)
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)