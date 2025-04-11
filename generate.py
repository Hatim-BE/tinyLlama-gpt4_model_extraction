# response_generator.py
from transformers import pipeline, AutoTokenizer
import torch
from config.config import MERGED_MODEL_DIR, GENERATION_CONFIG


tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)
generator = pipeline(
    "text-generation",
    model=MERGED_MODEL_DIR,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    **GENERATION_CONFIG
)

def generate_response(prompt):
    """Generate a response for the given prompt using the fine-tuned model."""
    messages = [{"role": "user", "content": prompt}]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    result = generator(
        formatted_prompt,
        return_full_text=False
    )
    
    return result[0]["generated_text"]

if __name__ == "__main__":
    test_prompt = "How to add padding between elements in a Row?"
    response = generate_response(test_prompt)
    print("\nTest Response:")
    print(response)