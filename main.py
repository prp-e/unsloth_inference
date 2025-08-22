import torch
from unsloth import FastLanguageModel

model_name = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= model_name,
    load_in_4bit = True
)

print(f"Model is loaded on {model.device}")