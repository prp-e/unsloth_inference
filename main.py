import torch
from datetime import datetime
from unsloth import FastLanguageModel

model_name = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= model_name,
    load_in_4bit = True
)

print(f"Model is loaded on {model.device}")

prompt = "What is the meaning of life, universe and everything?"

messages = [
    {
        "role" : "user", 
        "content" : prompt 
    }
]

tokenized_text = tokenizer.apply_chat_template(messages,
tokenize = False,
add_generation_prompt = True
)

print(tokenized_text)

tokenized_text = tokenizer(tokenized_text, return_tensors="pt").to("cuda")
# print("The tokenizer makes your input like:")
# print(tokenized_text)

then = datetime.now()
generated_text = model.generate(**tokenized_text, max_new_tokens=1024)
result = tokenizer.decode(generated_text[0][len(model_inputs.input_ids[0]):].tolist(), skip_special_tokens=True)
now = datetime.now()

generate_time = then - now 
print(result)
print(f"\nTook {generate_time} seconds.")