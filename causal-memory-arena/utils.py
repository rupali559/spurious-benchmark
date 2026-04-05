import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cuda:0",
)
model.eval()

print("Model loaded.")


def call_llm(prompt, system_prompt="", model_name=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def call_llm_json(prompt, system_prompt="", model_name=None):
    raw = call_llm(prompt, system_prompt)
    return extract_json(raw)


def extract_json(text):
    # handle ```json blocks
    code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except:
            pass

    # fallback: find outermost { }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass

    raise ValueError(f"No valid JSON found in: {repr(text)}")