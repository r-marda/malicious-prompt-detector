import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
import json

import transformers, peft

warnings.simplefilter("ignore")

BASE_MODEL = "google/gemma-3-270m"
ADAPTER = "rmarda/prompt-injection-detector-local-inference-gemma-reasoning"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.float32,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
)

model.eval()

print("=" * 60)
print("Prompt Injection Detector (Gemma Reasoning)")
print("Type 'quit' to exit.")
print("=" * 60)

while True:

    user_prompt = input("Enter prompt: ").strip()

    if user_prompt.lower() in {"quit", "exit"}:
        break

    prompt = f"""Classify the following prompt as either Benign or Malicious, and provide a reason why.

Prompt:
{user_prompt}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0]

    # Decode only newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    try:
        obj = json.loads(prediction)
    except json.JSONDecodeError:
        obj = {
            "label": "Malicious",
            "reasoning": "Default response due to invalid model output."
        }

    print("\nPrediction")
    print("-" * 60)
    print(f"Label      : {obj['label']}")
    print(f"Reasoning  : {obj['reasoning']}")
    print("-" * 60)