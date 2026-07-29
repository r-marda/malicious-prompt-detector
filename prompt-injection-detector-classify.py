import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings

warnings.simplefilter("ignore")

BASE_MODEL_G = "google/gemma-3-270m"
BASE_MODEL_Q = "Qwen/Qwen3-0.6B"

ADAPTER_G = "rmarda/prompt-injection-detector-local-inference-gemma"
ADAPTER_Q = "rmarda/prompt-injection-detector-local-inference-qwen"

# ----------------------------
# Choose model
# ----------------------------
model_choice = input("Choose a model (G for Gemma, Q for Qwen): ").strip().upper()

if model_choice == "G":
    BASE_MODEL = BASE_MODEL_G
    ADAPTER = ADAPTER_G
elif model_choice == "Q":
    BASE_MODEL = BASE_MODEL_Q
    ADAPTER = ADAPTER_Q
else:
    raise ValueError("Invalid model choice. Please choose 'G' or 'Q'.")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    dtype=torch.float32,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
)

model.eval()

print("=" * 60)
print("Prompt Injection Detector")
print("Type 'quit' to exit.")
print("=" * 60)

if model_choice == "G":
    print("Using Gemma model.\n")

    while True:
        user_prompt = input("Enter prompt: ").strip()

        if user_prompt.lower() in {"quit", "exit"}:
            break

        prompt = f"""Classify the following prompt as either Benign or Malicious.

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
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        prediction = generated[len(prompt):].strip()

        print("\nPrediction:")
        print(prediction)
        print("-" * 60)

else:
    print("Using Qwen model.\n")

    while True:
        user_prompt = input("Enter prompt: ").strip()

        if user_prompt.lower() in {"quit", "exit"}:
            break

        messages = [
            {
                "role": "user",
                "content": f"""Classify the following prompt as either Benign or Malicious.

Prompt:
{user_prompt}

Answer with only one word: Benign or Malicious."""
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        prediction = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        prediction = prediction.split()[0].strip(".,:;")

        print("\nPrediction:")
        print(prediction)
        print("-" * 60)