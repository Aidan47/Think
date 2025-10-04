import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import subprocess
import time
from datasets import load_dataset



def load_model():
    model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return model, tokenizer


def eval(input, output):
    filename = "temp.lean"
    proof = input + output

    try:
        with open(filename, "w") as f:
            f.write(proof)

        result = subprocess.run(
            ["lean", filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        return result.returncode != 0

    finally:    # remove temp file
        if os.path.exists(filename):
            os.remove(filename)



torch.manual_seed(30)
model, tokenizer = load_model()
minif2f_test = load_dataset("AI-MO/minif2f_test", split="train")

prompt = """
    Complete the following Lean 4 code:

    ```lean4
    {}
    ```

    Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
    The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
    """.strip()

correct = 0
for theorem in minif2f_test[:]['formal_statement']:
    start = time.time()
    chat = [
        {"role": "user", "content": prompt.format(theorem)}
    ]
    input = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    output = model.generate(input, max_new_tokens=8192)
    proof = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # get validity of proof
    correct += eval(theorem, proof)
    print(f"correct: {correct}")
    print(time.time() - start)

print(f"{correct / len(minif2f_test)}")