import os
import re
import subprocess
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model():
    model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return model, tokenizer


def clean(input, output):
    # input: get headers before theorem
    theorem_start_index = input.find("theorem ")

    if theorem_start_index != -1:
        preamble = input[:theorem_start_index]
    else:
        preamble = input

    preamble = preamble.strip()

    # output: theorem and proof statements
    pattern = r"### Complete Lean 4 Proof.*?```lean4\s*\n(.*?)\n\s*```"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        # group(1) contains the captured proof code.
        # .strip() removes any leading/trailing whitespace.
        proof_code = match.group(1).strip()
        return f"{preamble}\n\n{proof_code}"
    else:
        # error: no proof
        print("Error: Could not find a valid Lean 4 code block in the model's output.")
        print(f"output: {output}")
        return None


# unfinished (incorrect file testing)
def eval(proof):
    filename = "temp.lean"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(proof)

        result = subprocess.run(
            ["lean", filename],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        print(type(result))
        print(result)

        return result.returncode == 0

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
for i, theorem in enumerate(minif2f_test[:]['formal_statement']): # type: ignore
    print(f"Theorem {i}:")
    start = time.time()
    chat = [
        {"role": "user", "content": prompt.format(theorem)}
    ]
    input = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    output = model.generate(input, max_new_tokens=4096)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    proof = clean(theorem, output)
    print(f"solve time: {(time.time() - start):.4f}")

    # get validity of proof
    start = time.time()
    correct += eval(proof) if proof else 0
    print(f"verify time: {(time.time() - start):.4f}")
    print(f"correct: {correct}")
    print()


print(f"{correct / len(minif2f_test)}")
