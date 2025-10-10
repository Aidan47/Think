import os
import re
import subprocess
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from log import *


def load_model():
    model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
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
def verify(proof):
    project_dir = os.path.join(os.getcwd(), "lean-project")
    filename = "temp.lean"
    file_path = os.path.join(project_dir, "LeanProject", filename)

    try:
        with open(file_path, "w") as f:
            f.write(proof)

        result = subprocess.run(
            ["lake", "env", "lean", file_path],
            cwd=project_dir,    # lean project directory
            capture_output=True,
            text=True,
        )
        print(proof)
        print(type(result))
        print(result)
        

        return result.returncode == 0

    finally:    # remove temp file
        if os.path.exists(file_path):
            os.remove(file_path)


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
    

fieldnames = ['index', 'correct', 'number_of_tests', 'tokens_produced', 'solve_time', 'verify_time']
    
with TaskLogger('log.csv', fieldnames) as logger:
    ds_size = len(minif2f_test) # size of dataset
    logger.set_total_tasks(ds_size)
    progress = ProgressBar(total=ds_size, description="Progress")
    
    for i, theorem in enumerate(minif2f_test[:]['formal_statement']):
        start = time.time()
        # generate proof
        chat = [
            {"role": "user", "content": prompt.format(theorem)}
        ]
        input = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        output = model.generate(input, max_new_tokens=4096)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        proof = clean(theorem, output)
        solve_time = time.time() - start

        # get validity of proof
        start = time.time()
        correct = bool(verify(proof) if proof else 0)
        verify_time = time.time() - start
        
        # log info in csv
        info = {
            'index': i,
            'correctness': correct,
            'number_of_tests': 0,
            'tokens_produced': int,
            'solve_time': solve_time,
            'verify_time': verify_time
        }
        logger.log_task(info)
        
        # update progress bar
        progress.update()
    progress.finish()