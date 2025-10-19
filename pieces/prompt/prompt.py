import os
import re
import subprocess
import time

import torch
from datasets import load_dataset
from log import *
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model():
    model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
    return model, tokenizer


def get_prompt(x):
    if x == "OG":
        return """
            Complete the following Lean 4 code:

            ```lean4
            {}
            ```

            Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
            The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
            """.strip()
    # modified prompt
    elif x == "PE":
        return """
            Complete the following Lean 4 code:

            ```lean4
            {}
            ```

            Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
            The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
            You can test ideas by writing Lean 4 code between [TEST] and [/TEST]; it will be compiled and the compiler’s output will be returned in this conversation.
            Use this capability during your reasoning to validate your ideas and mitigate illogical arguments.
            """.strip()


# get final lean file w/ proof
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


def verify(proof):
    project_dir = os.path.join(os.getcwd(), "lean-project")
    filename = "temp.lean"
    file_path = os.path.join(project_dir, "LeanProject", filename)

    try:
        with open(file_path, "w") as f:
            f.write(proof)

        result = subprocess.run(
            ["lake", "env", "lean", file_path],
            cwd=project_dir,  # lean project directory
            capture_output=True,
            text=True,
        )

        return result

    finally:    # remove temp file
        if os.path.exists(file_path):
            os.remove(file_path)


# get test from CoT
def extract_test(output):
    i = len(output)
    while i > 7:
        if output[i - 7 : i] == "[TEST]":
            break
        i -= 1
    return output[i:]


def feedback(proof):
    results = verify(proof)
    # simplifiy compiler response
    feedback = results
    return feedback


def generate_proof(theorem):
    chat = [{"role": "user", "content": prompt.format(theorem)}]

    num_of_tests = tokens_produced = 0

    while True:
        input = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        input_size = input.shape[1]

        output = model.generate(
            input,
            max_new_tokens=4096,
            eos_token_id=tokenizer.encode("[/TEST]")[0],  # Stop at [/TEST]
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

        tokens_produced += output.shape[1] - input_size

        output = tokenizer.decode(output[0], skip_special_tokens=True)

        # if proof is done
        if "[/TEST]" not in output[-10:]:
            break

        num_of_tests += 1
        test = extract_test(output)
        compiler_feedback = str(feedback(test))

        # put output & feedback into input
        chat.append({"role": "assistant", "content": output})
        chat.append({"role": "user", "content": compiler_feedback})

    proof = clean(theorem, output)

    return proof, num_of_tests, tokens_produced


if __name__ == "__main__":
    torch.manual_seed(30)
    model, tokenizer = load_model()
    minif2f_test = load_dataset("AI-MO/minif2f_test", split="train")

    fieldnames = [
        "index",
        "correct",
        "number_of_tests",
        "tokens_produced",
        "solve_time",
        "verify_time",
    ]

    for i in ["OG", "PE"]:
        prompt = get_prompt(i)

        with TaskLogger(f"metrics/{i}-log.csv", fieldnames, overwrite=True) as logger:
            ds_size = len(minif2f_test)  # size of dataset
            logger.set_total_tasks(ds_size)
            progress = ProgressBar(total=ds_size, description=f"Progress ({i})")

            for i, theorem in enumerate(minif2f_test[:]["formal_statement"]):
                # generate proof
                start = time.time()
                proof, num_of_tests, tokens_produced = generate_proof(theorem)
                solve_time = time.time() - start

                # get validity of proof
                start = time.time()
                correct = verify(proof).returncode == 0 if proof else False
                verify_time = time.time() - start

                # log info in csv
                info = {
                    "index": i,
                    "correct": correct,
                    "number_of_tests": num_of_tests,
                    "tokens_produced": tokens_produced,
                    "solve_time": solve_time,
                    "verify_time": verify_time,
                }
                logger.log_task(info)

                # update progress bar
                progress.update()
            progress.finish()
