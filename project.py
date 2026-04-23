print("Starting project script...")

from transformers import AutoModelForCausalLM, AutoTokenizer
print("Transformers imported successfully.")
from datasets import load_dataset
print("Datasets imported successfully.")
from tqdm import tqdm
print("TQDM imported successfully.")
import torch
print("PyTorch imported successfully.")

print("Imports successful, loading model and dataset...")

def gate_grad_bit_rank(gate_weights, gate_grads, p):
    num_bits = 16
    sign_bits = 1
    exponent_bits = 8
    mantissa_bits = 7

    all_scores = []
    min_score = 0


    for layer_idx, (weights, grads) in enumerate(tqdm(zip(gate_weights, gate_grads), total=len(gate_weights), desc="Processing layers")):
        M, N = weights.shape
        
        for i in tqdm(range(M), desc="M"):
            for j in tqdm(range(N), desc="N", leave=False):
                
                w = weights[i, j]
                g = grads[i, j]

                for k in range(num_bits):
                    if(k < sign_bits):
                        delta_x = -2.0 * w
                        score = torch.abs(g * delta_x)
                        bit_str = 'sign'

                    elif(k < sign_bits + exponent_bits):
                        bit_idx = k - sign_bits
                        shamt = mantissa_bits + bit_idx

                        # extract exponent field
                        w_bits = w.view(torch.int16)
                        bit_val = w_bits >> shamt & 1

                        # correct signed flip effect
                        delta_E = (1 - 2 * bit_val) * (1 << bit_idx)
                        delta_x = w * (2 ** delta_E - 1)
                        score = torch.abs(g * delta_x)

                        bit_str = 'exp'
                    else:
                        bit_idx = k - sign_bits - exponent_bits

                        w_bits = w.view(torch.int16)
                        bit_val = w_bits >> bit_idx & 1

                        delta_f = (1 - 2 * bit_val) * (2 ** -(bit_idx + 1))

                        score = torch.abs(g * w * delta_f)

                    if(len(all_scores) < p):
                        all_scores.append((
                            score,
                            layer_idx,
                            i,
                            j,
                            bit_str,
                            0  # bit index within field
                        ))
                    elif(len(all_scores) == p):
                        all_scores.sort(key=lambda x: x[0])
                        all_scores.append((
                            score,
                            layer_idx,
                            i,
                            j,
                            bit_str,
                            0
                        ))
                    else:
                        if(score > all_scores[0][0]):
                            all_scores[0] = (
                                score,
                                layer_idx,
                                i,
                                j,
                                bit_str,
                                0
                            )
                            all_scores.sort(key=lambda x: x[0])

        return all_scores

def load_model():
    model_name = "allenai/OLMoE-1B-7B-0125"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16)
    model.zero_grad(set_to_none=True)

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    text = None
    for sample in dataset:
        if sample["text"] and sample["text"].strip():
            text = sample["text"]
            break
    if text is None:
        raise ValueError("No valid non-empty text found in dataset.")

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    return model, inputs

def gates_run(model, inputs):
    gates = [model.model.layers[i].mlp.gate for i in range(len(model.model.layers))]
    for gate in gates:
        gate.weight.requires_grad_(True)

    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()

    gate_weights = [gate.weight for gate in gates]
    gate_grads = [gate.weight.grad for gate in gates]

    bit_scores = gate_grad_bit_rank(gate_weights, gate_grads, p=10)
    print(bit_scores)