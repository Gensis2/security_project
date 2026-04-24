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

def gate_grad_bit_rank(model, inputs, gate_weights, gate_grads, p, n):
    num_bits = 16
    sign_bits = 1
    exponent_bits = 8
    mantissa_bits = 7

    flipped_bits = []
    flipped_bits_idx = []

    for _ in tqdm(range(n), desc="Bit flip iterations"):
        all_scores = []
        for layer_idx, (weights, grads) in enumerate(tqdm(zip(gate_weights, gate_grads), total=len(gate_weights), desc="Processing layers", leave=False)):
            M, N = weights.shape
            layer_scores = []

            for i in tqdm(range(M), desc="M", leave=False):
                for j in tqdm(range(N), desc="N", leave=False):
                    
                    w = weights[i, j]
                    g = grads[i, j]

                    for k in range(num_bits):
                        if(k < sign_bits):
                            delta_x = -2.0 * w
                            score = torch.abs(g * delta_x)
                            bit_str = 'sign'
                            bit_idx = 0

                        elif(k < sign_bits + exponent_bits):
                            bit_idx = k - sign_bits
                            shamt = mantissa_bits + bit_idx

                            # extract exponent field
                            w_bits = w.view(torch.int16)
                            bit_val = (w_bits >> shamt) & 1

                            # correct signed flip effect
                            delta_E = (1 - 2 * bit_val) * (1 << bit_idx)
                            delta_x = w * (2 ** delta_E - 1)
                            score = torch.abs(g * delta_x)

                            bit_str = 'exp'

                        else:
                            bit_idx = k - sign_bits - exponent_bits

                            w_bits = w.view(torch.int16)
                            bit_val = (w_bits >> bit_idx) & 1

                            delta_f = (1 - 2 * bit_val) * (2 ** -(bit_idx + 1))

                            score = torch.abs(g * w * delta_f)

                        if( (layer_idx, i, j, bit_str, bit_idx) in flipped_bits_idx):
                            continue

                        if(len(layer_scores) < p):
                            layer_scores.append((
                                score,
                                layer_idx,
                                i,
                                j,
                                bit_str,
                                bit_idx
                            ))
                        elif(len(layer_scores) == p):
                            layer_scores.sort(key=lambda x: x[0])
                            layer_scores.append((
                                score,
                                layer_idx,
                                i,
                                j,
                                bit_str,
                                bit_idx
                            ))
                        else:
                            if(score > layer_scores[0][0]):
                                layer_scores[0] = (
                                    score,
                                    layer_idx,
                                    i,
                                    j,
                                    bit_str,
                                    bit_idx
                                )
                                layer_scores.sort(key=lambda x: x[0])
            all_scores.append(layer_scores)

        # now forward pass through each layer and compare loss
        losses = []

        for layer_scores in enumerate(all_scores):
            for score, layer_idx, i, j, bit_str, bit_idx in layer_scores:
                # flip the bit
                original_weight = gate_weights[layer_idx][i, j].item()
                if bit_str == 'sign':
                    flipped_weight = -original_weight
                elif bit_str == 'exp':
                    w_bits = gate_weights[layer_idx][i, j].view(torch.int16)
                    w_bits ^= (1 << (mantissa_bits + bit_idx))
                    flipped_weight = w_bits.view(torch.float16).item()
                else:
                    w_bits = gate_weights[layer_idx][i, j].view(torch.int16)
                    w_bits ^= (1 << bit_idx)
                    flipped_weight = w_bits.view(torch.float16).item()

                with torch.no_grad():
                    gate_weights[layer_idx][i, j] = flipped_weight
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    losses.append((loss, layer_idx, i, j, bit_str, flipped_weight, bit_idx))
                    gate_weights[layer_idx][i, j] = original_weight

        flipped_idx = sorted(losses, key=lambda x: x[0], reverse=True)[0]
        with torch.no_grad():
            gate_weights[flipped_idx[1]][flipped_idx[2], flipped_idx[3]] = flipped_idx[5]

        flipped_bits_idx.append((flipped_idx[1], flipped_idx[2], flipped_idx[3], flipped_idx[4], flipped_idx[6]))
        flipped_bits.append(flipped_idx)

        model.zero_grad(set_to_none=True)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

    return flipped_bits


model_name = "allenai/Qwen/Qwen1.5-MoE-A2.7B"
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

gates = [model.model.layers[i].mlp.shared_expert_gate for i in range(len(model.model.layers))]
for gate in gates:
    gate.weight.requires_grad_(True)

outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()
print(f"Initial loss: {loss.item()}")

gate_weights = [gate.weight for gate in gates]
gate_grads = [gate.weight.grad for gate in gates]

flipped_bits = gate_grad_bit_rank(model, inputs, gate_weights, gate_grads, p=24, n=24)

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    print(f"Final loss after bit flips: {loss}")
