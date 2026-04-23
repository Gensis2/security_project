from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import os

def gate_bit_rank(gate_layers):
    pass

model_name = "allenai/OLMoE-1B-7B-0125"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
text = None
for sample in dataset:
    if sample["text"] and sample["text"].strip():
        text = sample["text"]
        break
if text is None:
    raise ValueError("No valid non-empty text found in dataset.")

inputs = tokenizer(text, return_tensors="pt").to(model.device)

gate = model.model.layers[0].mlp.gate
gate.weight.requires_grad_(True)
gate.weight.grad = None

outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss

loss.backward()

grad = gate.weight.grad

print("loss:", loss.item())
print("gate grad shape:", grad.shape)
print(grad)