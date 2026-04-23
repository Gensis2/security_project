import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import os


def gate_bit_rank(gate_grads):
    for grad in gate_grads:
        M, W = grad.shape
        num_bits = grad.element_size * 8
        print(M, W, num_bits)

model_name = "allenai/OLMoE-1B-7B-0125"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
text = None
for sample in dataset:
    if sample["text"] and sample["text"].strip():
        text = sample["text"]
        break
if text is None:
    raise ValueError("No valid non-empty text found in dataset.")

inputs = tokenizer(text, return_tensors="pt").to(model.device)

gates = [model.model.layers[i].mlp.gate for i in range(len(model.model.layers))]
for gate in gates:
    gate.weight.requires_grad_(True)
    gate.weight.grad = None

outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()

gate_grads = [gate.weight.grad for gate in gates]

gate_bit_rank(gate_grads)