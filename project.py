print("Starting project script...")

import os
import time

# Disable HuggingFace telemetry and slow lookups
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading PyTorch...")
start_torch = time.time()
import torch
import torch.nn.functional as F
print(f"PyTorch loaded in {time.time() - start_torch:.2f}s")

print("Loading tqdm...")
from tqdm import tqdm
print("tqdm imported successfully.")

print("Loading Datasets...")
start_ds = time.time()
from datasets import load_dataset
print(f"Datasets loaded in {time.time() - start_ds:.2f}s")

print("Loading Transformers...")
start_tf = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"Transformers loaded in {time.time() - start_tf:.2f}s")

print("\nImports successful, loading model and dataset...")

_BIT_MASKS_CACHE: dict[torch.device, torch.Tensor] = {}


def _bf16_bit_masks(device: torch.device) -> torch.Tensor:
    """Returns 16 int16 masks (bit 0..15) on the given device.

    Note: bit 15 (0x8000) is represented as -32768 in int16.
    """
    masks = _BIT_MASKS_CACHE.get(device)
    if masks is None:
        masks = torch.tensor([1 << i for i in range(15)] + [-32768], dtype=torch.int16, device=device)
        _BIT_MASKS_CACHE[device] = masks
    return masks


def _bitpos_to_field(bit_pos: int) -> tuple[str, int]:
    # bfloat16: [sign:1][exp:8][mantissa:7]
    if bit_pos == 15:
        return "sign", 0
    if 7 <= bit_pos <= 14:
        return "exp", bit_pos - 7
    return "mantissa", bit_pos


def _eval_lm_loss(model, inputs) -> float:
    with torch.no_grad():
        loss_val = _forward_lm_loss_fp32(model, inputs)
        return float(loss_val.detach().cpu().item())


def _forward_lm_loss_fp32(model, inputs) -> torch.Tensor:
    """Compute CausalLM loss in float32 for numerical stability."""
    outputs = model(**inputs)
    logits = outputs.logits.float()

    # Shift for next-token prediction (same convention as HF CausalLM heads).
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss


def _eval_lm_loss_legacy(model, inputs) -> float:
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss_val = float(outputs.loss.detach().cpu().item())
        return loss_val


def _ask_model(model, tokenizer, question: str, max_new_tokens: int = 80) -> str:
    q_inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **q_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if decoded.startswith(question):
        return decoded[len(question):].strip()
    return decoded.strip()


def _top_p_vulnerable_bits_bf16(
    weight: torch.Tensor,
    grad: torch.Tensor,
    p: int,
    *,
    layer_idx: int,
    flipped_set: set[tuple[int, int, int]],
) -> list[tuple[float, int, int, int]]:
    """Returns (score_est, layer_idx, flat_idx, bit_pos) for top-p candidates in one layer.

    score_est uses the first-order approximation |g * Δw| where Δw is the exact bf16
    value change induced by toggling the selected bit.
    """
    if grad is None:
        return []

    flat_w = weight.detach().flatten()
    flat_g = grad.detach().flatten()
    numel = flat_w.numel()
    if numel == 0:
        return []

    device = flat_w.device
    masks = _bf16_bit_masks(device)

    # Bitcast bf16 -> int16, flip each bit position, bitcast back -> bf16.
    w_i16 = flat_w.view(torch.int16)
    flipped_i16 = w_i16.unsqueeze(0) ^ masks.unsqueeze(1)  # (16, numel)
    flipped_w = flipped_i16.view(torch.bfloat16)

    flat_w_fp32 = flat_w.float()
    flat_g_fp32 = flat_g.float()
    delta_fp32 = flipped_w.float() - flat_w_fp32.unsqueeze(0)
    scores = (delta_fp32 * flat_g_fp32.unsqueeze(0)).abs()  # (16, numel)

    scores_flat = scores.flatten()  # (16 * numel,)
    if flipped_set:
        # Exclude previously flipped bits from re-selection.
        excluded = [bit_pos * numel + flat_idx for (li, flat_idx, bit_pos) in flipped_set if li == layer_idx]
        if excluded:
            scores_flat[torch.tensor(excluded, device=device, dtype=torch.long)] = -float("inf")

    k = min(int(p), int(scores_flat.numel()))
    if k <= 0:
        return []
    top_vals, top_idxs = torch.topk(scores_flat, k=k)

    bit_pos = (top_idxs // numel).to(torch.int64)
    flat_idx = (top_idxs % numel).to(torch.int64)

    return [
        (float(s), int(layer_idx), int(fi), int(bp))
        for s, fi, bp in zip(top_vals.detach().cpu().tolist(), flat_idx.detach().cpu().tolist(), bit_pos.detach().cpu().tolist())
    ]


def gate_grad_bit_rank(
    model,
    inputs,
    gate_weights: list[torch.Tensor],
    *,
    p: int,
    n: int,
    page_size_bytes: int = 4096,
):
    """Gradient-based bit ranking (GBR) for bf16 gate weights.

    Iteration loop:
      1) forward+backward to get gradients at current (already-flipped) weights
      2) for each layer, select top-p vulnerable bits using |g * Δw|
      3) evaluate each of the p×l candidates by temporarily flipping and measuring loss
      4) permanently apply the flip that maximizes loss increase

    Returns:
      selected_flips: list of chosen flips (length n)
      per_iter_rankings: list of full p×l candidate rankings for each iteration
    """
    model.eval()

    flipped_set: set[tuple[int, int, int]] = set()  # (layer_idx, flat_idx, bit_pos)
    selected_flips: list[dict] = []
    per_iter_rankings: list[list[dict]] = []

    for it in tqdm(range(n), desc="GBR iterations"):
        model.zero_grad(set_to_none=True)
        base_loss_t = _forward_lm_loss_fp32(model, inputs)
        base_loss = float(base_loss_t.detach().cpu().item())
        base_loss_t.backward()

        # 1) Build p×l candidate set.
        candidates: list[tuple[float, int, int, int]] = []
        for layer_idx, w in enumerate(gate_weights):
            g = w.grad
            candidates.extend(
                _top_p_vulnerable_bits_bf16(
                    w,
                    g,
                    int(p),
                    layer_idx=layer_idx,
                    flipped_set=flipped_set,
                )
            )

        # 2) Evaluate each candidate by flipping it, measuring, and restoring.
        ranking: list[dict] = []
        best = None

        for score_est, layer_idx, flat_idx, bit_pos in tqdm(
            candidates,
            desc=f"Eval candidates (iter {it})",
            leave=False,
        ):
            w = gate_weights[layer_idx]
            device = w.device
            masks = _bf16_bit_masks(device)

            w_i16 = w.view(torch.int16).view(-1)
            original_i16 = int(w_i16[flat_idx].item())
            original_u16 = original_i16 & 0xFFFF
            flipped_u16 = original_u16 ^ (1 << bit_pos)
            bit_before = (original_u16 >> bit_pos) & 1
            bit_after = 1 - bit_before

            # Temporarily flip.
            with torch.no_grad():
                w_i16[flat_idx] = torch.tensor(flipped_u16, dtype=torch.uint16, device=device).view(torch.int16)
                cand_loss = _eval_lm_loss(model, inputs)
                # Restore.
                w_i16[flat_idx] = torch.tensor(original_i16, dtype=torch.int16, device=device)

            # Handle NaNs robustly.
            if cand_loss != cand_loss:  # NaN
                cand_loss = float("inf")

            loss_after_flip = cand_loss
            loss_before = base_loss
            bit_gradient = loss_after_flip - loss_before

            field, field_bit = _bitpos_to_field(bit_pos)
            byte_offset = int(flat_idx) * 2
            page_num = int(byte_offset // page_size_bytes)
            page_off = int(byte_offset % page_size_bytes)

            item = {
                "iter": int(it),
                "layer_idx": int(layer_idx),
                "flat_idx": int(flat_idx),
                "bit_pos": int(bit_pos),
                "field": field,
                "field_bit": int(field_bit),
                "score_est": float(score_est),
                "bit_before": int(bit_before),
                "bit_after": int(bit_after),
                "original_u16": int(original_u16),
                "flipped_u16": int(flipped_u16),
                "byte_offset": byte_offset,
                "page_num": page_num,
                "page_offset": page_off,
                "base_loss": float(base_loss),
                "cand_loss": float(cand_loss),
                "loss_after_flip": float(loss_after_flip),
                "loss_before": float(loss_before),
                "bit_gradient": float(bit_gradient),
            }
            ranking.append(item)

            if best is None:
                best = item
            else:
                # Maximize loss increase.
                if item["bit_gradient"] > best["bit_gradient"]:
                    best = item

        # Sort full candidate ranking for this iteration by finite-difference bit gradient.
        ranking.sort(key=lambda d: d["bit_gradient"], reverse=True)
        per_iter_rankings.append(ranking)

        if best is None:
            break

        # 3) Permanently apply the best flip for this iteration.
        w = gate_weights[best["layer_idx"]]
        device = w.device
        with torch.no_grad():
            w_i16 = w.view(torch.int16).view(-1)
            original_i16 = int(w_i16[best["flat_idx"]].item())
            flipped_u16 = (original_i16 & 0xFFFF) ^ (1 << best["bit_pos"])
            w_i16[best["flat_idx"]] = torch.tensor(flipped_u16, dtype=torch.uint16, device=device).view(torch.int16)

        iter_loss_after_flip = _eval_lm_loss(model, inputs)
        print(
            f"[GBR] iter={it} layer={best['layer_idx']} flat_idx={best['flat_idx']} "
            f"bit_pos={best['bit_pos']} bit_gradient={best['bit_gradient']:.6f} "
            f"loss_after_flip={iter_loss_after_flip:.6f}"
        )

        flipped_set.add((best["layer_idx"], best["flat_idx"], best["bit_pos"]))
        selected_flips.append(best)

    return selected_flips, per_iter_rankings


model_name = "allenai/OLMoE-1B-7B-0125"
print(f"\nLoading tokenizer from {model_name}...")
start_tok = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"Tokenizer loaded in {time.time() - start_tok:.2f}s")

print(f"Loading model from {model_name}...")
start_model = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded in {time.time() - start_model:.2f}s")
model.zero_grad(set_to_none=True)

probe_question = "In one sentence, what is the capital of France?"
print(f"\n[Probe] Question: {probe_question}")
answer_before = _ask_model(model, tokenizer, probe_question)
print(f"[Probe] Before flips: {answer_before}")
print("\nLoading dataset and finding a non-empty text sample...")
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

initial_loss = _eval_lm_loss(model, inputs)
print(f"Initial loss: {initial_loss}")

gate_weights = [gate.weight for gate in gates]

selected_flips, per_iter_rankings = gate_grad_bit_rank(model, inputs, gate_weights, p=20, n=10)

answer_after = _ask_model(model, tokenizer, probe_question)
print(f"[Probe] After flips: {answer_after}")

final_loss = _eval_lm_loss(model, inputs)
print(f"Final loss after bit flips: {final_loss}")

print(f"Total flips applied: {len(selected_flips)}")
