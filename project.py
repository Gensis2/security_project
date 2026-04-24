print("Starting project script...")

import os
import csv
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


def _eval_avg_lm_loss(model, inputs_list: list[dict[str, torch.Tensor]]) -> float:
    if not inputs_list:
        raise ValueError("inputs_list must contain at least one sample")
    with torch.no_grad():
        total = 0.0
        for sample_inputs in tqdm(inputs_list, desc="Evaluating", leave=False):
            total += float(_forward_lm_loss_fp32(model, sample_inputs).detach().cpu().item())
    return total / float(len(inputs_list))


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


def _probe_next_token_stats(model, tokenizer, question: str, top_k: int = 5) -> str:
    """Return a compact summary of the next-token distribution for a prompt.

    This is a better debugging signal than decoded text alone, because the text can stay
    the same even when logits/probabilities shift after a bit flip.
    """
    q_inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**q_inputs)
        next_logits = outputs.logits[:, -1, :].float().squeeze(0)
        probs = torch.softmax(next_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=min(int(top_k), int(probs.numel())))

    pieces = []
    for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        token_text = token_text.replace("\n", "\\n")
        pieces.append(f"{token_id}:{token_text}:{prob:.4f}")
    return " | ".join(pieces)


def _collect_inputs_list(tokenizer, num_samples: int, model_device: torch.device) -> list[dict[str, torch.Tensor]]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    for sample in dataset:
        if sample["text"] and sample["text"].strip():
            texts.append(sample["text"])
            if len(texts) >= num_samples:
                break

    if not texts:
        raise ValueError("No valid non-empty text found in dataset.")

    return [tokenizer(t, return_tensors="pt").to(model_device) for t in texts]


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
    tokenizer,
    probe_question: str,
    inputs_list: list[dict[str, torch.Tensor]],
    gate_weights: list[torch.Tensor],
    *,
    p: int,
    n: int,
    page_size_bytes: int = 4096,
    csv_path: str = "bitflip_metadata.csv",
    probe_max_new_tokens: int = 80,
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
    csv_fieldnames = [
        "iter",
        "layer_idx",
        "flat_idx",
        "bit_pos",
        "field",
        "field_bit",
        "score_est",
        "bit_before",
        "bit_after",
        "original_u16",
        "flipped_u16",
        "byte_offset",
        "page_num",
        "page_offset",
        "base_loss",
        "cand_loss",
        "loss_after_flip",
        "loss_before",
        "bit_gradient",
        "probe_question",
        "probe_response",
        "probe_top_tokens",
        "probe_loss_after_flip",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()

    if not inputs_list:
        raise ValueError("inputs_list must contain at least one sample")

    num_samples = len(inputs_list)

    for it in tqdm(range(n), desc="GBR iterations", leave=False):
        model.zero_grad(set_to_none=True)
        base_loss_vals = []
        for sample_inputs in inputs_list:
            sample_loss_t = _forward_lm_loss_fp32(model, sample_inputs)
            base_loss_vals.append(float(sample_loss_t.detach().cpu().item()))
            # Average gradients over samples.
            (sample_loss_t / float(num_samples)).backward()
        base_loss = sum(base_loss_vals) / float(num_samples)

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
                cand_loss = _eval_avg_lm_loss(model, inputs_list)
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

        iter_loss_after_flip = _eval_avg_lm_loss(model, inputs_list)
        probe_response = _ask_model(model, tokenizer, probe_question, max_new_tokens=probe_max_new_tokens)
        probe_top_tokens = _probe_next_token_stats(model, tokenizer, probe_question, top_k=5)
        print(f"[Probe] iter={it} response: {probe_response}")
        print(f"[Probe] iter={it} top_tokens: {probe_top_tokens}")
        print(
            f"[GBR] iter={it} layer={best['layer_idx']} flat_idx={best['flat_idx']} "
            f"bit_pos={best['bit_pos']} bit_gradient={best['bit_gradient']:.6f} "
            f"avg_loss_after_flip={iter_loss_after_flip:.6f}"
        )

        csv_row = dict(best)
        csv_row.update(
            {
                "probe_question": probe_question,
                "probe_response": probe_response,
                "probe_top_tokens": probe_top_tokens,
                "probe_loss_after_flip": iter_loss_after_flip,
            }
        )
        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            writer.writerow(csv_row)

        flipped_set.add((best["layer_idx"], best["flat_idx"], best["bit_pos"]))
        selected_flips.append(best)

    return selected_flips, per_iter_rankings


def gate_hess_bit_rank(
    model,
    tokenizer,
    probe_question: str,
    inputs_list: list[dict[str, torch.Tensor]],
    gate_weights: list[torch.Tensor],
    *,
    p: int,
    n: int,
    page_size_bytes: int = 4096,
    hessian_eps: float = 1e-3,
    csv_path: str = "bitflip_metadata_hess.csv",
    probe_max_new_tokens: int = 80,
):
    """Hessian-informed bit ranking for bf16 gate weights (unused helper).

    This mirrors gate_grad_bit_rank but ranks candidates by a second-order Taylor
    approximation for each bit flip:

        Delta L ~= g * Delta w + 0.5 * h_diag * (Delta w^2)

    where h_diag is a finite-difference estimate of the diagonal Hessian entry
    at the candidate weight coordinate.
    """
    model.eval()

    if not inputs_list:
        raise ValueError("inputs_list must contain at least one sample")
    if hessian_eps <= 0:
        raise ValueError("hessian_eps must be > 0")

    num_samples = len(inputs_list)
    flipped_set: set[tuple[int, int, int]] = set()  # (layer_idx, flat_idx, bit_pos)
    selected_flips: list[dict] = []
    per_iter_rankings: list[list[dict]] = []
    csv_fieldnames = [
        "iter",
        "layer_idx",
        "flat_idx",
        "bit_pos",
        "field",
        "field_bit",
        "score_est",
        "original_u16",
        "flipped_u16",
        "byte_offset",
        "page_num",
        "page_offset",
        "base_loss",
        "cand_loss",
        "loss_after_flip",
        "loss_before",
        "bit_gradient",
        "g_ij",
        "h_ii",
        "delta_w",
        "second_order_score",
        "probe_question",
        "probe_response",
        "probe_top_tokens",
        "probe_loss_after_flip",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()

    def _avg_grad_at_coord(target_weight: torch.Tensor, flat_idx: int) -> float:
        model.zero_grad(set_to_none=True)
        for sample_inputs in inputs_list:
            loss_t = _forward_lm_loss_fp32(model, sample_inputs)
            (loss_t / float(num_samples)).backward()
        grad_t = target_weight.grad
        if grad_t is None:
            return 0.0
        return float(grad_t.view(-1)[flat_idx].detach().cpu().item())

    for it in tqdm(range(n), desc="Hessian GBR iterations", leave=False):
        model.zero_grad(set_to_none=True)
        base_loss_vals = []
        for sample_inputs in inputs_list:
            sample_loss_t = _forward_lm_loss_fp32(model, sample_inputs)
            base_loss_vals.append(float(sample_loss_t.detach().cpu().item()))
            (sample_loss_t / float(num_samples)).backward()
        base_loss = sum(base_loss_vals) / float(num_samples)

        # Snapshot base gradients since finite-difference Hessian probes will recompute grads.
        base_grads = [
            (w.grad.detach().clone() if w.grad is not None else torch.zeros_like(w))
            for w in gate_weights
        ]

        # Build p*l candidate set from first-order screening (same as gradient method).
        candidates: list[tuple[float, int, int, int]] = []
        for layer_idx, w in enumerate(gate_weights):
            candidates.extend(
                _top_p_vulnerable_bits_bf16(
                    w,
                    base_grads[layer_idx],
                    int(p),
                    layer_idx=layer_idx,
                    flipped_set=flipped_set,
                )
            )

        ranking: list[dict] = []
        best = None

        for score_est, layer_idx, flat_idx, bit_pos in tqdm(
            candidates,
            desc=f"Eval Hess candidates (iter {it})",
            leave=False,
        ):
            w = gate_weights[layer_idx]
            device = w.device

            w_i16 = w.view(torch.int16).view(-1)
            w_flat = w.view(-1)

            original_i16 = int(w_i16[flat_idx].item())
            original_u16 = original_i16 & 0xFFFF
            flipped_u16 = original_u16 ^ (1 << bit_pos)

            original_val = float(w_flat[flat_idx].detach().float().item())
            flipped_val_t = torch.tensor([flipped_u16], dtype=torch.uint16, device=device).view(torch.int16).view(torch.bfloat16)
            flipped_val = float(flipped_val_t[0].float().item())
            delta_w = flipped_val - original_val

            g_ij = float(base_grads[layer_idx].view(-1)[flat_idx].detach().cpu().item())

            # Finite-difference diagonal Hessian estimate at this coordinate.
            eps = float(hessian_eps * max(1.0, abs(original_val)))
            with torch.no_grad():
                w_flat[flat_idx] = torch.tensor(original_val + eps, dtype=w.dtype, device=device)
            g_plus = _avg_grad_at_coord(w, flat_idx)

            with torch.no_grad():
                w_flat[flat_idx] = torch.tensor(original_val - eps, dtype=w.dtype, device=device)
            g_minus = _avg_grad_at_coord(w, flat_idx)

            h_ii = (g_plus - g_minus) / (2.0 * eps)
            second_order_score = (g_ij * delta_w) + (0.5 * h_ii * (delta_w ** 2))

            # Evaluate true average loss change for reporting/debugging.
            with torch.no_grad():
                w_i16[flat_idx] = torch.tensor(flipped_u16, dtype=torch.uint16, device=device).view(torch.int16)
                cand_loss = _eval_avg_lm_loss(model, inputs_list)
                # Always restore exact original bits.
                w_i16[flat_idx] = torch.tensor(original_i16, dtype=torch.int16, device=device)

            if cand_loss != cand_loss:  # NaN
                cand_loss = float("inf")

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
                "original_u16": int(original_u16),
                "flipped_u16": int(flipped_u16),
                "byte_offset": byte_offset,
                "page_num": page_num,
                "page_offset": page_off,
                "base_loss": float(base_loss),
                "cand_loss": float(cand_loss),
                "loss_after_flip": float(cand_loss),
                "loss_before": float(base_loss),
                "bit_gradient": float(cand_loss - base_loss),
                "g_ij": float(g_ij),
                "h_ii": float(h_ii),
                "delta_w": float(delta_w),
                "second_order_score": float(second_order_score),
            }
            ranking.append(item)

            if best is None or item["second_order_score"] > best["second_order_score"]:
                best = item

            # Restore baseline gradients for subsequent first-order screening assumptions.
            for gw, bg in zip(gate_weights, base_grads):
                if gw.grad is None:
                    gw.grad = bg.clone()
                else:
                    gw.grad.copy_(bg)

        ranking.sort(key=lambda d: d["second_order_score"], reverse=True)
        per_iter_rankings.append(ranking)

        if best is None:
            break

        # Permanently apply one best bit flip for this iteration.
        w = gate_weights[best["layer_idx"]]
        device = w.device
        with torch.no_grad():
            w_i16 = w.view(torch.int16).view(-1)
            original_i16 = int(w_i16[best["flat_idx"]].item())
            flipped_u16 = (original_i16 & 0xFFFF) ^ (1 << best["bit_pos"])
            w_i16[best["flat_idx"]] = torch.tensor(flipped_u16, dtype=torch.uint16, device=device).view(torch.int16)

        iter_loss_after_flip = _eval_avg_lm_loss(model, inputs_list)
        probe_response = _ask_model(model, tokenizer, probe_question, max_new_tokens=probe_max_new_tokens)
        probe_top_tokens = _probe_next_token_stats(model, tokenizer, probe_question, top_k=5)
        print(f"[Probe] iter={it} response: {probe_response}")
        print(f"[Probe] iter={it} top_tokens: {probe_top_tokens}")
        print(
            f"[HESS-GBR] iter={it} layer={best['layer_idx']} flat_idx={best['flat_idx']} "
            f"bit_pos={best['bit_pos']} hess_score={best['second_order_score']:.6f} "
            f"avg_loss_after_flip={iter_loss_after_flip:.6f}"
        )

        csv_row = dict(best)
        csv_row.update(
            {
                "probe_question": probe_question,
                "probe_response": probe_response,
                "probe_top_tokens": probe_top_tokens,
                "probe_loss_after_flip": iter_loss_after_flip,
            }
        )
        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            writer.writerow(csv_row)

        flipped_set.add((best["layer_idx"], best["flat_idx"], best["bit_pos"]))
        selected_flips.append(best)

    return selected_flips, per_iter_rankings


def run_standardized_model_workflow(
    model,
    tokenizer,
    gate_weights: list[torch.Tensor],
    *,
    probe_question: str,
    num_grad_samples: int,
    p: int,
    n: int,
    grad_csv_path: str = "bitflip_metadata.csv",
    hess_csv_path: str = "bitflip_metadata_hess.csv",
) -> None:
    model.zero_grad(set_to_none=True)

    print(f"\n[Probe] Question: {probe_question}")
    answer_before = _ask_model(model, tokenizer, probe_question)
    print(f"[Probe] Before flips: {answer_before}")

    print(f"\nLoading dataset and collecting {num_grad_samples} non-empty text sample(s)...")
    inputs_list = _collect_inputs_list(tokenizer, num_grad_samples, model.device)

    for gate in [model.model.layers[i].mlp.gate for i in range(len(model.model.layers))]:
        gate.weight.requires_grad_(True)

    initial_loss = _eval_avg_lm_loss(model, inputs_list)
    print(f"Initial average loss ({len(inputs_list)} sample(s)): {initial_loss}")

    selected_flips, _ = gate_grad_bit_rank(
        model,
        tokenizer,
        probe_question,
        inputs_list,
        gate_weights,
        p=p,
        n=n,
        csv_path=grad_csv_path,
    )

    answer_after = _ask_model(model, tokenizer, probe_question)
    print(f"[Probe] After flips: {answer_after}")

    final_loss = _eval_avg_lm_loss(model, inputs_list)
    print(f"Final average loss after bit flips: {final_loss}")
    print(f"Total flips applied: {len(selected_flips)}")

    model.zero_grad(set_to_none=True)

    print("\nStarting second rerun with Hessian ranking...")
    inputs_list_hess = _collect_inputs_list(tokenizer, num_grad_samples, model.device)
    initial_hess_loss = _eval_avg_lm_loss(model, inputs_list_hess)
    print(f"Initial Hessian average loss ({len(inputs_list_hess)} sample(s)): {initial_hess_loss}")

    selected_flips_hess, _ = gate_hess_bit_rank(
        model,
        tokenizer,
        probe_question,
        inputs_list_hess,
        gate_weights,
        p=p,
        n=n,
        csv_path=hess_csv_path,
    )

    answer_after_hess = _ask_model(model, tokenizer, probe_question)
    print(f"[Probe] After Hessian flips: {answer_after_hess}")

    final_hess_loss = _eval_avg_lm_loss(model, inputs_list_hess)
    print(f"Final Hessian average loss after bit flips: {final_hess_loss}")
    print(f"Total Hessian flips applied: {len(selected_flips_hess)}")


def qwen() -> None:
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"
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

    gate_weights = [model.model.layers[i].mlp.gate.weight for i in range(len(model.model.layers))]
    run_standardized_model_workflow(
        model,
        tokenizer,
        gate_weights,
        probe_question="In one sentence, what is the capital of France?",
        num_grad_samples=int(os.getenv("NUM_GRAD_SAMPLES", "1")),
        p=20,
        n=10,
        grad_csv_path="bitflip_metadata.csv",
        hess_csv_path="bitflip_metadata_hess.csv",
    )

def olmoe() -> None:
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

    gate_weights = [model.model.layers[i].mlp.gate.weight for i in range(len(model.model.layers))]
    run_standardized_model_workflow(
        model,
        tokenizer,
        gate_weights,
        probe_question="In one sentence, what is the capital of France?",
        num_grad_samples=int(os.getenv("NUM_GRAD_SAMPLES", "1")),
        p=20,
        n=10,
        grad_csv_path="bitflip_metadata.csv",
        hess_csv_path="bitflip_metadata_hess.csv",
    )


if __name__ == "__main__":
    olmoe()
