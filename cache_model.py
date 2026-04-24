"""
Pre-cache the OLMoE model on the SLURM cluster to avoid repeated downloads.
Run this once before your main experiment.

Usage:
    python cache_model.py
"""

import os
import sys

# Disable telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

print("Pre-caching OLMoE model and tokenizer...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_name = "allenai/OLMoE-1B-7B-0125"
    
    print(f"Downloading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer cached")
    
    print(f"Downloading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # cache on CPU, will move to GPU at runtime
    )
    print("✓ Model cached")
    
    print("\n✓ All assets pre-cached successfully!")
    print("Your main script will now load much faster.")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error during caching: {e}")
    sys.exit(1)
