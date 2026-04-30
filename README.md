# GateHammer Simulation Framework

This repository contains a simulation framework for **GateHammer**, which extends the *DeepHammer bitflip algorithm* to Mixture-of-Experts (MoE) gating networks.

The goal of this project is to simulate and evaluate bit-level perturbations in MoE routing mechanisms, focusing on how gating decisions are affected under controlled fault injection.

---

## ⚙️ Requirements

To run this framework, you **must have access to a GPU capable of loading both models**:

- OLMoE 1B
- Qwen 1.5 2.7B

Due to model size and memory requirements, CPU-only execution is not supported.

---

## 🧪 Environment Setup (Conda)

We provide an `environment.yaml` file to recreate the exact environment.

### Step 1: Create the environment

```bash
conda env create -f environment.yaml