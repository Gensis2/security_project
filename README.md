# GateHammer Simulation Framework

This repository contains a simulation framework for GateHammer, which extends the DeepHammer bitflip algorithm to Mixture-of-Experts (MoE) gating networks.

The goal of this project is to simulate and evaluate bit-level perturbations in MoE routing mechanisms, focusing on how gating decisions are affected under controlled fault injection.

---

## ⚙️ Requirements

To run this framework, you must have access to a GPU capable of loading both models and datasets:

- OLMoE 1B
- Qwen 1.5 2.7B
- 10 input queries from wikitext-103

Due to model size and memory requirements, CPU-only execution is not supported.

---

## 🧪 Environment Setup (Conda)

We provide an environment.yml file to recreate the exact environment.

### Step 1: Create the environment

conda env create -f environment.yml

### Step 2: Activate the environment

conda activate gatehammer

### Step 3: Verify installation

python -c "import torch; print(torch.cuda.is_available())"

If this returns True, your GPU setup is working correctly.

---

## 🚀 Running the Simulation

You can run the framework in two ways depending on your system setup.

---

### Option 1: SLURM Cluster

If you are using a SLURM-managed computing cluster, run:

sbatch slurm_script.sh

This submits the job to the scheduler and runs the full simulation pipeline.

---

### Option 2: Local Machine

If you are running on a local or standalone GPU machine, use:

bash local_script.sh

Make sure your conda environment is activated before running.

---

## 📊 Outputs

After running the simulation, the outputs will remain in the base directory, containing a:
png and pdf output of the perplexity comparison
csv files for both models (for both gradient and hessian analysis).

These files are also included in the repository so you can immediately inspect example outputs without rerunning the full pipeline.
There additionally exists a qwen_nonfinite csv, with is from previous runs that did not consider inf/NaN values. It remains as a resource to support removing such values when processing bit-flips.
---

## 📌 Notes

- Ensure sufficient GPU memory before running (both models must fit in memory simultaneously, as well as the dataset).
- Results may vary slightly depending on hardware and CUDA version.
- This framework is intended for research and educational purposes.