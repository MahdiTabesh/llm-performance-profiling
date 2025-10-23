# ⚡ LLM Performance Profiling – DistilBERT Optimization Experiments

This repository presents a complete end-to-end **performance profiling and optimization study** of the DistilBERT transformer model using **PyTorch** and **Weights & Biases (W&B)**.  
It focuses on **efficient training, GPU utilization, hyperparameter sensitivity, and backend acceleration**, showcasing both engineering depth and practical MLOps skills.

---

## Overview

| Notebook | Focus | Key Skills Demonstrated |
|-----------|--------|--------------------------|
| **DistilBERT_FineTuning_and_Timing.ipynb** | Fine-tuning DistilBERT on the IMDB sentiment dataset and recording GPU timing metrics. | Model fine-tuning • Training loop design • Timing analysis • W&B experiment logging |
| **DataLoader_Profiling_and_ProfilerBreakdown.ipynb** | Benchmarking DataLoader throughput and analyzing PyTorch Profiler traces. | Data pipeline optimization • PyTorch Profiler • Bottleneck identification |
| **GPU_Performance_and_Hparam_Sweep.ipynb** | Measuring GPU training performance and conducting hyperparameter sweeps (batch size × learning rate). | GPU utilization • Hyperparameter tuning • W&B sweeps • Performance visualization |
| **Optimizer_and_Compile_Comparison.ipynb** | Comparing optimizers (SGD, Adam, AdamW) and evaluating Torch Compile (Inductor backend). | Optimizer evaluation • torch.compile benchmarking • Performance interpretation |

---

## Technologies & Tools

- **Frameworks:** PyTorch 2.x, Hugging Face Transformers  
- **Experiment Tracking:** Weights & Biases (W&B)  
- **Profiling Tools:** PyTorch Profiler, Matplotlib, Seaborn  
- **Hardware:** NVIDIA A100 GPU (Google Cloud Deep Learning VM)  
- **Optimization Topics Covered:**
  - DataLoader parallelism and memory pinning  
  - GPU timing and performance benchmarking  
  - Learning rate & batch size sensitivity  
  - Optimizer and backend (Eager vs. Inductor) comparison  

---

## Highlights

- Logged and visualized all runs on [Weights & Biases](https://wandb.ai/mah3i-tabesh500-mahdi-saleh-tabesh/hpml-hw2-llm)  
- Produced reproducible tables and figures (T5–T8, F5–F8) summarizing performance results  
- Identified GPU bottlenecks between data loading and compute phases  
- Quantified compile-time overhead vs. runtime gains of **`torch.compile`**  
- Followed structured, experiment-driven workflow aligned with **MLOps best practices**

---

## Key Takeaways

- **AdamW** consistently offered the best balance between convergence speed and stability.  
- **torch.compile (Inductor)** introduced initial overhead but delivered faster steady-state epochs.  
- **Efficient data pipelines** (num_workers tuning, pin_memory) significantly improved training throughput.  
- Profiling and systematic experimentation enabled data-driven optimization rather than intuition-based tuning.

---

## Author

**Mahdi Saleh Tabesh**  
M.S. in Electrical Engineering – *Data-Driven Analysis & Computation*, Columbia University  
Focus Areas: Machine Learning • Deep Learning • High-Performance ML • LLM Optimization  
[LinkedIn](https://www.linkedin.com/in/mahditabesh) | [GitHub](https://github.com/MahdiTabesh)

---



