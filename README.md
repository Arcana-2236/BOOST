# BOOST: Bottleneck-Optimized Scalable Training Framework For Low-Rank Large Language Models

**BOOST** is a Nanotron-based research framework for scalable training of **low-rank bottleneck LLM**s. It implements **Bottleneck-aware Tensor Parallelism (BTP)** along with several system-level optimizations for efficient distributed training.  

## News

- `2026-03-10`: **Code released!**
- `2026-01-26`: 🎉 BOOST is accepted to **MLSys 2026**!
- `2025-10-30`: We are excited to announce BOOST, a scalable **3D-parallel training framework** for low-rank bottleneck LLMs, featuring efficient communication and computation optimizations.

## Setup

Use the NGC PyTorch container to keep dependencies consistent and avoid host environment drift.

```bash
# Clone the repository on the host
git clone https://github.com/Arcana-2236/BOOST.git
cd BOOST

# Pull the base container
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# Launch the container and mount the repo
docker run --rm --gpus all \
  -v $(pwd):/workspace/BOOST \
  --entrypoint=/bin/bash \
  --shm-size=1g \
  -it nvcr.io/nvidia/pytorch:24.01-py3

# Env setup, Inside the container
cd /workspace/BOOST
pip install datasets transformers
pip install triton "flash-attn==2.5.1.post1" --no-build-isolation
pip install -e .
```

## Quickstart

### 1) Full-rank baseline

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 run_train.py --config-file examples/config_tiny_llama.yaml
```

### 2) CoLA-BTP run

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/cola/train_cola.py --config-file examples/cola/config_tiny_cola_llama.yaml
```

## Motivation

*Low-Rank Bottleneck Architecture and Iter time in TP setting*

Low-rank bottleneck architectures decompose dense projections into low-rank factors, reducing parameter count and computational cost while largely preserving model quality. However, when scaling such architectures to multi-GPU systems, **naïvely applying standard Tensor Parallelism (TP)** introduces new inefficiencies.

First, the deeper structure of low-rank layers can introduce **additional communication synchronization points**, increasing communication overhead. Second, the **irregular placement of low-rank factors** often leads to inefficient computation kernel execution, which reduces hardware utilization. As a result, the theoretical FLOP reduction from low-rank model training may not translate into real training speedups.

This repository focuses on optimizing **Tensor Parallel implementations for low-rank bottleneck LLMs**. In particular, we study how TP design affects **throughput and scalability on multi-GPU and multi-node systems**, and propose optimizations that reduce **communication overhead** and mitigate **kernel-level performance bottlenecks**.

## Methodology

*Bottleneck-aware Tensor Parallelism Design*

**BOOST proposes Bottleneck-aware Tensor Parallelism, which:**

- **shifts TP chunk boundaries** to align with the bottleneck structure
- **shards along the large hidden dimension d** instead of the low-rank dimension r --> Improves **GEMM arithmetic intensity** and **GPU utilization**
- performs communication on **low-rank activations [b,s,r]** rather than full hidden states [b,s,d] --> Reduces communication volume

In addition, BOOST introduces several system-level optimizations to further improve training efficiency, including:

- Online RMSNorm to eliminate latency-dominated normalization collectives
- Low-rank linear layer grouping to increase kernel efficiency and reduce launches overhead
- Communication-free low-rank activation checkpointing to reduce memory overhead without introducing additional communication

Together, these techniques enable efficient and scalable distributed training of low-rank bottleneck LLMs.

## Results

### System Performance

```bash
bash ./run_iter_compare.sh
```


| Model | GPUs | TP  | PP  | FullRank (s) | Vanilla TP (s) | BOOST (s) | Speedup vs FullRank | Speedup vs Vanilla |
| ----- | ---- | --- | --- | ------------ | -------------- | --------- | ------------------- | ------------------ |
| 1B    | 1    | 1   | 1   | 0.85         | 0.56           | 0.59      | 1.44×               | 0.95×              |
| 3B    | 2    | 2   | 1   | 1.14         | 1.41           | 0.78      | 1.46×               | **1.81×**          |
| 7B    | 4    | 4   | 1   | 1.06         | 1.64           | 0.72      | 1.47×               | **2.28×**          |
| 13B   | 8    | 4   | 2   | 2.07         | 2.42           | 1.30      | 1.59×               | **1.86×**          |


### Loss Curve

### Ablation study

#### GEMM Kernel Efficiency (LLaMA-7B, Batch Size = 4)


| Method     | Attn HFU (%) | MLP HFU (%) | Attn GEMM Time (ms) | MLP GEMM Time (ms) |
| ---------- | ------------ | ----------- | ------------------- | ------------------ |
| Vanilla-TP | ~59          | ~59         | ~0.20               | ~0.90              |
| **BTP**    | **~70**      | **~75**     | **~0.16**           | **~0.50**          |


#### Communication Efficiency (LLaMA-7B, Batch Size = 4, Seq Length = 4096)


| Method      | Communication Volume (GB) | Communication Time (ms) | Reduction vs Vanilla |
| ----------- | ------------------------- | ----------------------- | -------------------- |
| FullRank-TP | ~0.25                     | ~2.01                   | –                    |
| Vanilla-TP  | ~1.32                     | ~9.87                   | 1.0×                 |
| **TP**      | **~0.22**                 | **~1.85**               | **5.3× faster**      |


## Citation & Acknowledgement

```bibtex
@article{wang2025boost,
  title={BOOST: BOttleneck-Optimized Scalable Training Framework for Low-Rank Large Language Models},
  author={Wang, Zhengyang and Liu, Ziyue and Zhang, Ruijie and Maurya, Avinash and Hovland, Paul and Nicolae, Bogdan and Cappello, Franck and Zhang, Zheng},
  journal={arXiv preprint arXiv:2512.12131},
  year={2025}
}
```

### Acknowledgement

This project builds on the Nanotron ecosystem and open-source LLM training work from the broader community, including Hugging Face Nanotron, NVIDIA Megatron-LM/Apex, and FlashAttention contributors.