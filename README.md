# Exploring the Impact of a Transformer's Latent Space Geometry on Downstream Task Performance

Code for the paper:

> **Exploring the Impact of a Transformer's Latent Space Geometry on Downstream Task Performance**  
> Anna C. Marbut, J. Chandler, Travis J. Wheeler  
> *arXiv preprint arXiv:2406.12159*  
> [arXiv](https://arxiv.org/abs/2406.12159)

## Overview

How does the spatial structure of a transformer's contextual representations relate to what it can do on downstream tasks? We apply a suite of geometric measures to the latent spaces of BERT-family models at multiple layers and find that **quantized cell density** — which we term Point Patchiness (PP) — predicts GLUE benchmark performance with r = 0.9.

## Repository Structure

```
sample_build_and_metrics.py   Build representation samples and compute geometric metrics
model_perturb.py              Perturb model representations for controlled experiments
run_glue.py                   Run GLUE fine-tuning and evaluation
glue_parse.py                 Parse and aggregate GLUE evaluation outputs
sample_sequences.pkl          Pre-built sample token sequences
```

## Reproducing Results

1. Build samples and compute metrics: `python sample_build_and_metrics.py`
2. Run GLUE evaluation: `python run_glue.py`
3. Parse results: `python glue_parse.py`

Experiments were run on a SLURM HPC cluster; adapt resource requests as needed for your environment.

## Key Dependencies

```
transformers
torch
faiss
sklearn
numpy
pandas
```

## Citation

```bibtex
@article{marbut2024exploring,
  title   = {Exploring the Impact of a Transformer's Latent Space Geometry on Downstream Task Performance},
  author  = {Marbut, Anna C. and Chandler, J. and Wheeler, Travis J.},
  journal = {arXiv preprint arXiv:2406.12159},
  year    = {2024}
}
```
