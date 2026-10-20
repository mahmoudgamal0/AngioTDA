# AngioTDA

Code for **Graph-Based Self-Supervised Contrastive Learning for Phenotypic
Drug-Screen Profiling**, WACV 2027.

Mahmoud Gamal, Mahmood Alzubaidi, Uzair Shah, Ayman Al Haj Zen, and Marco Agus.

AngioTDA converts vascular microscopy images into skeleton graphs and learns
representations through contrastive learning between replicate wells.

## Install

Use Python 3.10+ and install PyTorch 2.5+ for your CUDA environment first.

```bash
git clone https://github.com/mahmoudgamal0/AngioTDA.git
cd AngioTDA
python -m venv .venv
source .venv/bin/activate
pip install -e '.[graphs,analysis]'
```

## Prepare the data

Datasets and generated outputs are kept outside the repository:

```bash
export ANGIOTDA_WORKSPACE="$HOME/angiotda-workspace"
```

Place the assay images, sample index and reference annotations in this workspace.
The source dataset is the LOPAC endothelial
tube-formation screen of [Sailem and Al Haj Zen (2020)](https://doi.org/10.1038/s41598-020-70440-0).

## Reproduce the paper

```bash
# Extract graphs and construct the shared five-fold compound partition.
angiotda prepare-graphs
angiotda make-splits --seed 42 --folds 5

# Train GIN-E with three seeds across all five folds.
angiotda train --preset configs/gine.yaml

# Evaluate seed 0; repeat for seeds 1 and 2.
angiotda evaluate-biology --architecture gine --seed 0
angiotda evaluate-jump --models gine --model-seed 0 \
  --output-name gine_seed0_jump_external
```

Training uses 500 epochs and the final checkpoint. Both wells of each held-out
compound are excluded from training. Evaluation reports replicate cosine,
replicate AUROC, neighbour correlation, and Target, Reactome and JUMP nDCG@10.

| Main comparison | Configuration or command |
|---|---|
| GIN-E | `configs/gine.yaml` |
| Fine-tuned DINOv3 | `configs/dinov3.yaml` |
| Late fusion | `configs/late-fusion.yaml` |
| Multimodal | `configs/multimodal.yaml` |
| Frozen DINOv3, Cell-DINO, graph statistics | `angiotda evaluate-baseline` |
| Published morphology | `angiotda evaluate-morphology` |
| Random ranking | `angiotda evaluate-random` |

## Code layout

```text
angiotda/data/       image preprocessing, graph extraction and data loading
angiotda/models/     graph and image encoders
angiotda/training/   compound splits and contrastive training
angiotda/eval/       metrics, baselines and statistical comparisons
configs/            experiment settings
```

No datasets, checkpoints or experiment reports are included in this repository.

## Citation

```bibtex
@inproceedings{gamal2027graph,
  title={Graph-Based Self-Supervised Contrastive Learning for Phenotypic Drug-Screen Profiling},
  author={Gamal, Mahmoud and Alzubaidi, Mahmood and Shah, Uzair and Al Haj Zen, Ayman and Agus, Marco},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2027}
}
```

The code license is pending author selection. External data and pretrained
models are subject to their providers' licenses.
