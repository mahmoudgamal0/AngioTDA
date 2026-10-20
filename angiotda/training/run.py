"""Run a seed/fold grid from an experiment YAML preset."""
from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

MULTIMODAL_ARCHITECTURES = {'dino_repeat', 'late_fusion_repeat', 'four_view_repeat'}


def commands(config, *, seeds=None, folds=None, prefix=None, epochs=None,
             graphs_dir=None, graph_ablation=None, resume=False):
    architecture = config["architecture"]
    seeds = config.get("seeds", [0, 1, 2]) if seeds is None else seeds
    folds = config.get("folds", [0, 1, 2, 3, 4]) if folds is None else folds
    if not seeds or not folds or len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ValueError("seeds and folds must be nonempty and unique")
    if any(seed < 0 for seed in seeds) or any(fold < 0 or fold > 4 for fold in folds):
        raise ValueError("seeds must be nonnegative and folds must be in 0..4")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    overrides = {}
    for item in config.get('fit_overrides', []):
        identity = (item['seed'], item['fold'])
        if identity in overrides:
            raise ValueError('duplicate seed/fold override')
        if identity[0] < 0 or not 0 <= identity[1] <= 4:
            raise ValueError('invalid seed/fold override')
        overrides[identity] = item['training']
    prefix = prefix or config.get("run_prefix", architecture)
    module = ('angiotda.training.train_multimodal_fold' if architecture in MULTIMODAL_ARCHITECTURES
              else 'angiotda.training.train_fold')
    for seed in seeds:
        for fold in folds:
            training = dict(config.get("training", {}))
            training.update(overrides.get((seed, fold), {}))
            if epochs is not None: training["epochs"] = epochs
            if graphs_dir is not None: training["graphs_dir"] = graphs_dir
            if graph_ablation is not None: training["graph_ablation"] = graph_ablation
            command = [sys.executable, "-m", module,
                       "--architecture", architecture, "--seed", str(seed), "--fold", str(fold),
                       "--split", str(config.get("split", "splits/compound_folds_seed42.json")),
                       "--run-name", f"{prefix}_seed{seed}_fold{fold}"]
            for key, value in training.items():
                if value is not None: command += ["--" + key.replace("_", "-"), str(value)]
            if resume: command += ["--resume"]
            yield command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--run-prefix")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--graphs-dir")
    parser.add_argument("--graph-ablation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.preset.read_text())
    for command in commands(config, seeds=args.seeds, folds=args.folds,
                            prefix=args.run_prefix, epochs=args.epochs,
                            graphs_dir=args.graphs_dir, graph_ablation=args.graph_ablation,
                            resume=args.resume):
        print(shlex.join(command), flush=True)
        if not args.dry_run: subprocess.run(command, check=True)

if __name__ == "__main__": main()
