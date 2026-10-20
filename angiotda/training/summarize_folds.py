"""Aggregate held-out fold metrics without mixing incompatible latent spaces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=Path("reports/biological_crossfit"))
    parser.add_argument("--architecture", choices=("gine", "pna"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()
    rows = []
    digests = set()
    for fold in range(args.folds):
        run = root / f"{args.architecture}_seed{args.seed}_fold{fold}"
        path = run / "heldout_evaluation_final.json"
        if not path.exists():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text())
        if (row["architecture"], row["seed"], row["fold"]) != (
                args.architecture, args.seed, fold):
            raise RuntimeError(f"identity mismatch in {path}")
        digests.add(row["split_sha256"])
        rows.append(row)
    if len(digests) != 1:
        raise RuntimeError("fold reports use different compound partitions")

    scalar_keys = ("rep_delta", "rep_auroc", "n_bonferroni_terms")
    summary = {}
    for key in scalar_keys:
        values = [row["metrics"][key] for row in rows]
        summary[key] = {
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "per_fold": values,
        }
    for mode, path in (
        ("primary", ("metrics", "bonferroni_terms")),
        ("multilabel", ("metrics", "multilabel", "bonferroni_terms")),
    ):
        counts = Counter()
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            counts.update(value)
        summary[f"{mode}_term_fold_frequency"] = dict(sorted(counts.items()))
    payload = {
        "evaluation": "within_fold_heldout_metrics_only",
        "architecture": args.architecture,
        "seed": args.seed,
        "n_folds": args.folds,
        "split_sha256": next(iter(digests)),
        "warning": (
            "Fold encoders define different latent coordinate systems. Embeddings "
            "were not pooled and cross-fold cosine similarities are invalid."
        ),
        "summary": summary,
        "fold_reports": rows,
    }
    output = args.output or root / f"{args.architecture}_seed{args.seed}_crossfit_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
