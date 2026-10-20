"""Paired out-of-fold comparison of two completed biology reports.

The same held-out compound is the pairing unit. Confidence intervals resample
paired compounds; fold-wise deltas are also retained so a pooled improvement
cannot hide a reversal in one training fold.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def paired_difference(reference: dict, contender: dict, *, n_bootstrap: int,
                      seed: int) -> dict:
    keys = sorted(set(reference) & set(contender))
    if set(reference) != set(contender):
        raise RuntimeError(
            f"paired query sets differ: reference={len(reference)}, "
            f"contender={len(contender)}, intersection={len(keys)}")
    if not keys:
        return {"n_paired_compounds": 0}
    values = np.asarray([contender[key] - reference[key] for key in keys], dtype=float)
    rng = np.random.RandomState(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        bootstrap[index] = values[rng.randint(0, len(values), len(values))].mean()
    folds = defaultdict(list)
    for key, value in zip(keys, values):
        folds[key.split(":", 1)[0]].append(float(value))
    return {
        "n_paired_compounds": len(keys),
        "difference": "contender_minus_reference",
        "mean": float(values.mean()),
        "ci_low": (float(np.percentile(bootstrap, 2.5))
                   if n_bootstrap else float(values.mean())),
        "ci_high": (float(np.percentile(bootstrap, 97.5))
                    if n_bootstrap else float(values.mean())),
        "n_bootstrap": n_bootstrap,
        "bootstrap_unit": "paired_out_of_fold_compound",
        "per_fold_mean": {
            fold: float(np.mean(fold_values))
            for fold, fold_values in sorted(folds.items())
        },
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def report_fold_indices(report: dict) -> list[int]:
    indices = report.get("fold_indices")
    return list(indices) if indices is not None else list(range(report["n_folds"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--contender", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-confirmation-subset", action="store_true",
        help="Allow matching architecture-confirmation folds (normally full partitions only).")
    args = parser.parse_args()
    reference, contender = json.loads(args.reference.read_text()), json.loads(
        args.contender.read_text())
    for name, report in (("reference", reference), ("contender", contender)):
        if report.get("status") != "complete":
            raise RuntimeError(f"{name} report is not complete")
        complete_partition = (
            report.get("claim_scope") == "out_of_fold_complete_partition" or
            (report.get("n_folds") == 5 and
             report.get("protocol", {}).get("ranking_boundary") ==
             "within_crossfit_fold_only")
        )
        confirmation = (
            args.allow_confirmation_subset and
            report.get("claim_scope") == "architecture_confirmation_folds")
        if not (complete_partition or confirmation):
            raise RuntimeError(f"{name} is not a complete out-of-fold partition")
    if reference.get("split_sha256") != contender.get("split_sha256"):
        raise RuntimeError("reports use different compound partitions")
    if report_fold_indices(reference) != report_fold_indices(contender):
        raise RuntimeError("reports use different out-of-fold subsets")

    comparisons = {}
    for endpoint in sorted(set(reference["endpoints"]) & set(contender["endpoints"])):
        left = reference["endpoints"][endpoint]["per_compound_metrics"]
        right = contender["endpoints"][endpoint]["per_compound_metrics"]
        metric_names = sorted(next(iter(left.values()))) if left else []
        comparisons[endpoint] = {
            metric: paired_difference(
                {key: row[metric] for key, row in left.items()},
                {key: row[metric] for key, row in right.items()},
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
            for metric in metric_names
        }
    for section in ("neighborhood_stability", "replicate_identity"):
        per_compound = f"{section}_per_compound"
        if per_compound not in reference or per_compound not in contender:
            continue
        left = reference[per_compound]
        right = contender[per_compound]
        comparisons[section] = {
            metric: paired_difference(
                {key: row[metric] for key, row in left.items()},
                {key: row[metric] for key, row in right.items()},
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
            for metric in sorted(next(iter(left.values())))
        }
    payload = {
        "status": "complete",
        "reference": str(args.reference),
        "contender": str(args.contender),
        "split_sha256": reference["split_sha256"],
        "inference": "paired out-of-fold compound bootstrap confidence interval",
        "comparisons": comparisons,
    }
    atomic_json(args.output, payload)
    lines = [
        "# Paired cross-fit model comparison", "",
        "Differences are contender minus reference. Confidence intervals resample "
        "the same out-of-fold compounds as paired units.", "",
        "| Endpoint | Metric | N | Difference | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for endpoint in ("sailem57", "chembl_offtargets",
                     "jump_cellpainting",
                     "chembl37_human_reactome", "neighborhood_stability",
                     "replicate_identity"):
        for metric in ("average_precision", "ndcg_at_10", "rank_correlation",
                       "replicate_consistency", "replicate_auroc",
                       "paired_cosine", "reciprocal_rank", "hit_at_1",
                       "hit_at_5", "hit_at_10"):
            row = comparisons.get(endpoint, {}).get(metric)
            if not row:
                continue
            lines.append(
                f"| {endpoint} | {metric} | {row['n_paired_compounds']} | "
                f"{row['mean']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
