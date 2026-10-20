"""Create or verify the deterministic compound cross-fit partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from angiotda import PROJECT_ROOT

from angiotda.data.dataset import GraphDatasetV2
from angiotda.data.paths import DATASET
from angiotda.eval.metrics import load_moa
from angiotda.training.splits import (
    atomic_write_json,
    make_multilabel_folds,
    write_split_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        PROJECT_ROOT / "splits"
        / f"compound_folds_seed{args.seed}.json"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    dataset = GraphDatasetV2(DATASET.sample_index, DATASET.graphs_dir)
    payload = make_multilabel_folds(
        dataset.compounds, load_moa("sailem57"),
        n_folds=args.folds, seed=args.seed,
    )
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise RuntimeError(
                f"existing split differs from deterministic result: {output}")
        print(f"Verified existing split: {output}")
    else:
        atomic_write_json(output, payload)
        print(f"Wrote split: {output}")
    write_split_report(output.with_suffix(".md"), payload)
    print(payload["compound_partition_sha256"])


if __name__ == "__main__":
    main()
