"""Deterministic multilabel-aware compound K-fold assignment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path


def _tie_key(seed: int, *parts: object) -> str:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode()).hexdigest()


def make_multilabel_folds(
    compounds: list[str],
    label_map: dict[str, list[str]],
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Assign each compound to exactly one fold using iterative stratification.

    Assignment is deterministic for the same compound set, annotations, seed,
    and fold count. Rare labels are balanced greedily where possible; labels
    occurring fewer than ``n_folds`` times are explicitly reported as not
    capable of appearing in every fold.
    """
    input_compounds = list(compounds)
    if len(input_compounds) != len(set(input_compounds)):
        raise ValueError("compound identifiers must be unique")
    compounds = sorted(input_compounds)
    if n_folds < 2 or len(compounds) < n_folds:
        raise ValueError("n_folds must be >=2 and no larger than the compound count")

    labels = {
        compound: tuple(sorted(set(label_map.get(compound, []))))
        for compound in compounds
    }
    totals = Counter(term for compound in compounds for term in labels[compound])
    base, remainder = divmod(len(compounds), n_folds)
    capacities = [base + int(fold < remainder) for fold in range(n_folds)]
    target = {
        (fold, term): totals[term] * capacities[fold] / len(compounds)
        for fold in range(n_folds) for term in totals
    }
    fold_compounds = [[] for _ in range(n_folds)]
    fold_counts = [Counter() for _ in range(n_folds)]
    remaining = set(compounds)
    remaining_counts = totals.copy()

    while remaining:
        labelled = [compound for compound in remaining if labels[compound]]
        if labelled:
            positive = [count for count in remaining_counts.values() if count > 0]
            rarest = min(positive)
            rare_terms = {
                term for term, count in remaining_counts.items() if count == rarest
            }
            candidates = [
                compound for compound in labelled
                if rare_terms.intersection(labels[compound])
            ]
            compound = min(
                candidates,
                key=lambda value: (
                    -len(labels[value]),
                    _tie_key(seed, "compound", value),
                ),
            )
        else:
            compound = min(remaining, key=lambda value: _tie_key(seed, "unlabelled", value))

        available = [
            fold for fold in range(n_folds)
            if len(fold_compounds[fold]) < capacities[fold]
        ]
        if not available:
            raise AssertionError("fold capacity accounting failed")

        def fold_priority(fold: int):
            # Larger unmet label need is better. Size deficit is a secondary
            # criterion and the seeded digest makes exact ties reproducible.
            label_need = sum(
                (target[(fold, term)] - fold_counts[fold][term])
                / max(totals[term], 1)
                for term in labels[compound]
            )
            size_need = capacities[fold] - len(fold_compounds[fold])
            return (-label_need, -size_need, _tie_key(seed, "fold", compound, fold))

        chosen = min(available, key=fold_priority)
        fold_compounds[chosen].append(compound)
        fold_counts[chosen].update(labels[compound])
        remaining.remove(compound)
        remaining_counts.subtract(labels[compound])

    fold_compounds = [sorted(values) for values in fold_compounds]
    assigned = [compound for values in fold_compounds for compound in values]
    if sorted(assigned) != compounds or len(assigned) != len(set(assigned)):
        raise AssertionError("folds do not form an exact compound partition")

    folds = []
    universe = set(compounds)
    for fold, heldout in enumerate(fold_compounds):
        heldout_set = set(heldout)
        train = sorted(universe - heldout_set)
        if heldout_set.intersection(train):
            raise AssertionError("compound leakage between train and heldout")
        folds.append({
            "fold": fold,
            "n_train_compounds": len(train),
            "n_heldout_compounds": len(heldout),
            "train_compounds": train,
            "heldout_compounds": heldout,
            "heldout_label_counts": dict(sorted(fold_counts[fold].items())),
        })

    canonical = json.dumps(
        {"seed": seed, "n_folds": n_folds, "fold_compounds": fold_compounds},
        sort_keys=True, separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "method": "deterministic_greedy_iterative_multilabel_stratification",
        "seed": seed,
        "n_folds": n_folds,
        "n_compounds": len(compounds),
        "fold_capacities": capacities,
        "compound_partition_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "label_counts_full": dict(sorted(totals.items())),
        "labels_too_rare_for_all_folds": sorted(
            term for term, count in totals.items() if count < n_folds
        ),
        "n_unannotated_compounds": sum(not labels[compound] for compound in compounds),
        "folds": folds,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
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


def write_split_report(path: Path, payload: dict) -> None:
    lines = [
        "# Compound-grouped biological cross-fit split", "",
        f"- Compounds: `{payload['n_compounds']}`",
        f"- Folds: `{payload['n_folds']}`",
        f"- Split seed: `{payload['seed']}`",
        f"- Partition SHA-256: `{payload['compound_partition_sha256']}`",
        f"- Unannotated compounds: `{payload['n_unannotated_compounds']}`",
        "", "Every compound is held out exactly once and absent from that fold's training set.",
        "", "| Fold | Train | Held out | Powered held-out labels (n>=5) |",
        "|---:|---:|---:|---|",
    ]
    for fold in payload["folds"]:
        powered = ", ".join(
            f"{term} ({count})" for term, count in fold["heldout_label_counts"].items()
            if count >= 5
        ) or "-"
        lines.append(
            f"| {fold['fold']} | {fold['n_train_compounds']} | "
            f"{fold['n_heldout_compounds']} | {powered} |"
        )
    lines.extend([
        "", "Labels occurring fewer times than the number of folds cannot be "
        "represented in every held-out fold:", "",
        ", ".join(payload["labels_too_rare_for_all_folds"]) or "None.",
    ])
    Path(path).write_text("\n".join(lines) + "\n")
