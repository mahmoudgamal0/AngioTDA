"""Evaluate unseen-compound biological retrieval and candidate neighbours.

The holdout checkpoint is scored only on compounds excluded from its training
split. Physical R1/R2 images are matched from their image names, averaged
within replicate when a library compound appears at multiple well locations,
and kept separate across replicates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from angiotda import PROJECT_ROOT

from angiotda.data.dataset import GraphDatasetV2, replicate_pair_key
from angiotda.data.paths import DATASET
from angiotda.eval.embed import embed_gine
from angiotda.eval.metrics import load_moa
from angiotda.eval.retrieval import (
    cross_replicate_retrieval,
    summarize_retrieval,
)
from angiotda import CONTROLS


REPLICATE_RE = re.compile(r"^P\d+R([12])_R\d+C\d+$")
DEFAULT_CANDIDATES = (
    ("DMH4", "Eupatorin"),
    ("JS-K", "Sanguinarine chloride"),
    ("Nocodazole", "Piperlongumine"),
)


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def holdout_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    for key in ("holdout_compounds", "test_compounds", "heldout_compounds"):
        if key in payload:
            return set(payload[key])
    raise KeyError(f"no holdout compound list in {path}")


def replicate_views(well_embeddings: dict[tuple[str, int], np.ndarray],
                    keep: set[str] | None = None) -> tuple[dict, dict, dict]:
    """Recover matched physical R1/R2 views from canonical well embeddings."""
    dataset = GraphDatasetV2(
        DATASET.sample_index, DATASET.graphs_dir, controls=CONTROLS)
    counters = defaultdict(int)
    physical = defaultdict(dict)
    for sample in dataset.samples:
        compound = sample["compound"]
        well_key = (compound, counters[compound])
        counters[compound] += 1
        if keep is not None and compound not in keep:
            continue
        match = REPLICATE_RE.fullmatch(sample["image_name"])
        if match is None or well_key not in well_embeddings:
            continue
        physical[(compound, replicate_pair_key(sample["image_name"]))][
            int(match.group(1))] = well_embeddings[well_key]

    by_compound = defaultdict(lambda: {1: [], 2: []})
    n_physical_pairs = 0
    for (compound, _), views in physical.items():
        if 1 not in views or 2 not in views:
            continue
        n_physical_pairs += 1
        by_compound[compound][1].append(views[1])
        by_compound[compound][2].append(views[2])

    replicate_1, replicate_2 = {}, {}
    for compound, views in by_compound.items():
        if views[1] and views[2]:
            replicate_1[compound] = unit(np.mean(views[1], axis=0))
            replicate_2[compound] = unit(np.mean(views[2], axis=0))
    audit = {
        "n_compounds_with_matched_replicates": len(replicate_1),
        "n_physical_replicate_pairs": n_physical_pairs,
        "replicate_assignment": "image_name R1/R2; multiple physical wells averaged within replicate",
    }
    return replicate_1, replicate_2, audit


def permutation_pvalues(replicate_1, replicate_2, label_map, observed,
                        n_permutations: int, seed: int = 42) -> dict:
    """Shuffle complete label sets across compounds, preserving multiplicity."""
    annotated = sorted(
        compound for compound in set(replicate_1) & set(replicate_2)
        if label_map.get(compound))
    values = [label_map[compound] for compound in annotated]
    metric_names = sorted(observed["metrics"])
    observed_values = {
        metric: observed["metrics"][metric]["mean"] for metric in metric_names
    }
    exceed = {metric: 0 for metric in metric_names}
    rng = np.random.RandomState(seed)
    for _ in range(n_permutations):
        order = rng.permutation(len(annotated))
        permuted = {
            compound: values[int(order[index])]
            for index, compound in enumerate(annotated)
        }
        rows = cross_replicate_retrieval(
            replicate_1, replicate_2, permuted)
        summary = summarize_retrieval(rows, n_bootstrap=0)
        for metric in metric_names:
            if summary["metrics"][metric]["mean"] >= observed_values[metric]:
                exceed[metric] += 1
    return {
        metric: float((exceed[metric] + 1) / (n_permutations + 1))
        for metric in metric_names
    }


def rank_of(query: str, target: str, embeddings: dict[str, np.ndarray],
            allowed: list[str]) -> int:
    candidates = [compound for compound in allowed if compound != query]
    similarities = np.asarray([
        float(embeddings[query] @ embeddings[compound])
        for compound in candidates
    ])
    order = np.argsort(-similarities, kind="stable")
    return int(np.flatnonzero(np.asarray(candidates, dtype=object)[order] == target)[0] + 1)


def cross_replicate_rank(query: str, target: str, query_view, gallery_view,
                         allowed: list[str]) -> int:
    candidates = [compound for compound in allowed if compound != query]
    similarities = np.asarray([
        float(query_view[query] @ gallery_view[compound])
        for compound in candidates
    ])
    order = np.argsort(-similarities, kind="stable")
    return int(np.flatnonzero(np.asarray(candidates, dtype=object)[order] == target)[0] + 1)


def candidate_rows(checkpoint: Path, seed: int, label_map: dict) -> list[dict]:
    compound_embeddings, well_embeddings = embed_gine(checkpoint)
    replicate_1, replicate_2, _ = replicate_views(well_embeddings)
    annotated = sorted(
        compound for compound in compound_embeddings
        if label_map.get(compound)
        and compound in replicate_1 and compound in replicate_2)
    rows = []
    for left, right in DEFAULT_CANDIDATES:
        if left not in annotated or right not in annotated:
            rows.append({"seed": seed, "left": left, "right": right,
                         "status": "missing"})
            continue
        compound_ranks = [
            rank_of(left, right, compound_embeddings, annotated),
            rank_of(right, left, compound_embeddings, annotated),
        ]
        replicate_ranks = [
            cross_replicate_rank(left, right, replicate_1, replicate_2, annotated),
            cross_replicate_rank(left, right, replicate_2, replicate_1, annotated),
            cross_replicate_rank(right, left, replicate_1, replicate_2, annotated),
            cross_replicate_rank(right, left, replicate_2, replicate_1, annotated),
        ]
        rows.append({
            "seed": seed,
            "left": left,
            "right": right,
            "status": "complete",
            "left_labels": label_map[left],
            "right_labels": label_map[right],
            "labels_disjoint": not bool(set(label_map[left]) & set(label_map[right])),
            "compound_cosine": float(
                compound_embeddings[left] @ compound_embeddings[right]),
            "compound_reciprocal_ranks": compound_ranks,
            "cross_replicate_four_direction_ranks": replicate_ranks,
            "n_annotated_candidates": len(annotated) - 1,
        })
    return rows


def render_markdown(payload: dict) -> str:
    result = payload["unseen_compound_retrieval"]
    lines = [
        "# Cross-replicate biological-neighbour retrieval",
        "",
        "The identical compound is excluded from every ranking. Labels are used only for evaluation.",
        "",
        f"- Held-out query compounds with a known positive: `{result['summary']['n_query_compounds']}`",
        f"- Matched held-out replicate compounds: `{result['replicate_audit']['n_compounds_with_matched_replicates']}`",
        f"- Label permutations: `{result['n_permutations']}` (complete multi-label sets shuffled by compound)",
        "",
        "| Metric | Mean | 95% compound-bootstrap CI | Permutation p |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in result["summary"]["metrics"].items():
        lines.append(
            f"| {metric} | {values['mean']:.4f} | "
            f"[{values['ci_low']:.4f}, {values['ci_high']:.4f}] | "
            f"{result['permutation_p'][metric]:.5f} |")
    lines.extend(["", "## Candidate relationship stability", "",
                  "Ranks are among annotated compounds; lower is better.", "",
                  "| Seed | Pair | Cosine | Compound reciprocal ranks | Cross-replicate ranks |",
                  "|---:|---|---:|---:|---:|"])
    for row in payload["candidate_relationships"]:
        if row["status"] != "complete":
            lines.append(f"| {row['seed']} | {row['left']} / {row['right']} | missing | — | — |")
            continue
        lines.append(
            f"| {row['seed']} | {row['left']} / {row['right']} | "
            f"{row['compound_cosine']:.6f} | "
            f"{row['compound_reciprocal_ranks']} | "
            f"{row['cross_replicate_four_direction_ranks']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-checkpoint", type=Path, default=Path(
        "checkpoints/experiments/gine_holdout_80_20_seed0/seed0/best.pt"))
    parser.add_argument("--holdout-split", type=Path, default=Path(
        "experiments/gine_holdout_80_20_seed0/split/split_seed0.json"))
    parser.add_argument("--candidate-seeds", type=int, nargs="*", default=range(5))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--output", type=Path, default=Path(
        "reports/biological_retrieval/gine_unseen_and_candidates.json"))
    args = parser.parse_args()
    for name in ("holdout_checkpoint", "holdout_split", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, PROJECT_ROOT / value)

    label_map = load_moa("sailem57", project_root=PROJECT_ROOT)
    print("Embedding compound-holdout checkpoint", flush=True)
    _, holdout_wells = embed_gine(args.holdout_checkpoint)
    replicate_1, replicate_2, audit = replicate_views(
        holdout_wells, holdout_ids(args.holdout_split))
    retrieval_rows = cross_replicate_retrieval(
        replicate_1, replicate_2, label_map)
    summary = summarize_retrieval(
        retrieval_rows, n_bootstrap=args.n_bootstrap)
    permutation_p = permutation_pvalues(
        replicate_1, replicate_2, label_map, summary,
        n_permutations=args.n_permutations)
    payload = {
        "evaluation": "unseen_compound_cross_replicate_biological_retrieval",
        "label_variant": "sailem57_all_annotations",
        "self_compound_excluded": True,
        "unseen_compound_retrieval": {
            "checkpoint": str(args.holdout_checkpoint.relative_to(PROJECT_ROOT)),
            "split": str(args.holdout_split.relative_to(PROJECT_ROOT)),
            "replicate_audit": audit,
            "summary": summary,
            "permutation_p": permutation_p,
            "n_permutations": args.n_permutations,
            "per_query": retrieval_rows,
        },
        "candidate_relationships": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.candidate_seeds:
        checkpoint = PROJECT_ROOT / f"checkpoints/gine_seed{seed}_uniqbatch_maskneg/best.pt"
        print(f"Validating candidate relationships, seed {seed}", flush=True)
        payload["candidate_relationships"].extend(
            candidate_rows(checkpoint, seed, label_map))
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        args.output.with_suffix(".md").write_text(render_markdown(payload))

    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(payload))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

