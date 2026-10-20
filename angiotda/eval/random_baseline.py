"""Random-ranking nDCG@10 baselines for the paper's biological endpoints.

Candidate pools and relevance profiles match the five-fold cross-fit evaluation,
but candidate order is sampled uniformly and independently for every query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from angiotda import PROJECT_ROOT

from angiotda.data.paths import DATASET  # noqa: E402
from angiotda.eval.metrics import load_offtargets  # noqa: E402
from angiotda.eval.retrieval import profile_cosine, tfidf_profile_vectors  # noqa: E402
from angiotda.eval.biology import (  # noqa: E402
    _atomic_json,
    _relevance_matrix,
    load_chembl37_human_reactome,
)
from angiotda.eval.jump import load_jump_profiles  # noqa: E402


def endpoint_matrices(split: dict, raw_profiles: dict, *, tfidf: bool) -> list[np.ndarray]:
    matrices = []
    for fold in split["folds"]:
        heldout = set(fold["heldout_compounds"])
        profiles = (tfidf_profile_vectors(raw_profiles, compounds=heldout)
                    if tfidf else {
                        name: vector for name, vector in raw_profiles.items()
                        if name in heldout
                    })
        compounds = sorted(profiles)
        if len(compounds) >= 3:
            matrices.append(_relevance_matrix(profiles, compounds, profile_cosine))
    return matrices


def random_ndcg_at_10(matrices: list[np.ndarray], n_random: int,
                      seed: int) -> dict:
    rng = np.random.RandomState(seed)
    discounts = 1.0 / np.log2(np.arange(2, 12))
    prepared = []
    expected_rows = []
    for relevance in matrices:
        n = len(relevance)
        k = min(10, n - 1)
        ideal = np.sort(relevance, axis=1)[:, ::-1][:, :k]
        idcg = (ideal * discounts[:k]).sum(axis=1)
        eligible = idcg > 0
        mean_relevance = relevance.sum(axis=1) / max(n - 1, 1)
        expected = np.divide(
            mean_relevance * discounts[:k].sum(), idcg,
            out=np.full(n, np.nan), where=eligible)
        expected_rows.extend(expected[eligible].tolist())
        prepared.append((relevance, idcg, eligible, k))

    null_means = np.empty(n_random, dtype=np.float64)
    for iteration in range(n_random):
        fold_values = []
        for relevance, idcg, eligible, k in prepared:
            n = len(relevance)
            keys = rng.random((n, n))
            np.fill_diagonal(keys, np.inf)
            selected = np.argpartition(keys, kth=k - 1, axis=1)[:, :k]
            selected_keys = np.take_along_axis(keys, selected, axis=1)
            order = np.argsort(selected_keys, axis=1)
            selected = np.take_along_axis(selected, order, axis=1)
            ranked = np.take_along_axis(relevance, selected, axis=1)
            dcg = (ranked * discounts[:k]).sum(axis=1)
            ndcg = np.divide(
                dcg, idcg, out=np.full(n, np.nan), where=eligible)
            fold_values.append(ndcg[eligible])
        null_means[iteration] = np.concatenate(fold_values).mean()

    return {
        "n_query_compounds": len(expected_rows),
        "expected_ndcg_at_10": float(np.mean(expected_rows)),
        "monte_carlo_mean": float(null_means.mean()),
        "ci_low": float(np.percentile(null_means, 2.5)),
        "ci_high": float(np.percentile(null_means, 97.5)),
        "n_random_rankings": n_random,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--jump-profiles", type=Path, default=Path(
        "data/external/jump_compound_profiles_overlap.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "reports/biological_crossfit/random_ranking_baselines.json"))
    parser.add_argument("--n-random", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("split", "jump_profiles", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, PROJECT_ROOT / path)

    split = json.loads(args.split.read_text())
    jump, jump_source = load_jump_profiles(args.jump_profiles)
    endpoints = {
        "Target": endpoint_matrices(
            split,
            load_offtargets(path=DATASET.index_dir / "offtargets_full291.json"),
            tfidf=True),
        "Reactome": endpoint_matrices(
            split, load_chembl37_human_reactome(), tfidf=True),
        "JUMP": endpoint_matrices(split, jump, tfidf=False),
    }
    payload = {
        "status": "complete",
        "protocol": {
            "ranking": "uniform random candidate order independently per query",
            "ranking_boundary": "within_crossfit_fold_only",
            "identical_compound_excluded": True,
            "k": 10,
        },
        "split_sha256": split["compound_partition_sha256"],
        "jump_source": jump_source,
        "endpoints": {
            name: random_ndcg_at_10(matrices, args.n_random, args.seed)
            for name, matrices in endpoints.items()
        },
    }
    _atomic_json(args.output, payload)
    lines = [
        "# Random-ranking biological retrieval baselines", "",
        "Candidate rankings are sampled uniformly within each held-out fold. "
        "The identical compound is excluded.", "",
        "| Endpoint | Queries | Expected nDCG@10 | Monte Carlo mean | 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["endpoints"].items():
        lines.append(
            f"| {name} | {row['n_query_compounds']} | "
            f"{row['expected_ndcg_at_10']:.4f} | {row['monte_carlo_mean']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
