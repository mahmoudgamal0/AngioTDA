"""Test whether GIN-E biological retrieval exceeds simple network scale.

Two leakage-safe controls are reported:

1. Seven simple network-scale statistics and all 26 graph statistics are
   normalized using training-compound wells only, then evaluated on each
   held-out fold.
2. GIN-E, 26D statistics, and a random ranking are evaluated on identical
   direction-specific candidate pools containing the nearest compounds in the
   seven-statistic space.  R1 queries select candidates from R2 statistics and
   conversely, so matching never averages a query with its paired replicate.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


from angiotda import PROJECT_ROOT

from angiotda import CONTROLS  # noqa: E402
from angiotda.data.paths import DATASET  # noqa: E402
from angiotda.eval.metrics import load_offtargets  # noqa: E402
from angiotda.eval.retrieval import (  # noqa: E402
    compound_metric_means,
    profile_cosine,
    summarize_retrieval,
    tfidf_profile_vectors,
)
from angiotda.eval.biology import (  # noqa: E402
    _atomic_json,
    _metric_summary,
    evaluate_endpoint,
    load_chembl37_human_reactome,
    load_fold_artifact,
)
from angiotda.eval.jump import load_jump_profiles  # noqa: E402


REPLICATE_RE = re.compile(r"^(P\d+)R([12])_(R\d+C\d+)$")
SIMPLE_INDICES = (0, 1, 2, 3, 8, 15, 16)
SIMPLE_NAMES = (
    "node_count", "edge_count", "junction_count", "endpoint_count",
    "connected_component_count", "total_centerline_length", "mean_branch_length",
)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def load_graph_rows() -> list[dict]:
    samples = json.loads(DATASET.sample_index.read_text())["samples"]
    controls = {name.lower() for name in CONTROLS}
    rows = []
    for sample in samples:
        compound = str(sample["compound"])
        if compound.lower() in controls:
            continue
        image_name = str(sample["image_name"])
        match = REPLICATE_RE.fullmatch(image_name)
        if match is None:
            raise ValueError(f"cannot parse replicate from {image_name!r}")
        path = DATASET.graphs_dir / f"{image_name}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            stats = np.asarray(payload["graph_stats"], dtype=np.float64)
        if stats.shape != (26,) or not np.all(np.isfinite(stats)):
            raise ValueError(f"invalid graph statistics in {path}")
        rows.append({
            "compound": compound,
            "image_name": image_name,
            "replicate": int(match.group(2)),
            "physical_key": f"{match.group(1)}_{match.group(3)}",
            "stats": stats,
        })
    return rows


def fit_transform(rows: list[dict], train_compounds: set[str], indices) -> list[dict]:
    """Log/min-max transform with limits fitted only on training compounds."""
    indices = np.asarray(indices, dtype=int)
    train = np.stack([
        np.log1p(row["stats"][indices])
        for row in rows if row["compound"] in train_compounds
    ])
    lower = train.min(axis=0)
    span = train.max(axis=0) - lower
    span[span <= 1e-12] = 1.0
    output = []
    for row in rows:
        copy = dict(row)
        copy["vector"] = (np.log1p(row["stats"][indices]) - lower) / span
        output.append(copy)
    return output


def replicate_views(rows: list[dict], heldout: set[str]) -> tuple[dict, dict, dict, dict]:
    grouped = defaultdict(lambda: {1: [], 2: []})
    by_physical = defaultdict(dict)
    for row in rows:
        if row["compound"] in heldout:
            by_physical[(row["compound"], row["physical_key"])][row["replicate"]] = row["vector"]
    for (compound, _), views in by_physical.items():
        if set(views) == {1, 2}:
            grouped[compound][1].append(views[1])
            grouped[compound][2].append(views[2])
    first_raw = {name: np.mean(views[1], axis=0) for name, views in grouped.items()}
    second_raw = {name: np.mean(views[2], axis=0) for name, views in grouped.items()}
    first = {name: unit(vector) for name, vector in first_raw.items()}
    second = {name: unit(vector) for name, vector in second_raw.items()}
    return first, second, first_raw, second_raw


def build_stat_folds(rows: list[dict], split: dict, indices) -> list[dict]:
    folds = []
    for fold in split["folds"]:
        heldout = set(fold["heldout_compounds"])
        transformed = fit_transform(rows, set(fold["train_compounds"]), indices)
        first, second, first_raw, second_raw = replicate_views(transformed, heldout)
        if set(first) != heldout or set(second) != heldout:
            raise RuntimeError(f"statistics do not cover fold {fold['fold']}")
        folds.append({
            "fold": fold["fold"], "replicate_1": first, "replicate_2": second,
            "match_1": first_raw, "match_2": second_raw,
        })
    return folds


def load_gine_folds(split: dict, report_root: Path, seed: int) -> list[dict]:
    folds = []
    for fold in split["folds"]:
        index = fold["fold"]
        path = report_root / f"gine_seed{seed}_fold{index}" / "heldout_embeddings_final.npz"
        folds.append(load_fold_artifact(
            path, "gine", seed, index, set(fold["heldout_compounds"])))
    return folds


def endpoint_profiles(kind: str, heldout: set[str], source: dict) -> dict:
    if kind in {"target", "reactome"}:
        return tfidf_profile_vectors(source, compounds=heldout)
    return {name: source[name] for name in heldout if name in source}


def ndcg(relevance: np.ndarray, k: int = 10) -> float:
    cutoff = min(k, len(relevance))
    discount = np.log2(np.arange(2, cutoff + 2))
    dcg = float((relevance[:cutoff] / discount).sum())
    ideal = np.sort(relevance)[::-1][:cutoff]
    idcg = float((ideal / discount).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def matched_rows(gine: dict, stats26: dict, simple: dict, profiles: dict,
                 pool_size: int) -> dict[str, list[dict]]:
    """Evaluate all rankings on pools fixed by simple opposite-view distance."""
    output = {"gine": [], "stats_26d": [], "random": []}
    shared = set(gine["replicate_1"]) & set(gine["replicate_2"])
    shared &= set(stats26["replicate_1"]) & set(stats26["replicate_2"])
    shared &= set(simple["replicate_1"]) & set(simple["replicate_2"]) & set(profiles)
    compounds = sorted(shared)
    directions = (
        ("R1_to_R2", "replicate_1", "replicate_2", "match_1", "match_2"),
        ("R2_to_R1", "replicate_2", "replicate_1", "match_2", "match_1"),
    )
    for direction, query_key, gallery_key, match_query_key, match_gallery_key in directions:
        for query in compounds:
            candidates = [name for name in compounds if name != query]
            distances = np.asarray([
                np.linalg.norm(simple[match_query_key][query] - simple[match_gallery_key][name])
                for name in candidates
            ])
            order = np.argsort(distances, kind="stable")[:min(pool_size, len(candidates))]
            candidates = [candidates[index] for index in order]
            relevance = np.asarray([
                profile_cosine(profiles[query], profiles[name]) for name in candidates
            ])
            if not np.any(relevance > 0):
                continue
            for model in ("gine", "stats_26d"):
                views = gine if model == "gine" else stats26
                similarity = np.asarray([
                    views[gallery_key][name] @ views[query_key][query]
                    for name in candidates
                ])
                ranked = relevance[np.argsort(-similarity, kind="stable")]
                output[model].append({
                    "compound": query, "direction": direction,
                    "n_candidates": len(candidates), "ndcg_at_10": ndcg(ranked),
                })
            cutoff = min(10, len(candidates))
            expected_dcg = float(
                relevance.mean() * (1 / np.log2(np.arange(2, cutoff + 2))).sum())
            ideal = np.sort(relevance)[::-1][:cutoff]
            ideal_dcg = float((ideal / np.log2(np.arange(2, cutoff + 2))).sum())
            output["random"].append({
                "compound": query, "direction": direction,
                "n_candidates": len(candidates),
                "ndcg_at_10": expected_dcg / ideal_dcg,
            })
    return output


def paired_difference(left: dict, right: dict, n_bootstrap: int, seed: int) -> dict:
    shared = sorted(set(left) & set(right))
    values = np.asarray([left[name]["ndcg_at_10"] - right[name]["ndcg_at_10"]
                         for name in shared])
    rng = np.random.RandomState(seed)
    boot = np.asarray([
        values[rng.randint(0, len(values), len(values))].mean()
        for _ in range(n_bootstrap)
    ])
    return {
        "n_paired_compounds": len(shared), "mean_difference": float(values.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "two_sided_bootstrap_p": float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0))),
    }


def summarize_matched(all_rows: dict, n_bootstrap: int, seed: int) -> dict:
    summaries = {}
    per_compound = {}
    for model, rows in all_rows.items():
        summaries[model] = summarize_retrieval(rows, n_bootstrap, seed)
        per_compound[model] = compound_metric_means(rows)
    return {
        "models": summaries,
        "paired_gine_minus_26d": paired_difference(
            per_compound["gine"], per_compound["stats_26d"], n_bootstrap, seed),
        "paired_gine_minus_random": paired_difference(
            per_compound["gine"], per_compound["random"], n_bootstrap, seed),
    }


def append_rows(target: dict, source: dict) -> None:
    for model, rows in source.items():
        target[model].extend(rows)


def average_per_compound(results: list[dict]) -> dict:
    """Average each compound-level metric across training seeds."""
    shared = set(results[0])
    for result in results[1:]:
        shared &= set(result)
    output = {}
    for compound in sorted(shared):
        metrics = set(results[0][compound])
        for result in results[1:]:
            metrics &= set(result[compound])
        output[compound] = {
            metric: float(np.mean([result[compound][metric] for result in results]))
            for metric in sorted(metrics)
        }
    return output


def average_endpoint_results(results: list[dict], n_bootstrap: int,
                             seed: int) -> dict:
    """Combine independently evaluated seeds at the compound-metric level."""
    per_compound = average_per_compound([
        result["per_compound_metrics"] for result in results
    ])
    return {
        "relevance": results[0]["relevance"],
        "ranking_boundary": results[0]["ranking_boundary"],
        "summary": _metric_summary(per_compound, n_bootstrap, seed),
        "per_compound_metrics": per_compound,
        "aggregation": "mean per out-of-fold compound across training seeds",
        "per_seed_summary": [result["summary"] for result in results],
    }


def average_directional_rows(seed_rows: list[list[dict]]) -> list[dict]:
    """Average matched-pool directional nDCG rows across training seeds."""
    indexed = []
    for rows in seed_rows:
        indexed.append({(row["compound"], row["direction"]): row for row in rows})
    shared = set(indexed[0])
    for rows in indexed[1:]:
        shared &= set(rows)
    output = []
    for key in sorted(shared):
        first = indexed[0][key]
        output.append({
            "compound": first["compound"],
            "direction": first["direction"],
            "n_candidates": first["n_candidates"],
            "ndcg_at_10": float(np.mean([rows[key]["ndcg_at_10"] for rows in indexed])),
        })
    return output


def render_markdown(payload: dict) -> str:
    lines = [
        "# Topology-versus-network-scale controls", "",
        "All transforms are fitted on training-compound wells only. Biological rankings "
        "remain inside held-out folds.", "",
        "## Strict cross-fit statistic baselines", "",
        "| Endpoint | GIN-E | 7D simple | 26D | GIN-E−7D (95% CI) | GIN-E−26D (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("target", "Target"), ("reactome", "Reactome"), ("jump", "JUMP")):
        row = payload["strict_crossfit_baselines"][key]
        gine = row["gine"]["summary"]["metrics"]["ndcg_at_10"]["mean"]
        simple = row["simple_7d"]["summary"]["metrics"]["ndcg_at_10"]["mean"]
        full = row["stats_26d"]["summary"]["metrics"]["ndcg_at_10"]["mean"]
        ds, df = row["paired_gine_minus_simple_7d"], row["paired_gine_minus_26d"]
        lines.append(
            f"| {label} | {gine:.4f} | {simple:.4f} | {full:.4f} | "
            f"{ds['mean_difference']:+.4f} [{ds['ci_low']:+.4f}, {ds['ci_high']:+.4f}] | "
            f"{df['mean_difference']:+.4f} [{df['ci_low']:+.4f}, {df['ci_high']:+.4f}] |")
    lines += ["", f"## Retrieval among {payload['protocol']['matched_pool_size']} scale-matched candidates", "",
              "| Endpoint | GIN-E | 26D | Random | GIN-E−26D (95% CI) | GIN-E−random (95% CI) |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("target", "Target"), ("reactome", "Reactome"), ("jump", "JUMP")):
        row = payload["matched_retrieval"][key]
        value = lambda model: row["models"][model]["metrics"]["ndcg_at_10"]["mean"]
        d26, dr = row["paired_gine_minus_26d"], row["paired_gine_minus_random"]
        lines.append(
            f"| {label} | {value('gine'):.4f} | {value('stats_26d'):.4f} | "
            f"{value('random'):.4f} | {d26['mean_difference']:+.4f} "
            f"[{d26['ci_low']:+.4f}, {d26['ci_high']:+.4f}] | "
            f"{dr['mean_difference']:+.4f} [{dr['ci_low']:+.4f}, {dr['ci_high']:+.4f}] |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/biological_crossfit"))
    parser.add_argument("--jump-profiles", type=Path, default=Path(
        "data/external/jump_compound_profiles_overlap.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "reports/biological_crossfit/topology_severity_controls.json"))
    parser.add_argument("--gine-seed", type=int, default=None,
                        help="Legacy single-seed option.")
    parser.add_argument("--gine-seeds", type=int, nargs="+", default=None,
                        help="GIN-E seeds averaged per compound before bootstrap.")
    parser.add_argument("--matched-pool-size", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    gine_seeds = (args.gine_seeds if args.gine_seeds is not None else
                  [args.gine_seed if args.gine_seed is not None else 0])
    for name in ("split", "report_root", "jump_profiles", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, PROJECT_ROOT / value)

    split = json.loads(args.split.read_text())
    rows = load_graph_rows()
    simple_folds = build_stat_folds(rows, split, SIMPLE_INDICES)
    stats26_folds = build_stat_folds(rows, split, range(26))
    gine_folds_by_seed = [
        load_gine_folds(split, args.report_root, seed) for seed in gine_seeds
    ]
    target = load_offtargets(path=DATASET.index_dir / "offtargets_full291.json")
    reactome = load_chembl37_human_reactome()
    jump, jump_metadata = load_jump_profiles(args.jump_profiles)
    sources = {"target": target, "reactome": reactome, "jump": jump}

    strict = {}
    matched_accum = {}
    for endpoint_index, (kind, source) in enumerate(sources.items()):
        profile_factory = lambda fold, kind=kind, source=source: endpoint_profiles(
            kind, set(fold["replicate_1"]) & set(fold["replicate_2"]), source)
        relevance_name = ("independent_jump_profile_cosine" if kind == "jump"
                          else f"within_fold_tfidf_{kind}_profile_cosine")
        gine_seed_endpoints = [
            evaluate_endpoint(folds, profile_factory, profile_cosine,
                              relevance_name, 0, 0, args.seed)
            for folds in gine_folds_by_seed
        ]
        gine_endpoint = average_endpoint_results(
            gine_seed_endpoints, args.n_bootstrap, args.seed)
        simple_endpoint = evaluate_endpoint(
            simple_folds, profile_factory, profile_cosine, relevance_name,
            args.n_bootstrap, 0, args.seed)
        stats26_endpoint = evaluate_endpoint(
            stats26_folds, profile_factory, profile_cosine, relevance_name,
            args.n_bootstrap, 0, args.seed)
        strict[kind] = {
            "gine": gine_endpoint,
            "simple_7d": simple_endpoint,
            "stats_26d": stats26_endpoint,
            "paired_gine_minus_simple_7d": paired_difference(
                gine_endpoint["per_compound_metrics"],
                simple_endpoint["per_compound_metrics"], args.n_bootstrap, args.seed),
            "paired_gine_minus_26d": paired_difference(
                gine_endpoint["per_compound_metrics"],
                stats26_endpoint["per_compound_metrics"], args.n_bootstrap, args.seed),
        }
        accumulated_by_seed = []
        for gine_folds in gine_folds_by_seed:
            accumulated = {"gine": [], "stats_26d": [], "random": []}
            for gine, full, simple in zip(gine_folds, stats26_folds, simple_folds):
                heldout = set(gine["replicate_1"]) & set(gine["replicate_2"])
                profiles = endpoint_profiles(kind, heldout, source)
                append_rows(accumulated, matched_rows(
                    gine, full, simple, profiles, args.matched_pool_size))
            accumulated_by_seed.append(accumulated)
        accumulated = {
            "gine": average_directional_rows([
                result["gine"] for result in accumulated_by_seed
            ]),
            "stats_26d": accumulated_by_seed[0]["stats_26d"],
            "random": accumulated_by_seed[0]["random"],
        }
        matched_accum[kind] = summarize_matched(
            accumulated, args.n_bootstrap, args.seed + endpoint_index)

    payload = {
        "status": "complete",
        "analysis": "topology_versus_simple_network_scale",
        "split_sha256": split["compound_partition_sha256"],
        "protocol": {
            "gine_seeds": gine_seeds,
            "gine_seed_aggregation": "mean per compound metric before bootstrap",
            "normalization_fit_scope": "training_compound_wells_only_per_fold",
            "normalization": "featurewise log1p then training-fold min-max; no test clipping",
            "simple_feature_indices": list(SIMPLE_INDICES),
            "simple_features": list(SIMPLE_NAMES),
            "ranking_boundary": "within_heldout_fold_only",
            "matched_pool_size": args.matched_pool_size,
            "matching": "opposite-replicate Euclidean distance in normalized simple 7D space",
            "candidate_pools_identical_across_rankers": True,
            "random_ranking": "exact expected nDCG for a uniform random permutation",
            "bootstrap_unit": "out_of_fold_compound",
            "n_bootstrap": args.n_bootstrap,
        },
        "jump_source": jump_metadata,
        "strict_crossfit_baselines": strict,
        "matched_retrieval": matched_accum,
    }
    _atomic_json(args.output, payload)
    args.output.with_suffix(".md").write_text(render_markdown(payload))
    print(args.output)


if __name__ == "__main__":
    main()
