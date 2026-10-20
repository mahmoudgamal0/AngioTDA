"""Evaluate biological retrieval from compound-held-out cross-fit artifacts.

Every fold has its own encoder and latent coordinate system. Similarities and
rankings are therefore computed strictly within a fold. Only per-compound
directional metric values are pooled across folds for the final bootstrap.
Complete annotation profiles are permuted independently inside each fold.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from angiotda import PROJECT_ROOT

from angiotda.data.paths import DATASET  # noqa: E402
from angiotda.eval.metrics import load_moa, load_offtargets  # noqa: E402
from angiotda.eval.retrieval import (  # noqa: E402
    compound_metric_means,
    cross_replicate_retrieval,
    label_jaccard,
    profile_cosine,
    summarize_retrieval,
    tfidf_profile_vectors,
)


REPLICATE_RE = re.compile(r"^(P\d+)R([12])_(R\d+C\d+)$")
K_VALUES = (5, 10)


def load_chembl37_human_reactome() -> dict[str, list[str]]:
    path = DATASET.index_dir / "chembl37_compound_biology_profiles.json"
    payload = json.loads(path.read_text())
    if payload.get("provenance", {}).get("chembl_db_version") != "ChEMBL_37":
        raise RuntimeError(f"unexpected or missing ChEMBL release in {path}")
    return payload["profiles"]["human_reactome_pathways"]


def profile_coverage_by_fold(split: dict, profiles: dict[str, list[str]]) -> list[dict]:
    """Report annotated compounds and distinct profile terms in each test fold."""
    coverage = []
    for fold in split["folds"]:
        compounds = fold["heldout_compounds"]
        terms = set().union(*(set(profiles.get(compound, [])) for compound in compounds))
        coverage.append({
            "fold": fold["fold"],
            "heldout_compounds": len(compounds),
            "annotated_compounds": sum(bool(profiles.get(compound, []))
                                       for compound in compounds),
            "unique_profile_terms": len(terms),
        })
    return coverage


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def reconstruct_replicate_views(compounds, image_names, embeddings):
    """Build one R1 and R2 vector per drug from stored physical-well names."""
    if not (len(compounds) == len(image_names) == len(embeddings)):
        raise ValueError("artifact arrays have different row counts")
    physical = defaultdict(dict)
    for compound, image_name, embedding in zip(compounds, image_names, embeddings):
        match = REPLICATE_RE.fullmatch(str(image_name))
        if match is None:
            raise ValueError(f"cannot identify physical replicate from {image_name!r}")
        physical_key = f"{match.group(1)}_{match.group(3)}"
        replica = int(match.group(2))
        key = (str(compound), physical_key)
        if replica in physical[key]:
            raise ValueError(f"duplicate R{replica} row for {key}")
        physical[key][replica] = unit(embedding)

    by_compound = defaultdict(lambda: {1: [], 2: []})
    matched_physical = 0
    for (compound, _), views in physical.items():
        if set(views) != {1, 2}:
            continue
        matched_physical += 1
        by_compound[compound][1].append(views[1])
        by_compound[compound][2].append(views[2])
    first, second = {}, {}
    for compound, views in by_compound.items():
        if views[1] and views[2]:
            first[compound] = unit(np.mean(views[1], axis=0))
            second[compound] = unit(np.mean(views[2], axis=0))
    return first, second, {
        "n_artifact_rows": len(compounds),
        "n_physical_pairs": matched_physical,
        "n_compounds_with_matched_replicates": len(first),
        "assignment": "stored image_names; same plate/row/column R1 matched to R2",
    }


def load_fold_artifact(path: Path, architecture: str, seed: int, fold: int,
                       expected_compounds: set[str] | None = None) -> dict:
    with np.load(path, allow_pickle=False) as artifact:
        required = {"compounds", "image_names", "embeddings", "metadata_json"}
        missing = required - set(artifact.files)
        if missing:
            raise KeyError(f"{path}: missing arrays {sorted(missing)}")
        compounds = artifact["compounds"].astype(str).tolist()
        image_names = artifact["image_names"].astype(str).tolist()
        embeddings = np.asarray(artifact["embeddings"], dtype=np.float64)
        metadata = json.loads(str(artifact["metadata_json"].item()))
    identity = (metadata.get("architecture"), metadata.get("seed"), metadata.get("fold"))
    if identity != (architecture, seed, fold):
        raise RuntimeError(
            f"{path}: identity {identity} != {(architecture, seed, fold)}")
    observed = set(compounds)
    if expected_compounds is not None and observed != expected_compounds:
        raise RuntimeError(
            f"{path}: held-out compounds differ from split "
            f"(missing={len(expected_compounds - observed)}, "
            f"extra={len(observed - expected_compounds)})")
    first, second, audit = reconstruct_replicate_views(
        compounds, image_names, embeddings)
    return {
        "fold": fold, "path": str(path), "metadata": metadata,
        "replicate_1": first, "replicate_2": second, "audit": audit,
        "compounds": observed,
    }


def _metric_summary(per_compound: dict[str, dict[str, float]],
                    n_bootstrap: int, seed: int) -> dict:
    if not per_compound:
        return {"n_query_compounds": 0, "metrics": {}}
    names = sorted(next(iter(per_compound.values())))
    compounds = sorted(per_compound)
    values = np.asarray([[per_compound[c][name] for name in names]
                         for c in compounds])
    observed = values.mean(axis=0)
    rng = np.random.RandomState(seed)
    bootstrap = np.empty((n_bootstrap, len(names)), dtype=float)
    for index in range(n_bootstrap):
        sample = rng.randint(0, len(compounds), len(compounds))
        bootstrap[index] = values[sample].mean(axis=0)
    return {
        "n_query_compounds": len(compounds),
        "bootstrap_unit": "out_of_fold_compound",
        "n_bootstrap": n_bootstrap,
        "metrics": {
            name: {
                "mean": float(observed[i]),
                "ci_low": (float(np.percentile(bootstrap[:, i], 2.5))
                           if n_bootstrap else float(observed[i])),
                "ci_high": (float(np.percentile(bootstrap[:, i], 97.5))
                            if n_bootstrap else float(observed[i])),
            }
            for i, name in enumerate(names)
        },
    }


def neighborhood_rows(first: dict, second: dict, fold: int) -> dict:
    compounds = sorted(set(first) & set(second))
    if len(compounds) < 3:
        return {}
    a = np.stack([first[c] for c in compounds])
    b = np.stack([second[c] for c in compounds])
    forward, reverse = a @ b.T, b @ a.T
    rows = {}
    for index, compound in enumerate(compounds):
        keep = np.arange(len(compounds)) != index
        left, right = forward[index, keep], reverse[index, keep]
        left_order = np.argsort(-left, kind="stable")
        right_order = np.argsort(-right, kind="stable")
        left_rank = np.empty(len(left_order)); left_rank[left_order] = np.arange(len(left_order))
        right_rank = np.empty(len(right_order)); right_rank[right_order] = np.arange(len(right_order))
        row = {"rank_correlation": float(np.corrcoef(left_rank, right_rank)[0, 1])}
        for k in K_VALUES:
            cutoff = min(k, len(left_order))
            row[f"overlap_at_{k}"] = float(
                len(set(left_order[:cutoff]) & set(right_order[:cutoff]))
                / max(cutoff, 1))
        rows[f"fold{fold}:{compound}"] = row
    return rows


def replicate_identity_rows(first: dict, second: dict, fold: int) -> dict:
    """Measure whether each replicate retrieves its matching compound replicate.

    Rankings stay inside one held-out fold. Metrics average the R1->R2 and
    R2->R1 directions so every compound remains the independent unit.
    """
    compounds = sorted(set(first) & set(second))
    if len(compounds) < 2:
        return {}
    a = np.stack([first[c] for c in compounds])
    b = np.stack([second[c] for c in compounds])
    similarity = a @ b.T
    directional_ranks = []
    directional_aurocs = []
    for matrix in (similarity, similarity.T):
        order = np.argsort(-matrix, axis=1, kind="stable")
        ranks = np.empty(len(compounds), dtype=np.int64)
        aurocs = np.empty(len(compounds), dtype=np.float64)
        for index in range(len(compounds)):
            ranks[index] = int(np.flatnonzero(order[index] == index)[0]) + 1
            positive = matrix[index, index]
            negatives = np.delete(matrix[index], index)
            aurocs[index] = float(np.mean(
                (positive > negatives) + 0.5 * (positive == negatives)))
        directional_ranks.append(ranks)
        directional_aurocs.append(aurocs)
    rows = {}
    for index, compound in enumerate(compounds):
        ranks = np.asarray([directional_ranks[0][index],
                            directional_ranks[1][index]], dtype=float)
        replicate_cosine = float(similarity[index, index])
        rows[f"fold{fold}:{compound}"] = {
            "replicate_consistency": replicate_cosine,
            "replicate_auroc": float(np.mean([
                directional_aurocs[0][index], directional_aurocs[1][index]])),
            "paired_cosine": replicate_cosine,
            "reciprocal_rank": float(np.mean(1.0 / ranks)),
            "hit_at_1": float(np.mean(ranks <= 1)),
            "hit_at_5": float(np.mean(ranks <= 5)),
            "hit_at_10": float(np.mean(ranks <= 10)),
        }
    return rows


def _relevance_matrix(profiles: dict, compounds: list[str], relevance_fn):
    if relevance_fn is profile_cosine:
        vectors = np.stack([profiles[c] for c in compounds])
        matrix = np.clip(vectors @ vectors.T, 0.0, 1.0)
    else:
        matrix = np.zeros((len(compounds), len(compounds)), dtype=np.float64)
        for i, left in enumerate(compounds):
            for j in range(i + 1, len(compounds)):
                value = relevance_fn(profiles[left], profiles[compounds[j]])
                matrix[i, j] = matrix[j, i] = value
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _fold_orders(first: dict, second: dict, compounds: list[str]):
    a = np.stack([first[c] for c in compounds])
    b = np.stack([second[c] for c in compounds])
    output = []
    for similarity in (a @ b.T, b @ a.T):
        np.fill_diagonal(similarity, -np.inf)
        output.append(np.argsort(-similarity, axis=1, kind="stable")[:, :-1])
    return output


def _per_compound_metric_arrays(relevance: np.ndarray, orders):
    directional = []
    for order in orders:
        ranked = np.take_along_axis(relevance, order, axis=1)
        binary = ranked > 0
        positives = binary.sum(axis=1)
        eligible = positives > 0
        rank = np.arange(1, ranked.shape[1] + 1)
        precision = np.cumsum(binary, axis=1) / rank
        metrics = {
            "average_precision": np.divide(
                (precision * binary).sum(axis=1), positives,
                out=np.full(len(ranked), np.nan), where=eligible),
        }
        ideal = np.sort(relevance, axis=1)[:, ::-1][:, :ranked.shape[1]]
        for k in K_VALUES:
            cutoff = min(k, ranked.shape[1])
            discount = np.log2(np.arange(2, cutoff + 2))
            dcg = (ranked[:, :cutoff] / discount).sum(axis=1)
            idcg = (ideal[:, :cutoff] / discount).sum(axis=1)
            metrics[f"ndcg_at_{k}"] = np.divide(
                dcg, idcg, out=np.full(len(ranked), np.nan), where=idcg > 0)
            metrics[f"recall_at_{k}"] = np.divide(
                binary[:, :cutoff].sum(axis=1), positives,
                out=np.full(len(ranked), np.nan), where=eligible)
        directional.append(metrics)
    output = {}
    for name in directional[0]:
        stacked = np.stack([directional[0][name], directional[1][name]])
        finite = np.isfinite(stacked)
        totals = np.where(finite, stacked, 0.0).sum(axis=0)
        counts = finite.sum(axis=0)
        output[name] = np.divide(
            totals, counts, out=np.full(stacked.shape[1], np.nan), where=counts > 0)
    return output


def blocked_profile_permutations(blocks: list[dict], observed: dict,
                                 n_permutations: int, seed: int) -> dict:
    """Independently permute complete profiles inside each fold/block."""
    if not observed.get("metrics") or not blocks:
        return {}
    metric_names = sorted(observed["metrics"])
    threshold = {name: observed["metrics"][name]["mean"] for name in metric_names}
    exceed = {name: 0 for name in metric_names}
    rng = np.random.RandomState(seed)
    for _ in range(n_permutations):
        accum = defaultdict(list)
        for block in blocks:
            permutation = rng.permutation(len(block["compounds"]))
            relevance = block["relevance"][np.ix_(permutation, permutation)]
            values = _per_compound_metric_arrays(relevance, block["orders"])
            for name, array in values.items():
                accum[name].append(array)
        for name in metric_names:
            null_mean = float(np.nanmean(np.concatenate(accum[name])))
            exceed[name] += null_mean >= threshold[name]
    return {name: float((exceed[name] + 1) / (n_permutations + 1))
            for name in metric_names}


def evaluate_endpoint(folds: list[dict], profile_factory, relevance_fn,
                      relevance_name: str, n_bootstrap: int,
                      n_permutations: int, seed: int) -> dict:
    all_per_compound = {}
    fold_summaries = []
    blocks = []
    for fold_data in folds:
        first, second = fold_data["replicate_1"], fold_data["replicate_2"]
        profiles = profile_factory(fold_data)
        rows = cross_replicate_retrieval(
            first, second, profiles, K_VALUES, relevance_fn)
        fold_metrics = compound_metric_means(rows)
        for compound, metrics in fold_metrics.items():
            key = f"fold{fold_data['fold']}:{compound}"
            if key in all_per_compound:
                raise RuntimeError(f"duplicate out-of-fold query {key}")
            all_per_compound[key] = metrics
        fold_summaries.append({
            "fold": fold_data["fold"],
            "n_query_compounds": len(fold_metrics),
            "metrics": summarize_retrieval(rows, n_bootstrap=0)["metrics"],
        })
        compounds = sorted(set(first) & set(second) & set(profiles))
        if len(compounds) >= 3:
            blocks.append({
                "fold": fold_data["fold"], "compounds": compounds,
                "relevance": _relevance_matrix(profiles, compounds, relevance_fn),
                "orders": _fold_orders(first, second, compounds),
            })
    summary = _metric_summary(all_per_compound, n_bootstrap, seed)
    return {
        "relevance": relevance_name,
        "ranking_boundary": "within_fold_only",
        "summary": summary,
        "per_compound_metrics": all_per_compound,
        "per_fold": fold_summaries,
        "blocked_profile_permutation_p": blocked_profile_permutations(
            blocks, summary, n_permutations, seed),
        "n_blocked_profile_permutations": n_permutations if blocks else 0,
        "permutation_blocks": [block["fold"] for block in blocks],
    }


def _atomic_json(path: Path, payload) -> None:
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


def render_markdown(payload: dict) -> str:
    lines = [
        f"# {payload['architecture'].upper()} seed {payload['seed']} cross-fit biology", "",
        "All rankings are strictly within held-out folds. Fold embeddings are never "
        "compared or pooled. Only out-of-fold per-compound metrics are aggregated.", "",
        f"- Claim scope: `{payload.get('claim_scope', 'out_of_fold_complete_partition')}`",
        f"- Folds: `{payload['n_folds']}`", f"- Split: `{payload['split']}`",
        f"- Status: `{payload['status']}`", "",
        "| Endpoint | Queries | MAP | nDCG@10 | MAP blocked p | nDCG@10 blocked p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("sailem57", "Sailem57"),
                       ("chembl_offtargets", "ChEMBL targets"),
                       ("chembl37_human_reactome", "ChEMBL37 human Reactome")):
        if key not in payload.get("endpoints", {}):
            continue
        row = payload["endpoints"][key]
        metrics = row["summary"]["metrics"]
        p = row["blocked_profile_permutation_p"]
        def value(name):
            return metrics.get(name, {}).get("mean")
        def fmt(number):
            return "—" if number is None else f"{number:.4f}"
        lines.append(
            f"| {label} | {row['summary']['n_query_compounds']} | "
            f"{fmt(value('average_precision'))} | {fmt(value('ndcg_at_10'))} | "
            f"{fmt(p.get('average_precision'))} | {fmt(p.get('ndcg_at_10'))} |")
    if "neighborhood_stability" in payload:
        lines.extend(["", "## Cross-replicate neighbourhood stability", ""])
        for name, metric in payload["neighborhood_stability"]["metrics"].items():
            lines.append(
                f"- {name}: {metric['mean']:.4f} "
                f"[{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]")
    if "replicate_identity" in payload:
        lines.extend(["", "## Same-compound replicate retrieval", ""])
        for name, metric in payload["replicate_identity"]["metrics"].items():
            lines.append(
                f"- {name}: {metric['mean']:.4f} "
                f"[{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]")
    if payload.get("profile_coverage_by_fold"):
        lines.extend(["", "## Held-out profile coverage", "",
                      "| Endpoint | Fold | Held-out compounds | Annotated compounds | Unique terms |",
                      "|---|---:|---:|---:|---:|"])
        for endpoint, rows in payload["profile_coverage_by_fold"].items():
            for row in rows:
                lines.append(
                    f"| {endpoint} | {row['fold']} | {row['heldout_compounds']} | "
                    f"{row['annotated_compounds']} | {row['unique_profile_terms']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("gine", "pna", "gatv2", "transformer", "gps", "deepsets", "gcn",
                                    "dino_repeat", "four_view_repeat",
                                    "late_fusion_repeat"),
        required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold-indices", type=int, nargs="*",
        help="Optional held-out fold subset for development-only comparisons.")
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--report-root", type=Path, default=Path(
        "reports/biological_crossfit"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reports/biological_crossfit"))
    parser.add_argument(
        "--run-name-template",
        default="{architecture}_seed{seed}_fold{fold}",
        help="Report directory template; supports architecture, seed, and fold.")
    parser.add_argument(
        "--artifact-name", default="heldout_embeddings_final.npz",
        help="Per-fold embedding artifact (used for fused/image/graph tower audits).")
    parser.add_argument(
        "--output-name",
        help="Output stem override, required to distinguish aligned variants.")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    for name in ("split", "report_root", "output_dir"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, PROJECT_ROOT / path)
    selected_folds = (list(range(args.folds)) if args.fold_indices is None
                      else sorted(set(args.fold_indices)))
    if not selected_folds or any(fold < 0 or fold >= args.folds for fold in selected_folds):
        raise ValueError("fold-indices must select at least one valid fold")
    output_stem = args.output_name or f"{args.architecture}_seed{args.seed}_biology"
    output = args.output_dir / f"{output_stem}.json"
    markdown = output.with_suffix(".md")
    config = {
        "analysis_version": 5,
        "fold_indices": selected_folds,
        "run_name_template": args.run_name_template,
        "artifact_name": args.artifact_name,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "permutation_seed": args.permutation_seed,
    }
    resumable = None
    if output.exists() and not args.no_resume:
        existing = json.loads(output.read_text())
        if existing.get("status") == "complete" and existing.get("analysis_config") == config:
            print(f"SKIP complete {output}")
            return
        if (existing.get("status") == "partial" and
                existing.get("analysis_config") == config and
                existing.get("architecture") == args.architecture and
                existing.get("seed") == args.seed):
            resumable = existing

    split = json.loads(args.split.read_text())
    if len(split["folds"]) != args.folds:
        raise RuntimeError("requested fold count does not match split")
    fold_data = []
    seen = set()
    digests = set()
    for fold in selected_folds:
        expected = set(split["folds"][fold]["heldout_compounds"])
        run_name = args.run_name_template.format(
            architecture=args.architecture, seed=args.seed, fold=fold)
        path = args.report_root / run_name / args.artifact_name
        data = load_fold_artifact(path, args.architecture, args.seed, fold, expected)
        overlap = seen & data["compounds"]
        if overlap:
            raise RuntimeError(f"compounds occur in multiple held-out folds: {sorted(overlap)[:3]}")
        seen.update(data["compounds"])
        digests.add(data["metadata"].get("split_sha256"))
        fold_data.append(data)
    if len(digests) != 1 or next(iter(digests)) != split["compound_partition_sha256"]:
        raise RuntimeError("fold artifacts and split do not share one partition digest")

    neighborhood = {}
    replicate_identity = {}
    for data in fold_data:
        neighborhood.update(neighborhood_rows(
            data["replicate_1"], data["replicate_2"], data["fold"]))
        replicate_identity.update(replicate_identity_rows(
            data["replicate_1"], data["replicate_2"], data["fold"]))
    split_digest = next(iter(digests))
    retained_endpoints = (
        resumable.get("endpoints", {})
        if resumable is not None and resumable.get("split_sha256") == split_digest
        else {})
    payload = {
        "status": "partial", "architecture": args.architecture, "seed": args.seed,
        "n_folds": len(selected_folds),
        "fold_indices": selected_folds,
        "claim_scope": (
            "out_of_fold_complete_partition"
            if len(selected_folds) == args.folds else
            "architecture_confirmation_folds"
            if selected_folds == list(range(1, args.folds)) else
            "development_fold_subset_only"),
        "split": str(args.split),
        "split_sha256": split_digest,
        "analysis_config": config,
        "aggregation_boundary": (
            "rank within fold; aggregate only out-of-fold per-compound directional metrics"),
        "fold_embeddings_compared_across_folds": False,
        "fold_audits": [{"fold": d["fold"], "artifact": d["path"], **d["audit"]}
                        for d in fold_data],
        "neighborhood_stability": _metric_summary(
            neighborhood, args.n_bootstrap, args.permutation_seed),
        "neighborhood_stability_per_compound": neighborhood,
        "replicate_identity": _metric_summary(
            replicate_identity, args.n_bootstrap, args.permutation_seed),
        "replicate_identity_per_compound": replicate_identity,
        "endpoints": retained_endpoints,
    }
    _atomic_json(output, payload); markdown.write_text(render_markdown(payload))
    sailem57 = load_moa("sailem57")
    target_map = load_offtargets(path=DATASET.index_dir / "offtargets_full291.json")
    pathway_map = load_chembl37_human_reactome()
    payload["excluded_endpoints"] = {
        "lopac291": "removed: catalogue fields are not validated MOA ground truth",
    }
    payload["profile_coverage_by_fold"] = {
        "chembl_targets": profile_coverage_by_fold(split, target_map),
        "chembl37_human_reactome": profile_coverage_by_fold(split, pathway_map),
    }
    # Never carry a retired endpoint forward from an older resumable report.
    payload["endpoints"].pop("lopac291", None)
    _atomic_json(output, payload); markdown.write_text(render_markdown(payload))
    endpoint_specs = (
        ("sailem57", lambda _: sailem57, label_jaccard, "multilabel_jaccard"),
        ("chembl_offtargets", lambda fold: tfidf_profile_vectors(
            target_map, compounds=set(fold["replicate_1"]) & set(fold["replicate_2"])),
         profile_cosine, "within_fold_tfidf_target_profile_cosine"),
        ("chembl37_human_reactome", lambda fold: tfidf_profile_vectors(
            pathway_map, compounds=set(fold["replicate_1"]) & set(fold["replicate_2"])),
         profile_cosine, "within_fold_tfidf_human_reactome_profile_cosine"),
    )
    for key, factory, relevance, name in endpoint_specs:
        if key in payload["endpoints"] and not args.no_resume:
            continue
        print(f"Evaluating {key}", flush=True)
        payload["endpoints"][key] = evaluate_endpoint(
            fold_data, factory, relevance, name,
            args.n_bootstrap, args.n_permutations, args.permutation_seed)
        _atomic_json(output, payload); markdown.write_text(render_markdown(payload))
    payload["status"] = "complete"
    _atomic_json(output, payload); markdown.write_text(render_markdown(payload))
    print(output)


if __name__ == "__main__":
    main()
