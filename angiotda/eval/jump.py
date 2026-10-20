"""Evaluate cross-fit embeddings against independent JUMP Cell Painting profiles.

The JUMP profiles come from a different cell line, assay, consortium, and
feature pipeline. Rankings remain restricted to each original held-out fold;
only compounds mapped by standardized connectivity InChIKey are eligible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from angiotda import PROJECT_ROOT

from angiotda.eval.retrieval import profile_cosine  # noqa: E402
from angiotda.eval.biological_retrieval import replicate_views  # noqa: E402
from angiotda.eval.biology import (  # noqa: E402
    _atomic_json,
    _metric_summary,
    evaluate_endpoint,
    load_fold_artifact,
    neighborhood_rows,
    replicate_identity_rows,
)
from angiotda.eval.baselines import (  # noqa: E402
    SUPPORTED,
    select_spec,
)


LEARNED = {
    **{name: (name, name + "_seed{seed}_fold{fold}", "heldout_embeddings_final.npz")
       for name in ("gatv2", "gps", "deepsets", "gcn", "pna", "transformer")},
    "gine": ("gine", "gine_seed{seed}_fold{fold}", "heldout_embeddings_final.npz"),
    "pna": ("pna", "pna_seed{seed}_fold{fold}", "heldout_embeddings_final.npz"),
    "gatv2": ("gatv2", "gatv2_seed{seed}_fold{fold}", "heldout_embeddings_final.npz"),
    "transformer": (
        "transformer", "transformer_seed{seed}_fold{fold}", "heldout_embeddings_final.npz"),
    "dino_repeat": (
        "dino_repeat", "dino_repeat_seed{seed}_fold{fold}", "heldout_embeddings_final.npz"),
    "dino_repeat_400": (
        "dino_repeat", "dino_repeat_400_seed{seed}_fold{fold}",
        "heldout_embeddings_final.npz"),
    "dino_photometric_400": (
        "dino_repeat", "dino_photometric_400_seed{seed}_fold{fold}",
        "heldout_embeddings_final.npz"),
    "four_view_repeat": (
        "four_view_repeat", "four_view_repeat_seed{seed}_fold{fold}",
        "heldout_embeddings_final.npz"),
    "four_view_repeat_400": (
        "four_view_repeat", "four_view_repeat_400_seed{seed}_fold{fold}",
        "heldout_embeddings_final.npz"),
    "late_fusion_repeat": (
        "late_fusion_repeat", "late_fusion_repeat_seed{seed}_fold{fold}",
        "heldout_embeddings_final.npz"),
    "late_fusion_repeat_400": (
        "late_fusion_repeat", "late_fusion_repeat_400_seed{seed}_fold{fold}",
        "heldout_embeddings_final.npz"),
}


def load_jump_profiles(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as payload:
        compounds = payload["compounds"].astype(str).tolist()
        profiles = np.asarray(payload["profiles"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata_json"].item()))
    if len(compounds) != len(profiles) or len(compounds) != len(set(compounds)):
        raise RuntimeError("invalid JUMP overlap artifact")
    norms = np.linalg.norm(profiles, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("JUMP overlap contains empty profiles")
    profiles = profiles / norms[:, None]
    return dict(zip(compounds, profiles)), metadata


def learned_folds(model: str, split: dict, report_root: Path,
                  model_seed: int, run_name_template: str | None = None) -> list[dict]:
    architecture, template, artifact_name = LEARNED[model]
    template = run_name_template or template
    output, seen = [], set()
    for fold in split["folds"]:
        index = fold["fold"]
        expected = set(fold["heldout_compounds"])
        path = report_root / template.format(architecture=architecture, seed=model_seed, fold=index) / artifact_name
        data = load_fold_artifact(path, architecture, model_seed, index, expected)
        if seen & data["compounds"]:
            raise RuntimeError("compound occurs in multiple fold artifacts")
        seen.update(data["compounds"])
        output.append(data)
    return output


def fixed_folds(model: str, split: dict) -> tuple[list[dict], dict]:
    spec = select_spec(model)
    _, embeddings = spec.embedder() if spec.checkpoint is None else spec.embedder(spec.checkpoint)
    first, second, audit = replicate_views(embeddings)
    expected = {compound for fold in split["folds"]
                for compound in fold["heldout_compounds"]}
    if set(first) & set(second) != expected:
        raise RuntimeError(f"{model} does not cover the split compound universe")
    folds = []
    for fold in split["folds"]:
        heldout = set(fold["heldout_compounds"])
        folds.append({
            "fold": fold["fold"],
            "replicate_1": {name: first[name] for name in heldout},
            "replicate_2": {name: second[name] for name in heldout},
        })
    return folds, audit


def auxiliary_metrics(folds: list[dict], n_bootstrap: int, seed: int):
    neighbourhood, identity = {}, {}
    for fold in folds:
        neighbourhood.update(neighborhood_rows(
            fold["replicate_1"], fold["replicate_2"], fold["fold"]))
        identity.update(replicate_identity_rows(
            fold["replicate_1"], fold["replicate_2"], fold["fold"]))
    return {
        "neighborhood_stability": _metric_summary(neighbourhood, n_bootstrap, seed),
        "neighborhood_stability_per_compound": neighbourhood,
        "replicate_identity": _metric_summary(identity, n_bootstrap, seed),
        "replicate_identity_per_compound": identity,
    }


def render_markdown(payload: dict) -> str:
    endpoint = payload["endpoints"]["jump_cellpainting"]
    metrics = endpoint["summary"]["metrics"]
    pvalues = endpoint["blocked_profile_permutation_p"]
    lines = [
        f"# {payload['model']} external JUMP validation", "",
        "JUMP profiles are independent U2OS Cell Painting measurements. Rankings "
        "are restricted to the original held-out-compound folds.", "",
        f"- Matched assay compounds: `{payload['coverage']['matched_assay_compounds']}`",
        f"- Evaluable query compounds: `{endpoint['summary']['n_query_compounds']}`",
        f"- JUMP features: `{payload['coverage']['jump_features']}`", "",
        "| Metric | Mean | 95% CI | Blocked permutation p |",
        "|---|---:|---:|---:|",
    ]
    for name in ("average_precision", "ndcg_at_5", "ndcg_at_10", "recall_at_10"):
        if name not in metrics:
            continue
        row = metrics[name]
        ptext = f"{pvalues[name]:.4f}" if name in pvalues else "n/a"
        lines.append(
            f"| {name} | {row['mean']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{ptext} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gine", "cell_dino"],
                        choices=sorted(set(LEARNED) | SUPPORTED))
    parser.add_argument("--profiles", type=Path, default=Path(
        "data/external/jump_compound_profiles_overlap.npz"))
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--report-root", type=Path, default=Path(
        "reports/biological_crossfit"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reports/external_validation"))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-seed", type=int, default=0,
        help="Training seed for learned fold artifacts; fixed baselines ignore it.")
    parser.add_argument(
        "--output-name",
        help="Output stem override; only valid when evaluating one model.")
    parser.add_argument("--run-name-template", help="Custom artifact directory template; requires one learned model.")
    args = parser.parse_args()
    if args.run_name_template and (len(args.models) != 1 or args.models[0] not in LEARNED):
        parser.error("--run-name-template requires exactly one learned model")
    if args.output_name and len(args.models) != 1:
        parser.error("--output-name requires exactly one model")
    for attribute in ("profiles", "split", "report_root", "output_dir"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, PROJECT_ROOT / path)

    profiles, metadata = load_jump_profiles(args.profiles)
    split = json.loads(args.split.read_text())
    for model in args.models:
        print(f"Evaluating external JUMP endpoint for {model}", flush=True)
        if model in LEARNED:
            folds = learned_folds(model, split, args.report_root, args.model_seed, args.run_name_template)
            audit = {"representation": "fold-specific held-out artifact"}
            model_seed = args.model_seed
        else:
            folds, audit = fixed_folds(model, split)
            model_seed = 0
        endpoint = evaluate_endpoint(
            folds, lambda fold: {
                name: profiles[name]
                for name in set(fold["replicate_1"]) & set(fold["replicate_2"])
                if name in profiles
            },
            profile_cosine, "independent_jump_profile_cosine",
            args.n_bootstrap, args.n_permutations, args.seed,
        )
        payload = {
            "status": "complete",
            "model": model,
            "architecture": LEARNED[model][0] if model in LEARNED else model,
            "seed": model_seed,
            "n_folds": len(split["folds"]),
            "fold_indices": [fold["fold"] for fold in split["folds"]],
            "claim_scope": "out_of_fold_complete_partition",
            "split_sha256": split["compound_partition_sha256"],
            "protocol": {
                "ranking_boundary": "within_crossfit_fold_only",
                "fold_embeddings_compared_across_folds": False,
                "external_endpoint_selected_post_hoc": True,
            },
            "representation_audit": audit,
            "external_source": metadata,
            "coverage": {
                "matched_assay_compounds": len(profiles),
                "jump_features": metadata["n_features"],
            },
            "endpoints": {"jump_cellpainting": endpoint},
            **auxiliary_metrics(folds, args.n_bootstrap, args.seed),
        }
        output_stem = args.output_name or (
            f"{model}_jump_external" if model_seed == 0
            else f"{model}_seed{model_seed}_jump_external")
        output = args.output_dir / f"{output_stem}.json"
        _atomic_json(output, payload)
        output.with_suffix(".md").write_text(render_markdown(payload))
        print(output, flush=True)


if __name__ == "__main__":
    main()
