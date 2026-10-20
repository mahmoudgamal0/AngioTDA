"""Evaluate fixed raw/image baselines on the cross-fit fold candidate pools.

These representations are not fitted on the assay compounds. For direct
comparability with trained graph cross-fit models, every neighbour ranking and
profile permutation is nevertheless restricted to one held-out fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from angiotda import PROJECT_ROOT

from angiotda.data.paths import DATASET  # noqa: E402
from angiotda.eval.metrics import load_moa, load_offtargets  # noqa: E402
from angiotda.eval.retrieval import (  # noqa: E402
    label_jaccard,
    profile_cosine,
    tfidf_profile_vectors,
)
from dataclasses import dataclass
from angiotda.eval.embed import (embed_raw_26d, embed_cell_dino, embed_dino_repeat_frozen, embed_dino_repeat_frozen_400)
from angiotda.eval.biological_retrieval import replicate_views  # noqa: E402
from angiotda.eval.biology import (  # noqa: E402
    _atomic_json,
    _metric_summary,
    evaluate_endpoint,
    load_chembl37_human_reactome,
    neighborhood_rows,
    profile_coverage_by_fold,
    replicate_identity_rows,
    render_markdown,
)


SUPPORTED = {
    "raw_26d",
    "dino_repeat_frozen_direct",
    "dino_repeat_frozen_direct_400",
    "cell_dino",
}


@dataclass(frozen=True)
class ModelSpec:
    run_id: str
    embedder: object
    checkpoint: Path | None = None
    family: str = ""
    seed: int = 0


def select_spec(run_id: str):
    embedders = {
        "raw_26d": embed_raw_26d,
        "cell_dino": embed_cell_dino,
        "dino_repeat_frozen_direct": embed_dino_repeat_frozen,
        "dino_repeat_frozen_direct_400": embed_dino_repeat_frozen_400,
    }
    if run_id not in embedders:
        raise ValueError(f"unknown fixed baseline: {run_id}")
    labels = {"raw_26d": "Graph stats (26D)", "cell_dino": "Cell-DINO",
              "dino_repeat_frozen_direct": "DINOv3 frozen",
              "dino_repeat_frozen_direct_400": "DINOv3 frozen (400px)"}
    return ModelSpec(run_id, embedders[run_id], family=labels[run_id])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=sorted(SUPPORTED), required=True)
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reports/biological_crossfit"))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--permutation-seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("split", "output_dir"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, PROJECT_ROOT / path)

    spec = select_spec(args.run_id)
    _, well_embeddings = (
        spec.embedder() if spec.checkpoint is None
        else spec.embedder(spec.checkpoint))
    first, second, replicate_audit = replicate_views(well_embeddings)
    split = json.loads(args.split.read_text())
    expected_universe = {
        compound for fold in split["folds"] for compound in fold["heldout_compounds"]
    }
    observed_universe = set(first) & set(second)
    if observed_universe != expected_universe:
        raise RuntimeError(
            f"fixed baseline compound universe differs from split: "
            f"missing={len(expected_universe - observed_universe)}, "
            f"extra={len(observed_universe - expected_universe)}")

    folds = []
    neighborhood = {}
    replicate_identity = {}
    for fold in split["folds"]:
        heldout = set(fold["heldout_compounds"])
        data = {
            "fold": fold["fold"],
            "replicate_1": {compound: first[compound] for compound in heldout},
            "replicate_2": {compound: second[compound] for compound in heldout},
        }
        folds.append(data)
        neighborhood.update(neighborhood_rows(
            data["replicate_1"], data["replicate_2"], data["fold"]))
        replicate_identity.update(replicate_identity_rows(
            data["replicate_1"], data["replicate_2"], data["fold"]))

    sailem57 = load_moa("sailem57")
    target_map = load_offtargets(path=DATASET.index_dir / "offtargets_full291.json")
    pathway_map = load_chembl37_human_reactome()
    specs = (
        ("sailem57", lambda _: sailem57, label_jaccard, "multilabel_jaccard"),
        ("chembl_offtargets", lambda fold: tfidf_profile_vectors(
            target_map, compounds=set(fold["replicate_1"])),
         profile_cosine, "within_fold_tfidf_target_profile_cosine"),
        ("chembl37_human_reactome", lambda fold: tfidf_profile_vectors(
            pathway_map, compounds=set(fold["replicate_1"])),
         profile_cosine, "within_fold_tfidf_human_reactome_profile_cosine"),
    )
    payload = {
        "status": "partial",
        "architecture": args.run_id,
        "run_id": args.run_id,
        "family": spec.family,
        "seed": spec.seed,
        "n_folds": len(folds),
        "claim_scope": "out_of_fold_complete_partition",
        "split": str(args.split),
        "split_sha256": split["compound_partition_sha256"],
        "analysis_config": {
            "analysis_version": 5,
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": args.n_permutations,
            "permutation_seed": args.permutation_seed,
        },
        "protocol": {
            "representation_fit_scope": "fixed_no_assay_compound_optimization",
            "ranking_boundary": "within_crossfit_fold_only",
            "fold_embeddings_compared_across_folds": False,
            "aggregation": "out_of_fold_per_compound_metrics",
        },
        "replicate_audit": replicate_audit,
        "excluded_endpoints": {
            "lopac291": "removed: catalogue fields are not validated MOA ground truth",
        },
        "profile_coverage_by_fold": {
            "chembl_targets": profile_coverage_by_fold(split, target_map),
            "chembl37_human_reactome": profile_coverage_by_fold(split, pathway_map),
        },
        "neighborhood_stability": _metric_summary(
            neighborhood, args.n_bootstrap, args.permutation_seed),
        "neighborhood_stability_per_compound": neighborhood,
        "replicate_identity": _metric_summary(
            replicate_identity, args.n_bootstrap, args.permutation_seed),
        "replicate_identity_per_compound": replicate_identity,
        "endpoints": {},
    }
    output = args.output_dir / f"{args.run_id}_crossfit_biology.json"
    markdown = output.with_suffix(".md")
    _atomic_json(output, payload)
    markdown.write_text(render_markdown(payload))
    for key, factory, relevance_fn, relevance_name in specs:
        print(f"Evaluating {args.run_id} {key}", flush=True)
        payload["endpoints"][key] = evaluate_endpoint(
            folds, factory, relevance_fn, relevance_name,
            args.n_bootstrap, args.n_permutations, args.permutation_seed)
        _atomic_json(output, payload)
        markdown.write_text(render_markdown(payload))
    payload["status"] = "complete"
    _atomic_json(output, payload)
    markdown.write_text(render_markdown(payload))
    print(output)


if __name__ == "__main__":
    main()
