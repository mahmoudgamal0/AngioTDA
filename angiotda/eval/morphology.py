"""Evaluate Sailem et al.'s published 101D compound morphology profiles.

The source table contains one already-aggregated profile per compound, so
replicate consistency and cross-replicate neighbourhood stability are not
defined.  Biological retrieval remains fold blocked and directly comparable
to the target, Reactome, and JUMP endpoints used for the learned models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from angiotda import PROJECT_ROOT

from angiotda.data.paths import DATASET  # noqa: E402
from angiotda.eval.metrics import load_offtargets  # noqa: E402
from angiotda.eval.retrieval import profile_cosine, tfidf_profile_vectors  # noqa: E402
from angiotda.eval.morphology_features import (  # noqa: E402
    load_feature_embeddings,
)
from angiotda.eval.biology import (  # noqa: E402
    _atomic_json,
    evaluate_endpoint,
    load_chembl37_human_reactome,
    render_markdown,
)
from angiotda.eval.jump import load_jump_profiles  # noqa: E402


def folds_from_profiles(split: dict, profiles: dict) -> list[dict]:
    folds = []
    for row in split["folds"]:
        selected = {
            compound: profiles[compound]
            for compound in row["heldout_compounds"] if compound in profiles
        }
        folds.append({
            "fold": row["fold"],
            # The descriptor table has one compound-level vector.  Duplicating
            # the dictionary only satisfies the bidirectional retrieval API;
            # no replicate metric is computed or reported.
            "replicate_1": selected,
            "replicate_2": selected,
        })
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--jump-profiles", type=Path, default=Path(
        "data/external/jump_compound_profiles_overlap.npz"))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_path = args.split if args.split.is_absolute() else PROJECT_ROOT / args.split
    jump_path = (args.jump_profiles if args.jump_profiles.is_absolute()
                 else PROJECT_ROOT / args.jump_profiles)
    split = json.loads(split_path.read_text())
    profiles, source = load_feature_embeddings()
    folds = folds_from_profiles(split, profiles)
    target_map = load_offtargets(path=DATASET.index_dir / "offtargets_full291.json")
    pathway_map = load_chembl37_human_reactome()

    internal = {
        "status": "complete",
        "architecture": "published_morphology_101d",
        "seed": 0,
        "n_folds": len(folds),
        "claim_scope": "fixed_compound_profiles_fold_blocked_retrieval",
        "split": str(split_path.relative_to(PROJECT_ROOT)),
        "split_sha256": split["compound_partition_sha256"],
        "protocol": {
            "representation_fit_scope": "published_fixed_compound_descriptors",
            "ranking_boundary": "within_crossfit_fold_only",
            "replicate_metrics_defined": False,
        },
        "source": source,
        "endpoints": {},
    }
    specs = (
        ("chembl_offtargets", target_map, "within_fold_tfidf_target_profile_cosine"),
        ("chembl37_human_reactome", pathway_map,
         "within_fold_tfidf_human_reactome_profile_cosine"),
    )
    for key, annotations, relevance_name in specs:
        internal["endpoints"][key] = evaluate_endpoint(
            folds,
            lambda fold, annotations=annotations: tfidf_profile_vectors(
                annotations, compounds=set(fold["replicate_1"])),
            profile_cosine, relevance_name,
            args.n_bootstrap, args.n_permutations, args.seed)
    internal_path = (PROJECT_ROOT / "reports" / "biological_crossfit" /
                     "published_morphology_101d_crossfit_biology.json")
    _atomic_json(internal_path, internal)
    internal_path.with_suffix(".md").write_text(render_markdown(internal))

    jump_profiles, jump_source = load_jump_profiles(jump_path)
    external_endpoint = evaluate_endpoint(
        folds,
        lambda fold: {
            name: jump_profiles[name]
            for name in fold["replicate_1"] if name in jump_profiles
        },
        profile_cosine, "independent_jump_profile_cosine",
        args.n_bootstrap, args.n_permutations, args.seed)
    external = {
        "status": "complete",
        "model": "published_morphology_101d",
        "n_folds": len(folds),
        "split_sha256": split["compound_partition_sha256"],
        "protocol": {
            "ranking_boundary": "within_crossfit_fold_only",
            "replicate_metrics_defined": False,
        },
        "source": source,
        "external_source": jump_source,
        "endpoints": {"jump_cellpainting": external_endpoint},
    }
    external_path = (PROJECT_ROOT / "reports" / "external_validation" /
                     "published_morphology_101d_jump_external.json")
    _atomic_json(external_path, external)
    metrics = external_endpoint["summary"]["metrics"]
    pvalues = external_endpoint["blocked_profile_permutation_p"]
    external_path.with_suffix(".md").write_text(
        "# Published morphology descriptors: external JUMP validation\n\n"
        "The source provides one compound-level profile, so replicate metrics are "
        "undefined. Rankings remain restricted to held-out folds.\n\n"
        "| Metric | Mean | 95% CI | Blocked permutation p |\n"
        "|---|---:|---:|---:|\n" + "\n".join(
            f"| {name} | {row['mean']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{pvalues[name]:.4f} |"
            for name, row in metrics.items()) + "\n")
    print(internal_path)
    print(external_path)


if __name__ == "__main__":
    main()
