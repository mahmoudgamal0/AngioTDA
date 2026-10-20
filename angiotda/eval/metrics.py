"""
Evaluation metrics for morphological drug repurposing.

Primary: TF-IDF weighted target similarity correlation (with permutation test)
Secondary: Replicate consistency, per-MoA compound-label permutation enrichment
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from angiotda.data.paths import DATASET, resolve_path

from angiotda import PROJECT_ROOT


def _coerce_label_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    return [str(value)]


def load_label_map(path: str | Path,
                   key: str | None = None) -> dict[str, list[str]]:
    """Load a generic compound → labels mapping from JSON."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and key is not None:
        data = data[key]
    elif isinstance(data, dict):
        for candidate in (
            "drug_moa",
            "labels",
            "compound_labels",
            "compound_to_labels",
            "perturbation_labels",
            "biology_labels",
        ):
            if candidate in data and isinstance(data[candidate], dict):
                data = data[candidate]
                break

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object mapping compounds to labels in {path}")

    return {str(compound): _coerce_label_values(labels) for compound, labels in data.items()}


def load_offtargets(project_root: Path = PROJECT_ROOT,
                    path: str | Path | None = None) -> dict[str, list[str]]:
    """Load compound → ChEMBL target IDs."""
    if path is None:
        dataset_paths = DATASET
        path = dataset_paths.index_dir / "offtargets_full291.json"
    else:
        path = resolve_path(project_root, path)
    with open(path) as f:
        data = json.load(f)
    return data["offtargets"]


def load_moa(variant: str = "lopac291",
             project_root: Path = PROJECT_ROOT) -> dict[str, list[str]]:
    """Load compound → MoA terms.

    Args:
        variant: 'lopac291' (full) or 'sailem57' (enriched subset)
    """
    dataset_paths = DATASET
    if variant == "sailem57":
        path = dataset_paths.index_dir / "lopac_moa_sailem57.json"
    else:
        path = dataset_paths.index_dir / "lopac_moa.json"
    return load_label_map(path, key="drug_moa")


def summarize_enrichment(rows: list[dict]) -> dict[str, int]:
    n_tested = len(rows)
    return {
        "n_tested": n_tested,
        "n_bonferroni": sum(1 for r in rows if r["p"] * max(n_tested, 1) < 0.05),
        "n_nominal": sum(1 for r in rows if r["p"] < 0.05),
    }


def _build_tfidf_sim(compound_embs, offtargets):
    """Shared helper: build morphological and TF-IDF target similarity matrices."""
    common = [c for c in compound_embs if c in offtargets and len(offtargets[c]) > 0]
    if len(common) < 3:
        return None

    all_targets = set()
    for c in common:
        all_targets.update(offtargets[c])
    target_list = sorted(all_targets)
    t2i = {t: i for i, t in enumerate(target_list)}

    N = len(common)
    doc_freq = np.zeros(len(target_list))
    for c in common:
        for t in offtargets[c]:
            doc_freq[t2i[t]] += 1
    idf = np.log(N / np.maximum(doc_freq, 1))

    target_vecs = np.zeros((N, len(target_list)), dtype=np.float32)
    for i, c in enumerate(common):
        for t in offtargets[c]:
            target_vecs[i, t2i[t]] = idf[t2i[t]]

    tn = np.linalg.norm(target_vecs, axis=1, keepdims=True)
    target_vecs_n = target_vecs / np.maximum(tn, 1e-8)
    target_sim = target_vecs_n @ target_vecs_n.T

    emb_matrix = np.stack([compound_embs[c] for c in common])
    morph_sim = emb_matrix @ emb_matrix.T

    return {
        "common": common, "N": N,
        "target_sim": target_sim, "morph_sim": morph_sim,
    }


def tfidf_target_correlation(compound_embs: dict[str, np.ndarray],
                              offtargets: dict[str, list[str]],
                              n_perm: int = 1000) -> dict:
    """
    Primary metric: Spearman correlation between morphological similarity
    and TF-IDF weighted ChEMBL target similarity, with permutation test.

    Returns:
        dict with 'spearman_r', 'spearman_p', 'perm_p', 'n_compounds', 'n_pairs'
    """
    data = _build_tfidf_sim(compound_embs, offtargets)
    if data is None:
        return {"spearman_r": 0.0, "spearman_p": 1.0, "perm_p": 1.0,
                "n_compounds": 0, "n_pairs": 0}

    N = data["N"]
    iu = np.triu_indices(N, k=1)
    target_upper = data["target_sim"][iu]
    morph_upper = data["morph_sim"][iu]

    r_obs, p_obs = spearmanr(morph_upper, target_upper)

    # Permutation test
    rng = np.random.RandomState(42)
    n_greater = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        morph_perm = data["morph_sim"][np.ix_(perm, perm)]
        r_p, _ = spearmanr(morph_perm[iu], target_upper)
        if r_p >= r_obs:
            n_greater += 1
    perm_p = (n_greater + 1) / (n_perm + 1)

    return {
        "spearman_r": float(r_obs),
        "spearman_p": float(p_obs),
        "perm_p": float(perm_p),
        "n_compounds": len(data["common"]),
        "n_pairs": len(target_upper),
    }


def replicate_consistency(well_embs: dict[tuple, np.ndarray]) -> dict:
    """Exactly compare all same-compound and different-compound well pairs.

    Args:
        well_embs: {(compound, well_idx): L2-normalized embedding}

    Returns:
        dict with 'delta', 'auroc', 'mannwhitney_p', etc.
    """
    keys = list(well_embs)
    matrix = np.stack([well_embs[key] for key in keys])
    similarity = matrix @ matrix.T
    iu = np.triu_indices(len(keys), k=1)
    compound = np.asarray([key[0] for key in keys], dtype=object)
    same_mask = compound[iu[0]] == compound[iu[1]]
    pair_sims = similarity[iu]
    same_arr = pair_sims[same_mask]
    diff_arr = pair_sims[~same_mask]

    # U/(n_same*n_different) is exactly the pairwise AUROC (with half credit
    # for ties) without constructing a multi-million-element label vector.
    stat, _ = mannwhitneyu(same_arr, diff_arr, alternative="greater")
    auroc = float(stat / (len(same_arr) * len(diff_arr)))
    n_compounds = sum(n >= 2 for n in Counter(compound).values())

    return {
        "same_mean": float(same_arr.mean()),
        "diff_mean": float(diff_arr.mean()),
        "delta": float(same_arr.mean() - diff_arr.mean()),
        # Pairwise observations overlap, so no inferential rank-sum p-value is
        # reported. AUROC and delta are descriptive exact summaries.
        "mannwhitney_p": None,
        "auroc": auroc,
        "n_pairs": len(same_arr),
        "n_compounds": n_compounds,
        "n_within_well_pairs": len(same_arr),
        "n_different_well_pairs": len(diff_arr),
    }


def _enrichment_deltas(sim: np.ndarray, label_ids: np.ndarray,
                       n_labels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return within means, between means, and deltas for every label.

    The implementation works from undirected edges in the similarity matrix.
    Besides being faster than materialising one array per class, this makes the
    exchangeable unit explicit: labels live on compounds, not on pairwise
    similarities.
    """
    n = len(label_ids)
    counts = np.bincount(label_ids, minlength=n_labels).astype(float)
    iu = np.triu_indices(n, k=1)
    left, right = label_ids[iu[0]], label_ids[iu[1]]
    same = left == right
    within_sums = np.bincount(
        left[same], weights=sim[iu][same], minlength=n_labels).astype(float)

    # Each cross-class edge contributes once to each endpoint class.  Summing
    # the off-diagonal row totals by label therefore gives 2*within + between.
    row_totals = sim.sum(axis=1) - np.diag(sim)
    group_totals = np.bincount(
        label_ids, weights=row_totals, minlength=n_labels).astype(float)
    between_sums = group_totals - 2.0 * within_sums

    within_den = counts * (counts - 1.0) / 2.0
    between_den = counts * (n - counts)
    within = np.divide(within_sums, within_den,
                       out=np.full(n_labels, np.nan), where=within_den > 0)
    between = np.divide(between_sums, between_den,
                        out=np.full(n_labels, np.nan), where=between_den > 0)
    return within, between, within - between


def per_label_enrichment(compound_embs: dict[str, np.ndarray],
                         label_map: dict[str, list[str]],
                         min_compounds: int = 5,
                         n_permutations: int = 9999,
                         seed: int = 42) -> list[dict]:
    """Test per-label cosine enrichment by permuting compound labels.

    The previous rank-sum test treated all pairwise similarities as independent,
    even though every compound occurs in many pairs.  Here the primary labels
    are shuffled across compounds while preserving every class size, and the
    one-sided p-value is the fraction of shuffled effect sizes at least as large
    as the observed within-minus-between mean.  Thus compounds, rather than
    overlapping pairs, are the exchangeable units.
    """
    common = [c for c in compound_embs if c in label_map and len(label_map[c]) > 0]
    if len(common) < 10:
        return []

    primary = {c: label_map[c][0] for c in common}
    term_counts = Counter(primary.values())
    powered = sorted(t for t, n in term_counts.items() if n >= min_compounds)

    emb_matrix = np.stack([compound_embs[c] for c in common])
    sim = emb_matrix @ emb_matrix.T
    N = len(common)

    if not powered:
        return []

    # Unpowered labels remain in the permutation as distinct labels.  Collapsing
    # them into one artificial OTHER class would change the label allocation.
    terms = sorted(term_counts)
    term_to_id = {term: i for i, term in enumerate(terms)}
    label_ids = np.asarray([term_to_id[primary[c]] for c in common], dtype=int)
    observed_within, observed_between, observed_delta = _enrichment_deltas(
        sim, label_ids, len(terms))

    rng = np.random.RandomState(seed)
    exceed = np.zeros(len(terms), dtype=np.int64)
    for _ in range(n_permutations):
        _, _, null_delta = _enrichment_deltas(
            sim, rng.permutation(label_ids), len(terms))
        exceed += np.asarray(null_delta >= observed_delta, dtype=np.int64)

    results = []
    for term in powered:
        idx = term_to_id[term]
        results.append({
            "term": term,
            "n_compounds": term_counts[term],
            "within_mean": float(observed_within[idx]),
            "between_mean": float(observed_between[idx]),
            "delta": float(observed_delta[idx]),
            "p": float((exceed[idx] + 1) / (n_permutations + 1)),
            "n_permutations": n_permutations,
            "test": "compound_label_permutation",
        })

    return results


def per_multilabel_enrichment(compound_embs: dict[str, np.ndarray],
                              label_map: dict[str, list[str]],
                              min_compounds: int = 5,
                              n_permutations: int = 9999,
                              seed: int = 42) -> list[dict]:
    """Test all annotations instead of discarding secondary MoA labels.

    Each term is a separate binary compound-level hypothesis. Positives are
    every compound carrying the term; negatives are annotated compounds that
    do not. Permuting this membership vector preserves term size and keeps the
    compound as the exchangeable unit. Terms may overlap, so tests are
    dependent; Bonferroni correction remains valid and conservative.
    """
    common = [c for c in compound_embs if c in label_map and label_map[c]]
    if len(common) < 10:
        return []

    annotations = {c: set(label_map[c]) for c in common}
    term_counts = Counter(term for labels in annotations.values() for term in labels)
    powered = sorted(term for term, count in term_counts.items()
                     if count >= min_compounds)
    if not powered:
        return []

    embeddings = np.stack([compound_embs[c] for c in common])
    similarity = embeddings @ embeddings.T
    diagonal = np.diag(similarity)
    row_totals = similarity.sum(axis=1) - diagonal
    n = len(common)

    def delta_for(indices: np.ndarray) -> tuple[float, float, float]:
        k = len(indices)
        block = similarity[np.ix_(indices, indices)]
        within_sum = (float(block.sum()) - float(diagonal[indices].sum())) / 2.0
        within = within_sum / (k * (k - 1) / 2.0)
        between_sum = float(row_totals[indices].sum()) - 2.0 * within_sum
        between = between_sum / (k * (n - k))
        return within, between, within - between

    results = []
    for term_index, term in enumerate(powered):
        positive = np.asarray(
            [i for i, compound in enumerate(common) if term in annotations[compound]],
            dtype=int,
        )
        within, between, observed = delta_for(positive)
        rng = np.random.RandomState(seed + term_index)
        exceed = 0
        for _ in range(n_permutations):
            permuted = rng.choice(n, size=len(positive), replace=False)
            _, _, null_delta = delta_for(permuted)
            exceed += null_delta >= observed
        results.append({
            "term": term,
            "n_compounds": len(positive),
            "within_mean": float(within),
            "between_mean": float(between),
            "delta": float(observed),
            "p": float((exceed + 1) / (n_permutations + 1)),
            "n_permutations": n_permutations,
            "test": "compound_membership_permutation_multilabel",
            "label_assignment": "all_annotations",
        })
    return results


def per_moa_enrichment(compound_embs: dict[str, np.ndarray],
                       drug_moa: dict[str, list[str]],
                       min_compounds: int = 5) -> list[dict]:
    """Backward-compatible wrapper for MoA enrichment."""
    return per_label_enrichment(compound_embs, drug_moa, min_compounds=min_compounds)


def evaluate_all(compound_embs: dict[str, np.ndarray],
                 well_embs: dict[tuple, np.ndarray] = None,
                 offtargets: dict[str, list[str]] | None = None,
                 label_sets: list[dict] | None = None,
                 project_root: Path = PROJECT_ROOT) -> dict:
    """Run all evaluation metrics."""
    results = {}

    if offtargets is None:
        try:
            offtargets = load_offtargets(project_root=project_root)
        except FileNotFoundError:
            offtargets = None
    if offtargets is not None:
        results.update(tfidf_target_correlation(compound_embs, offtargets))

    # Recorded on every result so two rows computed over different well sets
    # cannot sit side by side in a table looking comparable.
    results["n_entities"] = len(compound_embs)
    if well_embs is not None:
        rc = replicate_consistency(well_embs)
        results["rep_delta"] = rc["delta"]
        results["rep_auroc"] = rc["auroc"]
        results["n_wells"] = len(well_embs)
        results["n_replicate_pairs"] = rc["n_pairs"]
        results["n_replicate_compounds"] = rc["n_compounds"]
        results["n_within_well_pairs"] = rc["n_within_well_pairs"]

    if label_sets is None:
        label_sets = [
            {
                "metric_key": "wilcox_sailem57",
                "label_map": load_moa("sailem57", project_root=project_root),
                "min_compounds": 5,
            },
            {
                "metric_key": "wilcox_lopac291",
                "label_map": load_moa("lopac291", project_root=project_root),
                "min_compounds": 5,
            },
        ]

    tests_used = []
    permutations_used = []
    for spec in label_sets:
        label_mode = spec.get("label_mode", "primary")
        enrichment_fn = (per_multilabel_enrichment
                         if label_mode == "multilabel"
                         else per_label_enrichment)
        rows = enrichment_fn(
            compound_embs,
            spec["label_map"],
            min_compounds=spec.get("min_compounds", 5),
            n_permutations=spec.get("n_permutations", 9999),
            seed=spec.get("seed", 42),
        )
        summary = summarize_enrichment(rows)
        metric_key = spec["metric_key"]
        results[metric_key] = summary["n_bonferroni"]
        results[f"{metric_key}_tested"] = summary["n_tested"]
        results[f"{metric_key}_terms"] = sorted(
            row["term"] for row in rows
            if row["p"] * max(summary["n_tested"], 1) < 0.05
        )
        tests_used.append(rows[0]["test"] if rows else
                          ("compound_membership_permutation_multilabel"
                           if label_mode == "multilabel"
                           else "compound_label_permutation"))
        permutations_used.append(spec.get("n_permutations", 9999))

    results["enrichment_test"] = (tests_used[0] if len(set(tests_used)) == 1
                                  else tests_used)
    results["enrichment_permutations"] = (permutations_used[0]
                                          if len(set(permutations_used)) == 1
                                          else permutations_used)

    return results


def bootstrap_enrichment(compound_embs: dict[str, np.ndarray],
                         drug_moa: dict[str, list[str]],
                         n_resamples: int = 100,
                         baseline_min_compounds: int = 5,
                         bootstrap_min_compounds: int = 3,
                         n_permutations: int = 999,
                         baseline_permutations: int = 9999,
                         seed: int = 42) -> list[dict]:
    """Per-MoA enrichment with a bootstrap stability estimate.

    Each resample draws 80% of the annotated compounds without replacement and
    re-runs the enrichment. ``boot_sig`` is the fraction of resamples in which a
    term stays Bonferroni-significant under a fresh compound-label permutation
    test, correcting against the *baseline* number
    of tested terms so the threshold does not drift as terms drop in and out.
    ``ci_low``/``ci_high`` are the 2.5/97.5 percentiles of the resampled delta.

    Returns baseline rows augmented with ci_low, ci_high, boot_sig, boot_present,
    sorted by p-value.
    """
    from collections import defaultdict as _dd

    common = [c for c in compound_embs if c in drug_moa and len(drug_moa[c]) > 0]
    baseline = per_label_enrichment(
        compound_embs, drug_moa, min_compounds=baseline_min_compounds,
        n_permutations=baseline_permutations, seed=seed)
    baseline_by_term = {r["term"]: r for r in baseline}
    n_tested = len(baseline)
    rng = np.random.RandomState(seed)
    size = int(round(0.8 * len(common)))

    deltas, sig_hits, counts = _dd(list), _dd(int), _dd(int)
    for resample_idx in range(n_resamples):
        keep = sorted(rng.choice(common, size=size, replace=False).tolist())
        rows = per_label_enrichment(
            {c: compound_embs[c] for c in keep},
            {c: drug_moa[c] for c in keep if c in drug_moa},
            min_compounds=bootstrap_min_compounds,
            n_permutations=n_permutations,
            seed=seed + resample_idx + 1)
        seen = {r["term"] for r in rows}
        for r in rows:
            deltas[r["term"]].append(r["delta"])
            counts[r["term"]] += 1
            if n_tested > 0 and r["p"] * n_tested < 0.05:
                sig_hits[r["term"]] += 1
        for term in baseline_by_term:
            if term not in seen:
                deltas[term].append(float("nan"))

    out = []
    for term, row in baseline_by_term.items():
        arr = np.asarray([x for x in deltas[term] if not np.isnan(x)], dtype=float)
        out.append({
            **row,
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "boot_sig": sig_hits[term] / n_resamples,
            "boot_present": counts[term] / n_resamples,
        })
    return sorted(out, key=lambda x: x["p"])
