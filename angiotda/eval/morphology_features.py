"""
Run the same per-MoA compound-label permutation analysis on the original
Sailem supplementary feature table.

This treats each compound as a single 101D feature vector from
`data/external/sailem2020_supp6_drug_features.xlsx`, L2-normalized for cosine-based
similarity, then applies the same `per_moa_enrichment` procedure used
for the learned embeddings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from angiotda import PROJECT_ROOT

from angiotda.data.paths import DATASET
from angiotda.eval.metrics import load_moa, per_moa_enrichment

EXPERIMENT_ROOT = PROJECT_ROOT / "reports" / "morphology_features"
FEATURE_XLSX = DATASET.sailem_features
SHEET = "Supplementary Table 6"
CONTROLS = {"CTRL", "DMSO", "Suramin"}


def normalize_name(x: str) -> str:
    return str(x).strip().strip(chr(39)).replace(chr(39) + chr(39), chr(39))


def load_feature_embeddings() -> tuple[dict[str, np.ndarray], dict]:
    df = pd.read_excel(FEATURE_XLSX, sheet_name=SHEET)
    df = df.copy()
    df["compound"] = df["Perturbation"].map(normalize_name)
    df = df[~df["compound"].isin({"DMSO", "Suramin"})].copy()

    feature_cols = [c for c in df.columns if c not in {"Perturbation", "compound"}]
    X = df[feature_cols].to_numpy(dtype=float)

    # Treat the published feature table as fixed compound-level descriptors.
    # L2 normalization puts it into the same cosine-similarity regime used
    # elsewhere in the repo for MoA enrichment.
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)

    compound_embs = {compound: X[i] for i, compound in enumerate(df["compound"].tolist())}
    metadata = {
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
    }
    return compound_embs, metadata


def render_markdown(payload: dict) -> str:
    lines = [
        "# Sailem Original Feature Permutation Test",
        "",
        f"- Source file: `{FEATURE_XLSX.name}`",
        f"- Source sheet: `{SHEET}`",
        f"- Compound rows analyzed: `{payload['metadata']['n_rows']}`",
        f"- Feature dimensions: `{payload['metadata']['n_features']}`",
        f"- Tested Sailem57 terms with `n >= 5`: `{payload['summary']['n_tested']}`",
        f"- Bonferroni-significant terms: `{payload['summary']['n_bonferroni']}`",
        "",
        "| Term | n | Δ | p | Bonferroni |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in payload["rows"]:
        bonf = row["p"] * payload["summary"]["n_tested"]
        lines.append(
            f"| {row['term']} | {row['n_compounds']} | {row['delta']:+.4f} | "
            f"{row['p']:.4g} | {bonf:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Significant Terms",
            "",
            ", ".join(payload["summary"]["bonf_terms"]) if payload["summary"]["bonf_terms"] else "none",
            "",
        ]
    )
    return "\n".join(lines)


def compute_analysis() -> dict:
    compound_embs, metadata = load_feature_embeddings()
    sailem57 = load_moa("sailem57")
    rows = per_moa_enrichment(compound_embs, sailem57, min_compounds=5)
    n_tested = len(rows)
    bonf_terms = [row["term"] for row in rows if row["p"] * n_tested < 0.05]

    return {
        "metadata": metadata,
        "summary": {
            "n_tested": n_tested,
            "n_bonferroni": len(bonf_terms),
            "bonf_terms": bonf_terms,
        },
        "rows": rows,
    }

def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = compute_analysis()
    json_path = EXPERIMENT_ROOT / "sailem_feature_wilcoxon.json"
    md_path = EXPERIMENT_ROOT / "sailem_feature_wilcoxon.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_markdown(payload))
    print(md_path.read_text())


if __name__ == "__main__":
    main()
