"""Periodic, checkpoint-free evaluation and durable training reports.

Training scripts pass the current encoder's well embeddings to
``TrainingEvaluationReporter``.  The reporter uses the same compound-level
permutation test and exact replicate summaries as the final paper evaluator,
then atomically rewrites JSON and Markdown histories after every evaluation.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import (
    load_moa,
    per_label_enrichment,
    per_multilabel_enrichment,
    replicate_consistency,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def pool_well_embeddings(
    per_well: list[tuple[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], np.ndarray]]:
    """Pool ordered ``(compound, embedding)`` rows without dropping wells."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    well_embeddings: dict[tuple[str, int], np.ndarray] = {}
    for compound, embedding in per_well:
        embedding = _unit(embedding)
        well_embeddings[(compound, len(grouped[compound]))] = embedding
        grouped[compound].append(embedding)
    compound_embeddings = {
        compound: _unit(np.mean(embeddings, axis=0))
        for compound, embeddings in grouped.items()
    }
    return compound_embeddings, well_embeddings


def evaluate_representation(
    per_well: list[tuple[str, np.ndarray]],
    *,
    label_map: dict[str, list[str]],
    n_permutations: int = 999,
    permutation_seed: int = 42,
    include_multilabel: bool = True,
) -> dict:
    """Evaluate one representation using the final pipeline's statistical units."""
    compound_embeddings, well_embeddings = pool_well_embeddings(per_well)
    replicate = replicate_consistency(well_embeddings)
    rows = per_label_enrichment(
        compound_embeddings,
        label_map,
        min_compounds=5,
        n_permutations=n_permutations,
        seed=permutation_seed,
    )
    threshold = 0.05 / max(len(rows), 1)
    significant = sorted(row["term"] for row in rows if row["p"] < threshold)
    result = {
        "n_compounds": len(compound_embeddings),
        "n_wells": len(well_embeddings),
        "rep_delta": replicate["delta"],
        "rep_auroc": replicate["auroc"],
        "n_replicate_pairs": replicate["n_pairs"],
        "n_tested_terms": len(rows),
        "n_bonferroni_terms": len(significant),
        "bonferroni_terms": significant,
        "enrichment_rows": rows,
        "enrichment_test": "compound_label_permutation",
        "enrichment_permutations": n_permutations,
    }
    if include_multilabel:
        multi_rows = per_multilabel_enrichment(
            compound_embeddings,
            label_map,
            min_compounds=5,
            n_permutations=n_permutations,
            seed=permutation_seed,
        )
        multi_threshold = 0.05 / max(len(multi_rows), 1)
        multi_significant = sorted(
            row["term"] for row in multi_rows if row["p"] < multi_threshold)
        result["multilabel"] = {
            "n_tested_terms": len(multi_rows),
            "n_bonferroni_terms": len(multi_significant),
            "bonferroni_terms": multi_significant,
            "enrichment_rows": multi_rows,
            "enrichment_test": "compound_membership_permutation_multilabel",
            "enrichment_permutations": n_permutations,
        }
    return result


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class TrainingEvaluationReporter:
    """Append epoch evaluations and keep machine- and human-readable reports."""

    def __init__(self, report_dir: Path, *, run_name: str,
                 label_map: dict[str, list[str]] | None = None):
        self.report_dir = Path(report_dir)
        self.run_name = run_name
        self.label_map = label_map if label_map is not None else load_moa("sailem57")
        self.history_path = self.report_dir / "evaluation_history.json"
        self.markdown_path = self.report_dir / "evaluation_history.md"
        if self.history_path.exists():
            payload = json.loads(self.history_path.read_text())
            self.history = list(payload.get("evaluations", []))
        else:
            self.history = []

    def record(self, *, epoch: int, training_loss: float,
               representations: dict[str, list[tuple[str, np.ndarray]]],
               n_permutations: int = 999, final: bool = False) -> dict:
        entry = {
            "epoch": int(epoch),
            "training_loss": float(training_loss),
            "final": bool(final),
            "representations": {
                name: evaluate_representation(
                    rows,
                    label_map=self.label_map,
                    n_permutations=n_permutations,
                )
                for name, rows in representations.items()
            },
        }
        # A resumed epoch replaces its earlier partial result instead of
        # silently creating duplicate x-axis entries.
        self.history = [row for row in self.history if row["epoch"] != entry["epoch"]]
        self.history.append(entry)
        self.history.sort(key=lambda row: row["epoch"])
        self._write()
        return entry

    def _write(self) -> None:
        payload = {
            "run_name": self.run_name,
            "evaluation_unit": "compound_label_permutation",
            "evaluations": self.history,
        }
        _atomic_write(self.history_path, json.dumps(payload, indent=2) + "\n")

        lines = [
            f"# Periodic evaluation: {self.run_name}",
            "",
            "Each row is evaluated on all wells. Replicate summaries enumerate exact well "
            "pairs; MoA inference permutes labels across compounds.",
            "",
            "| Epoch | Representation | Train loss | Replicate Delta | AUROC | Primary terms | Multi-label terms | Primary identities | Multi-label identities | Permutations |",
            "|---:|---|---:|---:|---:|---:|---:|---|---|---:|",
        ]
        for entry in self.history:
            for name, result in entry["representations"].items():
                identities = ", ".join(result["bonferroni_terms"]) or "-"
                multi = result.get("multilabel", {})
                multi_identities = ", ".join(multi.get("bonferroni_terms", [])) or "-"
                lines.append(
                    f"| {entry['epoch']} | {name} | {entry['training_loss']:.6f} | "
                    f"{result['rep_delta']:+.6f} | {result['rep_auroc']:.6f} | "
                    f"{result['n_bonferroni_terms']} | "
                    f"{multi.get('n_bonferroni_terms', 0)} | {identities} | "
                    f"{multi_identities} | "
                    f"{result['enrichment_permutations']} |"
                )
        _atomic_write(self.markdown_path, "\n".join(lines) + "\n")
