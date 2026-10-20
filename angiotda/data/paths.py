"""One canonical dataset root under the external ANGIOTDA_WORKSPACE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from angiotda import PROJECT_ROOT


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    index_dir: Path
    sample_index: Path
    raw_dir: Path
    graphs_dir: Path
    skeleton_dir: Path
    splits_dir: Path
    cv_folds: Path
    external_dir: Path
    sailem_features: Path


def _build(root: Path) -> DatasetPaths:
    return DatasetPaths(
        root=root,
        index_dir=root / "index",
        sample_index=root / "index" / "sample_index.json",
        raw_dir=root / "raw",
        graphs_dir=root / "graphs",
        skeleton_dir=root / "skeleton_images",
        splits_dir=root / "splits",
        cv_folds=root / "cv_folds.json",
        # Third-party source data, as published — distinct from the derived
        # annotations in index/. Sailem & Al Haj Zen 2020, supplementary 6.
        external_dir=root / "external",
        sailem_features=root / "external" / "sailem2020_supp6_drug_features.xlsx",
    )


DATASET = _build(PROJECT_ROOT / "data")


def resolve_path(project_root: Path | str, path_like: str | Path | None) -> Path | None:
    """Absolute paths pass through; relative ones resolve against the project."""
    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else Path(project_root) / path


def dataset_paths() -> DatasetPaths:
    """Kept as a function so call sites read the same as before the change."""
    return DATASET
