"""Create a compact assay-overlap matrix from public JUMP Cell Painting profiles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np


from angiotda import PROJECT_ROOT
COMPOUND_URL = (
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/"
    "v0.11.0/metadata/compound.csv.gz"
)
PROFILE_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/"
    "source_all/workspace/profiles_assembled/COMPOUND/v1.0/"
    "profiles_var_mad_int_featselect_harmony.parquet"
)
PROFILE_ETAG = "c9371af57a36a51e021935c9ca78e506"


def connectivity_key(inchikey: str | None) -> str | None:
    if not inchikey or "-" not in inchikey:
        return None
    block = inchikey.split("-", 1)[0].upper()
    return block if len(block) == 14 else None


def download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    request = urllib.request.Request(
        url, headers={"User-Agent": "AngioTDA-external-validation/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def jcp_to_connectivity(metadata_gzip: bytes, wanted: set[str]) -> dict[str, str]:
    output = {}
    with gzip.GzipFile(fileobj=io.BytesIO(metadata_gzip)) as compressed:
        rows = csv.DictReader(io.TextIOWrapper(compressed, encoding="utf-8"))
        for row in rows:
            key = connectivity_key(row.get("Metadata_InChIKey"))
            if key in wanted:
                output[row["Metadata_JCP2022"]] = key
    return output


def aggregate_profiles(parquet_path: Path, jcp_mapping: dict[str, str]):
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_path, format="parquet")
    identifier = "Metadata_JCP2022"
    feature_names = [name for name in dataset.schema.names
                     if not name.startswith("Metadata_")]
    if identifier not in dataset.schema.names or not feature_names:
        raise RuntimeError("unexpected JUMP profile schema")
    scanner = dataset.scanner(
        columns=[identifier, *feature_names],
        filter=ds.field(identifier).isin(sorted(jcp_mapping)),
        batch_size=512,
    )
    sums: dict[str, np.ndarray] = {}
    counts = defaultdict(int)
    for batch in scanner.to_batches():
        ids = batch.column(identifier).to_pylist()
        features = np.column_stack([
            batch.column(name).to_numpy(zero_copy_only=False)
            for name in feature_names
        ]).astype(np.float64, copy=False)
        features = np.nan_to_num(features)
        for jcp_id, vector in zip(ids, features):
            key = jcp_mapping.get(jcp_id)
            if key is None:
                continue
            if key not in sums:
                sums[key] = np.zeros(len(feature_names), dtype=np.float64)
            sums[key] += vector
            counts[key] += 1
    profiles = {}
    for key, total in sums.items():
        vector = total / counts[key]
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            profiles[key] = (vector / norm).astype(np.float32)
    return profiles, dict(counts), feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identifiers", type=Path, default=Path(
        "data/external/pubchem_compound_identifiers.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/external/jump_compound_profiles_overlap.npz"))
    parser.add_argument("--profile-cache", type=Path,
                        help="Optional retained path for the 2.8 GB profile parquet.")
    args = parser.parse_args()
    for attribute in ("identifiers", "output", "profile_cache"):
        path = getattr(args, attribute)
        if path is not None and not path.is_absolute():
            setattr(args, attribute, PROJECT_ROOT / path)

    identifiers = json.loads(args.identifiers.read_text())
    name_to_key = {
        name: row["connectivity_key"]
        for name, row in identifiers["records"].items()
        if row.get("connectivity_key")
    }
    key_to_names = defaultdict(list)
    for name, key in name_to_key.items():
        key_to_names[key].append(name)

    request = urllib.request.Request(
        COMPOUND_URL, headers={"User-Agent": "AngioTDA-external-validation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        metadata_gzip = response.read()
    jcp_mapping = jcp_to_connectivity(metadata_gzip, set(key_to_names))

    temporary = None
    if args.profile_cache is None:
        fd, name = tempfile.mkstemp(prefix="jump_compound_", suffix=".parquet")
        os.close(fd)
        parquet_path = Path(name)
        temporary = parquet_path
    else:
        parquet_path = args.profile_cache
    try:
        if not parquet_path.exists() or parquet_path.stat().st_size == 0:
            print(f"Downloading {PROFILE_URL} to {parquet_path}", flush=True)
            profile_sha256 = download(PROFILE_URL, parquet_path)
        else:
            digest = hashlib.sha256()
            with parquet_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            profile_sha256 = digest.hexdigest()
        profiles, counts, feature_names = aggregate_profiles(parquet_path, jcp_mapping)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    compounds, vectors, connectivity = [], [], []
    for compound, key in sorted(name_to_key.items()):
        if key in profiles:
            compounds.append(compound)
            vectors.append(profiles[key])
            connectivity.append(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "JUMP Cell Painting Consortium compound profiles",
        "profile_url": PROFILE_URL,
        "profile_manifest_etag": PROFILE_ETAG,
        "profile_sha256": profile_sha256,
        "compound_metadata_url": COMPOUND_URL,
        "release": "jump-cellpainting/datasets v0.11.0; COMPOUND profiles v1.0",
        "matching": "PubChem-to-JUMP first InChIKey block",
        "n_input_compounds": len(name_to_key),
        "n_jcp_ids_matched": len(jcp_mapping),
        "n_assay_compounds_matched": len(compounds),
        "n_features": len(feature_names),
        "profile_rows_per_connectivity_key": counts,
    }
    np.savez_compressed(
        args.output,
        compounds=np.asarray(compounds, dtype=str),
        connectivity_keys=np.asarray(connectivity, dtype=str),
        profiles=np.stack(vectors).astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
