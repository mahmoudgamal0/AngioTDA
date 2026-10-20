"""Fetch compact ChEMBL target biology and derive compound pathway profiles.

The input target IDs already exist in ``offtargets_full291.json``.  This script
does not remap compound names or change those profiles; it adds release-pinned
target names, genes, Reactome pathways, and GO annotations from the official
ChEMBL REST API. Progress is written atomically after every batch so acquisition
can resume without repeating completed target requests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


from angiotda import PROJECT_ROOT
DEFAULT_API = "https://www.ebi.ac.uk/chembl/api/data"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def request_json(url: str, retries: int = 4, timeout: int = 90) -> dict:
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "ECCV-BIC2026-biology-evaluation/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last_error}")


def _named_xrefs(component: dict, source: str) -> list[dict[str, str | None]]:
    rows = {
        (xref.get("xref_id"), xref.get("xref_name"))
        for xref in component.get("target_component_xrefs", [])
        if xref.get("xref_src_db") == source and xref.get("xref_id")
    }
    return [
        {"id": identifier, "name": name}
        for identifier, name in sorted(rows)
    ]


def compact_target(target: dict) -> dict:
    """Retain only target biology used by the evaluation."""
    genes, accessions = set(), set()
    reactome, go_process, go_function = {}, {}, {}
    family_xrefs = {source: {} for source in ("InterPro", "PANTHER", "Pfam")}
    for component in target.get("target_components", []):
        if component.get("accession"):
            accessions.add(component["accession"])
        for synonym in component.get("target_component_synonyms", []):
            if (synonym.get("syn_type") == "GENE_SYMBOL" and
                    synonym.get("component_synonym")):
                genes.add(synonym["component_synonym"])
        for source, destination in (
            ("Reactome", reactome),
            ("GoProcess", go_process),
            ("GoFunction", go_function),
        ):
            for row in _named_xrefs(component, source):
                destination[row["id"]] = row["name"]
        for source in family_xrefs:
            for row in _named_xrefs(component, source):
                family_xrefs[source][row["id"]] = row["name"]
    return {
        "target_chembl_id": target.get("target_chembl_id"),
        "pref_name": target.get("pref_name"),
        "target_type": target.get("target_type"),
        "organism": target.get("organism"),
        "tax_id": target.get("tax_id"),
        "accessions": sorted(accessions),
        "gene_symbols": sorted(genes),
        "reactome": [
            {"id": identifier, "name": reactome[identifier]}
            for identifier in sorted(reactome)
        ],
        "go_process": [
            {"id": identifier, "name": go_process[identifier]}
            for identifier in sorted(go_process)
        ],
        "go_function": [
            {"id": identifier, "name": go_function[identifier]}
            for identifier in sorted(go_function)
        ],
        "family_xrefs": {
            source: [
                {"id": identifier, "name": values[identifier]}
                for identifier in sorted(values)
            ]
            for source, values in family_xrefs.items()
        },
    }


def derive_compound_profiles(target_profiles: dict[str, list[str]],
                             target_metadata: dict[str, dict]) -> dict:
    """Project target-level annotations back to compound-level profiles."""
    fields = {
        "reactome": "reactome_pathways",
        "go_process": "go_biological_processes",
        "go_function": "go_molecular_functions",
        "gene_symbols": "gene_symbols",
    }
    output = {destination: {} for destination in fields.values()}
    output["human_reactome_pathways"] = {}
    for compound, target_ids in sorted(target_profiles.items()):
        records = [target_metadata[target_id] for target_id in target_ids
                   if target_id in target_metadata]
        for source, destination in fields.items():
            values = set()
            for record in records:
                items = record.get(source, [])
                if source == "gene_symbols":
                    values.update(items)
                else:
                    values.update(item["id"] for item in items)
            output[destination][compound] = sorted(values)
        human_pathways = {
            item["id"]
            for record in records if record.get("tax_id") == 9606
            for item in record.get("reactome", [])
        }
        output["human_reactome_pathways"][compound] = sorted(human_pathways)
    return output


def acquisition_url(api_base: str, target_ids: list[str]) -> str:
    joined = ";".join(target_ids)
    path = f"{api_base.rstrip('/')}/target/set/{joined}.json"
    return path + "?" + urllib.parse.urlencode({
        "only": "target_chembl_id,pref_name,target_type,organism,tax_id,target_components"
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "data/index/offtargets_full291.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/external/chembl37_target_biology.json"))
    parser.add_argument("--profiles-output", type=Path, default=Path(
        "data/index/chembl37_compound_biology_profiles.json"))
    parser.add_argument("--api-base", default=DEFAULT_API)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    for name in ("input", "output", "profiles_output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, PROJECT_ROOT / value)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    source = json.loads(args.input.read_text())
    target_profiles = source["offtargets"]
    requested = sorted({target for values in target_profiles.values() for target in values})
    status = request_json(f"{args.api_base.rstrip('/')}/status.json",
                          retries=args.retries)
    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    metadata = existing.get("targets", {})
    missing = set(existing.get("missing_target_ids", []))
    completed = set(metadata) | missing
    remaining = [target for target in requested if target not in completed]
    retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat()

    for offset in range(0, len(remaining), args.batch_size):
        batch = remaining[offset:offset + args.batch_size]
        response = request_json(
            acquisition_url(args.api_base, batch), retries=args.retries)
        returned = {}
        for target in response.get("targets", []):
            row = compact_target(target)
            identifier = row.get("target_chembl_id")
            if identifier:
                returned[identifier] = row
        metadata.update(returned)
        missing.update(set(batch) - set(returned))
        payload = {
            "provenance": {
                "api_base": args.api_base,
                "chembl_db_version": status.get("chembl_db_version"),
                "chembl_release_date": status.get("chembl_release_date"),
                "api_status_snapshot": status,
                "retrieved_at_utc": retrieved_at,
                "source_profile": str(args.input.relative_to(PROJECT_ROOT)),
                "source_profile_note": source.get("meta", {}).get("note"),
                "requested_target_ids": len(requested),
            },
            "targets": dict(sorted(metadata.items())),
            "missing_target_ids": sorted(missing),
        }
        atomic_json(args.output, payload)
        print(
            f"targets {len(metadata)}/{len(requested)}; missing {len(missing)}",
            flush=True,
        )

    profiles = derive_compound_profiles(target_profiles, metadata)
    coverage = {
        name: {
            "n_compounds_nonempty": sum(bool(values) for values in mapping.values()),
            "n_unique_terms": len({term for values in mapping.values() for term in values}),
        }
        for name, mapping in profiles.items()
    }
    atomic_json(args.profiles_output, {
        "provenance": {
            "target_metadata": str(args.output.relative_to(PROJECT_ROOT)),
            "chembl_db_version": status.get("chembl_db_version"),
            "chembl_release_date": status.get("chembl_release_date"),
            "retrieved_at_utc": retrieved_at,
            "projection": "union of annotations across each compound's derived ChEMBL target profile",
        },
        "coverage": coverage,
        "profiles": profiles,
    })
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()

