"""Resolve assay compound names to release-stamped PubChem identifiers.

The output is resumable and records unsuccessful queries explicitly.  The
connectivity block of each InChIKey is used later to tolerate salt and
stereochemistry differences when matching public perturbation datasets.
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
API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def connectivity_key(inchikey: str | None) -> str | None:
    if not inchikey or "-" not in inchikey:
        return None
    block = inchikey.split("-", 1)[0].upper()
    return block if len(block) == 14 else None


def property_url(name: str) -> str:
    encoded = urllib.parse.quote(name, safe="")
    properties = "Title,IUPACName,CanonicalSMILES,IsomericSMILES,InChIKey"
    return f"{API_BASE}/compound/name/{encoded}/property/{properties}/JSON"


def resolve_name(name: str, retries: int, timeout: int) -> dict | None:
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                property_url(name),
                headers={"User-Agent": "AngioTDA-external-validation/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            rows = payload.get("PropertyTable", {}).get("Properties", [])
            if not rows:
                return None
            row = rows[0]
            inchikey = row.get("InChIKey")
            return {
                "cid": row.get("CID"),
                "title": row.get("Title"),
                "iupac_name": row.get("IUPACName"),
                "canonical_smiles": row.get("ConnectivitySMILES", row.get("CanonicalSMILES")),
                "isomeric_smiles": row.get("SMILES", row.get("IsomericSMILES")),
                "inchikey": inchikey,
                "connectivity_key": connectivity_key(inchikey),
                "query_name": name,
            }
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            last_error = error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"failed to resolve {name!r}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/index/lopac_compounds.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/external/pubchem_compound_identifiers.json"))
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.21,
                        help="Delay between PubChem requests; default respects 5 requests/s.")
    parser.add_argument("--limit", type=int,
                        help="Optional smoke-test limit on unresolved names.")
    args = parser.parse_args()
    for attribute in ("input", "output"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, PROJECT_ROOT / path)

    source = json.loads(args.input.read_text())
    names = sorted(source["label_index"])
    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    records = existing.get("records", {})
    missing = set(existing.get("missing", []))
    attempted = set(records) | missing
    remaining = [name for name in names if name not in attempted]
    if args.limit is not None:
        remaining = remaining[:args.limit]

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    for index, name in enumerate(remaining, 1):
        record = resolve_name(name, args.retries, args.timeout)
        if record is None:
            missing.add(name)
        else:
            records[name] = record
        atomic_json(args.output, {
            "provenance": {
                "api": API_BASE,
                "retrieved_at_utc": started,
                "source": str(args.input.relative_to(PROJECT_ROOT)),
                "matching_key": "first 14 characters of PubChem InChIKey",
            },
            "records": dict(sorted(records.items())),
            "missing": sorted(missing),
            "summary": {
                "n_source_names": len(names),
                "n_resolved": len(records),
                "n_missing": len(missing),
                "n_unattempted": len(set(names) - set(records) - missing),
            },
        })
        if index % 25 == 0 or index == len(remaining):
            print(f"resolved={len(records)} missing={len(missing)} "
                  f"remaining={len(names) - len(records) - len(missing)}", flush=True)
        if index < len(remaining):
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
