"""Small, lazy command dispatcher for training, preprocessing and evaluation."""
from __future__ import annotations
import argparse
import runpy
import sys

COMMANDS = {
    "prepare-graphs": "angiotda.data.prepare_graphs",
    "make-splits": "angiotda.training.make_splits",
    "train": "angiotda.training.run",
    "train-fold": "angiotda.training.train_fold",
    "train-multimodal": "angiotda.training.train_multimodal_fold",
    "evaluate-fold": "angiotda.training.evaluate_fold",
    "evaluate-biology": "angiotda.eval.biology",
    "evaluate-jump": "angiotda.eval.jump",
    "evaluate-baseline": "angiotda.eval.baselines",
    "evaluate-morphology": "angiotda.eval.morphology",
    "evaluate-random": "angiotda.eval.random_baseline",
    "evaluate-topology": "angiotda.eval.topology",
    "compare": "angiotda.eval.paired",
    "compare-models": "angiotda.eval.statistics",
    "fetch-identifiers": "angiotda.data.fetch_pubchem_identifiers",
    "fetch-targets": "angiotda.data.fetch_chembl_target_biology",
    "prepare-jump": "angiotda.data.prepare_jump_profiles",
}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__, usage="angiotda COMMAND [options]")
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    previous = sys.argv
    try:
        sys.argv = [COMMANDS[args.command], *argv[1:]]
        runpy.run_module(COMMANDS[args.command], run_name="__main__")
    finally:
        sys.argv = previous
