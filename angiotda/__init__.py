"""AngioTDA: graph representation learning for phenotypic angiogenesis screens."""
from pathlib import Path
import os

__version__ = "0.2.0"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("ANGIOTDA_WORKSPACE", Path.home() / "angiotda-workspace")).expanduser().resolve()
DATA_ROOT = PROJECT_ROOT / "data"
SAMPLE_INDEX = DATA_ROOT / "index" / "sample_index.json"
GRAPHS_DIR = DATA_ROOT / "graphs"
RAW_DIR = DATA_ROOT / "raw"
SKELETON_DIR = DATA_ROOT / "skeleton_images"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CONTROLS = ["CTRL", "DMSO", "Suramin"]
