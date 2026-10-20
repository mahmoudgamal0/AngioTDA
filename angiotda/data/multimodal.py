"""Shared graph+image data plumbing for the multimodal encoders.

Both the late-fusion and CLIP scripts (train and eval) pull the dataset,
sampler and TIFF loading from here so preprocessing cannot drift between
training and evaluation.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from angiotda.data.paths import DATASET
from angiotda.data.dataset import (
    CompoundPairSampler,
    GraphDatasetV2,
    collate_fn as graph_collate_fn,
)

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False
from PIL import Image as PILImage

CONTROLS = ["CTRL", "DMSO", "Suramin"]


def stem_to_tiff(stem):
    m = re.match(r"P(\d+)R(\d+)_R(\d+)C(\d+)", stem)
    p, r, row, col = m.groups()
    return f"plate{p}_replica{r}_{int(row):03d}{int(col):03d}-1-001001001.tif"


def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
    original_dtype = arr.dtype
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        arr = arr / max(float(info.max), 1.0)
    else:
        hi = float(np.max(arr)) if arr.size else 1.0
        if hi > 1.0:
            arr = arr / hi
    mu, sigma = arr.mean(), arr.std()
    return (arr - mu) / (sigma + 1e-6)


def load_image(path, img_size=224, image_transform=None):
    """Generic grayscale image → (1, img_size, img_size) z-scored tensor."""
    path = Path(path)
    if _HAS_TIFFFILE and path.suffix.lower() in {".tif", ".tiff"}:
        try:
            arr = tifffile.imread(str(path))
        except Exception:
            arr = np.array(PILImage.open(str(path)))
    else:
        arr = np.array(PILImage.open(str(path)))
    if image_transform is None:
        # Preserve the established preprocessing path exactly for every
        # non-augmented experiment.
        arr = _normalize_image_array(arr)
    else:
        original_dtype = arr.dtype
        arr = arr.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if np.issubdtype(original_dtype, np.integer):
            arr /= max(float(np.iinfo(original_dtype).max), 1.0)
        else:
            hi = float(np.max(arr)) if arr.size else 1.0
            if hi > 1.0:
                arr /= hi
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    if tuple(t.shape[-2:]) != (img_size, img_size):
        t = F.interpolate(
            t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)
    if image_transform is not None:
        t = image_transform(t)
        t = (t - t.mean()) / (t.std(correction=0) + 1e-6)
    return t


class GraphImageDataset(torch.utils.data.Dataset):
    """Wraps GraphDatasetV2 and adds fluorescence image loading."""

    def __init__(self, sample_index, graphs_dir, raw_dir, root=None,
                 controls=None, img_size=224, strict_images=True,
                 image_transform=None):
        self.graph_ds = GraphDatasetV2(sample_index, graphs_dir,
                                       controls=controls or CONTROLS)
        self.raw_dir = Path(raw_dir)
        self.root = Path(root) if root is not None else Path(sample_index).parents[2]
        self.img_size = img_size
        self.strict_images = strict_images
        self.image_transform = image_transform

        with open(sample_index) as f:
            data = json.load(f)
        samples = data["samples"] if isinstance(data, dict) else data
        self._stem_map = {s["image_name"]: s for s in samples}

        self._image_paths = {
            sample["image_name"]: self._resolve_image_path(sample["image_name"])
            for sample in self.graph_ds.samples
        }
        missing = [str(path) for path in self._image_paths.values()
                   if not path.exists()]
        if missing and strict_images:
            preview = "\n".join(missing[:10])
            suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"{len(missing)} microscopy images are missing; refusing to "
                f"replace them with zero tensors:\n{preview}{suffix}"
            )

        self.compounds = self.graph_ds.compounds
        # Canonical sample metadata is part of the GraphDatasetV2-compatible
        # interface and is required by compound-held-out subset wrappers.
        self.samples = self.graph_ds.samples
        self.compound_to_indices = self.graph_ds.compound_to_indices
        self.compound_to_replicate_pairs = self.graph_ds.compound_to_replicate_pairs
        # Back-compat alias used by CompoundPairSampler.
        self._compound_to_idx = self.compound_to_indices

    def __len__(self):
        return len(self.graph_ds)

    def __getitem__(self, idx):
        item = self.graph_ds[idx]
        item["image"] = self.load_image(item["image_name"])
        return item

    def _resolve_image_path(self, stem):
        info = self._stem_map.get(stem)
        if info and "image_path" in info:
            return self.root / info["image_path"]
        return self.raw_dir / stem_to_tiff(stem)

    def load_image(self, stem):
        tiff_path = self._image_paths.get(stem, self._resolve_image_path(stem))
        if tiff_path.exists():
            return load_image(tiff_path, self.img_size, self.image_transform)
        if self.strict_images:
            raise FileNotFoundError(f"microscopy image does not exist: {tiff_path}")
        return torch.zeros(1, self.img_size, self.img_size)


def collate_graph_image(samples):
    """Collate graph fields (padded) + stack images."""
    batch = graph_collate_fn(samples)
    batch["image"] = torch.stack([s["image"] for s in samples])
    return batch


def build_dataset(project_root=None, img_size=224, controls=None,
                  image_transform=None):
    paths = DATASET
    return GraphImageDataset(
        sample_index=paths.sample_index,
        graphs_dir=paths.graphs_dir,
        raw_dir=paths.raw_dir,
        root=Path(project_root),
        controls=controls,
        img_size=img_size,
        image_transform=image_transform,
    )


def build_train_loader(dataset, batch_compounds=32, seed=0, num_workers=2):
    sampler = CompoundPairSampler(dataset, batch_compounds=batch_compounds, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_graph_image,
        num_workers=num_workers, pin_memory=True, generator=generator,
        worker_init_fn=seed_worker)
    return loader, sampler


def build_eval_loader(dataset, batch_size=16, num_workers=2):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_graph_image, num_workers=num_workers)


def seed_worker(worker_id):
    """Seed Python and NumPy from the distinct seed assigned by DataLoader."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    # Import locally to keep the module's public RNG dependencies minimal.
    import random
    random.seed(worker_seed)


def batch_to_device(batch, device):
    """Returns (images, x, edge_index, edge_attr, node_mask)."""
    return (
        batch["image"].to(device),
        batch["x"].to(device),
        [e.to(device) for e in batch["edge_index"]],
        [e.to(device) for e in batch["edge_attr"]],
        batch["node_mask"].to(device),
    )
