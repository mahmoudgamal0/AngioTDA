"""Train one compound-held-out graph cross-fit fold with replicate NT-Xent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from angiotda import PROJECT_ROOT
from angiotda.training.common import format_optional

from angiotda.data.dataset import CompoundPairSampler, GraphDatasetV2, collate_fn
from angiotda.data.graph_ablation import GRAPH_ABLATIONS, ablate_graph_batch
from angiotda.data.multimodal import seed_worker
from angiotda.data.paths import DATASET
from angiotda.eval.metrics import load_moa, load_offtargets
from angiotda.eval.retrieval import (
    cross_replicate_retrieval,
    profile_cosine,
    summarize_retrieval,
    tfidf_profile_vectors,
)
from angiotda.losses import NTXentLoss
from angiotda.models.projection_head import ProjectionHead
from angiotda.training.common import augment_batch
from angiotda.training.common import (
    CompoundSubsetDataset,
    audit_physical_pairs,
    audit_training_batch,
    build_crossfit_encoder,
    collect_embeddings,
    save_embedding_artifact,
    training_degree_histogram,
)
from angiotda.training.biology_alignment import (
    WeightedMultiPositiveContrastiveLoss,
    build_moa_relevance,
    build_target_relevance,
)
from angiotda.eval.biology import (
    load_chembl37_human_reactome,
    neighborhood_rows,
    reconstruct_replicate_views,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_replicates(embedding, head, objective, compounds, precision, use_amp):
    with torch.autocast(device_type=embedding.device.type, dtype=torch.bfloat16,
                        enabled=use_amp and precision == "legacy_bf16"):
        projected = head(embedding if precision == "legacy_bf16" else embedding.float())
        return projected, objective(projected, compounds)


def jsonable_args(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def upsert_history(path: Path, row: dict) -> None:
    history = []
    if path.exists():
        history = [json.loads(line) for line in path.read_text().splitlines() if line]
    history = [item for item in history if item["epoch"] != row["epoch"]]
    history.append(row)
    history.sort(key=lambda item: item["epoch"])
    path.write_text("".join(json.dumps(item) + "\n" for item in history))


def quick_endpoint(first, second, raw_profiles):
    profiles = tfidf_profile_vectors(
        raw_profiles, compounds=set(first) & set(second))
    summary = summarize_retrieval(cross_replicate_retrieval(
        first, second, profiles, (10,), profile_cosine), n_bootstrap=0)
    return {
        "n_query_compounds": summary["n_query_compounds"],
        "ndcg_at_10": summary["metrics"].get("ndcg_at_10", {}).get("mean"),
    }


def periodic_biology(compounds, image_names, embeddings, fold):
    first, second, pair_audit = reconstruct_replicate_views(
        compounds, image_names, embeddings)
    stability = neighborhood_rows(first, second, fold)
    correlations = [row["rank_correlation"] for row in stability.values()]
    return {
        "heldout_compounds": len(first),
        "pair_audit": pair_audit,
        "neighbor_rank_correlation": (
            float(np.mean(correlations)) if correlations else None),
        "chembl_targets": quick_endpoint(
            first, second,
            load_offtargets(path=DATASET.index_dir / "offtargets_full291.json")),
        "chembl37_human_reactome": quick_endpoint(
            first, second, load_chembl37_human_reactome()),
    }


def render_periodic(path: Path, run_name: str, history: list[dict]) -> None:
    lines = [
        f"# Periodic held-out biology: {run_name}", "",
        "These metrics are monitoring only. They never select a checkpoint.", "",
        "| Epoch | Train loss | Neighbour rho | Target queries | Target nDCG@10 | Reactome queries | Reactome nDCG@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda value: "—" if value is None else f"{value:.4f}"
    for row in history:
        biology = row["biology"]
        target, reactome = (biology["chembl_targets"],
                            biology["chembl37_human_reactome"])
        lines.append(
            f"| {row['epoch']} | {row['training_loss']:.4f} | "
            f"{fmt(biology['neighbor_rank_correlation'])} | "
            f"{target['n_query_compounds']} | {fmt(target['ndcg_at_10'])} | "
            f"{reactome['n_query_compounds']} | {fmt(reactome['ndcg_at_10'])} |")
    path.write_text("\n".join(lines) + "\n")


def checkpoint_payload(
    *, args, config, split, epoch, loss, best_loss, global_step,
    encoder, head, optimizer, scheduler, scaler, sampler,
):
    payload = {
        "arch": "biological_crossfit_graph_ssl",
        "architecture": args.architecture,
        "fold": args.fold,
        "seed": args.seed,
        "split_seed": split["seed"],
        "split_sha256": split["compound_partition_sha256"],
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,
        "global_step": global_step,
        "encoder_state": encoder.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "sampler_rng_state": sampler._rng.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config,
        "objective": config["objective"],
        "biology_alignment_coverage": config["biology_alignment_coverage"],
    }
    return payload


def validate_split(split: dict, fold_index: int):
    if fold_index < 0 or fold_index >= split["n_folds"]:
        raise ValueError(f"fold must be in [0,{split['n_folds'] - 1}]")
    fold = split["folds"][fold_index]
    if fold["fold"] != fold_index:
        raise ValueError("split fold records are not indexed canonically")
    train, heldout = set(fold["train_compounds"]), set(fold["heldout_compounds"])
    if train & heldout:
        raise AssertionError("compound leakage in persisted split")
    all_heldout = [
        compound for record in split["folds"]
        for compound in record["heldout_compounds"]
    ]
    if len(all_heldout) != len(set(all_heldout)) or len(all_heldout) != split["n_compounds"]:
        raise AssertionError("held-out folds are not an exact compound partition")
    return fold


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = args.split
    if not split_path.is_absolute():
        split_path = PROJECT_ROOT / split_path
    split = json.loads(split_path.read_text())
    fold = validate_split(split, args.fold)

    graphs_dir = args.graphs_dir or DATASET.graphs_dir
    if not graphs_dir.is_absolute():
        graphs_dir = PROJECT_ROOT / graphs_dir
    base_dataset = GraphDatasetV2(DATASET.sample_index, graphs_dir)
    index = json.loads(DATASET.sample_index.read_text())
    expected_wells = {s["image_name"] for s in index["samples"]
                      if s["compound"] not in {"CTRL", "DMSO", "Suramin"}}
    if {s["image_name"] for s in base_dataset.samples} != expected_wells:
        raise RuntimeError("graph directory does not cover every non-control well")
    manifest = graphs_dir / "manifest.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    if args.graphs_dir:
        if not manifest.exists():
            raise RuntimeError("derived graph cache requires a completed extraction manifest")
        graph_audit = json.loads(manifest.read_text())
        if (graph_audit.get("status") != "complete"
                or graph_audit.get("scope") != "all_noncontrol_wells"
                or {s["image_name"] for s in graph_audit["wells"]} != expected_wells):
            raise RuntimeError("derived graph manifest is incomplete")
    if set(base_dataset.compounds) != {
        compound for record in split["folds"] for compound in record["heldout_compounds"]
    }:
        raise RuntimeError("split compound universe differs from the current graph dataset")
    train_dataset = CompoundSubsetDataset(base_dataset, fold["train_compounds"])
    heldout_dataset = CompoundSubsetDataset(base_dataset, fold["heldout_compounds"])
    if set(train_dataset.compounds) & set(heldout_dataset.compounds):
        raise AssertionError("compound leakage after dataset construction")
    pair_audit = audit_physical_pairs(train_dataset)
    if pair_audit["unpairable_compounds"]:
        raise RuntimeError(
            "training compounds without a physical R1/R2 pair: "
            f"{pair_audit['unpairable_compounds'][:10]}")

    moa_labels = load_moa("sailem57")
    if args.biology_alignment == "moa":
        biological_relevance = build_moa_relevance(
            train_dataset.compounds, moa_labels)
    elif args.biology_alignment == "targets":
        biological_relevance = build_target_relevance(
            train_dataset.compounds,
            load_offtargets(path=DATASET.index_dir / "offtargets_full291.json"),
        )
    else:
        biological_relevance = None
    alignment_audit = (
        biological_relevance.audit if biological_relevance is not None else {
            "mode": "none",
            "fit_scope": "not_applicable",
            "n_training_compounds": len(train_dataset.compounds),
            "n_annotated_training_compounds": 0,
            "annotation_coverage": 0.0,
            "missing_annotations_policy": "not_applicable",
        }
    )

    sampler = CompoundPairSampler(
        train_dataset, batch_compounds=args.batch_compounds, seed=args.seed)
    if len(sampler) == 0:
        raise RuntimeError("no valid contrastive batches")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=sampler, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=generator, worker_init_fn=seed_worker,
    )
    heldout_loader = torch.utils.data.DataLoader(
        heldout_dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )

    degree_histogram = (
        training_degree_histogram(train_dataset) if args.architecture == "pna" else None
    )
    config = {
        **jsonable_args(args),
        "split": str(split_path),
        "split_sha256": split["compound_partition_sha256"],
        "embed_dim": args.embed_dim,
        "layers": args.layers,
        "heads": args.heads,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "degree_histogram": degree_histogram,
        "device": str(device),
        "graphs_dir": str(graphs_dir.relative_to(PROJECT_ROOT)),
        "graph_manifest_sha256": manifest_hash,
        "objective": {
            "replicate": "nt_xent_always_enabled",
            "biology_alignment": args.biology_alignment,
            "biology_alignment_loss": (
                "weighted_multi_positive_contrastive"
                if biological_relevance is not None else "disabled"),
            "biology_lambda": args.biology_lambda,
            "total": "replicate_nt_xent + biology_lambda * biology_alignment",
            "model_selection": "fixed_final_epoch",
            "heldout_labels_used_for_optimization": False,
            "heldout_labels_used_for_model_selection": False,
        },
        "biology_alignment_coverage": alignment_audit,
    }
    encoder = build_crossfit_encoder(args.architecture, config).to(device)
    head = ProjectionHead(args.embed_dim).to(device)
    replicate_objective = NTXentLoss(args.temperature)
    alignment_objective = WeightedMultiPositiveContrastiveLoss(args.temperature)
    parameters = list(encoder.parameters()) + list(head.parameters())
    config["parameter_count"] = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(sampler)
    warmup = min(args.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * (
            0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda" and args.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    run_name = args.run_name or f"{args.architecture}_seed{args.seed}_fold{args.fold}"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "biological_crossfit" / run_name
    report_dir = PROJECT_ROOT / "reports" / "biological_crossfit" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (report_dir / "data_audit.json").write_text(json.dumps({
        "split_sha256": split["compound_partition_sha256"],
        "fold": args.fold,
        "train_compounds": len(train_dataset.compounds),
        "heldout_compounds": len(heldout_dataset.compounds),
        "train_wells": len(train_dataset),
        "heldout_wells": len(heldout_dataset),
        "compound_overlap": [],
        "positive_pair_audit": pair_audit,
        "pna_degree_histogram_training_only": degree_histogram,
        "objective": config["objective"],
        "biology_alignment_coverage": alignment_audit,
        "alignment_compounds_subset_of_training": (
            biological_relevance is None or
            biological_relevance.training_compounds == frozenset(train_dataset.compounds)
        ),
        "heldout_annotations_optimizer_access": False,
        "graph_ablation": args.graph_ablation,
        "graph_ablation_scope": "training_and_heldout_evaluation",
        "graphs_dir": config["graphs_dir"],
        "graph_manifest_sha256": manifest_hash,
    }, indent=2) + "\n")
    start_epoch, global_step, best_loss = 1, 0, float("inf")
    latest = checkpoint_dir / "latest.pt"
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"--resume requested but missing {latest}")
        state = torch.load(latest, map_location=device, weights_only=False)
        state["config"].setdefault("graphs_dir", "data/graphs")
        state["config"].setdefault("graph_manifest_sha256", None)
        state["config"].setdefault("precision", "legacy_bf16")
        identity = (state["architecture"], state["fold"], state["seed"], state["split_sha256"])
        expected = (args.architecture, args.fold, args.seed, split["compound_partition_sha256"])
        if identity != expected:
            raise RuntimeError(f"resume identity mismatch: {identity} != {expected}")
        resume_keys = (
            "epochs", "batch_compounds", "embed_dim", "hidden", "layers",
            "dropout", "temperature", "lr", "lr_min", "weight_decay",
            "warmup_steps", "node_dropout", "edge_dropout", "graph_ablation",
            "graphs_dir", "graph_manifest_sha256", "precision",
        )
        if args.architecture in {"gatv2", "transformer", "gps"}:
            resume_keys += ("heads",)
        if args.biology_alignment != "none":
            resume_keys += ("biology_alignment", "biology_lambda")
        changed = {
            key: (state["config"].get(key), config.get(key))
            for key in resume_keys
            if state["config"].get(key) != config.get(key)
        }
        if changed:
            raise RuntimeError(f"resume hyperparameters differ: {changed}")
        encoder.load_state_dict(state["encoder_state"])
        head.load_state_dict(state["head_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and state.get("scaler_state") is not None:
            scaler.load_state_dict(state["scaler_state"])
        sampler._rng.setstate(state["sampler_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([
                rng_state.cpu() for rng_state in state["cuda_rng_state"]])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_loss = state["best_loss"]
        print(f"Resuming {run_name} at epoch {start_epoch}", flush=True)
    elif latest.exists():
        raise FileExistsError(
            f"existing resumable run at {latest}; pass --resume or choose another run name")

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train(); head.train()
        loss_total = 0.0; replicate_total = 0.0; alignment_total = 0.0
        dispersion_total = 0.0; uniform_loss_total = 0.0
        alignment_anchors = 0; alignment_known_pairs = 0; n_batches = 0
        for batch in train_loader:
            audit_training_batch(batch)
            batch = ablate_graph_batch(batch, args.graph_ablation, args.seed)
            batch = augment_batch(batch, args.node_dropout, args.edge_dropout)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                embedding = encoder(
                    batch["x"].to(device),
                    [value.to(device) for value in batch["edge_index"]],
                    [value.to(device) for value in batch["edge_attr"]],
                    batch["node_mask"].to(device),
                )
            # Keep the projection, normalization, and similarity calculations
            # in FP32: BF16 can round distinct near-collinear wells to ties.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp and args.precision == "legacy_bf16"):
                projected, replicate_result = project_replicates(
                    embedding, head, replicate_objective, batch["compound"],
                    args.precision, use_amp)
                if replicate_result["n_pairs"] * 2 != len(batch["compound"]):
                    raise AssertionError("NT-Xent did not recognize every physical pair")
                if biological_relevance is not None:
                    relevance_matrix = biological_relevance.batch_matrix(
                        batch["compound"], device=projected.device)
                    alignment_result = alignment_objective(
                        projected, batch["compound"], relevance_matrix)
                else:
                    alignment_result = {
                        "loss": projected.sum() * 0.0,
                        "n_anchors": 0, "n_positive_pairs": 0,
                        "n_known_pairs": 0,
                    }
                loss = (replicate_result["loss"]
                        + args.biology_lambda * alignment_result["loss"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
            scheduler.step(); global_step += 1
            loss_total += loss.item()
            replicate_total += replicate_result["loss"].item()
            alignment_total += alignment_result["loss"].item()
            alignment_anchors += alignment_result["n_anchors"]
            alignment_known_pairs += alignment_result["n_known_pairs"]
            with torch.no_grad():
                normalized = torch.nn.functional.normalize(projected.float(), dim=-1)
                dispersion_total += float((normalized-normalized.mean(0)).square().sum(-1).mean().sqrt())
            uniform_loss_total += math.log(len(batch["compound"])-1)
            n_batches += 1

        average_loss = loss_total / max(n_batches, 1)
        upsert_history(report_dir / "train_history.jsonl", {
            "epoch": epoch, "loss": average_loss,
            "total_loss": average_loss,
            "replicate_ntxent_loss": replicate_total / max(n_batches, 1),
            "biology_alignment_loss": alignment_total / max(n_batches, 1),
            "biology_alignment": args.biology_alignment,
            "biology_lambda": args.biology_lambda,
            "biology_alignment_anchors": alignment_anchors,
            "biology_alignment_known_pairs": alignment_known_pairs,
            "lr": scheduler.get_last_lr()[0], "n_batches": n_batches,
            "projected_rms_dispersion": dispersion_total / max(n_batches, 1),
            "uniform_ntxent_loss": uniform_loss_total / max(n_batches, 1),
        })
        if average_loss < best_loss:
            best_loss = average_loss
            atomic_torch_save(checkpoint_payload(
                args=args, config=config, split=split, epoch=epoch, loss=average_loss,
                best_loss=best_loss, global_step=global_step, encoder=encoder, head=head,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler, sampler=sampler,
            ), checkpoint_dir / "best.pt")

        should_evaluate = (
            epoch == 1 or epoch == args.epochs
            or (args.eval_every > 0 and epoch % args.eval_every == 0)
        )
        if should_evaluate:
            compounds, image_names, embeddings = collect_embeddings(
                encoder, heldout_loader, device, args.graph_ablation, args.seed)
            biology = periodic_biology(
                compounds, image_names, embeddings, args.fold)
            periodic_path = report_dir / "periodic_biology.json"
            history = (json.loads(periodic_path.read_text())
                       if periodic_path.exists() else [])
            history = [row for row in history if row["epoch"] != epoch]
            history.append({"epoch": epoch, "training_loss": average_loss,
                            "biology": biology})
            history.sort(key=lambda row: row["epoch"])
            periodic_path.write_text(json.dumps(history, indent=2) + "\n")
            render_periodic(report_dir / "periodic_biology.md", run_name, history)
            print(
                f"{run_name} epoch={epoch} loss={average_loss:.5f} "
                f"rho={format_optional(biology['neighbor_rank_correlation'])} "
                f"target_ndcg10={format_optional(biology['chembl_targets']['ndcg_at_10'])} "
                f"reactome_ndcg10={format_optional(biology['chembl37_human_reactome']['ndcg_at_10'])}",
                flush=True,
            )

        state = checkpoint_payload(
            args=args, config=config, split=split, epoch=epoch, loss=average_loss,
            best_loss=best_loss, global_step=global_step, encoder=encoder, head=head,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, sampler=sampler,
        )
        atomic_torch_save(state, latest)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch and epoch < args.epochs:
            print(f"Stopped diagnostic prefix at epoch {epoch}; no final checkpoint emitted", flush=True)
            return report_dir

    if start_epoch > args.epochs:
        state = torch.load(latest, map_location=device, weights_only=False)
        average_loss = state["loss"]
    atomic_torch_save(state, checkpoint_dir / "final.pt")
    compounds, image_names, embeddings = collect_embeddings(
        encoder, heldout_loader, device, args.graph_ablation, args.seed)
    metadata = {
        "architecture": args.architecture, "seed": args.seed, "fold": args.fold,
        "split_sha256": split["compound_partition_sha256"],
        "checkpoint": str((checkpoint_dir / "final.pt").relative_to(PROJECT_ROOT)),
        "embedding_space": "fold_specific_do_not_pool_across_folds",
        "objective": config["objective"],
        "biology_alignment_coverage": alignment_audit,
        "graph_ablation": args.graph_ablation,
        "graph_ablation_scope": "training_and_heldout_evaluation",
        "graphs_dir": config["graphs_dir"],
        "graph_manifest_sha256": manifest_hash,
    }
    save_embedding_artifact(
        report_dir / "heldout_embeddings_final.npz",
        compounds, image_names, embeddings, metadata)
    return report_dir


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("gine", "pna", "gatv2", "transformer", "gps", "deepsets", "gcn"),
        required=True)
    parser.add_argument("--graphs-dir", type=Path,
                        help="Explicit derived graph cache for extraction sensitivity.")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--precision", choices=("legacy_bf16", "fp32_objective", "fp32"),
                        default="fp32_objective",
                        help="Keep projection and contrastive loss in FP32 by default.")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop a diagnostic prefix without changing the planned LR schedule or writing final artifacts.")
    parser.add_argument("--batch-compounds", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--biology-alignment", choices=("none", "moa", "targets"),
        default="none")
    parser.add_argument(
        "--biology-lambda", type=nonnegative_float, default=0.1,
        help="Weight on alignment; replicate NT-Xent always retains weight one.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--node-dropout", type=float, default=0.1)
    parser.add_argument("--edge-dropout", type=float, default=0.05)
    parser.add_argument("--graph-ablation", choices=GRAPH_ABLATIONS, default="none")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-permutations", type=int, default=999)
    parser.add_argument("--final-eval-permutations", type=int, default=9999)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
