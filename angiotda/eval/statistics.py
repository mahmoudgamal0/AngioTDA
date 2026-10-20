"""Six-metric paired model comparisons with a shared Holm correction."""
from __future__ import annotations
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from angiotda.eval.paired import paired_difference, atomic_json

METRICS = ("Replicate cosine","Replicate AUROC","Neighbour correlation","Target","Reactome","JUMP")

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("p-values must be a finite vector in [0,1]")
    if not len(p): return p.copy()
    order=np.argsort(p,kind="stable")
    adjusted=np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty_like(p);out[order]=adjusted
    return out

def sign_flip_p(values,n_permutations=99999,seed=42):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired differences must be a nonempty finite vector")
    if n_permutations < 1: raise ValueError("n_permutations must be positive")
    observed=abs(values.mean());tolerance=1e-14
    if len(values)<=16:
        signs=np.asarray(list(itertools.product((-1.,1.),repeat=len(values))))
        return float(np.mean(np.abs(signs@values/len(values))>=observed-tolerance))
    rng=np.random.RandomState(seed);exceed=generated=0
    while generated<n_permutations:
        size=min(256,n_permutations-generated)
        signs=rng.randint(0,2,size=(size,len(values)),dtype=np.int8)*2-1
        exceed+=int(np.count_nonzero(np.abs((signs*values).mean(axis=1))>=observed-tolerance))
        generated+=size
    return (exceed+1)/(n_permutations+1)

def compound_scores(biology,jump):
    for report in (biology,jump):
        if report.get("status")!="complete" or report.get("fold_indices")!=list(range(5)):
            raise ValueError("complete five-fold reports are required")
    if biology.get("architecture") != jump.get("architecture"):
        raise ValueError("biology and JUMP inputs refer to different encoders")
    if biology.get("split_sha256")!=jump.get("split_sha256") or biology.get("seed")!=jump.get("seed"):
        raise ValueError("biology and JUMP inputs must match in split and training seed")
    return {
        "Replicate cosine":{c:r["paired_cosine"] for c,r in biology["replicate_identity_per_compound"].items()},
        "Replicate AUROC":{c:r["replicate_auroc"] for c,r in biology["replicate_identity_per_compound"].items()},
        "Neighbour correlation":{c:r["rank_correlation"] for c,r in biology["neighborhood_stability_per_compound"].items()},
        **{label:{c:r["ndcg_at_10"] for c,r in report["endpoints"][endpoint]["per_compound_metrics"].items()}
           for label,report,endpoint in [("Target",biology,"chembl_offtargets"),("Reactome",biology,"chembl37_human_reactome"),("JUMP",jump,"jump_cellpainting")]}}

def compare(groups,n_bootstrap=10000,n_permutations=99999,seed=42):
    if len(groups)<2: raise ValueError("at least two models are required")
    if n_bootstrap<1: raise ValueError("n_bootstrap must be positive")
    seeds=None;digest=None;averaged={}
    for model,inputs in groups.items():
        runs={}
        for biology,jump in inputs:
            current=biology["split_sha256"]
            if digest is not None and current!=digest: raise ValueError("models use different partitions")
            digest=current;key=biology["seed"]
            if key in runs: raise ValueError("duplicate training seed for a model")
            runs[key]=compound_scores(biology,jump)
        if not runs: raise ValueError("each model requires at least one seed")
        if seeds is not None and set(runs)!=seeds: raise ValueError("model seed sets differ")
        seeds=set(runs);averaged[model]={}
        for metric in METRICS:
            keys=set(next(iter(runs.values()))[metric])
            if not keys or any(set(r[metric])!=keys for r in runs.values()):
                raise ValueError("seed compound sets differ or are empty")
            averaged[model][metric]={c:float(np.mean([r[metric][c] for r in runs.values()])) for c in sorted(keys)}
    rows=[]
    for first,second in itertools.combinations(averaged,2):
        for metric in METRICS:
            a,b=averaged[first][metric],averaged[second][metric]
            row=paired_difference(b,a,n_bootstrap=n_bootstrap,seed=seed)
            differences=np.array([a[c]-b[c] for c in sorted(a)])
            row.update(first=first,second=second,metric=metric,difference="first_minus_second",
                       p_two_sided=sign_flip_p(differences,n_permutations,seed))
            rows.append(row)
    for row,p in zip(rows,holm([r["p_two_sided"] for r in rows])):
        row.update(p_holm=float(p),significant_holm=bool(p<.05))
    return dict(status="complete",split_sha256=digest,seeds=sorted(seeds),
                inference="paired compound bootstrap and sign flips, conditional on fitted models; seed-averaged scores",
                confidence_intervals="pointwise 95%; not adjusted for multiplicity",
                correction=f"Holm across all {len(rows)} pair-by-metric tests",
                means={m:{k:float(np.mean(list(v.values()))) for k,v in metrics.items()} for m,metrics in averaged.items()},
                comparisons=rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",action="append",nargs=3,metavar=("NAME","BIOLOGY_JSON","JUMP_JSON"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--n-bootstrap",type=int,default=10000)
    parser.add_argument("--n-permutations",type=int,default=99999)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args();groups=defaultdict(list)
    for name,bio,jump in args.model:
        groups[name].append((json.loads(Path(bio).read_text()),json.loads(Path(jump).read_text())))
    payload=compare(groups,args.n_bootstrap,args.n_permutations,args.seed)
    atomic_json(args.output,payload)
    lines=["# Paired model comparison","",payload["inference"],"",payload["correction"],"",
           "| First - second | Metric | Difference | Pointwise 95% CI | Raw p | Holm p |",
           "|---|---|---:|---|---:|---:|"]
    for r in payload["comparisons"]:
        lines.append(f"| {r['first']} - {r['second']} | {r['metric']} | {r['mean']:+.5f} | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] | {r['p_two_sided']:.5f} | {r['p_holm']:.5f} |")
    args.output.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print(args.output)

if __name__ == "__main__": main()
