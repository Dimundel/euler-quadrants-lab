"""Reproducible experiments for continuous selection of Euler term graphs.

The train/validation/test split is fixed before any graph search. Test targets
are never passed to the optimizer or intervention calibration.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy

from euler_lab.experiments import quadrant_state
from euler_lab.solver import EulerSolver, primitive_to_conservative
from .control_solver import solve_pressure_control
from .vendor import ivan_interventions as iv


@dataclass
class Dataset:
    name: str
    train: iv.RegressionData
    validation: iv.RegressionData
    test: iv.RegressionData
    perturbed: dict[str, iv.RegressionData]
    metadata: dict


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_dataset(root, name, *, n=96, intervention_seed=1701):
    """Recompute training perturbations; cache only deterministic trajectories."""
    root = Path(root)
    data_dir = root / "data" / "processed"
    if name == "well":
        with np.load(data_dir / "well_sample.npz", allow_pickle=False) as z:
            frames = z["conserved"][:, :, :, None, :].astype(float)
            times = z["time"].copy()
            spacing = (float(z["x"][1]-z["x"][0]), float(z["y"][1]-z["y"][0]), 1.0)
        train_centers, val_centers, test_centers = [1, 2], [3], [5]
        kw = dict(spacing=spacing, sigma=2.0, margin=7, block_size=8)
        source = "The Well, external Clawpack simulation, existing repository sample"
        warnings = [
            "The Well contains numerical simulations, not physical measurements.",
            "One trajectory only: the score does not establish cross-trajectory generalization.",
            "Derivative target centers are distinct. Neighboring finite-difference stencils share input frames across validation boundaries.",
            "The test derivative stencil [4,5,6] is disjoint from training stencils [0,1,2] and [1,2,3].",
        ]
        source_sha = _sha256(data_dir / "well_sample.npz")
    elif name in ("synthetic", "control"):
        case = json.loads((data_dir / "anu_cases.json").read_text())["6"]
        state_names = ("rho", "u", "v", "w", "p")
        short_names = {"UL":"upper_left", "UR":"upper_right", "LL":"lower_left", "LR":"lower_right"}
        quadrants = {short: [case["quadrants"][long][k] for k in state_names]
                     for short, long in short_names.items()}
        initial = quadrant_state(quadrants, n=n, nz=1)
        times = np.linspace(0, .3, 31)
        spacing = (1/n, 1/n, 1.0)
        cache = root / ".cache" / "lab3"
        cache.mkdir(parents=True, exist_ok=True)
        cache_version = "v2" if name == "control" else "v1"
        cache_file = cache / f"{name}_{n}_{cache_version}.npz"
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as z:
                frames = z["frames"]
        else:
            if name == "synthetic":
                primitive = EulerSolver(gamma=1.4, cfl=.4).solve(initial, times, spacing)
                frames = primitive_to_conservative(primitive, 1.4)
            else:
                frames = solve_pressure_control(initial, times, spacing, gamma=1.4)
            np.savez_compressed(cache_file, frames=frames, times=times)
        train_centers, val_centers, test_centers = [3,5,7,9,11,13], [17,19], [25,27,29]
        sigma = n/32
        kw = dict(spacing=spacing, sigma=sigma, margin=int(np.ceil(3*sigma))+1,
                  block_size=max(8, n//8))
        source = "ANU case 6, local Rusanov solver" if name == "synthetic" else "ANU case 6, modified generating mechanism p=(gamma-1)*E"
        warnings = [
            "The synthetic solver is first order; regression coefficients can absorb numerical diffusion.",
            "The control removes kinetic energy from pressure in the generating solver, not just in the regressor table.",
        ]
        source_sha = _sha256(data_dir / "anu_cases.json")
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    train = iv.build_regression(frames, times, train_centers, **kw)
    validation = iv.build_regression(frames, times, val_centers, scales=train.scales, **kw)
    test = iv.build_regression(frames, times, test_centers, scales=train.scales, **kw)
    parents = list(dict.fromkeys(iv.PARAM_PARENT.values()))
    perturbed = {
        parent: iv.build_with_intervention(
            frames, times, train_centers,
            iv.Intervention(parent, "shuffle", seed=intervention_seed + k*100),
            scales=train.scales, **kw)
        for k, parent in enumerate(parents)
    }
    metadata = {
        "name": name, "source": source, "source_sha256": source_sha,
        "trajectory_sha256":hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest(),
        "grid": list(frames.shape[1:4]), "times": times.tolist(),
        "train_centers": train_centers, "validation_centers": val_centers,
        "test_centers": test_centers,
        "train_times": times[train_centers].tolist(),
        "validation_times": times[val_centers].tolist(),
        "test_times": times[test_centers].tolist(),
        "regression": kw, "equation_scales_from_training": train.scales.tolist(),
        "n_rows": {"train":len(train.y), "validation":len(validation.y), "test":len(test.y)},
        "n_train_blocks": len(np.unique(train.block)),
        "intervention_seed": intervention_seed,
        "warnings": warnings,
        "expected_planar_support": [1,1,0,1,1,0,0,0,0] if name == "control" else [1,1,0,1,1,0,1,1,0],
        "support_note": "Reference support of the two-dimensional generating operator; zero z-columns do not establish absence of the three-dimensional physical mechanism.",
    }
    return Dataset(name, train, validation, test, perturbed, metadata)


def _json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=_json_default, allow_nan=False) + "\n")


def _normalize_run(raw, data, method, oracle):
    selected=raw["selected"]
    coefficients=np.asarray(selected["coefficients"])
    residual=data.test.y-data.test.X@coefficients
    expected=np.asarray(data.metadata["expected_planar_support"],dtype=bool)
    mask=np.asarray(selected["mask"],dtype=bool)
    archive=[]
    for row in raw["archive"]:
        row={**row,"val_nmse":row["validation_nmse"]}
        archive.append(row)
    return {
        "method":method, "seed":raw["seed"], "selected_mask":mask.tolist(),
        "coefficients":coefficients.tolist(), "weights":selected["weights"],
        "val_nmse":selected["validation_nmse"],
        "test_nmse":float(residual@residual/(data.test.y@data.test.y)),
        "score":selected["score"], "complexity":selected["complexity"],
        "intervention_penalty":selected["intervention_penalty"],
        "oracle_gap":float(selected["score"]-oracle["score"]),
        "support_hamming_distance":int(np.count_nonzero(mask!=expected)),
        "archive":archive, "trajectory":raw["trajectory"], "rounds":raw["rounds"],
        "settings":raw["settings"],
        "unique_graphs":len({tuple(row["mask"]) for row in archive}),
    }


def _rerank_sensitivity(run, reference_keep):
    """Rerank a fixed archive on validation only; do not touch test targets."""
    rows=[]
    reference=np.asarray(reference_keep,dtype=bool)
    for complexity in (.005,.02,.08):
        for intervention in (0.0,.03,.1,.3):
            candidates=[]
            for item in run["archive"]:
                mask=np.asarray(item["mask"],dtype=bool)
                penalty=float(np.mean(mask!=reference))
                score=item["val_nmse"]+complexity*float(mask.mean())+intervention*penalty
                candidates.append((score,item))
            score,item=min(candidates,key=lambda value:value[0])
            rows.append({"lambda_complexity":complexity,"lambda_intervention":intervention,
                         "selected_mask":item["mask"],"val_nmse":item["val_nmse"],"score":score})
    return {"procedure":"Rerank the fixed seed-0 intervention archive on validation. No new graph search or test evaluation.",
            "rows":rows}


def run_experiments(root, *, n=96):
    """Compute all pre-specified experiments, save the graph archive and figures."""
    from .selection import DiagnosticEngine, optimize_graphs, exact_oracle
    from .plotting import plot_dataset, plot_comparison
    root=Path(root)
    output=root/"results"/"lab3"
    output.mkdir(parents=True,exist_ok=True)
    config={"n_synthetic":n,"n_boot":100,"n_splits":16,"diagnostic_seed":0,
            "search_seeds":[0,1,2],"lambda_complexity":.02,"lambda_intervention":.1,
            "ridge":1e-5,"rounds":4,"steps":40,"samples_per_round":24,
            "intervention_seed":1701,
            "selection_rule":"Validation NMSE + lambda_complexity * active_groups/9 + lambda_intervention * training_evidence_disagreement/9",
            "seed_scope":"Search seeds vary gates and graph proposals. Intervention, bootstrap and null-calibration seeds remain fixed.",
            "precommitment":"Splits and default penalties fixed before evaluating test quality. Sensitivity uses validation only."}
    summary={"config":config,"runtime":{"python":platform.python_version(),
             "numpy":np.__version__,"scipy":scipy.__version__},"datasets":{}}
    started=time.perf_counter()
    with (output/"candidate_graphs.jsonl").open("w",encoding="utf-8") as archive_file:
        for name in ("synthetic","control","well"):
            start=time.perf_counter()
            print(f"[{name}] preparing data",flush=True)
            data=prepare_dataset(root,name,n=n,intervention_seed=config["intervention_seed"])
            engine=DiagnosticEngine(data.train,data.perturbed,n_boot=config["n_boot"],
                                    n_splits=config["n_splits"],seed=config["diagnostic_seed"])
            reference=engine.evaluate(np.ones(9,dtype=bool))
            reference_keep=np.asarray(reference["keep"],dtype=bool)
            expected=np.asarray(data.metadata["expected_planar_support"],dtype=bool)
            result={"metadata":data.metadata,"reference_diagnostics":reference,
                    "diagnostic_false_negatives":[iv.PARAMS[j] for j in range(9) if expected[j] and not reference_keep[j]],
                    "diagnostic_false_positives":[iv.PARAMS[j] for j in range(9) if not expected[j] and reference_keep[j]],
                    "runs":[]}
            result["oracle"]={}
            for method,eta,seeds in [("intervention",.1,[0,1,2]),("ablation",0.0,[0])]:
                oracle=exact_oracle(data.train,data.validation,reference_keep,
                    lambda_complexity=.02,lambda_intervention=eta,ridge=1e-5)
                result["oracle"][method]=oracle
                for seed in seeds:
                    print(f"[{name}] {method} seed={seed}",flush=True)
                    raw=optimize_graphs(data.train,data.validation,engine,
                        lambda_complexity=.02,lambda_intervention=eta,ridge=1e-5,
                        seed=seed,rounds=4,steps=40,samples_per_round=24)
                    run=_normalize_run(raw,data,method,oracle)
                    result["runs"].append(run)
                    for number,item in enumerate(run["archive"]):
                        record={"dataset":name,"method":method,"seed":seed,"sample_index":number,**item}
                        archive_file.write(json.dumps(record,ensure_ascii=False,default=_json_default,allow_nan=False)+"\n")
                    print(f"[{name}] mask={run['selected_mask']}, val NMSE={run['val_nmse']:.6g}, "
                          f"test L2={np.sqrt(run['test_nmse']):.4f}, oracle gap={run['oracle_gap']:.3g}",flush=True)
            main_run=result["runs"][0]
            result["sensitivity"]=_rerank_sensitivity(main_run,reference_keep)
            masks=np.array([r["selected_mask"] for r in result["runs"] if r["method"]=="intervention"],dtype=int)
            result["seed_stability"]={"selected_support_frequency":masks.mean(axis=0).tolist(),
                                      "number_distinct_masks":len(set(map(tuple,masks)))}
            result["elapsed_seconds"]=time.perf_counter()-start
            summary["datasets"][name]=result
            # Archive numerical coordinates actually plotted, not only rendered images.
            idx=np.random.default_rng(0).choice(len(data.test.y),min(6000,len(data.test.y)),replace=False)
            np.savez_compressed(output/f"{name}_approximation.npz",row_indices=idx,
                y=data.test.y[idx],prediction=(data.test.X@np.asarray(main_run["coefficients"]))[idx],
                equation=data.test.equation[idx],coefficients=main_run["coefficients"],
                selected_mask=main_run["selected_mask"],train_scales=data.train.scales)
            plot_dataset(data,result,output)
            write_json(output/"metrics.json",summary)
    summary["runtime"]["elapsed_seconds"]=time.perf_counter()-started
    plot_comparison(summary,output/"comparison.png")
    write_json(output/"metrics.json",summary)
    print(f"Saved results to {output}",flush=True)
    return summary


def rebuild_plots(root):
    from .plotting import plot_dataset, plot_comparison
    root=Path(root);output=root/"results"/"lab3"
    summary=json.loads((output/"metrics.json").read_text())
    for name,result in summary["datasets"].items():
        data=prepare_dataset(root,name,n=summary["config"]["n_synthetic"],
                             intervention_seed=summary["config"]["intervention_seed"])
        plot_dataset(data,result,output)
    plot_comparison(summary,output/"comparison.png")
