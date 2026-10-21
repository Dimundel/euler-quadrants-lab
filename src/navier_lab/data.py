"""Сохранённая выборка внешнего расчёта обтекания цилиндра, Re=100."""

import hashlib
import json
from pathlib import Path

import numpy as np
import requests
from scipy.io import loadmat

REVISION = "932f50a2d8ef4e80d1456bbae6887a73ff5166ef"
SOURCE_URL = (f"https://raw.githubusercontent.com/maziarraissi/PINNs/{REVISION}"
              "/main/Data/cylinder_nektar_wake.mat")
SOURCE_SHA256 = "99d9e1f9da7d2dff113ff2723373237b455e2de6c6747ac681846fe86fbba3d0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_cylinder(output_dir, source_mat=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if source_mat is None:
        cache = output.parent / "cache" / "lab2"
        cache.mkdir(parents=True, exist_ok=True)
        source_mat = cache / "cylinder_nektar_wake.mat"
        if not source_mat.exists():
            with requests.get(SOURCE_URL, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                payload = bytearray()
                for chunk in response.iter_content(1 << 20):
                    payload.extend(chunk)
                    if len(payload) > 26_000_000:
                        raise RuntimeError("Источник неожиданно превышает 26 МБ")
            source_mat.write_bytes(payload)
    if digest(source_mat) != SOURCE_SHA256:
        raise ValueError("Контрольная сумма исходного MAT не совпала")
    source = loadmat(source_mat)
    points = source["X_star"]
    x, y, time = np.unique(points[:, 0]), np.unique(points[:, 1]), source["t"].ravel()
    order = np.lexsort((points[:, 0], points[:, 1]))
    xx, yy = np.meshgrid(x, y)
    if not np.array_equal(points[order], np.column_stack((xx.ravel(), yy.ravel()))):
        raise ValueError("Исходные точки не образуют декартову сетку")
    def grid(values):
        return values[order].reshape(len(y), len(x), len(time)).transpose(2, 0, 1).astype(np.float32)
    path = output / "cylinder_wake.npz"
    np.savez_compressed(path, x=x, y=y, time=time,
                        u=grid(source["U_star"][:, 0]),
                        v=grid(source["U_star"][:, 1]), p=grid(source["p_star"]))
    manifest = {
        "name": "Cylinder wake, Re=100, PINNs / Nektar++ numerical data",
        "kind": "external numerical simulation, not a physical experiment",
        "source_url": SOURCE_URL, "revision": REVISION,
        "source_sha256": SOURCE_SHA256, "source_bytes": Path(source_mat).stat().st_size,
        "article": "https://doi.org/10.1016/j.jcp.2018.10.045",
        "author_description": "https://maziarraissi.github.io/PINNs/",
        "expected_parameters": {"beta": 1.0, "nu": 0.01},
        "coordinates": {"x": [float(x.min()), float(x.max())],
                        "y": [float(y.min()), float(y.max())],
                        "t": [float(time.min()), float(time.max())]},
        "shape": [len(time), len(y), len(x)],
        "transform": "all points and frames retained; reorder to (t,y,x); u,v,p cast to float32",
        "artifact": path.name, "artifact_sha256": digest(path),
        "model_repository": "https://github.com/papayiv/NavieStocks_project",
        "model_revision": "66eb7bc73cd802a3249c058ad69fc59a95fbd807",
        "reused_file": "navier_stokes_tree.py -> src/navier_lab/ivan_graph.py",
        "reason_for_data_change": "Ivan's PDEBench velocity file has no observed pressure; intervention requires p",
    }
    (output / "lab2_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_cylinder(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "lab2_sources.json").read_text())
    path = data_dir / manifest["artifact"]
    if digest(path) != manifest["artifact_sha256"]:
        raise ValueError("Сохранённые данные изменились")
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}, manifest
