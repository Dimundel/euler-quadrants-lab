"""Небольшая воспроизводимая выборка The Well и начальные данные ANU."""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import requests
from bs4 import BeautifulSoup

WELL_REPO = "polymathic-ai/euler_multi_quadrants_openBC"
WELL_REVISION = "84e3eb65faf286533714295515caf9b99b4717a7"
WELL_FILENAME = "euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_40.hdf5"
WELL_URL = (
    f"https://huggingface.co/datasets/{WELL_REPO}/resolve/"
    f"{WELL_REVISION}/data/test/{WELL_FILENAME}"
)
WELL_CARD_URL = (
    f"https://huggingface.co/datasets/{WELL_REPO}/raw/{WELL_REVISION}/README.md"
)
ANU_SETTINGS_URL = "https://www.mso.anu.edu.au/fyris/lw2driemann.html"
FRAME_INDICES = (0, 1, 2, 4, 6, 8, 10)
QUADRANT_ORDER = ("upper_right", "upper_left", "lower_left", "lower_right")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP-файл с кэшем и жёстким лимитом загруженных байт.

    Сервер обязан отвечать 206 и подтверждать запрошенный диапазон. Ответ
    200 не читается: иначе безобидный запрос HDF5 мог бы скачать весь файл.
    """

    def __init__(self, url, *, max_bytes=100_000_000, block_size=1 << 20, session=None):
        super().__init__()
        if max_bytes <= 0 or block_size <= 0 or block_size > max_bytes:
            raise ValueError("Invalid transfer limit or block size")
        self.url = url
        self.max_bytes = int(max_bytes)
        self.block_size = int(block_size)
        self.session = session or requests.Session()
        self._owns_session = session is None
        self._position = 0
        self.size = None
        self.bytes_downloaded = 0
        self.ranges = []
        self._blocks = {}
        self._fetch_block(0)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("Unknown seek mode")
        if position < 0:
            raise ValueError("Negative file position")
        self._position = position
        return position

    def _fetch_block(self, start):
        end = start + self.block_size - 1
        if self.size is not None:
            end = min(end, self.size - 1)
        length = end - start + 1
        if self.bytes_downloaded + length > self.max_bytes:
            raise RuntimeError("HTTP Range download budget exceeded")
        # Разные URL не позволяют промежуточному кэшу перепутать Range-запросы.
        separator = "&" if "?" in self.url else "?"
        url = f"{self.url}{separator}lab_offset={start}"
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        with self.session.get(
            url, headers=headers, stream=True, timeout=(15, 60)
        ) as response:
            if response.status_code != 206:
                raise RuntimeError(f"Expected HTTP 206, got {response.status_code}")
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match:
                raise RuntimeError("Missing or invalid Content-Range")
            actual_start, actual_end, total = map(int, match.groups())
            expected_end = min(end, total - 1)
            if actual_start != start or actual_end != expected_end:
                raise RuntimeError("Server returned a different byte range")
            if self.size is not None and self.size != total:
                raise RuntimeError("Remote file size changed")
            self.size = total
            expected_length = actual_end - actual_start + 1
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                self.bytes_downloaded += len(chunk)
                if (
                    self.bytes_downloaded > self.max_bytes
                    or len(payload) + len(chunk) > expected_length
                ):
                    raise RuntimeError("Response exceeded the requested byte range")
                payload.extend(chunk)
            if len(payload) != expected_length:
                raise RuntimeError("Truncated HTTP Range response")
        data = bytes(payload)
        self._blocks[start] = data
        self.ranges.append(
            {"start": start, "end": actual_end, "sha256": sha256_bytes(data)}
        )

    def read(self, size=-1):
        self._checkClosed()
        if size is None or size < 0:
            size = max(0, self.size - self._position)
        if size > self.max_bytes:
            raise RuntimeError("Refusing to read the entire remote file")
        result = bytearray()
        while size > 0 and self._position < self.size:
            start = self._position // self.block_size * self.block_size
            if start not in self._blocks:
                self._fetch_block(start)
            block = self._blocks[start]
            offset = self._position - start
            part = block[offset : offset + size]
            result.extend(part)
            self._position += len(part)
            size -= len(part)
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self):
        if self._owns_session:
            self.session.close()
        super().close()


def coarsen_cells(values, factor):
    """Средние по непересекающимся квадратам; пространственные оси идут первыми."""
    values = np.asarray(values)
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    nx, ny = values.shape[:2]
    if nx % factor or ny % factor:
        raise ValueError("Spatial dimensions must be divisible by factor")
    shape = (nx // factor, factor, ny // factor, factor, *values.shape[2:])
    return values.reshape(shape).mean(axis=(1, 3), dtype=np.float64)


def parse_anu_case(html, case_id):
    """Разбирает числовую таблицу ANU, включая различающийся порядок строк P/d."""
    soup = BeautifulSoup(html, "html.parser")
    tables = [
        table
        for table in soup.find_all("table")
        if "T_final" in table.get_text() and table.find("table") is None
    ]
    if len(tables) != 1:
        raise ValueError("Expected one ANU initial-state table")
    values = {name: {} for name in QUADRANT_ORDER}
    active = None
    final_time = None
    columns = {"d": "rho", "P": "p", "v_x": "u", "v_y": "v"}
    for row in tables[0].find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["td", "th"], recursive=False)
        ]
        if cells[:1] == ["T_final"]:
            final_time = float(cells[1])
        elif cells == ["Upper Left", "Upper Right"]:
            active = ("upper_left", "upper_right")
        elif cells == ["Lower Left", "Lower Right"]:
            active = ("lower_left", "lower_right")
        elif len(cells) == 4 and cells[0] in columns:
            if active is None or cells[2] not in columns:
                raise ValueError("Malformed quadrant table")
            for side, label, number in zip(active, cells[::2], cells[1::2]):
                values[side][columns[label]] = float(number)
    if final_time is None or not np.isfinite(final_time) or final_time <= 0:
        raise ValueError("Missing valid final time")
    for state in values.values():
        if set(state) != {"rho", "p", "u", "v"}:
            raise ValueError("Incomplete quadrant state")
        if (
            not all(np.isfinite(value) for value in state.values())
            or state["rho"] <= 0
            or state["p"] <= 0
        ):
            raise ValueError("Nonphysical quadrant state")
        state["w"] = 0.0
    return {
        "case_id": int(case_id),
        "gamma": 1.4,
        "final_time": final_time,
        "discontinuity": [0.5, 0.5],
        "domain": [[0.0, 1.0], [0.0, 1.0]],
        "boundary": "outflow",
        "native_spatial_dimensions": 2,
        "quadrant_order": list(QUADRANT_ORDER),
        "primitive_order": ["rho", "u", "v", "w", "p"],
        "primitive": [
            [values[name][field] for field in ("rho", "u", "v", "w", "p")]
            for name in QUADRANT_ORDER
        ],
        "quadrants": values,
    }


def _small_download(url, session, limit=1_000_000):
    with session.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > limit:
                raise RuntimeError("Text source exceeds download limit")
        return bytes(data)


def download_well_sample(output_dir):
    """Берёт семь кадров одной траектории, не загружая полный HDF5."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        card = _small_download(WELL_CARD_URL, session)
        if "0.015s" not in card.decode("utf-8"):
            raise ValueError(
                "Dataset card no longer confirms the assumed output interval"
            )
        tree_url = f"https://huggingface.co/api/datasets/{WELL_REPO}/tree/{WELL_REVISION}/data/test"
        tree = json.loads(_small_download(tree_url, session))
        file_info = next(
            item for item in tree if item["path"].endswith("/" + WELL_FILENAME)
        )
        with HTTPRangeReader(WELL_URL, session=session) as remote:
            with h5py.File(remote, "r") as handle:
                density_dataset = handle["t0_fields/density"]
                if density_dataset.shape != (10, 101, 512, 512):
                    raise ValueError("Unexpected source HDF5 shape")
                source_time = handle["dimensions/time"][:].astype(np.float64)
                if not np.array_equal(source_time, np.arange(101)):
                    raise ValueError(
                        "Time coordinates changed; revisit the documented conversion"
                    )
                x = handle["dimensions/x"][:].astype(np.float64)
                y = handle["dimensions/y"][:].astype(np.float64)
                gamma = float(handle["scalars/gamma"][()])
                schema = {}
                for path in (
                    "t0_fields/density",
                    "t0_fields/energy",
                    "t0_fields/pressure",
                    "t1_fields/momentum",
                ):
                    field = handle[path]
                    schema[path] = {
                        "shape": list(field.shape),
                        "dtype": str(field.dtype),
                        "chunks": field.chunks,
                        "compression": field.compression,
                    }
                conserved, pressures, eos_samples = [], [], {}
                indices = np.random.default_rng(11).choice(
                    512 * 512, size=1000, replace=False
                )
                for frame in FRAME_INDICES:
                    rho = density_dataset[0, frame].astype(np.float64)
                    energy = handle["t0_fields/energy"][0, frame].astype(np.float64)
                    pressure = handle["t0_fields/pressure"][0, frame].astype(np.float64)
                    momentum = handle["t1_fields/momentum"][0, frame].astype(np.float64)
                    state = np.stack(
                        (
                            rho,
                            momentum[..., 0],
                            momentum[..., 1],
                            np.zeros_like(rho),
                            energy,
                        ),
                        axis=-1,
                    )
                    conserved.append(coarsen_cells(state, 8))
                    pressures.append(coarsen_cells(pressure, 8))
                    # EOS нелинейна по rho и momentum: усреднение испортило бы подгонку gamma.
                    if frame in (2, 10):
                        raw = np.stack(
                            (rho, momentum[..., 0], momentum[..., 1], energy, pressure),
                            axis=-1,
                        )
                        eos_samples[frame] = raw.reshape(-1, 5)[indices]
                    print(
                        f"The Well: кадр {frame:2d}, получено {remote.bytes_downloaded / 1e6:.1f} МБ",
                        flush=True,
                    )
                stored_trajectory_attribute = int(handle.attrs["n_trajectories"])
            transfer = {
                "bytes_downloaded": remote.bytes_downloaded,
                "remote_file_size": remote.size,
                "requests": len(remote.ranges),
                "ranges": remote.ranges,
            }
    selection = np.array(FRAME_INDICES, dtype=int)
    path = output_dir / "well_sample.npz"
    np.savez_compressed(
        path,
        time=source_time[selection] * 0.015,
        source_time=source_time[selection],
        frame_indices=selection,
        x=x.reshape(64, 8).mean(axis=1),
        y=y.reshape(64, 8).mean(axis=1),
        conserved=np.array(conserved),
        pressure=np.array(pressures),
        gamma=np.array(gamma),
        eos_train=eos_samples[2],
        eos_test=eos_samples[10],
        eos_flat_indices=indices,
    )
    return {
        "name": "The Well / Euler Multi-quadrants, open boundaries",
        "provider": "Polymathic AI; Ohana et al. (2024)",
        "url": WELL_URL,
        "revision": WELL_REVISION,
        "upstream_file_sha256": file_info.get("lfs", {}).get("oid"),
        "upstream_sha256_verified_by_full_download": False,
        "card_url": WELL_CARD_URL,
        "card_sha256": sha256_bytes(card),
        "license": "CC-BY-4.0",
        "license_source": "https://arxiv.org/html/2412.00568v2",
        "license_note": "The Well paper, datasheet question Q43; HF card has no license field.",
        "artifact": path.name,
        "artifact_sha256": sha256_bytes(path.read_bytes()),
        "artifact_bytes": path.stat().st_size,
        "selection": {
            "trajectory": 0,
            "frames": list(FRAME_INDICES),
            "spatial_reduction": "512x512 -> 64x64 by non-overlapping 8x8 cell means",
            "conserved_order": [
                "rho",
                "momentum_x",
                "momentum_y",
                "momentum_z",
                "total_energy_density",
            ],
            "eos_order": [
                "rho",
                "momentum_x",
                "momentum_y",
                "total_energy_density",
                "pressure",
            ],
            "eos_sampling": "1000 deterministic random cells (seed 11), raw frame 2 train / frame 10 test",
        },
        "schema": schema,
        "transfer": transfer,
        "coordinates": {
            "x_first": float(x[0]),
            "x_last": float(x[-1]),
            "x_step": float(x[1] - x[0]),
            "y_first": float(y[0]),
            "y_last": float(y[-1]),
            "y_step": float(y[1] - y[0]),
        },
        "physical_time_assumption": "physical_time = HDF5 dimensions/time * 0.015 s; interval from pinned dataset card",
        "warnings": [
            "Native data are 2D + time. momentum_z=0 is an explicit planar embedding, not an independent 3D observation.",
            "HDF5 time stores indices 0..100 with no time-unit attribute. Conversion to seconds uses the dataset card, not per-file timing metadata.",
            "Source x/y coordinates include both 0 and 1 (spacing 1/511), unlike finite-volume cell centers. The sample preserves their block means; a [0,1] finite-volume solve uses dx=dy=1/N from the documented domain.",
            f"Root n_trajectories={stored_trajectory_attribute} is inconsistent; actual first dimension is 10.",
            "Averaged reference pressure is retained separately; it is not generally the EOS pressure of the averaged conservative state.",
        ],
    }


def download_anu_cases(output_dir, case_ids=(3, 6, 12)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, sources = {}, []
    with requests.Session() as session:
        settings = _small_download(ANU_SETTINGS_URL, session)
        text = BeautifulSoup(settings, "html.parser").get_text(" ", strip=True)
        if not re.search(r"Gamma\s*=\s*1\.4", text):
            raise ValueError("ANU settings no longer confirm gamma=1.4")
        sources.append(
            {
                "name": "ANU Fyris / common Riemann settings",
                "url": ANU_SETTINGS_URL,
                "source_sha256": sha256_bytes(settings),
                "downloaded_bytes": len(settings),
            }
        )
        for case_id in case_ids:
            url = f"https://www.mso.anu.edu.au/fyris/lw2drtst{case_id:02d}.html"
            html = _small_download(url, session)
            case = parse_anu_case(html, case_id)
            case["url"] = url
            cases[str(case_id)] = case
            sources.append(
                {
                    "name": f"ANU Fyris / Riemann test {case_id}",
                    "url": url,
                    "source_sha256": sha256_bytes(html),
                    "downloaded_bytes": len(html),
                    "data_kind": "Published numeric initial states; no final trajectory downloaded",
                }
            )
    path = output_dir / "anu_cases.json"
    path.write_text(
        json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "name": "ANU Fyris / Liska-Wendroff initial-state tables",
        "sources": sources,
        "provider": "Ralph Sutherland, Australian National University",
        "artifact": path.name,
        "artifact_sha256": sha256_bytes(path.read_bytes()),
        "artifact_bytes": path.stat().st_size,
        "license_note": "Only factual numerical initial conditions are retained; no ANU code or figures are redistributed.",
    }


def download_data(output_dir):
    output_dir = Path(output_dir)
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [download_well_sample(output_dir), download_anu_cases(output_dir)],
    }
    path = output_dir / "sources.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_well_sample(data_dir):
    with np.load(Path(data_dir) / "well_sample.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_anu_case(case_id, data_dir):
    cases = json.loads((Path(data_dir) / "anu_cases.json").read_text(encoding="utf-8"))
    return cases[str(case_id)]


def verify_local_data(data_dir):
    """Сверяет SHA-256 включённых файлов с сохранённым манифестом."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "sources.json").read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        path = data_dir / source["artifact"]
        if sha256_bytes(path.read_bytes()) != source["artifact_sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
    return manifest
