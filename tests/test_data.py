import re
from pathlib import Path

import numpy as np
import pytest

from euler_lab.data import (
    HTTPRangeReader,
    coarsen_cells,
    load_anu_case,
    load_well_sample,
    parse_anu_case,
    verify_local_data,
)


class FakeResponse:
    def __init__(self, payload, start, end, total, *, status=206, wrong_range=False, truncated=False):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start + int(wrong_range)}-{end}/{total}"}
        self.payload = payload[:-1] if truncated else payload
        self.body_read = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        self.body_read = True
        yield self.payload


class FakeSession:
    def __init__(self, data=b"abcdefghijklmnop", **response_options):
        self.data = data
        self.options = response_options
        self.responses = []

    def get(self, url, *, headers, stream, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        end = min(end, len(self.data) - 1)
        response = FakeResponse(self.data[start : end + 1], start, end, len(self.data), **self.options)
        self.responses.append(response)
        return response


def test_range_reader_seeks_caches_and_reads_eof():
    session = FakeSession()
    with HTTPRangeReader("https://example.org/data", block_size=4, max_bytes=16, session=session) as reader:
        assert reader.read(6) == b"abcdef"
        reader.seek(2)
        assert reader.read(4) == b"cdef"
        assert len(session.responses) == 2
        reader.seek(-2, 2)
        target = bytearray(4)
        assert reader.readinto(target) == 2
        assert target[:2] == b"op"
        assert reader.read(1) == b""
        assert reader.bytes_downloaded == 12
        assert all(len(entry["sha256"]) == 64 for entry in reader.ranges)


def test_range_reader_rejects_full_file_response_before_reading_body():
    session = FakeSession(status=200)
    with pytest.raises(RuntimeError, match="Expected HTTP 206"):
        HTTPRangeReader("https://example.org/data", block_size=4, max_bytes=16, session=session)
    assert not session.responses[0].body_read


@pytest.mark.parametrize("options, message", [
    ({"wrong_range": True}, "different byte range"),
    ({"truncated": True}, "Truncated"),
])
def test_range_reader_rejects_corrupted_response(options, message):
    with pytest.raises(RuntimeError, match=message):
        HTTPRangeReader("https://example.org/data", block_size=4, max_bytes=16,
                        session=FakeSession(**options))


def test_range_reader_enforces_total_transfer_budget():
    with HTTPRangeReader("https://example.org/data", block_size=4, max_bytes=8,
                         session=FakeSession()) as reader:
        with pytest.raises(RuntimeError, match="entire remote file"):
            reader.read()
        reader.seek(4)
        assert reader.read(4) == b"efgh"
        with pytest.raises(RuntimeError, match="budget exceeded"):
            reader.read(1)
        assert reader.bytes_downloaded == 8


def test_coarsening_preserves_cell_integrals_for_all_conservative_fields():
    values = np.random.default_rng(1).normal(size=(12, 8, 5))
    coarse = coarsen_cells(values, 4)
    assert coarse.shape == (3, 2, 5)
    np.testing.assert_allclose(coarse.sum(axis=(0, 1)) * 16, values.sum(axis=(0, 1)), atol=1e-13)
    with pytest.raises(ValueError, match="divisible"):
        coarsen_cells(values, 3)


def test_anu_parser_uses_labels_instead_of_assuming_row_order():
    html = """<table><tr><td>T_final</td><td>0.3</td></tr>
    <tr><td>Upper Left</td><td>Upper Right</td></tr>
    <tr><td>P</td><td>0.3</td><td>P</td><td>1.5</td></tr>
    <tr><td>d</td><td>0.5</td><td>d</td><td>1.5</td></tr>
    <tr><td>v_x</td><td>1.2</td><td>v_x</td><td>0</td></tr>
    <tr><td>v_y</td><td>0</td><td>v_y</td><td>0</td></tr>
    <tr><td>Lower Left</td><td>Lower Right</td></tr>
    <tr><td>d</td><td>0.1</td><td>d</td><td>0.5</td></tr>
    <tr><td>P</td><td>0.03</td><td>P</td><td>0.3</td></tr>
    <tr><td>v_x</td><td>1.2</td><td>v_x</td><td>0</td></tr>
    <tr><td>v_y</td><td>1.2</td><td>v_y</td><td>1.2</td></tr></table>"""
    case = parse_anu_case(html, 3)
    assert case["quadrants"]["upper_left"] == {"rho": 0.5, "p": 0.3, "u": 1.2, "v": 0.0, "w": 0.0}
    assert case["primitive"][0] == [1.5, 0.0, 0.0, 0.0, 1.5]
    with pytest.raises(ValueError, match="Incomplete"):
        parse_anu_case(html.replace("v_y", "missing"), 3)


def test_bundled_data_checksums_shape_and_raw_equation_of_state():
    directory = Path(__file__).resolve().parents[1] / "data" / "processed"
    manifest = verify_local_data(directory)
    sample = load_well_sample(directory)
    assert sample["conserved"].shape == (7, 64, 64, 5)
    assert sample["pressure"].shape == (7, 64, 64)
    np.testing.assert_allclose(sample["time"], sample["source_time"] * 0.015)
    assert np.all(sample["conserved"][..., 3] == 0.0)
    for split in ("eos_train", "eos_test"):
        rho, mx, my, energy, pressure = sample[split].T
        predicted = (sample["gamma"] - 1) * (energy - (mx * mx + my * my) / (2 * rho))
        np.testing.assert_allclose(predicted, pressure, rtol=1e-5, atol=1e-6)
    assert manifest["sources"][0]["transfer"]["bytes_downloaded"] < 100_000_000
    case = load_anu_case(3, directory)
    assert case["quadrants"]["upper_left"]["rho"] == 0.5323
    assert case["quadrants"]["lower_left"]["p"] == 0.029
