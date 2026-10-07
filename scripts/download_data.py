"""Повторно получить небольшую выборку из двух независимых источников."""

from argparse import ArgumentParser
from pathlib import Path

from euler_lab.data import download_data


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    args = parser.parse_args()
    manifest = download_data(args.output)
    for source in manifest["sources"]:
        print(f"Сохранено: {source['artifact']} ({source['artifact_bytes']:,} байт)")
