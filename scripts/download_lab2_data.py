"""Загрузить опубликованные данные с давлением для ЛР №2."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from navier_lab.data import prepare_cylinder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mat", type=Path, help="Использовать уже скачанный оригинал")
    args = parser.parse_args()
    result = prepare_cylinder(ROOT / "data" / "processed", args.source_mat)
    print(result["name"], result["shape"])
