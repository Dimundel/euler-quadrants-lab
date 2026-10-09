"""Выполнить ноутбук и сохранить результаты в нём."""

from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ["MPLCONFIGDIR"] = str(root / "data" / "cache" / "matplotlib")
    os.environ["PYTHONPATH"] = str(root / "src")
    # Ядро должно использовать то же окружение, что и команда запуска.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    notebook_path = root / "notebooks" / "lab11.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Выполнены {sum(c.cell_type == 'code' for c in notebook.cells)} ячеек")
    print(f"Ноутбук: {notebook_path.relative_to(root)}")


if __name__ == "__main__":
    main()
