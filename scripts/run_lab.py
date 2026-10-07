"""Выполнить ноутбук и сохранить HTML с теми же результатами."""

from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


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
    exporter = HTMLExporter(template_name="lab")
    html, _ = exporter.from_notebook_node(notebook)
    output = root / "results" / "lab11.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Выполнены {sum(c.cell_type == 'code' for c in notebook.cells)} ячеек")
    print(f"HTML: {output.relative_to(root)}")


if __name__ == "__main__":
    main()
