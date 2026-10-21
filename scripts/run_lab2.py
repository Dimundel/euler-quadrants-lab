"""Выполнить отчёт ЛР №2 и сохранить графики и метрики."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    os.environ['MPLCONFIGDIR'] = str(ROOT/'data'/'cache'/'matplotlib')
    os.environ['PYTHONPATH'] = str(ROOT/'src')
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    path = ROOT/'notebooks'/'lab2.ipynb'
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, path)
    print('ЛР №2 выполнена:', path.relative_to(ROOT))
