"""Run lab 3 experiments and optionally execute the report notebook."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Small dense systems are faster with one BLAS thread, and reproducible here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-notebook",action="store_true")
    parser.add_argument("--plot-only",action="store_true",help="Rebuild figures from saved search records")
    parser.add_argument("--n",type=int,default=96,help="Synthetic grid size (default 96)")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    from graph_lab.experiments import run_experiments, rebuild_plots
    if args.plot_only:
        rebuild_plots(root)
    else:
        run_experiments(root,n=args.n)
    if args.execute_notebook:
        import nbformat
        from nbclient import NotebookClient
        notebook_path=root/"notebooks"/"lab3.ipynb"
        notebook=nbformat.read(notebook_path,as_version=4)
        NotebookClient(notebook,timeout=600,kernel_name="python3",
                       resources={"metadata":{"path":str(root)}}).execute()
        nbformat.write(notebook,notebook_path)


if __name__=="__main__":
    main()
