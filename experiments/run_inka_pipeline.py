#!/usr/bin/env python
"""
Run the LIKA pipeline on the INKA cell-line dataset.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lika-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lika-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from pipeline import pipeline
from utils import plot_top_kinases


def main():
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    intensity_df = pd.read_csv(ROOT / "data" / "intensity_data_INKA.csv")
    network, rejection_set, results_df, top_kinases, p_values = pipeline(intensity_df)

    (output_dir / "rejection_set_INKA.txt").write_text(",".join(rejection_set))
    with open(output_dir / "INKA_p_values.json", "w") as f:
        json.dump(p_values, f, indent=2)
    network.save_to_graphml(output_dir / "INKA_network.graphml")
    results_df.to_csv(output_dir / "INKA_results.csv", index=False)
    plot_top_kinases(results_df, top_kinases)


if __name__ == "__main__":
    main()
