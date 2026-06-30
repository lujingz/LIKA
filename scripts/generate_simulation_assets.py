#!/usr/bin/env python
"""
Generate manuscript assets for the LIKA simulation setting.

The default command writes only files needed for manuscript reproducibility:
Supplementary Table S1, Figure 3 source data, and main Figure 3.
"""

import argparse
from pathlib import Path

import pandas as pd

from manuscript_assets import (
    FIGURE3_SOURCE_DATA,
    FIGURE_DIR,
    ROOT,
    SIMULATION_TABLE,
    ensure_dirs,
    figure_3_from_summary,
    write_simulation_structure_table,
)
from simulation_methods import (
    DEFAULT_BASE_SEED,
    DEFAULT_MAX_K,
    DEFAULT_RUNS,
    experiment_1_inputs,
    experiment_2_inputs,
    experiment_3_inputs,
    precision_recall_at_k,
    run_one_experiment,
    summarize_precision_recall,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of repeats per simulation.")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Base seed used to derive deterministic run seeds.",
    )
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K, help="Largest k for precision@k/recall@k.")
    parser.add_argument(
        "--figure3-source-data",
        type=Path,
        default=FIGURE3_SOURCE_DATA,
        help="Output CSV for Figure 3 source data.",
    )
    parser.add_argument(
        "--figure3-output",
        type=Path,
        default=FIGURE_DIR / "figure_3_simulation_precision_recall.pdf",
        help="Output PDF for main Figure 3.",
    )
    parser.add_argument(
        "--write-intermediate-dir",
        type=Path,
        default=None,
        help="Optional directory for per-run ranking and metric CSVs. Defaults to no intermediate files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    outputs = [write_simulation_structure_table(SIMULATION_TABLE)]
    experiments = {
        1: experiment_1_inputs,
        2: experiment_2_inputs,
        3: experiment_3_inputs,
    }

    ranking_outputs = []
    for experiment_number, input_fn in experiments.items():
        result_df = run_one_experiment(
            experiment_number,
            input_fn,
            runs=args.runs,
            base_seed=args.base_seed,
        )
        ranking_outputs.append(result_df)
        if args.write_intermediate_dir is not None:
            args.write_intermediate_dir.mkdir(parents=True, exist_ok=True)
            path = args.write_intermediate_dir / f"simulation_experiment_{experiment_number}_pvalue_rankings_{args.runs}runs.csv"
            result_df.to_csv(path, index=False)
            outputs.append(path)

    all_rankings = pd.concat(ranking_outputs, ignore_index=True)
    metric_df = precision_recall_at_k(all_rankings, max_k=args.max_k)
    summary_df = summarize_precision_recall(metric_df)

    args.figure3_source_data.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.figure3_source_data, index=False)
    outputs.append(args.figure3_source_data)

    if args.write_intermediate_dir is not None:
        metric_path = args.write_intermediate_dir / f"simulation_precision_recall_by_run_{args.runs}runs.csv"
        summary_path = args.write_intermediate_dir / f"simulation_precision_recall_summary_{args.runs}runs.csv"
        metric_df.to_csv(metric_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        outputs.extend([metric_path, summary_path])

    outputs.append(figure_3_from_summary(args.figure3_source_data, args.figure3_output))

    for path in outputs:
        try:
            label = path.relative_to(ROOT)
        except ValueError:
            label = path
        print(f"Wrote {label}")


if __name__ == "__main__":
    main()
