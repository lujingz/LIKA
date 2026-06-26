# Reproducibility

This document records the commands and file layout used to reproduce the LIKA analyses for the manuscript `manuscript/network_inference_for_schizophrenia.pdf`.

## Environment

Create a Python environment and install the package set:

```bash
conda create -n LIKA python=3.10 -y
conda activate LIKA
pip install -r requirements.txt
```

The current `requirements.txt` was exported from a local conda environment. For archival release, prefer replacing it with a concise `environment.yml` or a package-only `requirements.txt`.

## Data

The processed input files used by the scripts are in `data/`:

- `data/KSEA_dataset_processed.csv`: kinase-substrate network derived from the KSEA App website.
- `data/intensity_data_INKA.csv`: processed cell-line phosphoproteomics input.
- `data/residual_data_SCZ.csv`: processed SCZ/control residual phosphosite data.

See `data/README.md` for the data file map and cleanup notes.

## Main Analyses

Run the SCZ/control pipeline:

```bash
python experiments/run_scz_pipeline.py
```

This writes LIKA outputs under `results/`, including:

- `results/SCZ_results.csv`
- `results/SCZ_network.graphml`
- `results/SCZ_p_values.json`
- `results/rejection_set_SCZ.txt`

Run the INKA cell-line pipeline:

```bash
python experiments/run_inka_pipeline.py
```

## Simulation Study

Run the manuscript simulation study with 100 repeats:

```bash
python experiments/simulation_experiment.py --runs 100 --output-dir results --max-k 10
```

Expected output files:

- `results/simulation_experiment_1_pvalue_rankings_100runs.csv`
- `results/simulation_experiment_2_pvalue_rankings_100runs.csv`
- `results/simulation_experiment_3_pvalue_rankings_100runs.csv`
- `results/simulation_precision_recall_by_run_100runs.csv`
- `results/simulation_precision_recall_summary_100runs.csv`
- `results/simulation_precision_recall_summary_100runs.png`

The LIKA simulation ranking is by capped kinase p-value, with ties broken by LIKA influence score, where the influence score is the effective downstream substrate count `sum_s 1 / parent_degree(s)`.

## Manuscript Assets

Audit the manuscript asset set:

```bash
python scripts/generate_manuscript_assets.py --audit
```

Regenerate manuscript ranking/subnetwork figures from the included supplementary ranking/substrate tables:

```bash
python scripts/generate_manuscript_assets.py --figures
```

Regenerate Figure 3 after the 100-run simulation summary exists:

```bash
python scripts/generate_manuscript_assets.py --figures --include-figure3
```

Regenerate supplementary tables that do not require slow kinase refits:

```bash
python scripts/generate_manuscript_assets.py --tables
```

To recompute S2/S5 kinase ranking tables from the processed input data, add `--recompute-rankings`. This can be slow because it refits LIKA.

Main manuscript figures are stored in `manuscript/figures/main/`:

- `figure_3_simulation_precision_recall.pdf`
- `figure_4a_inka_ksea_pvalue_ranking.pdf`
- `figure_4b_inka_lika_ranking.pdf`
- `figure_5_abl1_egfr_subnetworks.pdf`
- `figure_6a_scz_ksea_pvalue_ranking.pdf`
- `figure_6b_scz_lika_ranking.pdf`

Supplementary/diagnostic figures are stored in `manuscript/figures/supplementary/`:

- `supplementary_figure_fdr_sensitivity_top10_overlap.pdf`

Regenerate the FDR sensitivity analysis and heatmap:

```bash
python scripts/run_fdr_sensitivity_analysis.py
```

This writes detailed sensitivity outputs to `results/`, refreshes `manuscript/figures/supplementary/supplementary_figure_fdr_sensitivity_top10_overlap.pdf`, and writes the heatmap source data to `manuscript/supplementary_tables/supplementary_figure_fdr_sensitivity_top10_overlap_source_data.csv`.

An older alternate LIKA cell-line ranking plot is preserved in `manuscript/figures/archive/`.

Supplementary tables are stored in `manuscript/supplementary_tables/`:

- `supplementary_table_s1_simulation_network_structures.csv`
- `supplementary_table_s2_inka_kinase_rankings.csv`
- `supplementary_table_s3_scz_substrate_statistics.csv`
- `supplementary_table_s4_inka_substrate_statistics.csv`
- `supplementary_table_s5_scz_kinase_rankings.csv`

## Final Release Notes

- The kinase-substrate network is derived from the KSEA App website: https://casecpb.shinyapps.io/ksea/.
- The relevant KSEA App reference is *The KSEA App: a web-based tool for kinase activity inference from quantitative phosphoproteomics*.
- The full SCZ/control phosphoproteomics dataset is not deposited here because LIKA uses a subset of the measured phosphoproteome. The analyzed values used by LIKA should be provided in the manuscript supplement, either at subject level or as group-level summaries depending on final journal/data-use requirements. Additional SCZ/control data can be made available from the corresponding author upon reasonable request.
- Manuscript-specific generation and sensitivity analyses are kept in `scripts/` so the core LIKA implementation in `src/` remains focused.
- Keep generated outputs in `results/` out of version control unless the journal specifically asks for static result files.
