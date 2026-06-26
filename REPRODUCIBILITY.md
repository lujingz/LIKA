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

- `data/KSEA_dataset_processed.csv`: kinase-substrate network filtered from the KSEA resource.
- `data/intensity_data_INKA.csv`: processed cell-line phosphoproteomics input.
- `data/residual_data_Schizo.csv`: processed schizophrenia-control residual phosphosite data.
- `data/Network_Data_for_Schizo.csv`, `data/category.csv`, and `data/Kinase_Substrate_Dataset.csv`: supporting network/source tables.

See `data/README.md` for the data file map and cleanup notes.

## Main Analyses

Run the schizophrenia pipeline:

```bash
python src/pipeline.py
```

This writes LIKA outputs under `results/`, including:

- `results/Schizo_results.csv`
- `results/Schizo_network.graphml`
- `results/Schizo_p_values.json`
- `results/rejection_set_Schizo.txt`

To run the INKA cell-line pipeline instead, call `pipeline_INKA()` from `src/pipeline.py` or switch the `__main__` block to `pipeline_INKA()`.

## Simulation Study

Run the manuscript simulation study with 100 repeats:

```bash
python src/simulation_experiment.py --runs 100 --output-dir results --max-k 10
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

## Notes For Final Journal Release

- Confirm the schizophrenia data availability statement and accession/controlled-access details.
- The FDR sensitivity script is available at `scripts/run_fdr_sensitivity_analysis.py`; it is slower than the table/figure formatting scripts because it refits LIKA across multiple first-stage FDR thresholds.
- Manuscript-specific generation and sensitivity analyses are kept in `scripts/` so the core LIKA implementation in `src/` remains focused.
- Keep generated outputs in `results/` out of version control unless the journal specifically asks for static result files.
