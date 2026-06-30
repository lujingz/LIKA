# Reproducibility

This document records the commands and file layout used to reproduce the LIKA analyses for the manuscript `manuscript/network_inference_for_schizophrenia.pdf`.

## Environment

Create a Python environment and install the package set:

```bash
conda env create -f environment.yml
conda activate lika
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

The environment is specified with Python 3.11 and a concise, platform-independent dependency list.

## Data

The processed input files used by the scripts are in `data/`:

- `data/KSEA_dataset_processed.csv`: kinase-substrate network derived from the KSEA App website.
- `data/intensity_data_INKA.csv`: processed cell-line phosphoproteomics input.
- `data/residual_data_SCZ.csv`: processed SCZ/control residual phosphosite data.

See `data/README.md` for the data file map and cleanup notes.

## Reproducibility Settings

The repository uses `scripts/` as the single location for reproducibility commands. The three primary settings are SCZ/control, INKA, and the simulation experiment. These scripts generate only manuscript-facing source data, figures, and supplementary tables under `manuscript/`.

The repository does not keep code paths for exporting rejection sets, GraphML files, or broad exploratory result folders because those files are not required to reproduce the manuscript.

### Setting 1: SCZ/control

```bash
python scripts/generate_scz_assets.py
```

This uses `data/residual_data_SCZ.csv` to regenerate:

- `manuscript/supplementary_tables/supplementary_table_s3_scz_substrate_statistics.csv`
- `manuscript/supplementary_tables/supplementary_table_s5_scz_kinase_rankings.csv`
- `manuscript/figures/main/figure_6a_scz_ksea_pvalue_ranking.pdf`
- `manuscript/figures/main/figure_6b_scz_lika_ranking.pdf`

Use `--use-existing-ranking` to regenerate S3 and Figure 6A/B from the committed S5 ranking table without refitting LIKA/KSEA.

### Setting 2: INKA cell-line

```bash
python scripts/generate_inka_assets.py
```

This uses `data/intensity_data_INKA.csv` to regenerate:

- `manuscript/supplementary_tables/supplementary_table_s2_inka_kinase_rankings.csv`
- `manuscript/supplementary_tables/supplementary_table_s4_inka_substrate_statistics.csv`
- `manuscript/figures/main/figure_4a_inka_ksea_pvalue_ranking.pdf`
- `manuscript/figures/main/figure_4b_inka_lika_ranking.pdf`
- `manuscript/figures/main/figure_5_abl1_egfr_subnetworks.pdf`

Use `--use-existing-ranking` to regenerate S4 and Figures 4/5 from the committed S2 ranking table without refitting LIKA/KSEA.

### Setting 3: Simulation experiment

```bash
python scripts/generate_simulation_assets.py --runs 100 --max-k 10
```

This reruns the 100-repeat simulation experiment and regenerates:

- `manuscript/supplementary_tables/supplementary_table_s1_simulation_network_structures.csv`
- `manuscript/supplementary_tables/figure_3_simulation_precision_recall_source_data.csv`
- `manuscript/figures/main/figure_3_simulation_precision_recall.pdf`

Use `--write-intermediate-dir /path/to/dir` only if you also want per-run ranking and metric CSV files for inspection. Those intermediate files are not required for manuscript reproducibility and are not written by default.

The LIKA simulation ranking is by capped kinase p-value, with ties broken by LIKA influence score, where the influence score is the effective downstream substrate count `sum_s 1 / parent_degree(s)`.

## Manuscript Assets

Audit the manuscript asset set:

```bash
python scripts/generate_manuscript_assets.py --audit
```

Asset generation is split by reproducibility setting:

- SCZ/control: `python scripts/generate_scz_assets.py`
- INKA: `python scripts/generate_inka_assets.py`
- Simulation: `python scripts/generate_simulation_assets.py --runs 100 --max-k 10`
- FDR sensitivity: `python scripts/run_fdr_sensitivity_analysis.py --use-existing-source-data`

Main manuscript figures are stored in `manuscript/figures/main/`:

- `figure_3_simulation_precision_recall.pdf`
- `figure_4a_inka_ksea_pvalue_ranking.pdf`
- `figure_4b_inka_lika_ranking.pdf`
- `figure_5_abl1_egfr_subnetworks.pdf`
- `figure_6a_scz_ksea_pvalue_ranking.pdf`
- `figure_6b_scz_lika_ranking.pdf`

Supplementary/diagnostic figures are stored in `manuscript/figures/supplementary/`:

- `supplementary_figure_fdr_sensitivity_top10_overlap.pdf`

Regenerate the FDR sensitivity heatmap from the committed source-data table:

```bash
python scripts/run_fdr_sensitivity_analysis.py --use-existing-source-data
```

Fully recompute the FDR sensitivity analysis and heatmap:

```bash
python scripts/run_fdr_sensitivity_analysis.py
```

Both commands refresh `manuscript/figures/supplementary/supplementary_figure_fdr_sensitivity_top10_overlap.pdf` and write the heatmap source data to `manuscript/supplementary_tables/supplementary_figure_fdr_sensitivity_top10_overlap_source_data.csv`. The full recomputation refits LIKA across first-stage FDR levels. Add `--output-dir /path/to/dir` only if you also want intermediate sensitivity matrices and diagnostic heatmaps.

Supplementary tables are stored in `manuscript/supplementary_tables/`:

- `supplementary_table_s1_simulation_network_structures.csv`
- `supplementary_table_s2_inka_kinase_rankings.csv`
- `supplementary_table_s3_scz_substrate_statistics.csv`
- `supplementary_table_s4_inka_substrate_statistics.csv`
- `supplementary_table_s5_scz_kinase_rankings.csv`

Additional figure source-data tables in the same folder:

- `figure_3_simulation_precision_recall_source_data.csv`
- `supplementary_figure_fdr_sensitivity_top10_overlap_source_data.csv`

## Final Release Notes

- The kinase-substrate network is derived from the KSEA App website: https://casecpb.shinyapps.io/ksea/.
- The relevant KSEA App reference is *The KSEA App: a web-based tool for kinase activity inference from quantitative phosphoproteomics*.
- The full SCZ/control phosphoproteomics dataset is not deposited here because LIKA uses a subset of the measured phosphoproteome. The analyzed values used by LIKA should be provided in the manuscript supplement, either at subject level or as group-level summaries depending on final journal/data-use requirements. Additional SCZ/control data can be made available from the corresponding author upon reasonable request.
- Reproducibility entry points and manuscript-specific generation scripts are kept in `scripts/` so the core LIKA implementation in `src/` remains focused.
- Keep optional intermediate outputs outside the repository unless the journal specifically asks for additional static result files.
