## LIKA: Likelihood Inference of Kinase Activity

This repository contains the code and processed analysis assets for the LIKA manuscript. LIKA infers kinase dysregulation from phosphoproteomic data while accounting for many-to-many kinase-substrate annotations.

## Repository Layout

- `src/`: reusable LIKA implementation and statistical methods.
- `scripts/`: reproducibility entry points and manuscript asset-generation utilities.
- `data/`: processed input data used by the analysis scripts.
- `manuscript/`: manuscript PDF, main figures, supplementary figures, and supplementary tables.
- `REPRODUCIBILITY.md`: commands and file map for reproducing the manuscript analyses.

## Installation

```bash
conda env create -f environment.yml
conda activate lika
```

Alternatively, install the Python dependencies directly:

```bash
python -m pip install -r requirements.txt
```

The manuscript scripts were tested with Python 3.11. The dependency files are
kept concise and platform-independent for review and archival release.

## Quick Validation

After installation, check that the committed manuscript assets are present:

```bash
python scripts/generate_manuscript_assets.py --audit
```

To verify the plotting and table-generation paths without refitting LIKA/KSEA,
run:

```bash
python scripts/generate_scz_assets.py --use-existing-ranking
python scripts/generate_inka_assets.py --use-existing-ranking
python scripts/run_fdr_sensitivity_analysis.py --use-existing-source-data
```

## Reproducibility Scripts

The repository has three explicit reproducibility settings. These scripts write
only the manuscript-facing files needed for reproducibility under `manuscript/`.
They do not export rejection sets, GraphML files, or other exploratory result
objects.

SCZ/control analysis:

```bash
python scripts/generate_scz_assets.py
```

INKA cell-line analysis:

```bash
python scripts/generate_inka_assets.py
```

Simulation experiment with the manuscript default of 100 repeats:

```bash
python scripts/generate_simulation_assets.py --runs 100 --max-k 10
```

The simulation script can optionally write per-run intermediate CSV files with
`--write-intermediate-dir`, but this is off by default because the manuscript
requires only the summarized Figure 3 source data.

Audit manuscript assets:

```bash
python scripts/generate_manuscript_assets.py --audit
```

Regenerate the FDR sensitivity heatmap:

```bash
python scripts/run_fdr_sensitivity_analysis.py --use-existing-source-data
```

Fully recompute the FDR sensitivity analysis:

```bash
python scripts/run_fdr_sensitivity_analysis.py
```

## Manuscript Assets

The manuscript PDF is stored at `manuscript/network_inference_for_schizophrenia.pdf`.

Main figures are in `manuscript/figures/main/`, supplementary figures are in `manuscript/figures/supplementary/`, and supplementary tables are in `manuscript/supplementary_tables/`.

## Data Sources And Availability

The kinase-substrate network used by LIKA was derived from the KSEA App website: https://casecpb.shinyapps.io/ksea/. The relevant KSEA App reference is *The KSEA App: a web-based tool for kinase activity inference from quantitative phosphoproteomics*.

For the SCZ/control phosphoproteomics analysis, the full phosphosite dataset is not deposited here because LIKA uses a subset of the measured phosphoproteome. The processed values used by the analysis are provided in `data/residual_data_SCZ.csv`, and the manuscript supplement should include the analyzed values used by LIKA, either at subject level or as group-level summaries depending on the final journal/data-use requirements. Additional SCZ/control data can be made available from the corresponding author upon reasonable request.

## Citation

If you use this repository, cite the LIKA manuscript once the final journal citation is available.

The cell-line data are derived from:

Beekhof, R., van Alphen, C., Henneman, A. A., Knol, J. C., Pham, T. V., Rolfs, F., Labots, M., Henneberry, E., Le Large, T. Y. S., de Haas, R. R., Piersma, S. R., Vurchio, V., Bertotti, A., Trusolino, L., Verheul, H. M. W., Jimenez, C. R., & Altelaar, A. F. M. (2019). INKA, an integrative data analysis pipeline for phosphoproteomic inference of active kinases. *Molecular Systems Biology*, 15(4), e8250. https://doi.org/10.15252/msb.20188526
