## LIKA: Likelihood Inference of Kinase Activity

This repository contains the code and processed analysis assets for the LIKA manuscript. LIKA infers kinase dysregulation from phosphoproteomic data while accounting for many-to-many kinase-substrate annotations.

## Repository Layout

- `src/`: reusable LIKA implementation and statistical methods.
- `experiments/`: executable analysis and simulation scripts.
- `data/`: processed input data used by the analysis scripts.
- `manuscript/`: manuscript PDF, main figures, supplementary figures, and supplementary tables.
- `scripts/`: manuscript-facing generation/audit utilities kept separate from the core LIKA implementation.
- `results/`: generated outputs from local runs. This directory is ignored by git.
- `REPRODUCIBILITY.md`: commands and file map for reproducing the manuscript analyses.

## Installation

```bash
conda create -n LIKA python=3.10 -y
conda activate LIKA
pip install -r requirements.txt
```

Some optional graph visualizations may require Graphviz system libraries.

## Run The Analyses

Run the SCZ/control analysis:

```bash
python experiments/run_scz_pipeline.py
```

Run the INKA cell-line analysis:

```bash
python experiments/run_inka_pipeline.py
```

Run the simulation study with the manuscript default of 100 repeats:

```bash
python experiments/simulation_experiment.py --runs 100 --output-dir results --max-k 10
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the expected outputs and manuscript asset mapping.

Audit manuscript assets:

```bash
python scripts/generate_manuscript_assets.py --audit
```

Regenerate the FDR sensitivity heatmap:

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
