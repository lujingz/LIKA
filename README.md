## LIKA: Inferring Disruption of Directed Graphs in Phosphorylation Networks

This repository contains the code for the project “Inferring disruption of directed graphs using LIKA reveals altered protein phosphorylation networks in schizophrenia.” It includes the full LIKA pipeline, simulation experiments, and utilities for analyzing and visualizing kinase–substrate networks.

### 1) Environment Setup

Prerequisites:
- Conda (Anaconda/Miniconda)
- Python (recommended via Conda environment)

Steps:
```bash
# 1) Create and activate the LIKA environment
conda create -n LIKA python=3.10 -y
conda activate LIKA

# 2) Install Python dependencies
pip install -r requirements.txt
```

Notes:
- If you use a different Python version, ensure package compatibility and re-generate your environment if needed.
- Some optional visualizations may require system packages like Graphviz; see error messages for guidance.

### 2) Download the Data?

I processed the raw data first, should I just put the processed data in the repository instead of asking people to download from the orinigal sources and process the data themselves?

### 3) Run the Experiments

1. Initialize and activate the LIKA environment:
```bash
conda activate LIKA
```

2. Run the experiments on the INKA dataset (produces results under `results/`):
```bash
python src/pipeline.py
```

3. Run the simulation experiments described in the paper (also writes outputs under `results/`):
```bash
python src/simulation_experiment.py
```

Outputs (examples):
- Processed results (CSV)
- Network files (GraphML)
- Figures/plots (PNG)

### Project Structure

- `src/`
  - `pipeline.py`: Main LIKA pipeline over experimental datasets (e.g., INKA, Schizo).
  - `simulation_experiment.py`: Reproducible synthetic/simulation experiments used in the paper.
  - `method.py`: Core statistical methods for LIKA (likelihood profiling, empirical Bayes, ranking).
  - `utils.py`: Network utilities, visualization helpers, and I/O utilities.
  - `vash.py`: Variance Adaptive Shrinkage (Empirical Bayes) implementation used by the pipeline.
- `experiments/`: (Optional) experiment artifacts or scripts.
- `results/`: Generated outputs (created on first run).

### Data Availability and Citation

- For cell line data, the dataset is derived from [INKA: an integrative data analysis pipeline for phosphoproteomic inference of active kinases](https://doi.org/10.1038/ncomms12851).

- For Schizophrenia data, the dataset is derived from []


### How to Cite

If you use LIKA in your work, please cite:

> [PLACEHOLDER] Full paper citation for the LIKA manuscript.

Please cite the datasets used to reproduce the experiments:

Wilkes, E. H., Terfve, C., Gribben, J. G., Saez-Rodriguez, J., & Cutillas, P. R. (2016).  
INKA, an integrative data analysis pipeline for phosphoproteomic inference of active kinases.  
*Nature Communications*, 7, 12851. https://doi.org/10.1038/ncomms12851

In the INKA paper, they mention that 'To further test our strategy for prioritizing active kinases, we also examined phosphoproteome data of oncogene-driven cell lines from the literature (Guo et al, 2008; Bai et al, 2012; Fig 4, Dataset EV4). INKA analysis of data on EGFR-mutant NSCLC cell line H3255 (Guo et al, 2008) uncovered major EGFR activity in these cells, with EGFR ranking first, followed by MET (Fig 4A). In another study, the rhab- domyosarcoma-derived cell line A204 was associated with PDGFRa signaling (Bai et al, 2012), and INKA scoring of the underlying data accordingly ranks PDGFRa in second place (Fig 4B). In the same study, osteosarcoma-derived MNNG/HOS cells were shown to be dependent on MET signaling and sensitive to MET inhibitors (Bai et al, 2012). In line with this, INKA analysis clearly pinpointed MET as the major driver candidate in this cell line (Fig 4C).' 

In a nutshell, they are using data from somewhere else. How should I cite them?