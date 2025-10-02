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

### 2) Run the Experiments

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

Please cite the datasets used to reproduce the experiments:

- [PLACEHOLDER] INKA dataset citation
- [PLACEHOLDER] Schizophrenia dataset citation

Add any required data download/placement instructions here (e.g., expected CSV paths under a `data/` directory) if the data are not versioned in the repository.

### Reproducibility Tips

- Ensure your `data/` directory contains the expected files referenced by `src/pipeline.py` and `src/simulation_experiment.py` (e.g., `data/intensity_data_INKA.csv`, `data/residual_data_Schizo.csv`, `data/KSEA_dataset_processed.csv`).
- Set a consistent random seed where applicable if you need exact reproducibility of simulation outputs.

### How to Cite

If you use LIKA in your work, please cite:

> [PLACEHOLDER] Full paper citation for the LIKA manuscript.


