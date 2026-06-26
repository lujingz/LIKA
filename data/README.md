# Data Files

This folder contains processed inputs and source/support tables used by the LIKA manuscript analyses.

## Required Processed Inputs

- `KSEA_dataset_processed.csv`: two-column kinase-substrate edge list used by the LIKA pipeline. Columns are `from` kinase and `to` phosphosite. This network was derived from the KSEA App website: https://casecpb.shinyapps.io/ksea/.
- `intensity_data_INKA.csv`: processed INKA cell-line phosphosite intensity table. The pipeline lower-cases columns at runtime; expected logical columns are `Name`, `Group`, and `intensity_*`.
- `residual_data_SCZ.csv`: processed SCZ/control residual phosphosite table. Expected logical columns are `Name`, `Group`, and `Intensity_*`.

## Data Availability Notes

- The KSEA App reference for the kinase-substrate network is *The KSEA App: a web-based tool for kinase activity inference from quantitative phosphoproteomics*.
- The full SCZ/control phosphoproteomics dataset is not deposited in this repository because the LIKA analysis uses a subset of the measured phosphoproteome.
- `residual_data_SCZ.csv` contains the processed subject-level values used by the SCZ/control LIKA analysis. The manuscript supplement should include the analyzed values used by LIKA, either at subject level or as group-level summaries depending on the final journal/data-use requirements.
- Additional SCZ/control data can be made available from the corresponding author upon reasonable request.

## Notes

- Zero intensity values are treated as missing values by the current preprocessing code.
- The committed files are already processed. Raw-data preprocessing and SCZ/control covariate residualization are not fully represented in this repository yet.
- Each current data file is a direct pipeline input. Generated outputs should stay in `results/`, which is ignored by git.
