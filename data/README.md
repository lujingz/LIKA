# Data Files

This folder contains processed inputs and source/support tables used by the LIKA manuscript analyses.

## Required Processed Inputs

- `KSEA_dataset_processed.csv`: two-column kinase-substrate edge list used by the LIKA pipeline. Columns are `from` kinase and `to` phosphosite.
- `intensity_data_INKA.csv`: processed INKA cell-line phosphosite intensity table. The pipeline lower-cases columns at runtime; expected logical columns are `Name`, `Group`, and `intensity_*`.
- `residual_data_SCZ.csv`: processed schizophrenia-control residual phosphosite table. Expected logical columns are `Name`, `Group`, and `Intensity_*`.

## Source And Supporting Tables

- `Kinase_Substrate_Dataset.csv`: larger source kinase-substrate annotation table used to derive `KSEA_dataset_processed.csv`.
- `Network_Data_for_Schizo.csv`: supporting network table for schizophrenia-related analyses.
- `category.csv`: node category/description support table.

## Notes

- Zero intensity values are treated as missing values by the current preprocessing code.
- The committed files are already processed. Raw-data preprocessing and schizophrenia covariate residualization are not fully represented in this repository yet.
- No data files were removed during cleanup because each current file is either a direct pipeline input or a source/support table. Generated outputs should stay in `results/`, which is ignored by git.
