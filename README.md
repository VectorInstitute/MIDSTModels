# MIDSTModels

Welcome to the Reference Implementations for the MIDST Challenge (Membership Inference over Diffusion-models-based Synthetic Tabular Data) - SaTML 2025!

This repository provides code implementations and resources to support participants in the MIDST challenge, focusing on membership inference attacks over synthetic tabular data generated by diffusion models. Our implementations are based on the [Diffusion Model Bootcamp](https://github.com/VectorInstitute/diffusion_model_bootcamp/tree/main) provided by the Vector Institute.



## Repository Structure
The midst_models directory is the main folder of this repository, containing implementations and resources for the MIDST Challenge. It is organized into the following subdirectories:
- multi_table_ClavaDDPM/: Contains code and resources for multi-table data synthesis using the ClavaDDPM model. This implementation is designed for generating synthetic data across multiple relational tables using diffusion-based generative models.
- single_table_TabDDPM/:  Includes implementations for single-table data synthesis using the TabDDPM algorithm. This directory provides code, notebooks, and resources focused on applying TabDDPM to synthesize data from single-table datasets.
- single_table_TabSyn/: Hosts the code and resources related to the TabSyn algorithm for single-table data synthesis. It contains notebooks and examples demonstrating how to use TabSyn for generating synthetic tabular data and evaluating its performance.

## Getting Started

To get started with this repository, follow these steps:
1. Clone this repository to your machine.
2. Activate your python environment. You can create a new environment using the following command:
```bash
pip install --upgrade pip poetry
poetry env use [name of your python] #python3.9
source $(poetry env info --path)/bin/activate
poetry install --with "synthcity, tabsyn, clavaddpm, csdi, tsdiff"
# If your system is not compatible with pykeops, you can uninstall it using the following command
pip uninstall pykeops
# Install the kernel for jupyter (only need to do it once)
ipython kernel install --user --name=diffusion_models
```
3. Begin with each model in the `midst_models/` directory, as guided by the README files.

## License
This project is licensed under the terms of the [LICENSE] file located in the root directory of this repository.


## Contact Information
For more information or help with navigating this repository, please contact Vector AI ENG Team at sana.ayromlou@vectorinstitute.ai .
