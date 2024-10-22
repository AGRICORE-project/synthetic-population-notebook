# AGRICORE-synthetic-population-generation
This repository contains all the code necessary for generating synthetic populations for the Agricore project.
The repository is divided into several files and notebooks that load, process, and transform data to generate synthetic populations.
The module uses tabular data sourced from FADN entities and other relevant databases such as EUROSTAT and specific regional use case databases.
Specifically, four main use cases can be managed: Andalusia, Italy, Greece, and Poland. Additionally, for each use case, several years can be simulated depending on the specifics of the use case.  
  

## Notebooks
This folder contains a set of notebooks created for different purposes during the process of generating synthetic populations and assessing the goodness of fit of such synthetic data. Each notebook's purpose is described in the title. Some of the notebooks were not published due to privacy concerns, as they explicitly show sensitive information extracted from microdata.  
The notebooks in the "notebooks" folder are not used during the generation of the synthetic population but are part of the work done to develop the SPG generator.  
  

## src
This folder contains the main modules and Python scripts used during the process of generating the synthetic population. It includes different scripts aimed at specific purposes, such as standardizing variable nomenclature, generating a predefined DAG for the Bayesian network, dataset fusion, or reporting the synthetic population generation process.  
  

## Generate_synthetic_populaiton.ipynb
This notebook orchestrates the generation of synthetic populations. Users can set the use case or country for which the synthetic population will be generated. A set of accountancy years can be selected depending on the use case and the Agricore project's objectives. By default, it generates all use cases for all available years. The notebook imports the SPG_module and passes all the necessary arguments for an accurate simulation. With minimal user interaction, the orchestrator will manage all the artifacts needed and produced during this process.  
  
  
## Upload_synthetic_population.ipynb
This notebook orchestrates the generation of synthetic populations. Users can set the use case or country for which the synthetic population will be generated. A set of accountancy years can be selected depending on the use case and the Agricore project's objectives. By default, it generates all use cases for all available years. The notebook imports the SPG_module and passes all the necessary arguments for an accurate simulation. With minimal user interaction, the orchestrator will manage all the artifacts needed and produced during this process.  
  

## Data
To use this repository, a specific data folder structure must be created. This folder will contain a subfolder for each use case. Each use case folder, in turn, will contain several folders:

* microdata: where microdata files by year will be stored. The microdata files must be saved separately by year with a specific nomenclature according to the use case.  
* metadata: containing all metadata files, including configuration files, inferred results, categorical variables, defined product groups, livestock product groups, land value, and subsidies data.
For privacy and anonymity reasons, this folder is not public, as it contains FADN data.  
  