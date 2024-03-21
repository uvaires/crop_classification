# Cropclassification

## Description
This repository is developed to process Harmonized Landsat Sentinel-2 (HLS) data, create training samples using gridded, random, clustered, and stratified sampling techniques, and employ an Artificial Neural Network for crop type mapping. To execute the code, it is necessary to have a ground truth image used to create the samples. In this example, CropScape Data Layers (CDL) were used as the base to generate the samples.

## Dependencies management
The following comand can be used to recreate the conda enviroment with all the dependencies needed to run the code in this repository.
```
conda env create -f environment.yml
```
After creating the new enviroment (or you can use an existing one) you need to activate it and install the package in development mode. To do so, from the repository root, run the command below. It will install the package in development mode, so you can make changes to the code and test it without the need to reinstall the package.
```
pip install -e .
```




# crop_classification
