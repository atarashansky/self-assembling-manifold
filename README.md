[![Build Status](https://travis-ci.com/atarashansky/self-assembling-manifold.svg?branch=master)](https://travis-ci.com/atarashansky/self-assembling-manifold)

# self-assembling-manifold
The Self-Assembling-Manifold (SAM) algorithm.

# Update (3/21/2019) -- SAM version 0.4.4

- Added Diffusion UMAP (`run_diff_umap()`) an experimental projection method in which UMAP is applied to a diffusion map calculated from the SAM nearest neighbor graph. Requires `scanpy`.
- Added a wrapper function for Leiden clustering, an improved version of Louvain clustering. Requires `scanpy`.
- Added yet another method for marker gene identification (in `identify_marker_genes_corr`). In the future, all the marker gene identification functions will be merged into a single function `identify_marker_genes`, with a parameter to choose which specific method to use.
- Can now directly load `h5ad` files (the native file format of AnnData) using `load_data`. `save_anndata` saves the `adata_raw` object to an `h5ad` file so that it can be used for faster loading in the future.
- Removed `save` and `load` functions as all of the SAM attributes are now contained within its `AnnData` objects, so saving the SAM attribute dictionary is no longer required. `sam.save_anndata(filename, data='adata')` should be used to save `sam.adata` to a `h5ad` file.

## Requirements
 - `numpy`
 - `scipy`
 - `pandas`
 - `scikit-learn`
 - `umap-learn`
 - `numba`
 - `anndata`

### Optional dependencies
 - Plotting
   - `matplotlib`
 - Clustering
   - `louvain`
   - `hdbscan`
 - `scanpy`

## Installation
SAM has been most extensively tested using python3.6 but presumably should work on python>=3.6. Python can be installed using Anaconda.

Download Anacodna from here:
    https://www.anaconda.com/download/

Create and activate a new environment with python3.6 as follows:
```
conda create -n environment_name python=3.6
conda activate environment_name
```

Having activated the environment, SAM can be downloaded from the PyPI repository using pip or, for the development version, downloaded from the github directly.

PIP install:
```
pip install sam-algorithm
```

Development version install:
```
git clone https://github.com/atarashansky/self-assembling-manifold.git
cd self-assembling-manifold
python setup.py install
```

## Tutorial
Please see the Jupyter notebooks in the 'tutorial' folder for basic tutorials. If you installed a fresh environment, do not forget to install jupyter into that environment! Please run
```
pip install jupyter
```
in your conda environment. The tutorial assumes that all optional dependencies are installed.

# Basic usage

There are a number of different ways to load data into the SAM object. 

## Using the SAM constructor
### Using preloaded scipy.sparse or numpy expression matrix, gene IDs, and cell IDs:
```
from SAM import SAM #import SAM
sam=SAM(counts=(matrix,geneIDs,cellIDs))
sam.preprocess_data() # log transforms and filters the data
sam.run() #run with default parameters
sam.scatter() #display resulting UMAP plot
```
### Using preloaded pandas.DataFrame (cells x genes):
```
from SAM import SAM #import SAM
sam=SAM(counts=dataframe)
sam.preprocess_data() # log transforms and filters the data
sam.run() #run with default parameters
sam.scatter() #display resulting UMAP plot
```

### Using an existing AnnData object:
```
from SAM import SAM #import SAM
sam=SAM(counts=adata)
sam.preprocess_data() # log transforms and filters the data
sam.run() #run with default parameters
sam.scatter() #display resulting UMAP plot
```

## Using the `load_data` function
### Loading data from a tabular file (e.g. csv or txt):
```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data('/path/to/expression_data_file.csv') #load data from a csv file
#sam.load_data('/path/to/expression_data_file.txt', sep='\t') #load data from a txt file with tab delimiters
sam.preprocess_data() # log transforms and filters the data
sam.load_annotations('/path/to/annotations_file.csv')
sam.run()
sam.scatter()
```
### Loading an existing AnnData `h5ad` file: 

If loading tabular data (e.g. from a `csv`), `load_data` by default saves the sparse data structure to a `h5ad` file in the same location as the tabular file for faster loading in subsequent analyses. This file can be loaded as:

```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data('/path/to/h5ad_file.h5ad') #load data from a h5ad file
sam.preprocess_data() # log transforms and filters the data
sam.run()
sam.scatter()
```

If you wish to save the SAM outputs and filtered data, you can write `sam.adata` to a `h5ad` file as follows:
`sam.save_anndata(filename, data = 'adata')`.

If for whatever reason you wish to save the raw, unfiltered AnnData object,
`sam.save_anndata(filename, data = 'adata_raw')`.

## Citation
If using the SAM algorithm, please cite the following preprint:
https://www.biorxiv.org/content/early/2018/07/07/364166

## Adding extra functionality
As always, please submit a new issue if you would like to see any functionalities / convenience functions / etc added.
