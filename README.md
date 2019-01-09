[![Build Status](https://travis-ci.com/atarashansky/self-assembling-manifold.svg?branch=master)](https://travis-ci.com/atarashansky/self-assembling-manifold)

# self-assembling-manifold
The Self-Assembling-Manifold (SAM) algorithm.

# Update (1/8/2019) -- SAM version 0.3.0

What was previously 'SAMsparse' is now just 'SAM'. Refer to the below code snippets or the updated tutorial notebook to see any usage changes (mainly, the loading and filtering functions in SAMsparse changed names to match their counterparts in the old SAM). Other tweaks here and there have been made to the SAM algorithm to improve convergence stability, run-time performance, etc.

## Requirements
 - `numpy>=1.14,<=1.15.2`
 - `scipy`
 - `pandas`
 - `scikit-learn==0.20.0`
 - `umap-learn<=0.3.6`
 - `numba>=0.37,<0.40`

### Optional dependencies
 - Plotting
   - `matplotlib`
 - Louvain clustering
   - `louvain`

## Installation
SAM has been most extensively tested using python3.6 but presumably should work on python>=3.5. Python can be installed using Anaconda.

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
Please see the Jupyter notebook in the 'tutorial' folder for a basic tutorial. If you installed a fresh environment, do not forget to install jupyter into that environment! Please run
```
pip install jupyter
```
in your conda environment. The tutorial assumes that all optional dependencies are installed.

## Basic usage

### Loading data from a file:
```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data_from_file('/path/to/expression_data_file.csv') #load data from a csv file and filter with default parameters
sam.load_annotations('/path/to/annotations_file.csv')
sam.run()
sam.scatter()
```

### Using a preloaded scipy.sparse matrix, gene IDs, and cell IDs:
```
from SAM import SAM #import SAM
sam=SAM(counts=(sparse_matrix,geneIDs,cellIDs)) #tuple
sam.filter_data() #filter data with default parameters
sam.run() #run with default parameters
sam.scatter() #display resulting UMAP plot
```

### Loading a pickle file into SAM (output from load_data_from_file):
```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_sparse_data('/path/to/sparse_expression_pickle_file.p') #load data from a pickle file and filter with default parameters
sam.load_annotations('/path/to/annotations_file.csv')
sam.run()
sam.scatter()
```
After loading the data for the first time using 'load_data_from_file', a pickle file of the sparse data will be automatically saved in the same location as the original file. Load the pickle file with 'load_sparse_data' in the future to greatly speed up the loading of data. 

## Citation
If using the SAM algorithm, please cite the following preprint:
https://www.biorxiv.org/content/early/2018/07/07/364166

## Adding extra functionality
As always, please submit a new issue if you would like to see any functionalities / convenience functions / etc added.
