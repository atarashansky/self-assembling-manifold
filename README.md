[![Build Status](https://travis-ci.com/atarashansky/self-assembling-manifold.svg?branch=master)](https://travis-ci.com/atarashansky/self-assembling-manifold)

# self-assembling-manifold
The Self-Assembling-Manifold (SAM) algorithm.

# Update (3/10/2019) -- SAM version 0.4.3

Please see the updated tutorials for any usage changes.

- Input and output has been further streamlined: `load_data` will now be used for loading both tabular csv/txt files as well as pickled sparse data structures. Note that `load_data` no longer preprocesses the data automatically, and `filter_data` was renamed to `preprocess_data`.
- In preparation for integrating with the Scanpy package (https://github.com/theislab/scanpy), SAM can now accept as input to its constructor (via the `counts` argument) an AnnData object. It also stores all key SAM outputs in an AnnData object (`.adata`).
- New clustering methods have been added (DBSCAN in `density_clustering` and HDBSCAN in `hdbknn_clustering`). `hdbknn_clustering` is a slightly extended version of HDBSCAN in which outlier cells (i.e. cells that were not assigned to a cluster) are assigned to the clusters that were found using a kNN classification approach.
- A Random Forest classification approach for marker gene identification was added (in `identify_marker_genes_rf`).


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
Please see the Jupyter notebooks in the 'tutorial' folder for basic tutorials. If you installed a fresh environment, do not forget to install jupyter into that environment! Please run
```
pip install jupyter
```
in your conda environment. The tutorial assumes that all optional dependencies are installed.

## Basic usage

There are a number of ways to load data into the SAM object:

### Loading data from a file:
```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data('/path/to/expression_data_file.csv') #load data from a csv file
sam.preprocess_data() # log transforms and filters the data (recommended)
sam.load_annotations('/path/to/annotations_file.csv')
sam.run()
sam.scatter()
```

### Using preloaded scipy.sparse or numpy expression matrix, gene IDs, and cell IDs:
```
from SAM import SAM #import SAM
sam=SAM(counts=(matrix,geneIDs,cellIDs))
sam.preprocess_data() # log transforms and filters the data (recommended)
sam.run() #run with default parameters
sam.scatter() #display resulting UMAP plot
```
### Using preloaded pandas.DataFrame (cells x genes):
```
from SAM import SAM #import SAM
sam=SAM(counts=dataframe)
sam.preprocess_data() # log transforms and filters the data (recommended)
sam.run() #run with default parameters
sam.scatter() #display resulting UMAP plot
```

### Using an existing AnnData object:
```
from SAM import SAM #import SAM
sam=SAM(counts=adata)
sam.preprocess_data() # log transforms and filters the data (recommended)
sam.run() #run with default parameters
sam.scatter() #display resulting UMAP plot
```

### Loading a pickle file previously output by SAM: 

Finally, `load_data` by default saves the sparse data structure to a pickle file (`_sparse.p`) for faster loading in subsequent analyses. This file can be loaded as:

```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data('/path/to/sparse_expression_pickle_file_sparse.p') #load data from a pickle file
sam.preprocess_data() # log transforms and filters the data (recommended)
sam.run()
sam.scatter()
```

### Saving and loading a pickled SAM object:
```
from SAM import SAM #import SAM

#Save
sam=SAM() #initialize SAM object
sam.load_data('/path/to/expression_data_file.csv') #load data from a csv file and filter with default parameters
sam.preprocess_data() # log transforms and filters the data (recommended)
sam.run()
sam.save('/desired/output/path') #pickle the SAM object with all its attributes

#Load
sam = SAM()
sam.load('/desired/output/path.p') #load the SAM object and all its attributes
sam.scatter() #visualize UMAP output
```
## Citation
If using the SAM algorithm, please cite the following preprint:
https://www.biorxiv.org/content/early/2018/07/07/364166

## Adding extra functionality
As always, please submit a new issue if you would like to see any functionalities / convenience functions / etc added.
