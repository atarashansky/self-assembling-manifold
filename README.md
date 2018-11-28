[![Build Status](https://travis-ci.com/atarashansky/self-assembling-manifold.svg?branch=master)](https://travis-ci.com/atarashansky/self-assembling-manifold)

# self-assembling-manifold
The Self-Assembling-Manifold (SAM) algorithm.

# Update (11/28/2018)

I have added SAMsparse, which uses scipy.sparse matrices to dramatically improve the speed and scalability of SAM applied to large (>8000 cells) datasets. Runs fairly quickly (< 5 minutes on a nothing-too-special desktop computer) on datasets tested up to 50,000 cells. SAM.py will eventually be phased out in favor of SAMsparse.py as any leftover kinks get ironed out. An updated tutorial notebook for interfacing with SAMsparse has been uploaded, under 'tutorial/'. Core usage has essentially remained the same. Please submit any issues! I will fix them ASAP.

## Requirements
 - `numpy`
 - `scipy`
 - `pandas`
 - `scikit-learn`
 - `umap-learn`
 - `numba>=0.37,<0.40`

### Optional dependencies
 - Plotting
   - `matplotlib`
 - Clustering
   - `scanpy`
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

### Using a preloaded Pandas DataFrame:
```
from SAM import SAM #import SAM
sam=SAM(counts=dataframe, #pandas.DataFrame
            annotations=ann) #numpy.ndarray
sam.filter_data() #filter data with default parameters, optional but recommended.
sam.runprojection='umap'() #run with default parameters
sam.scatter() #display resulting UMAP plot
```

### Loading data from a file:
```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data_from_file('/path/to/expression_data_file.csv') #load data from a csv file and filter with default parameters
sam.load_annotations('/path/to/annotations_file.csv')
sam.run(projection='umap')
sam.scatter()
```

### Loading data from a file using SAMsparse:
```
from SAMsparse import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_dense_data_from_file('/path/to/expression_data_file.csv') #load data from a csv file and filter with default parameters
sam.load_annotations('/path/to/annotations_file.csv',delimiter=',')
sam.run(projection='umap')
sam.scatter()
```

### Loading a scipy.sparse '.npz' file into SAMsparse (output from load_dense_data_from_file):
```
from SAMsparse import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_sparse_data('/path/to/sparse_expression_data_file.npz',
                      /path/to/gene_IDs_file.txt',
                      /path/to/cell_IDs_file.txt') #load data from a sparse npz file and filter with default parameters
sam.load_annotations('/path/to/annotations_file.csv',delimiter=',')
sam.run(projection='umap')
sam.scatter()
```
After loading the data for the first time using 'load_dense_data_from_file', use 'load_sparse_data' in the future to greatly speed up the loading of data. 

## Citation
If using the SAM algorithm, please cite the following preprint:
https://www.biorxiv.org/content/early/2018/07/07/364166

## Adding extra functionality
In its current form, this is just a lightweight implementation of the SAM algorithm. If there is any added functionality you would like to see added for downstream analysis, such as cell clustering, differential gene expression analysis, data exporting, etc, please let me know by submitting a new issue describing your request and I will do my best to add that feature.
