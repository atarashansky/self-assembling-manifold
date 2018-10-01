[![Build Status](https://travis-ci.com/atarashansky/self-assembling-manifold.svg?branch=master)](https://travis-ci.com/atarashansky/self-assembling-manifold)

# self-assembling-manifold
The Self-Assembling-Manifold (SAM) algorithm.


## Requirements
 - `numpy`
 - `pandas`
 - `scikit-learn`
 - `matplotlib`
 - `scanpy`
 - `louvain`
 - `umap-learn`
 - `numba>=0.37,<0.40`

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
in your conda environment.

## Basic usage

Using a preloaded Pandas DataFrame:
```
from SAM import SAM #import SAM
sam=SAM(counts=dataframe, #pandas.DataFrame
            annotations=ann) #numpy.ndarray
sam.filter_data() #filter data with default parameters, optional but recommended.
sam.run() #run with default parameters
sam.scatter() #display resulting t-SNE plot
```

Loading data from a file:
```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data_from_file('/path/to/expression_data_file.csv') #load data from a csv file and filter with default parameters
sam.load_annotations('/path/to/annotations_file.csv')
sam.run()
sam.scatter()
```

## Citation
If using the SAM algorithm, please cite the following preprint:
https://www.biorxiv.org/content/early/2018/07/07/364166

## Adding extra functionality
In its current form, this is just a lightweight implementation of the SAM algorithm. If there is any added functionality you would like to see added for downstream analysis, such as cell clustering, differential gene expression analysis, data exporting, etc, please let me know by submitting a new issue describing your request and I will do my best to add that feature.
