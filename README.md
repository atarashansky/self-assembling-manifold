[![Build Status](https://travis-ci.com/atarashansky/self-assembling-manifold.svg?branch=master)](https://travis-ci.com/atarashansky/self-assembling-manifold)

# self-assembling-manifold -- SAM version 0.6.6
The Self-Assembling-Manifold (SAM) algorithm.

## Requirements
 - `numpy`
 - `scipy<=1.2.0`
 - `pandas`
 - `scikit-learn`
 - `umap-learn`
 - `numba`
 - `anndata`

### Optional dependencies
 - Plotting
   - `plotly==4.0.0`
   - `ipythonwidgets`
   - `jupyter`
   - `colorlover`
   - `ipyevents`

 - Clustering
   - `louvain`
   - `leidenalg`
   - `hdbscan`
   - `cython`

 - `scanpy`

## Version 0.6.6

I scrapped the matplotlib GUI I was using before in favor of using the magical `Plotly` and `ipythonwidgets` packages. To interactively explore the data, use the new module `SAMGUI.py`. Run the following in a jupyter notebook:
```
from SAMGUI import SAMGUI
sam_gui = SAMGUI(sam) # sam is your SAM object
sam_gui.SamPlot
```
This will launch an interactive widget that you can use to explore the data. I tried my best to add tooltips (hover mouse over labels) wherever possible. Please submit an issue if there are any outstanding bugs / feature requests.

I still have to document this, but you can click on a cell to select / deselect all cells with matching labels.

I would recommend installing from the development version for now (not pip) as I currently do not have a stable release of this GUI.

![SAM GUI example image](samgui.png)

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

For interactive data exploration (in the `SAMGUI.py` module), `jupyter`, `ipythonwidgets`, `colorlover`, `ipyevents`, and `plotly` are required. Install them in the previously made environment like so:

```
conda install -c conda-forge -c plotly jupyter ipywidgets plotly=4.0.0 colorlover ipyevents
```

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
scatter(sam)
```
### Using preloaded pandas.DataFrame (cells x genes):
```
from SAM import SAM #import SAM
sam=SAM(counts=dataframe)
sam.preprocess_data() # log transforms and filters the data
sam.run() #run with default parameters
scatter(sam)
```

### Using an existing AnnData object:
```
from SAM import SAM #import SAM
sam=SAM(counts=adata)
sam.preprocess_data() # log transforms and filters the data
sam.run() #run with default parameters
scatter(sam)
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
scatter(sam)
```
### Loading an existing AnnData `h5ad` file: 

If loading tabular data (e.g. from a `csv`), `load_data` by default saves the sparse data structure to a `h5ad` file in the same location as the tabular file for faster loading in subsequent analyses. This file can be loaded as:

```
from SAM import SAM #import SAM
sam=SAM() #initialize SAM object
sam.load_data('/path/to/h5ad_file.h5ad') #load data from a h5ad file
sam.preprocess_data() # log transforms and filters the data
sam.run()
scatter(sam)
```

If you wish to save the SAM outputs and filtered data, you can write `sam.adata` to a `h5ad` file as follows:
`sam.save_anndata(filename, data = 'adata')`.

If for whatever reason you wish to save the raw, unfiltered AnnData object,
`sam.save_anndata(filename, data = 'adata_raw')`.


# SAM GUI
```
from SAM import scatter
# ... run SAM analysis ...

scatter(sam) # launch the GUI -- two windows should pop out
```

The SAM GUI consists of a scatter plot and a control panel window with various buttons / text boxes. (All buttons with little arrows drawn on them can be scrolled on with the scroll wheel to select different options)

Features include:

Scroll wheel:
 - Zoom in and out
 - Click and hold to pan

Left click:
 - Click and drag to highlight cells. Highlighted cell IDs are stored in `sam.ps.selected_cells`.

Right click:
 - Resets the plot and unselects all cells (if subclustering has been done, the plot is reset to the default view for the subclustering analysis), removes markers.

ESC (keyboard):
 - Resets the plot to the original SAM object (removes subclustering), unselects all cells, removes markers.

Enter (keyboard):
 - If cells are selected, pressing Enter will identify marker genes for those cells. The ranked gene slider can be used to scroll through and display the ranked list of marker genes.

Left/Right arrows (keyboard):
 - Can be used to scroll through the ranked gene slider.

Subcluster/Run (button widget):
 - Pressing this button will rerun SAM on the selected cells.

Cluster (button widget):
 - Pressing this button will perform clustering on the current UMAP projection. Scrolling while hovering over this button changes the clustering algorithm.
 
Display annotations (button widget):
 - Pressing this button will display the currently selected annotations. Scrolling hovering over this button changes the annotations to display.

Slider:
 - Selects the resolution parameter for the current clustering algorithm.

Slider:
 - If marker genes for a particular cluster have been identified, this slider scrolls through and displays the ranked list of marker genes.
 - If marker genes were not identified, this slider scrolls through and displays the ranked list of genes according to their SAM weights.

Slider:
- When showing the expression of a particular gene, slide the bottom slider to select cells with expression in that gene greater than the current value of the slider.

Text box:
 - Type a gene ID and press enter to display the corresponding expression pattern.

`a` (keyboard):
- pressing `a` toggles gene expression averaging

`x` (keyboard): 
- pressing `x` unselects all cells.



## Citation
If using the SAM algorithm, please cite the following preprint:
https://www.biorxiv.org/content/early/2018/07/07/364166

## Adding extra functionality
As always, please submit a new issue if you would like to see any functionalities / convenience functions / etc added.
