[![Build Status](https://travis-ci.com/atarashansky/self-assembling-manifold.svg?branch=master)](https://travis-ci.com/atarashansky/self-assembling-manifold)

# self-assembling-manifold -- SAM version 0.5.1
The Self-Assembling-Manifold (SAM) algorithm.

# Update (6/13/2019)

Created a new control panel figure. All widgets are now in this window.

Added a "Show plot history" button. Clicking this button opens a new window with an image of a previous plot shown. You can scroll hovering over this image to scroll through all previously generated plots.

## Important Note
`scatter` and `show_gene_expression` are now outside the SAM class but still in the SAM module. To use these functions, they must be imported:
`from SAM import scatter, show_gene_expression`
Both functions now take as the first required argument a SAM object:
`scatter(sam_object)`

# Update (6/6/2019)

Added a slider bar at the very bottom: When showing the expression of a particular gene, slide the bottom slider to select cells with expression in that gene greater than the current value of the slider.

Added a hotkey: pressing `a` toggles gene expression averaging

Added a hotkey: pressing `x` unselects all cells.

Added two new text boxes. Enter a key in the text box on the very left to select which annotation in `sam.adata.obs` you will modify with the text box next to it. If the annotation does not exist, a new one will be created. Typing a label and pressing 'enter' in the text box next to it labels the currently selected cells with that label.Any annotations made to cells while subclustering are also saved to the original SAM object.

Use case: Let's say you wanted to create a new annotation vector in `sam.adata.obs` with the key `anno`
 - Unselect all cells
 - Highlight population of interest with mouse (call it `pop1`)
 - Type `anno` in left text box
 - Type `pop1` in the right text box. Press Enter.
 - If you hover over the button above the left text box and scroll the wheel, you'll find a new annotation called `anno`. 
 - Click on it to show the current annotations. Cells not yet annotated are labeled by default with an empty string.
 - Unselect all cells
 - Highlight another population.
 - Type `pop2` in the right text box. Press Enter.
 - Click on the `anno` button to show the annotations. You should see the second population now.

Currently this GUI is mainly for my own usage, so I apologize for the total lack of clarity in terms of all the hotkeys and all the buttons...Improvements to the general usability and adding a help page will be coming soon...


# Update (6/2/2019)

Added a bunch of new features to the scatter plots -- a more detailed changelog will come soon. Also on my To-Do list is to make a more detailed tutorial for those who are interested in this GUI-esque interface.

In brief, 

You can now scroll while hovering over the clustering button to change the clustering algorithm. The clustering parameter slider also changes depending on the clustering algorithm.

Added a new button to display any existing annotations in the AnnData.obs slot. Hovering over the button and scrolling will scroll through any existing annotations. Click on the button to display the desired annotation.

When annotations (i.e. cluster assignments or annotations loaded from a file) are displayed, a legend appears on the righthand side in the light-gray axes. Clicking on the box next to the annotation label will unselect/select those corresponding cells. Unselected cells appear gray in the plot. The remaining cells can then be subclustered using the "Subcluster" button. You can scroll up and down to see all annotation labels if they do not all fit in the box.

You can interactively annotate the plot by double clicking and typing out a label. Pressing enter will place the label. Pressing escape will remove the label.

# Update (6/1/2019)

Made use of matplotlib widgets to add interactivity to the scatter plots. The interactive plot can be accessed via the `scatter` function.

Scroll wheel:
 - Zoom in and out
 - Click and hold to pan

Left click:
 - Click and drag to highlight cells. Highlighted indices and cell IDs are stored in `sam.ps.selected_points` and `sam.ps.selected_cells`, respectively.

Right click:
 - Resets the plot and unselects all cells (if subclustering has been done, the plot is reset to the default view for the subclustering analysis), removes markers.

ESC (keyboard):
 - Resets the plot to the original SAM object (removes subclustering), unselects all cells, removes markers.

Enter (keyboard):
 - If cells are selected, pressing Enter will identify marker genes for those cells. The bottom slider can be used to scroll through and display the ranked list of marker genes.

Left/Right arrows (keyboard):
 - Can be used to scroll through the bottom slider.

Subcluster (button widget):
 - If cells are selected, pressing this button will rerun SAM on the selected cells.

Density cluster (button widget):
 - Pressing this button will perform density clustering on the current UMAP projection.

Slider (upper widget):
 - Selects the epsilon (distance) parameter for the density clustering algorithm.

Slider (lower widget):
 - If marker genes for a particular cluster have been identified, this slider scrolls through and displays the ranked list of marker genes.
 - If marker genes were not identified, this slider scrolls through and displays the ranked list of genes according to their SAM weights.

Text box:
 - Type a gene ID and press enter to display the corresponding expression pattern.

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
