#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author:     Alexander Tarashansky
date:       10/01/2018
content:    Tests for SAM.
"""
# Modules
from SAM import SAM
import pandas as pd 
import numpy as np


# Script
if __name__ == '__main__':
    counts = pd.read_csv('example_data/GSE74596_data.csv.gz', index_col=0).T
    annotations = pd.read_csv('example_data/GSE74596_ann.csv', header=None)
    annotations = np.array(list(annotations.values.flatten()))
    sam = SAM(counts=counts, annotations=annotations)
    assert sam.D.shape == (203, 25890)
    sam.filter_data()
    assert sam.D.shape == (203, 15450)

    sam = SAM()
    sam.load_data_from_file('example_data/GSE74596_data.csv.gz')
    sam.load_annotations('example_data/GSE74596_ann.csv')

    assert sam.D.shape == (203, 15450)
    assert np.all(sam.annotations == annotations)

    sam.run()
    sam.run_tsne()
    # TODO: UMAP's and louvain's deps have everchanging APIs, skip for now
    #sam.run_umap()
    #sam.louvain_clustering()
    #sam.identify_marker_genes()
