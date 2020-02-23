#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author:     Alexander Tarashansky
date:       10/01/2018
content:    Tests for SAM.
"""
# Modules
from samalg import SAM


# Script
if __name__ == "__main__":

    sam = SAM()
    sam.load_data("example_data/darmanis_data.csv.gz")
    sam.load_obs_annotations("example_data/darmanis_ann.csv")
    sam.preprocess_data()
    sam.run(projection=None)
    sam.run_umap()
    sam.run_tsne()
    sam.kmeans_clustering(4)
    sam.identify_marker_genes_ratio()
    sam.identify_marker_genes_rf()
    umap_obj = sam.umap_obj  # checking to see if umap_obj is stored in sam object
