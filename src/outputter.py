'''
Transforms output data of starfish functions into tissuumaps compatible files

Functions are adapted from: https://github.com/TissUUmaps/TissUUmaps/blob/master/examples/starfish2tmap.py
'''

import numpy as np
import pandas as pd
from PIL import Image
from starfish.types import Axes
import os


def seurat_outputter(gem_nested, work_dir):
    '''
    Creates output files and folders required for seurat.
    '''
    #Extract cells and gene names
    cells = list(set(gem_nested.index.get_level_values(level=0).tolist()))
    features = list(set(gem_nested.index.get_level_values(level=1).tolist()))
    #Create empty matrix to be filled with expression values
    gem_np = np.zeros((len(cells), len(features)))

    #for each cell and gene, extract expression value and insert into matrix
    for i in range(len(cells)):
        for j in range(len(features)):
            gem_np[i,j] = gem_nested.loc[(cells[i], features[j]), 'expression_matrix']
    
    #create output folders for seurat
    os.makedirs(os.path.join(work_dir, "output/"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "output", "filtered_features_bc_matrices"), exist_ok=True)

    #create df gene expression matrix from cells, genes and expression values
    pd.DataFrame(cells, columns=["cells"]).to_csv(os.path.join(work_dir, "output", "filtered_features_bc_matrices", "barcodes.tsv"), sep = "\t", header = False, index = False)
    pd.DataFrame(features, columns=["feautures"]).to_csv(os.path.join(work_dir, "output", "filtered_features_bc_matrices", "features.tsv"), sep = "\t", header = False, index = False)
    pd.DataFrame(gem_np, index = cells, columns = features).to_csv(os.path.join(work_dir, "output", "filtered_features_bc_matrices", "matrix.mtx"), sep = "\t", header = False, index = False)

    #create metadata table
    #since the gem table has a nested index of "cells" and for each cell "genes" and each cell has only one metadata row,
    #we need to reduce the size of the table to one row per cell first. 
    jump = len(set(gem_nested.index.get_level_values('genes')))
    meta = gem_nested[["x", "y", "z", "xc", "yc", "zc", "area", "number_of_undecoded_spots"]].iloc[::jump]
    #Then we use the cellIDs as index for the metadata
    meta.index = cells
    meta.to_csv(os.path.join(work_dir, "output", "metadata.tsv"), sep = "\t", header = True, index = True)


def tissuumaps_csv(experiment, spot_intensities, output_name):
    """
    Create a CSV file from a starfish experiments compatible with the TissUUmaps "Spot Insepector" plugin

    Parameters:
        experiment (starfish Experiment): containing the codebook
        spot_intensities (starfish DecodedIntensityTable): result of the decoding process
        output_name (string): name of the output csv file
    """

    rounds_gt = []
    channels_gt = []

    x = spot_intensities.x.values
    y = spot_intensities.y.values
    target_names = spot_intensities.target.values

    for name in target_names:
        if not name == "undefined":
            idx = np.where(experiment.codebook.target.values == name)[0][0]
            rs = np.where(experiment.codebook.values[idx] > 0)[0]
            rounds = ""
            for i in range(len(rs)):
                rounds = rounds + f"{rs[i]};"
            rounds_gt.append(rounds.strip(";"))
            chans = []
            for r in np.unique(rs):
                ch = experiment.codebook.values[idx][r]
                chans.append(np.argwhere(ch == np.amax(ch)))
            chs = np.concatenate(chans)
            channels = ""
            for j in range(len(chs)):
                channels = channels + f"{chs[j][0]};"
            channels_gt.append(channels.strip(";"))
        else:
            rounds_gt.append("00000")
            channels_gt.append("00000")

    df = pd.DataFrame(
        np.stack([x, y, target_names, rounds_gt, channels_gt]).transpose(),
        columns=["x", "y", "target_name", "rounds", "channels"],
    )
    df.to_csv(output_name)

    return output_name



def tissuumaps_images(filtered_imgs, output_name, img_type):
    """
    Creates the images from a starfish experiments compatible with the TissUUmaps "Spot Inspector" plugin

    Parameters:
        filtered_imgs (starfish ImageStack): image stack after filtering and deconvolving the data
    """

    image_names = []

    for r in range(filtered_imgs.num_rounds):
        for c in range(filtered_imgs.num_chs):
            im = np.squeeze(
                filtered_imgs.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values
            )
            im = np.log(im + 1)

            mn = im.min()
            mx = im.max()
            mx -= mn
            im = ((im - mn) / mx) * 255
            im = im.astype(np.uint8)

            im = Image.fromarray(im)
            if img_type == "nuclei":
                image_name = "nuclei-R{}_C{}.tif".format(r, c)
            else:
                image_name = "R{}_C{}.tif".format(r, c)
            im.save(os.path.join(output_name, image_name))
            image_names.append(image_name)

    return image_names


def napari_csv(spot_intensities, output_name, palette_name="hsv"):
    """
    Creates a csv file that can be used to create a point layer in napari with different colors and 
    """
    import seaborn as sns
    import random

    x = list(spot_intensities.x.values)
    y = list(spot_intensities.y.values)
    z = list(spot_intensities.z.values)
    target_names = list(spot_intensities.target.values)

    all_symbols = ["clobber", "cross", "diamond", "disc", "hbar", "ring", "square", "star", 
                "triangle_down", "triangle_up", "vbar", "x"]
    
    unique_targets = set(target_names)
    palette = sns.color_palette(palette_name, len(unique_targets))
    rgba = [list(rgb) + [1.0] for rgb in palette]
    gene_colors = {gene: rgba[i] for i, gene in enumerate(unique_targets)}
    colors = [gene_colors[gene] for gene in target_names]
    
    random_symbols = {gene: random.choice(all_symbols) for gene in unique_targets}
    symbols = [random_symbols[gene] for gene in target_names]

    results = {"x" : x, "y" : y, "z" : z, "target_name" : target_names, "color" : colors, "symbol" : symbols}
    df = pd.DataFrame(results)
    df.to_csv(output_name, sep="\t")

    return output_name