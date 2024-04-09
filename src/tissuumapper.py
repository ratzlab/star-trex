'''
Transforms output data of starfish functions into tissuumaps compatible files

Functions are adapted from: https://github.com/TissUUmaps/TissUUmaps/blob/master/examples/starfish2tmap.py
'''

import numpy as np
import pandas as pd
from PIL import Image
from starfish.types import Axes
import os


def qc_csv(experiment, spot_intensities, output_name):
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

    df = pd.DataFrame(
        np.stack([x, y, target_names, rounds_gt, channels_gt]).transpose(),
        columns=["x", "y", "target_name", "rounds", "channels"],
    )
    df.to_csv(output_name)

    return output_name



def qc_images(filtered_imgs, output_name):
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
            image_name = "R{}_C{}.tif".format(r, c)
            im.save(os.path.join(output_name, image_name))
            image_names.append(image_name)

    return image_names