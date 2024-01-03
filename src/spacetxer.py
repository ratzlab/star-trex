"""
Converts image folders with one ome.tiff image per round ordered by round to structured format
which corresponds to a folder of one image per round, channel and z plane with name convetion
<image_type>-f<fov_id>-r<round_label>-c<ch_label>-z<zplane_label>.tiff. Creates coordinates.csv.
Transforms data to spaceTx format.
"""

import tifffile
from tifffile import TiffFile
import xml.etree.ElementTree as ET
import pandas as pd
from path_handler import *


def process_tiff(input_path, output_path, image_type="primary", extension=None, fov=0):
    """
    Splits single round tiff images into one tiff image per round, channel and zplane. 
    Creates coordinates.csv file. Both steps are required for transformation to SpaceTx 
    format.

    Input:
        - input_path: String with path to input folder containing either one (ome).tiff
          image per round OR, if fov="multi", one folder per FOV with each containg one 
          (ome).tiff per round
        - output_path: String with path to output folder
        - image_type: Distinguishes images of sequencing rounds or additional stainings
          such as DAPI or Nissl, can be "primary" or "nuclei", default: "primary"
        - extension: If input folder contains several file types, a specific file extension
          such as ".tiff" can be provided to select only certain files, default: None
        - fov: The given fov, can either be an number (int) or "multi". If "multi"
          is given, a folder with several input folders (one per FOV) needs to be provided,
          default: 0
    """
    #Check the paths and convert to Path object
    input_path = path_maker(input_path)
    output_path = path_maker(output_path)

    #Get the input image files
    if fov == "multi":
        path_list = []
        folder_list = get_folders_in_path(input_path)
        for folder in folder_list:
            path_list.append(get_files_in_path(folder, extension))
    else: 
        path_list = [get_files_in_path(input_path, extension)]

    #Initate a coordinate pandas table
    coord_df = pd.DataFrame(columns=["fov", "round", "ch","zplane","xc_min","yc_min","zc_min","xc_max","yc_max","zc_max"])

    #Loop through the FOVs
    for fov_num in range(len(path_list)):

        #Loop through each image/round
        for round in range(len(path_list[fov_num])):
            # Read the TIFF image
            tiff_stack = tifffile.imread(path_list[fov_num][round])
            
            #Read the metadata as XML tree
            with TiffFile(path_list[fov_num][round]) as tiff:
                root = ET.fromstring(tiff.ome_metadata)
            url = root.tag.strip("OME")
            
            # Extract the number of channels and zplanes
            num_zplanes, num_channels, pixelx, pixely = tiff_stack.shape

            #Extract the physical size of the planes
            for pixel in root.iter(url + "Pixels"):
                sizeX = pixel.attrib["PhysicalSizeX"]
                sizeY = pixel.attrib["PhysicalSizeY"]
                sizeZ = float(pixel.attrib["SizeZ"]) * (float(sizeX)/float(pixel.attrib["SizeX"]))

            # Iterate through each channel and z plane
            for channel in range(num_channels):
                for zplane in range(num_zplanes):
                    # Extract a specific plane
                    single_plane = tiff_stack[zplane,channel, :, :]

                    #Extract and build physical coordinates table
                    for plane in root.iter(url + 'Plane'):
                        if plane.attrib["TheZ"] == str(zplane) and plane.attrib["TheC"] == str(channel):
                            minX = float(plane.attrib["PositionX"])
                            minY = float(plane.attrib["PositionY"])
                            minZ = float(plane.attrib["PositionZ"])
                            maxX = minX + float(sizeX)
                            maxY = minY + float(sizeY)
                            maxZ = minZ + sizeZ
                            coord_df.loc[len(coord_df)] = [fov_num, round, channel, zplane, minX, minY, minZ, maxX, maxY, maxZ]

                    # Save the extracted image
                    full_output_path = f"{output_path}/{image_type}-f{fov_num}-r{round}-c{channel}-z{zplane}.tiff"
                    tifffile.imsave(full_output_path, single_plane)
                    print(f"Saved: {full_output_path}")
    
    #Save coordinates as csv file
    for col in ["fov", "round", "ch"]:
        coord_df[col] = coord_df[col].astype(int)                
    coord_df.to_csv(output_path / "coordinates.csv", index=False)