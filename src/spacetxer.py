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
from src.path_handler import *
import os

def channel_sorter(channel, nuclei):
    '''
    Re-sorts the channel number such as that the nuclei channel gets the last number. This is unfortunately necessary
    as starfish does not allow cropping out a channel from the beginning or middle of the list in order to perform 
    spot detection only on spot channels
    '''
    if channel == nuclei:
        channel = 0
    elif channel > nuclei:
        channel = channel - 1
    return channel


def process_tiff(input_path, output_path, extension=None, fov=0, nuclei = 0):
    """
    Splits single round tiff images into one tiff image per round, channel and zplane. 
    Creates coordinates.csv file. Both steps are required for transformation to SpaceTx 
    format.

    Parameters:
        - input_path: String with path to input folder containing either one (ome).tiff
          image per round OR, if fov="multi", one folder per FOV with each containg one 
          (ome).tiff per round
        - output_path: String with path to output folder
        - extension: If input folder contains several file types, a specific file extension
          such as ".tiff" can be provided to select only certain files, default: None
        - fov: The given fov, can either be an number (int) or "multi". If "multi"
          is given, a folder with several input folders (one per FOV) needs to be provided,
          default: 0
        - nuclei: Number of the nuclei channel (channel numbering starts with 0). Indicate None
          if there is no nuclei channel. Default: 0.
    """
    #Check the paths and convert to Path object
    input_path = path_maker(input_path)
    output_path = path_maker(output_path)
    counter = 0
    #Create the separate spot and nuclei output folders
    if not os.path.exists(os.path.join(output_path, "primary")):
        os.makedirs(os.path.join(output_path, "primary"))
    if nuclei != None:
        if not os.path.exists(os.path.join(output_path, "nuclei")):
            os.makedirs(os.path.join(output_path, "nuclei"))

    #Get the input image files
    if fov == "multi":
        path_list = []
        folder_list = get_folders_in_path(input_path)
        for folder in folder_list:
            path_list.append(get_files_in_path(folder, extension))
    else: 
        path_list = [get_files_in_path(input_path, extension)]

    #Initate a coordinate pandas table
    coord_primary = pd.DataFrame(columns=["fov", "round", "ch","zplane","xc_min","yc_min","zc_min","xc_max","yc_max","zc_max"])
    if nuclei != None:
        coord_nuclei = pd.DataFrame(columns=["fov", "round", "ch","zplane","xc_min","yc_min","zc_min","xc_max","yc_max","zc_max"])

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

            #The below line is useful for reading and understanding the xml metadata
            #print(ET.tostring(root, encoding='utf8').decode('utf8'))

            # Extract the number of channels and zplanes
            try:
                num_zplanes, num_channels, pixelx, pixely = tiff_stack.shape
            except:
                num_zplanes = 1
                num_channels, pixelx, pixely = tiff_stack.shape

            #Extract the physical size of the planes
            for pixel in root.iter(url + "Pixels"):
                sizeX = float(pixel.attrib["PhysicalSizeX"]) * int(pixelx)
                sizeY = float(pixel.attrib["PhysicalSizeY"]) * int(pixely)
                sizeZ = float(pixel.attrib["PhysicalSizeX"])

            # Iterate through each channel and z plane
            for channel in range(num_channels):
                for zplane in range(num_zplanes):
                    # Extract a specific plane
                    if num_zplanes > 1:
                        single_plane = tiff_stack[zplane,channel, :, :]
                    else:
                        single_plane = tiff_stack[channel, :, :]

                    #Extract and build physical coordinates table
                    for tiffdata in root.iter(url + 'TiffData'):
                        if tiffdata.attrib["FirstZ"] == str(zplane) and tiffdata.attrib["FirstC"] == str(channel):
                            #This code is not yet for multiple fovs in one image!
                            minX = 0.0
                            #minX = float(tiffdata.attrib["PositionX"])
                            minY = 0.0
                            #minY = float(tiffdata.attrib["PositionY"])
                            minZ = zplane * sizeZ
                            #minZ = float(tiffdata.attrib["PositionZ"])
                            maxX = minX + float(sizeX)
                            maxY = minY + float(sizeY)
                            maxZ = minZ + sizeZ
                            if channel == nuclei:
                                image_type = "nuclei"
                                channel_new = channel_sorter(channel, nuclei)
                                coord_nuclei.loc[len(coord_nuclei)] = [fov_num, round, channel_new, zplane, minX, minY, minZ, maxX, maxY, maxZ]
                            else:
                                image_type = "primary"
                                channel_new = channel_sorter(channel, nuclei)
                                coord_primary.loc[len(coord_primary)] = [fov_num, round, channel_new, zplane, minX, minY, minZ, maxX, maxY, maxZ]
                            
                    # Save the extracted image
                    full_output_path = f"{output_path}/{image_type}/{image_type}-f{fov_num}-r{round}-c{channel_new}-z{zplane}.tiff"
                    tifffile.imsave(full_output_path, single_plane)
                    print(f"Saved: {full_output_path}")

    #Save coordinates as csv file
    for col in ["fov", "round", "ch", "zplane"]:
        coord_primary[col] = coord_primary[col].astype(int) 
        if nuclei != None:
            coord_nuclei[col] = coord_nuclei[col].astype(int)                      
    coord_primary.to_csv(output_path / "primary/coordinates.csv", index=False)
    if nuclei != None:
        coord_nuclei.to_csv(output_path / "nuclei/coordinates.csv", index=False)



def process_forISS(input_path, output_path, image_type="primary", extension=None, fov=0):
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
        print(path_list[fov_num])

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
                sizeX = float(pixel.attrib["PhysicalSizeX"]) * int(pixelx)
                sizeY = float(pixel.attrib["PhysicalSizeY"]) * int(pixely)
                sizeZ = float(pixel.attrib["PhysicalSizeX"])

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
                    full_output_path = f"{output_path}/R{round}/TileScan 1--Stage1--Z{zplane}--C{channel}.tif"
                    tifffile.imsave(full_output_path, single_plane)
                    print(f"Saved: {full_output_path}")
    
    #Save coordinates as csv file
    for col in ["fov", "round", "ch", "zplane"]:
        coord_df[col] = coord_df[col].astype(int)                
    coord_df.to_csv(output_path / "coordinates.csv", index=False)

