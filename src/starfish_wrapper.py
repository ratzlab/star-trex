"""
Wraps starfish functions and code to allow a single line exectution in notebooks.
Adds some useful functions to create informative output.

Functions are adapted from www.github.com/spacetx/starfish
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
import xarray as xr
from starfish.types import Axes
from starfish import FieldOfView
from starfish import ImageStack
import warnings

#Suppress all annoying warnings
warnings.filterwarnings("ignore")


def stacker(exp, x_edges, y_edges, fov, channels, rounds, zplanes, stack_it=False):
    '''
    Loads specific image planes into memory and organises them into an ImageStack object.
    '''

    #if coordinates of different roudns are not aligned, it is not possible to load the data
    #directly as one stack. They are loaded as iterators of several stacks that need to be 
    #organised in one stack 
    if stack_it:
        #create the iterator containing one image stack/round in one loop
        stacks = exp[fov].get_images(FieldOfView.PRIMARY_IMAGES, chs=channels, rounds=rounds, zplanes=zplanes,
                                    x = slice(x_edges[0], x_edges[1]), y= slice(y_edges[0], y_edges[1]))

        #if iterator contains one round process can be as usual
        if len(stacks) == 1:
            stack = stacks
        else:
            #if iterator contains more than one round, those rounds need to be extracted
            #and stored in a list stacklist
            stacklist = []
            for stack in stacks:
                stacklist.append(stack)
            #Needs to be inverted because, strangely, get_images() returns an inverted iterator
            stacklist = stacklist[::-1]
            
            #Now we concatenate the xarrays of the two elemets in the list
            xarray_data = xr.concat([stacklist[0].xarray, stacklist[1].xarray], dim='r')
            #if the list contains more than two elements, we concatenate every new element with
            #the previous concatenated array
            if len(stacks) > 2:
                for i in range(2,len(stacklist)):
                    xarray_data = xr.concat([xarray_data, stacklist[i].xarray])
            #finally we transform the xarray back to an ImageStack
            stack = ImageStack.from_numpy(xarray_data)
    else:
       #Ideally coordinates are aligned and the stack can simply be loaded with the get_image() starfish function
       stack = exp[fov].get_image(FieldOfView.PRIMARY_IMAGES, chs=channels, rounds=rounds, zplanes=zplanes,
                                  x = slice(x_edges[0], x_edges[1]), y= slice(y_edges[0], y_edges[1]))
    return stack


def transformer(stack, transforms, save_transforms, just_register):
    '''
    Transforms an ImageStack object. Offset values can either be loaded from
    a json file or freshly calculated and saved as json here. 
    The function can also be used to calculate and store the offset only. 
    '''
    from starfish import image
    from starfish.types import Axes

    #if no transforms list is provided, the offset needs to be calculated
    if not transforms:
        print("...Calculates the offset to anchor round")
        projection = stack.reduce({Axes.CH, Axes.ZPLANE}, func="max")
        reference_image = projection.sel({Axes.ROUND: 0})

        ltt = image.LearnTransform.Translation(
            reference_stack=reference_image,
            axes=Axes.ROUND,
            upsampling=1000,
        )
        transforms = ltt.run(projection)
        print(transforms)
    
    #the transforms list with offset value can be stored as json
    if save_transforms:
        print("...Saves the offset values as json")
        transforms.to_json(save_transforms)
    
    #the pipeline can be interupted here, e.g. to calculate the offset values for
    #a large image and use the offset values to transform small tiles
    if just_register:
        print("...Finished!")
        return transforms

    #if a path to a transform list json file is provided it will be loaded here
    print("...Transforms images based on provided/calculated offset")
    if type(transforms) == str:
        from starfish.core.image._registration.transforms_list import TransformsList
        transformslist = TransformsList()
        transformslist.from_json(transforms)
        transforms = transformslist
    
    #transforms ImageStack with provided or calculated transforms list
    projection = stack.reduce({Axes.CH, Axes.ZPLANE}, func="max")
    warp = image.ApplyTransform.Warp()
    stack = warp.run(
        stack=stack,
        transforms_list=transforms,
    )
    return stack


def intensity_equalizer(stack):
    '''
    Wrapper of starfish functions that equalise intensities across rounds and channels
    '''
    from starfish import image
    from starfish.types import Axes

    mh = image.Filter.MatchHistograms({Axes.CH, Axes.ROUND})
    scaled = mh.run(stack, in_place=False, verbose=True, n_processes=8)
    return scaled


def spot_caller(stack):
    '''
    Wrapper of the starfish function BlobDetector to detect spots
    '''
    from starfish.spots import FindSpots

    bd = FindSpots.BlobDetector(
        min_sigma=1,
        max_sigma=8,
        num_sigma=10,
        threshold=np.percentile(np.ravel(stack.xarray.values), 95),
        exclude_border=None) 
    return bd.run(stack)


def spot_decoder(exp, spots):
    '''
    Wrapper of starfish functions to build traces (nearest neighbor) and
    decode spots with PerRoundMaxChannel method.
    '''
    from starfish.spots import DecodeSpots
    from starfish.types import TraceBuildingStrategies

    decoder = DecodeSpots.PerRoundMaxChannel(
        codebook=exp.codebook,
        anchor_round=0,
        search_radius=10,
        trace_building_strategy=TraceBuildingStrategies.NEAREST_NEIGHBOR)

    decoded = decoder.run(spots=spots)

    decode_mask = decoded['target'] != 'nan'
    return decode_mask


def segmenter(stack, nuclei):
    '''
    Wrapper of starfish segmenting functions
    '''
    import numpy as np
    from starfish.image import Segment
    from starfish.types import Axes

    dapi_thresh = .22  # binary mask for cell (nuclear) locations
    stain_thresh = 0.011  # binary mask for overall cells // binarization of stain
    min_dist = 56

    registered_mp = stack.reduce(dims=[Axes.CH, Axes.ZPLANE], func="max").xarray.squeeze()
    stain = np.mean(registered_mp, axis=0)
    stain = stain / stain.max()
    nuclei = nuclei.reduce(dims=[Axes.ROUND, Axes.CH, Axes.ZPLANE], func="max")

    seg = Segment.Watershed(
        nuclei_threshold=dapi_thresh,
        input_threshold=stain_thresh,
        min_distance=min_dist
    )
    masks = seg.run(stack, nuclei)

    return seg, masks


def make_expression_matrix(masks, decoded):
    '''
    Wrapper of starfish functions to create gene expression matrix from
    spots and segmented cells.
    '''
    from starfish.spots import AssignTargets
    
    al = AssignTargets.Label()
    labeled = al.run(masks, decoded[decoded.target != 'nan'])
    gem = labeled[labeled.cell_id != 'nan'].to_expression_matrix()
    return gem


def df_creator(xarray):
    '''
    Transforms an xarray object such as the decoded array returned from spot_decoder
    into a readable dataframe
    '''
    return xarray.to_features_dataframe()


def collector(full_decoded, decoded):
    '''
    Concatenates two xarrays. When working on separated tiles it joins the spot results from the recent tile
    with the results of previous tiles.
    '''
    return xr.concat([full_decoded, decoded], dim="features")


def gene_counter(spots_df):
    '''
    Creates a gene count table based on the spots dataframe.
    '''
    from starfish.types import Features

    genes, counts = np.unique(spots_df.loc[spots_df[Features.PASSES_THRESHOLDS]][Features.TARGET], return_counts=True)
    table = pd.Series(counts, index=genes).sort_values(ascending=False)
    return table


def seconds_converter(seconds):
    '''
    Converts seconds into days, hours, minutes and remaining seconds.
    '''
    # Calculate days, hours, minutes, and seconds
    days, remainder = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)
    return days, hours, minutes, seconds


def get_available_memory():
    '''
    Returns the amount of memory (in bytes) that is currently available for new processes.
    '''
    mem_info = psutil.virtual_memory()
    return mem_info.available


def format_bytes(size):
    '''
    Transforms bytes into largest possible unit.
    '''
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def correct_pixels(full_decoded, x_edges, y_edges):
    '''
    Transforms pixel coordinates that show position of spot within a tile
    to correct pixel coordinates for the position within the full image.
    '''
    #Extracts tile-related x and y coordinates from the xarray
    x_old = full_decoded["x"].values
    y_old = full_decoded["y"].values
    #Prepare two lists for collection the correct coordinates
    x_new = []
    y_new = []

    for i in range(len(x_old)):
        #Each coordinate in x and y is added to the position of the left x- or y-edge
        x_new.append(x_old[i] + x_edges[0])
        y_new.append(y_old[i] + y_edges[0])

    #Changes the tile-related coordinates in xarray to image-related
    full_decoded["x"].values = x_new
    full_decoded["y"].values = y_new

    return full_decoded


def run(exp, nuclei, x_step, y_step, x_max, y_max, fov=None, channels=None, 
             rounds=None, zplanes=None, test = False, full_transform=False, transforms=None,
             stack_it=False, save_transforms=None, just_register=False):
    '''
    Wrapper to run a full starfish pipeline from loading the relevant image planes, splitting them
    into tiles, transforming, spot detection and decoding and cell segmentation.

    Parameters:
        - exp: Starfish Experiment object containing the experiment to process.
        - nuclei: Can be either an ImageStack object containing the loaded nuclei images.
          Or can be an integer indicating the nuclei channel in the experiment.
        - x_step: Desired X-axis tile length in pixel.
        - y_step: Desired Y-axis tile length in pixel.
        - x_max: Total length of x-axis of FOV.
        - y_max: Total length of y-axis of FOV.
        - fov: Name of the Field of View to work on if a specific shall be selected, otherwise set it to None.
        - channels: List of channels to focus analysis on, e.g. [0,1,2] if the 4th channel is a nuclei channel.
          To use all channels, set to None. Default: None.
        - rounds: List of rounds to focus analysis on, e.g. [1,2,3,4]. To use all rounds, set to None.
          Default: None.
        - zplanes: List of zplanes to focus analysis on. To use all zplanes, set to None. Default: None.
        - test: If set to True, only runs the pipeline on the first tile and estimates and prints total 
          run time of the pipeline based on the run time of a single loop. Returns estimated time as 
          days, hours, minutes and seconds. Default: False.
        - full_transform: If set to True, loads the full FOV into memory and transforms it, then extracts
          sub-stacks and performs spot detection/decoding on sub-stacks. Requires sufficient RAM to load full
          image into memory. Can only be performed on one FOV, variable fov cannot be set to None.
          If set to False, it loads tiles into memory and transforms tiles. Requires only enough RAM to load a
          tile, not a full image. Default: False.
        - transforms: A TransformsList object or a path [str] to a TransformsList json file. Allows to jump over the 
          offset calculation step and use offset from the full image on tiles. If set to None, offset will be 
          calculated. Default: None.
        - stack_it: The image planes can only be directly loaded into memory as ImageStack if coordinates are 
          consistent across rounds (all rounds coordiantes start in same start point, e.g. (0,0,0)). Instead such
          unaligned rounds are split into several stacks and collected in an Iterator. If encountering an error in
          this step, set stack_it = True and a joint ImageStack will created from an IteratorObject that contains 
          several stacks. Default: False.
        - save_transforms: Path to location to save the calculated offset as json file transforms.json. This file
          can be loaded as a TransformsList into another run with the "transforms" parameter to avoid re-calculating
          offsets. If set to None it will not save the TransformsList. Default: None.
        - just_register: If set to True, stops pipeline after calculating the offset (i.e. registration). This is 
          useful when wanting to calculate the offset for a large image and apply it to a small tiles. Use "save_
          transforms" to saving and reusing the calculated offsets. Default: False.
    
    Return:
        - spots_df: Dataframe with spots and spot parameters.
        - seg: Segmentation results
        - gem: Gene expression matrix

    '''
    #Indicate start time
    start_time = datetime.now()
    print("The pipeline was started at:", datetime.now())
    loop_counter = 0
    total_loops = len(range(0,x_max,x_step)) * len(range(0,y_max,y_step))

    #isolate the FOV if not provided
    if not fov:
        fov = exp.fovs()[0].name
    
    #if full_transform, load the full image and transform
    if full_transform:
        print("Loads and transform the full image")
        stack = stacker(exp, x_edges=(0,x_max), y_edges=(0,y_max), fov=fov, channels=channels, rounds=rounds, zplanes=zplanes, stack_it=stack_it)
        stack = transformer(stack, transforms, save_transforms, just_register)
        #Stops the execution here if just_register is True
        if just_register:
            return
    #start loop
    for i in range(0, x_max, x_step):
        for j in range(0, y_max, y_step):
            loop_start = datetime.now()
            loop_counter += 1
            #extracts x and y dimensions of the tile
            x_edges = (i, i + x_step)
            y_edges = (j, j + y_step)

            #Loads a tile, either from a stack or from a file
            print(f"Starts processing tile {loop_counter} of {total_loops} at {datetime.now()}")
            print(f"...Available Memory before loading the tile: {format_bytes(get_available_memory())} bytes")
            #if the image has already been fully loaded a stack, it now extracts sub-stacks
            if full_transform:
                stack_now = stack.sel({Axes.X: x_edges, Axes.Y: y_edges})
            else:
                #loads a single tile as stack into memory
                stack_now = stacker(exp, x_edges, y_edges, fov, channels, rounds, zplanes, stack_it)
            #If the full image hasn't been loaded/transformed, transforms the tile
            if not full_transform:
                print("...Calculates offset and transforms tile")
                stack_now = transformer(stack_now, transforms, save_transforms, just_register)
                if just_register:
                    return
            #Calls the starfish spot detection functions
            print("...Searches for spots")
            spots = spot_caller(stack_now)
            #To free memory the current stack is deleted as it is not needed anymore
            del stack_now
            #Calls the starfish spot decoding functions
            print("...Decodes spots")
            decoded = spot_decoder(exp, spots)
            #Corrects pixel from tile-related to full image-related pixels
            print("...Corrects pixels to full image values")
            decoded = correct_pixels(decoded, x_edges, y_edges)
            #If this is the first loop, it just stores the results
            if loop_counter == 1:
                full_decoded = decoded
            else:
                #in the following loops it merges detected spots with the previous results
                print("...Merges spot results with previous round")
                full_decoded = collector(full_decoded, decoded)
            print(f"...Tile {x_edges},{y_edges} is finished! It took {datetime.now()-loop_start} to run")
            #If test == True the loop is stopped now as it now knows how long one loop takes
            if test and loop_counter == 2:
                break
        if test and loop_counter == 2:
            break
    if test and loop_counter == 2:
        loop_end = datetime.now()
        #calculates how long one loop took
        loop_time = loop_end - loop_start
        #involves the time for segmentation in the calculation
        print("Segments cells")
        if type(nuclei) == int:
            x_edges = (0,2048)
            y_edges = (0,2048)
            nuclei = stacker(exp, x_edges, y_edges, fov, channels=[nuclei], rounds=rounds, zplanes=zplanes, stack_it=stack_it)
        seg, masks = segmenter(stack, nuclei)
        print("Creates gene expression matrix")
        gem = make_expression_matrix(masks, decoded)
        #Runtime calculation based on the number of loops times the time one loop takes + the initation time before
        #looping starts + the end time after looping ends (segmentation)
        runtime = loop_time * len(list(exp.keys())) * total_loops + (loop_start - start_time) 
        + (datetime.now() - loop_end)
        #Converts the results into readable format
        days, hours, minutes, seconds = seconds_converter(runtime.seconds)
        print(f"\nResults: One loop takes {loop_time.seconds} seconds\n{total_loops} loops are expected."
              f"\nExecuting the code will take an estimated total time of {days} day(s), {hours} hour(s),"
              f" {minutes} minute(s) and {seconds} second(s).\n")
        return days, hours, minutes, seconds
    else:   
        print("Spot detection is finished! \nSegments cells")
        #For cell segmentation it is necessary to know which channel is the nuclei channel. Nuclei images
        #can directly be loaded as stack. Or as integer indicating the channel number with nuclei staining.
        #here it it tests whether the nuclei variable is int and if yes, creates the stack
        if type(nuclei) == int:
            x_edges = (0,2048)
            y_edges = (0,2048)
            nuclei = stacker(exp, x_edges, y_edges, fov, channels=[nuclei], rounds=rounds, zplanes=zplanes, stack_it=stack_it)
        seg, masks = segmenter(stack, nuclei)
        print("Creates gene expression matrix")
        #Calls starfish functions that transforms results of spot decoding and segmentation into a gene
        #expression matrix
        gem = make_expression_matrix(masks, decoded)

        #Estimates, converts and returns runtime 
        runtime = datetime.now() - start_time
        days, hours, minutes, seconds = seconds_converter(runtime.seconds)
        print(f"The pipeline is finished at {datetime.now()}.\nIt took " 
              f"{days} day(s), {hours} hour(s), {minutes} minute(s) and {seconds} second(s).")
        return full_decoded, stack, seg, gem
                