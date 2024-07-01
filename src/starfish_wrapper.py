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
import yaml
import inspect
from types import SimpleNamespace

#Suppress all annoying warnings
warnings.filterwarnings("ignore")


def get_default_args(func):
    '''
    Extracts the default arguments from a function and returns a dictionary of arg:value pairs.
    '''
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def settings_reader(settings_path, func, args, values):
    '''
    First, the code extracts all default settings and stores those that have been changed via the function
    call into a dictionary. Reads the provided settings.yaml file and changes all saves the settings of the 
    yaml file unless the setting has been modified in the function call, in that case it keeps the modification.
    Return a SimpleNamespace dictionary for easy access to settings.
    '''
    defaults = get_default_args(func)
    non_default = {arg: values[arg] for arg in args if arg in defaults.keys() and values[arg] != defaults[arg]}

    if settings_path != None:
        settings_dict = yaml.safe_load(open(settings_path))
        for key in non_default.keys():
                settings_dict[key] = non_default[key]
        #Makes access to dict easier by allowing to acces values with dict.key
        s = SimpleNamespace(**settings_dict)
    else: 
        for key in non_default.keys():
                defaults[key] = non_default[key]
        #Makes access to dict easier by allowing to acces values with dict.key
        s = SimpleNamespace(**defaults)
    return s


def stacker(exp, img_type, x_edges, y_edges, fov, channels, rounds, zplanes, stack_it=False):
    '''
    Loads specific image planes into memory and organises them into an ImageStack object.
    '''

    #if coordinates of different roudns are not aligned, it is not possible to load the data
    #directly as one stack. They are loaded as iterators of several stacks that need to be 
    #organised in one stack 
    if stack_it:
        #create the iterator containing one image stack/round in one loop
        stacks = exp[fov].get_images(img_type, chs=channels, rounds=rounds, zplanes=zplanes,
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
            if len(stacklist) > 2:
                for i in range(2,len(stacklist)):
                    xarray_data = xr.concat([xarray_data, stacklist[i].xarray])
            #finally we transform the xarray back to an ImageStack
            stack = ImageStack.from_numpy(xarray_data)
    else:
       #Ideally coordinates are aligned and the stack can simply be loaded with the get_image() starfish function
       stack = exp[fov].get_image(img_type, chs=channels, rounds=rounds, zplanes=zplanes,
                                  x = slice(x_edges[0], x_edges[1]), y= slice(y_edges[0], y_edges[1]))
       
       if stack.num_rounds == 1:
           print("WARNING: Your stack contains only one round. If you expect more than one round, check the coordinates.csv\n"
                 "files in work_dir/primary and work_dir/nuclei folder and make sure that image coordinates are aligned. \n"
                 "If the coordinates are supposed to be not aligned, please set stack_it = True but be aware that subsequent\n"
                 "results might be wrong.")

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

    mh = image.Filter.MatchHistograms([Axes.CH, Axes.ROUND])
    scaled = mh.run(stack, in_place=False, verbose=True, n_processes=8)
    return scaled


def autofluorescence_remover(stack, is_volume, masking_radius):
    '''
    Wraps the starfish White Top-Hat filter function to remove background
    '''
    from starfish.image import Filter
    
    masking_radius = masking_radius
    filt = Filter.WhiteTophat(masking_radius, is_volume=is_volume)
    filtered = filt.run(stack, verbose=True, in_place=False)
    return filtered


def deconvolver(stack, num_iter, sigma):
    from starfish.types import Levels
    from starfish.image import Filter

    dpsf = Filter.DeconvolvePSF(num_iter=num_iter, sigma=sigma, level_method=Levels.SCALE_SATURATED_BY_IMAGE)
    
    return dpsf.run(stack, verbose=True, in_place=False)


def spot_caller(stack, radius, is_volume, num_sigma, overlap, 
                threshold_percentile, measurement_type):
    '''
    Wrapper of the starfish function BlobDetector to detect spots
    '''
    from starfish.spots import FindSpots
    import math

    if is_volume:
        dev = 3
    else:
        dev = 2

    bd = FindSpots.BlobDetector(
        min_sigma=radius[0]/ math.sqrt(dev),
        max_sigma=radius[1]/ math.sqrt(dev),
        num_sigma=num_sigma,
        overlap = overlap,
        threshold=np.percentile(np.ravel(stack.xarray.values),threshold_percentile),
        exclude_border=False,
        is_volume = True,
        measurement_type=measurement_type) 
    return bd.run(stack)


def spot_decoder(exp, spots, anchor_round, search_radius):
    '''
    Wrapper of starfish functions to build traces (nearest neighbor) and
    decode spots with PerRoundMaxChannel method.
    '''
    from starfish.spots import DecodeSpots
    from starfish.types import TraceBuildingStrategies

    decoder = DecodeSpots.PerRoundMaxChannel(
        codebook=exp.codebook,
        anchor_round=anchor_round,
        search_radius=search_radius,
        trace_building_strategy=TraceBuildingStrategies.NEAREST_NEIGHBOR)

    decoded = decoder.run(spots=spots)

    decode_mask = decoded['target'] != 'nan'
    return decode_mask


def segmenter(stack, nuclei, dapi_threshold, stain_threshold, min_dist):
    '''
    Wrapper of starfish segmenting functions
    '''
    import numpy as np
    from starfish.image import Segment
    from starfish.types import Axes

    dapi_thresh = dapi_threshold  # binary mask for cell (nuclear) locations
    stain_thresh = stain_threshold #0.986 #0.014  # binary mask for overall cells // binarization of stain
    min_dist = min_dist

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


def run(exp, x_step = 2048, y_step = 2048, x_max = 2048, y_max = 2048, settings_path=None, fov=None, channels=[0,1,2,3], 
             rounds=None, zplanes=None, is_volume = True, test = False, transforms=None,
             stack_it=False, save_transforms=None, just_register=False, radius = (1, 5), num_sigma=10, overlap=1.0, 
             threshold_percentile=96.5, measurement_type="max", anchor_round=0, search_radius=19, deconvolution=True, num_iter=9, 
             sigma=1.5, whitetophat=True, masking_radius=5, equalise=True, roi_import=None,
             segmentation_starfish=True, dapi_threshold=0.26, stain_threshold=0.014, min_dist=56):

    '''
    Wrapper to run a full starfish pipeline from loading the relevant image planes, splitting them
    into tiles, transforming, spot detection and decoding and cell segmentation.

    Parameters:
        - exp: Starfish Experiment object containing the experiment to process.
        - x_step: Desired X-axis tile length in pixel.
        - y_step: Desired Y-axis tile length in pixel.
        - x_max: Total length of x-axis of FOV.
        - y_max: Total length of y-axis of FOV.
        - settings_path: Path to settings.yaml file. If None, will only consider function call settings. Default: None.
        - fov: Name of the Field of View to work on, if only one exists set it to None. This code is not suited to run
          on several FOVs, nor is it able to stitch FOVs into one image. Therefore, an already stitched image is required,
          which is here called a (stitched) FOV.
        - channels: List of channels to focus analysis on, e.g. [0,1,2] if the 4th channel is a nuclei channel.
          To use all channels, set to None. Default: [0,1,2,3].
        - rounds: List of rounds to focus analysis on, e.g. [1,2,3,4]. To use all rounds, set to None.
          Default: None.
        - zplanes: List of zplanes to focus analysis on. To use all zplanes, set to None. Default: None.
        - is_volume: Set to True if images have x, y and z dimensions. Default: True.
        - test: If set to True, only runs the pipeline on the first tile and estimates and prints total 
          run time of the pipeline based on the run time of a single loop. Returns estimated time as 
          days, hours, minutes and seconds. Default: False.
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
        - radius: Expected minimum and maximum spot radius in pixel, denoted as tuple in style (min_radius, max_radius).
          Default: (1,5)
        - num_sigma: Step between minimum and maximum radius to search for. Default: 10
        - overlap: Minimum overlap ratio between two spots required to consder them as one spot (values between 0 - 1). 
          Default: 1.0
        - threshold_percentile: Percentile of pixel intensity above which the average or maximum pixel intensity of a spot
          must lie to be counted, serves to sort out too dark spots/noise. Whether average or maximum is taken can be set
          with the "measurement_type" variable. Default: 96.5.
        - measurement_type: Defines for spot intensity thresholding whether average or maximum is taken. Possible values:
          "mean" and "max". Default: max.
        - anchor_round: Imaging round to use as anchor for spot decoding. All spots in this round will be traced down in 
          remaining rounds. Default: 0.
        - search_radius:  Radius [pixel] around spot to search for in remaining round for a spot found in anchor round. 
          Default: 19.
        - deconvolution: Set to True if the pipeline should perform deconvolution. Default: True.
        - num_iter: Number of iteration for point spread function estimation during deconvolution. Is ignored if 
          deconvolution = False. Default: 9
        - sigma: Sigma value for the devonvolution function. Is ignored if deconvolution = False. Default: 1.5.
        - whitetophat: Set to True if the pipeline should perform white Tophat filtering. Default: True.
        - masking radius: Masking radius [pixel] for white Tophat filtering. Is ignored if whitetophat = False. 
          Default: 5.
        - equalise: Set to True if the pipeline should perform intensity equalisation across channels and rounds.
          Default: True.
        - roi_import: Path to RoiSet.zip file with exported cell segmentation ROIs. Set to None to provide no ROIs.
        - segmentation_starfish: Set to True if the pipeline should perform cell segmentation. Is ignored if a ROI set
          is provided with the "roi_import" argument. Default: True.
        - dapi_threshold: Relative intensity threshold for the DAPI channel to perform binary thresholding for cell
          segmentation. Is ignored if segmentation_starfish is set to False. Default: 0.26.
        - stain_threshold: Relative intensity threshold for the spot channels to perform binary thresholding for cell
          segmentation. Is ignored if segmentation_starfish is set to False. Default: 0.014. 
        - min_dist: Minimum distance between two peaks (possibly nuclei) allowed during cell segmentation. Prevents over-
          segmentation but can also lead to merging of two nuclei into one. Is ignored if segmentation_starfish is set 
          to False. Default: 56. 
    
    Returns:
        - full_decoded: Dataframe with spots and spot parameters.
        - stack: ImageStack containing the imaging rounds with spots.
        - nuclei: ImageStack containing the nuclei images. 
        - gem: Gene expression matrix. Only returned if segmentation_starfish = True.
        - masks: Cell segmentation mask. Only returned if segmentation_starfish = True.
        - seg: Segmentation results. Only returned if roi_import = False.
    '''

    #Indicate start time
    start_time = datetime.now()
    print("The pipeline was started at:", datetime.now())
    
    #Retrieve all setting parameters from the settings file and keep settings that were set
    #through the function call, overwriting settings from the setting files (careful: only if they
    #do not match default settings). Simply put: The settings file contains permanent settings for 
    #the respective datasets. To test quick parameter changes, they can be adjusted in the function
    #call without need to change the settings file.

    #Retrieving the current parameters and values of the function run() defined through the function call
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    #Creating the settings dictionary with given parameters and the settings file
    s = settings_reader(settings_path, run, args, values)

    loop_counter = 0
    total_loops = len(range(0,s.x_max,s.x_step)) * len(range(0,s.y_max,s.y_step))

    #isolate the FOV if not provided
    if not s.fov:
        s.fov = exp.fovs()[0].name
    
    if s.transforms == None:
        print("Loads the full nuclei image")
        nuclei = stacker(exp, img_type="nuclei", x_edges=(0,s.x_max), y_edges=(0,s.y_max), fov=s.fov, channels=s.channels, rounds=s.rounds, zplanes=s.zplanes, stack_it=s.stack_it)
        print("Calculates offset between rounds by image registration")
        s.transforms = transformer(nuclei, transforms=s.transforms, save_transforms=s.save_transforms, just_register=True)
        #Stops the execution here if just_register is True
        if s.just_register:
            return
    print("Loads the full signal image")
    stack = stacker(exp, img_type="primary", x_edges=(0,s.x_max), y_edges=(0,s.y_max), fov=s.fov, channels=s.channels, rounds=s.rounds, zplanes=s.zplanes, stack_it=s.stack_it)
    print("Transforms the image according to calculated offset")
    stack = transformer(stack, transforms=s.transforms, save_transforms=False, just_register=False)
    #Equalise pixel intesities across channels and rounds
    if s.equalise:
        print("Equalises pixel intensities across channels and rounds")
        stack = intensity_equalizer(stack)
    #Uses a white top hat filter to remove background 
    if s.whitetophat:
        print("Removes background from images with White Top-Hat filter")
        stack = autofluorescence_remover(stack, s.is_volume, s.masking_radius)
    #Perform deconvolution to remove technical blur
    if s.deconvolution:
        print("Performs deconvolution on images")
        stack = deconvolver(stack, s.num_iter, s.sigma)
    
    #start loop
    for i in range(0, s.x_max, s.x_step):
        for j in range(0, s.y_max, s.y_step):
            loop_start = datetime.now()
            loop_counter += 1
            #extracts x and y dimensions of the tile
            x_edges = (i, i + s.x_step)
            y_edges = (j, j + s.y_step)

            #Loads a tile, either from a stack or from a file
            print(f"Starts processing tile {loop_counter} of {total_loops} at {datetime.now()}")
            print(f"...Available Memory before loading the tile: {format_bytes(get_available_memory())} bytes")
            #Extracts sub-stacks
            #stack_now = stack.sel({Axes.X: x_edges, Axes.Y: y_edges})
            stack_now = stack

            #Calls the starfish spot detection functions
            print("...Searches for spots")
            spots = spot_caller(stack_now, s.radius, s.is_volume, s.num_sigma, s.overlap, 
                                s.threshold_percentile, s.measurement_type)
            #To free memory the current stack is deleted as it is not needed anymore
            del stack_now
            #Calls the starfish spot decoding functions
            print("...Decodes spots")
            decoded = spot_decoder(exp, spots, s.anchor_round, s.search_radius)
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
            if s.test:
                break
        if s.test:
            break
    if s.test:
        loop_end = datetime.now()
        #calculates how long one loop took
        loop_time = loop_end - loop_start
        #involves the time for segmentation in the calculation
        print("Spot detection is finished!")
        #For cell segmentation it is necessary to know which channel is the nuclei channel. Nuclei images
        #can directly be loaded as stack. Or as integer indicating the channel number with nuclei staining.
        #here it it tests whether the nuclei variable is int and if yes, creates the stack
        nuclei = transformer(nuclei, transforms=None, save_transforms=False, just_register=False)
        #nuclei = autofluorescence_remover(nuclei, is_volume)
        #nuclei = nuclei.reduce(dims=[Axes.ROUND, Axes.CH, Axes.ZPLANE], func="max")
        if s.roi_import or s.segmentation_starfish:
            if s.roi_import != None:
                from starfish import BinaryMaskCollection
                dapi = nuclei.reduce(dims=[Axes.ROUND], func="max")
                masks = BinaryMaskCollection.from_fiji_roi_set(path_to_roi_set_zip=s.roi_import, original_image=dapi)
            elif s.segmentation_starfish:
                print("Segments cells")
                seg, masks = segmenter(stack, nuclei, s.dapi_threshold, s.stain_threshold, s.min_dist)
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
        print("Spot detection is finished!")
        #For cell segmentation it is necessary to know which channel is the nuclei channel. Nuclei images
        #can directly be loaded as stack. Or as integer indicating the channel number with nuclei staining.
        #here it it tests whether the nuclei variable is int and if yes, creates the stack
        nuclei = transformer(nuclei, transforms=None, save_transforms=False, just_register=False)
        #nuclei = autofluorescence_remover(nuclei, is_volume)
        #nuclei = nuclei.reduce(dims=[Axes.ROUND, Axes.CH, Axes.ZPLANE], func="max")
        if s.roi_import or s.segmentation_starfish:
            if s.roi_import != None:
                from starfish import BinaryMaskCollection
                dapi = nuclei.reduce(dims=[Axes.ROUND], func="max")
                masks = BinaryMaskCollection.from_fiji_roi_set(path_to_roi_set_zip=s.roi_import, original_image=dapi)
            elif s.segmentation_starfish:
                print("Segments cells")
                seg, masks = segmenter(stack, nuclei, s.dapi_threshold, s.stain_threshold, s.min_dist)

            print("Creates gene expression matrix")
            #Calls starfish functions that transforms results of spot decoding and segmentation into a gene
            #expression matrix
            gem = make_expression_matrix(masks, decoded)

        #Estimates, converts and returns runtime 
        runtime = datetime.now() - start_time
        days, hours, minutes, seconds = seconds_converter(runtime.seconds)
        print(f"The pipeline is finished at {datetime.now()}.\nIt took " 
              f"{days} day(s), {hours} hour(s), {minutes} minute(s) and {seconds} second(s).")
        if s.roi_import or s.segmentation_starfish:
            if s.roi_import:
                return full_decoded, stack, nuclei, gem, masks
            else:
                return full_decoded, stack, nuclei, gem, masks, seg
        else:
            return full_decoded, stack, nuclei
        
