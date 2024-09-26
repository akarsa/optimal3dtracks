"""
Created on Thu March 14 2024

@author: Anita Karsa, University of Cambridge, UK
"""

# In[]: Import all necessary tools

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

from tifffile import imread,imsave
from pathlib import Path
from csbdeep.utils import normalize
import pickle

from tqdm import tqdm

from skimage import measure
import pandas as pd
pd.options.mode.chained_assignment = None

import SimpleITK as sitk

import ot

import xml.etree.cElementTree as ET

# In[]

################### Helper functions for display and outputs ###################


# In[]: Display 3D + time image with segmentations

import ipywidgets as widgets
from ipywidgets import interact


def show_4d_with_contours(im, seg):
    """
    Visualize a 4D grayscale image with contour overlays of segmented regions.

    Args:
        im (ndarray): 4D grayscale image normalized between 0 and 1.
        seg (ndarray): 4D integer label map.

    Returns:
        None
    """
    
    # Get the number of timepoints and slices
    n_timepoints = im.shape[0]
    n_slices = im.shape[1]

    # Compute edges of segmented regions
    edges = scipy.ndimage.binary_erosion(seg, np.ones((1, 1, 5, 5), np.uint8), iterations=1)
    edges = seg * (1 - edges)
    max_label = np.max(edges)

    def update(timepoint, slice_num):
        # Round the timepoint and slice number
        timepoint = np.round(timepoint)
        slice_num = np.round(slice_num)

        # Create a new figure
        plt.figure()
        
        # Display the grayscale image
        plt.imshow(im[int(timepoint), int(slice_num), :, :], cmap='gray', interpolation='none', vmin=0, vmax=0.95)
        
        # Overlay contours of segmented regions
        plt.imshow(edges[int(timepoint), int(slice_num), :, :], vmax=max_label, cmap='prism',
                   alpha=0.5 * (edges[int(timepoint), int(slice_num), :, :] > 0), interpolation='none')
        
        # Add label numbers
        df = pd.DataFrame(measure.regionprops_table(edges[int(timepoint), int(slice_num), :, :].astype(int),
                                                     properties=["label", "centroid"]))
        for i in range(len(df)):
            plt.text(df["centroid-1"][i], df["centroid-0"][i], str(df["label"][i]), color='white', fontsize=12)

        # Show the plot
        plt.show()

    # Create widgets for timepoint and slice selection
    timepoint_widget = widgets.FloatSlider(
        value=int(n_timepoints / 2),
        min=0,
        max=n_timepoints - 1,
        step=1,
        description='Time: ',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='',
        style={'description_width': 'initial'}
    )
    slice_widget = widgets.FloatSlider(
        value=int(n_slices / 2),
        min=0,
        max=n_slices - 1,
        step=1,
        description='Slice: ',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='',
        style={'description_width': 'initial'}
    )
    
    # Interactively update the plot using widgets
    interact(update, timepoint=timepoint_widget, slice_num=slice_widget)
    
    
# In[]: Plot division tree

def generate_tree(track_df, split_df, merge_df):
    """
    Arrange nodes for better visualization of a cell lineage tree.

    Args:
        track_df (DataFrame): DataFrame containing the tracks from create_tracks.
                              Columns: 'label', 'track_id', 'timepoint'.
        split_df (DataFrame): DataFrame containing cell divisions from create_tracks.
                              Columns: 'parent', 'child_0', 'child_1', 'timepoint'.
        merge_df (DataFrame): DataFrame containing cell merges from create_tracks.
                              Columns: 'parent_0', 'parent_1', 'child', 'timepoint'.

    Returns:
        tuple: Tuple containing three arrays: tree, start, and end.
               - tree: An array representing the arranged nodes in the tree.
               - start: An array representing the start points for all nodes.
               - end: An array representing the end points for all nodes.
    """
    
    # Initialize tree
    tree = np.array([])

    # Loop through all time points
    for timepoint in np.unique(track_df['timepoint'].to_numpy()):

        # Turn DataFrame at timepoint into NumPy array
        split_np = split_df[['parent', 'child_0', 'child_1']][split_df['timepoint'] == timepoint].to_numpy()
        merge_np = merge_df[['parent_0', 'parent_1', 'child']][merge_df['timepoint'] == timepoint].to_numpy()
        track_np = track_df['track_id'][track_df['timepoint'] == timepoint].to_numpy()

        # First, add nodes based on split_df
        for parent in range(len(split_np)):
            # Find parent and children nodes
            parent_node = split_np[parent, 0]
            children = split_np[parent, 1:3]
            # Find position of parent node in the tree
            parent_id = np.squeeze(np.argwhere(tree == parent_node))
            # Insert children on both sides of the parent node
            tree = np.insert(tree, parent_id + 1, children[0])
            tree = np.insert(tree, parent_id, children[1])

        # Then, add the child node of the merging nodes
        for parent in range(len(merge_np)):
            # Find parents and child nodes
            parent_nodes = merge_np[parent, 0:2]
            child = merge_np[parent, 2]
            # Find position of parent nodes in the tree
            parent_ids = np.squeeze(np.argwhere(np.isin(tree, parent_nodes)))
            # Insert child node in between the parent nodes
            tree = np.insert(tree, int(np.floor(np.mean(parent_ids))), child)

        # Finally, add nodes from track_df
        track_np = track_np[track_np > 0]
        tree = np.insert(tree, len(tree), track_np[~np.isin(track_np, tree)])

    # Find starting and end points for all nodes
    start = np.zeros(tree.shape)
    end = np.zeros(tree.shape)
    all_frames = []
    for n in range(len(tree)):
        all_frames.append(np.concatenate(
            [track_df['timepoint'][track_df['track_id'] == tree[n]].to_numpy(),
             split_df['timepoint'][split_df['parent'] == tree[n]].to_numpy(),
             merge_df['timepoint'][merge_df['child'] == tree[n]].to_numpy()-1]))
        start[n] = np.min(all_frames[n])
        end[n] = np.max(all_frames[n])
        
    

    # Draw a graph
    fig,ax = plt.subplots(1)
    fig.set_size_inches(35, 20)

    for n in range(len(tree)):
        ax.plot([n,n], [-start[n],-end[n]], color='blue')
        ax.text(n, -start[n], str(int(tree[n])), fontsize=16)

    # Add horizontal lines for cell division events
    for parent_node in split_df['parent'].unique():
        all_children = split_df[['child_0', 'child_1']][split_df['parent'] == parent_node].to_numpy()
        all_xs = np.argwhere(np.isin(tree, all_children))
        y = end[np.argwhere(tree == parent_node)]
        ax.hlines(y=-y, xmin=np.min(all_xs), xmax=np.max(all_xs), colors='blue')
    # Add horizontal lines for cell merging events
    for child_node in merge_df['child'].unique():
        all_parents = merge_df[['parent_0', 'parent_1']][merge_df['child'] == child_node].to_numpy()
        all_xs = np.argwhere(np.isin(tree, all_parents))
        y = start[np.argwhere(tree == child_node)]
        ax.hlines(y=-y, xmin=np.min(all_xs), xmax=np.max(all_xs), colors='blue')
        
    ax.set_axis_off()
    
    # Add time bar
    t_end = np.max(track_df['timepoint'])
    bar_pos = len(tree) + 1
    ax.plot([bar_pos,bar_pos], [-t_end,-1], color = 'black', linewidth = 4)
    for t in np.arange(t_end)+1:
        ax.text(bar_pos+0.1,-t,str(int(t)),fontsize=32)

    return tree, start, end

# In[]: Top-down view plot of the segmented cells

def project_colours(label_image):
    """
    Project the colors of labeled regions in a 3D image volume onto a 2D projection.

    Args:
        label_image (ndarray): A 3D array representing labeled regions.

    Returns:
        ndarray: A 2D projection of the colors of labeled regions.
    """
    
    # Create a projection array with three channels (RGB)
    projection = np.zeros([label_image.shape[1], label_image.shape[2], 3])
    projection_red = projection[:, :, 0]  # Red channel
    projection_green = projection[:, :, 1]  # Green channel
    projection_blue = projection[:, :, 2]  # Blue channel
    
    # Compute the sum of label values along each slice
    slices = np.sum(label_image, axis=(1, 2))
    slices = np.where(slices > 0)
    
    # Determine the starting and ending slices for projection
    slice_0 = np.max([slices[0][0] - 10, 0])
    slice_end = np.min([slices[0][-1] + 10, label_image.shape[0] - 1])
    
    # Create a colormap
    cmap = np.zeros([1000, 3])
    for c in range(1000):
        np.random.seed(c)
        cmap[c, :] = np.random.uniform(low=0.0, high=1.0, size=[1, 3])
    
    # Project the colors of labeled regions onto the projection
    for s in range(int(slice_0), int(slice_end + 1)):
        current_slice = label_image[s, :, :].copy()
        # Red channel
        projection_red[current_slice > 0] = cmap[current_slice[current_slice > 0] % 1000, 0]
        # Green channel
        projection_green[current_slice > 0] = cmap[current_slice[current_slice > 0] % 1000, 1]
        # Blue channel
        projection_blue[current_slice > 0] = cmap[current_slice[current_slice > 0] % 1000, 2]

    return projection

# In[]

################### Helper functions for tracking ###################

# In[]: Calculate Gaussian mixture model parameters 

def calculate_Gaussian_parameters(label_files, intensity_files, save_folder, resolution):
    """
    Calculate Gaussian parameters for segmented cells based on intensity images.
    
    Parameters:
    - label_files (list of str): List of paths to segmentation label files.
    - intensity_files (list of str): List of paths to intensity image files.
    - save_folder (str): Path to the folder where results will be saved.
    - resolution (float): Resolution parameter for Gaussian fitting (eg. in um).
    
    Returns:
    - str: A message indicating the completion of the process.
    
    Raises:
    - AssertionError: If the first 7 characters of each filename in `label_files`
      and `intensity_files` are not equal.
    """

    # Assertion to check if the first 7 characters of each filename are equal
    # assert all(Path(input_file_1).name[0:6] == Path(input_file_2).name[0:6] 
    #            for input_file_1, input_file_2 in zip(intensity_files, label_files))

    # Loop over pairs of label and intensity files
    for label_file, intensity_file in zip(label_files, intensity_files):
        
        # Print the current label file being processed
        print(label_file)
        
        # Extract the name for the output file
        output_file_name = os.path.splitext(os.path.basename(intensity_file))[0][:] 
        
        # Load the segmentation image from the label file
        Y = imread(label_file,is_ome=False)
        
        # Load the intensity image from the intensity file
        X = imread(intensity_file,is_ome=False)     
        # Normalize intensity values to a certain percentile range
        X = normalize(X, 1, 99.8, axis=(0, 1, 2))
        
        # Check that images are the same size
        assert X.shape == Y.shape
        
        # Fit Gaussian mixture model to all segmented cells
        Gaussian_params = fit_Gaussian_mixture(X, Y, resolution)
            
        # Save the calculated Gaussian parameters using pickle
        with open(save_folder + '/' + output_file_name, "wb") as fp:   
            pickle.dump(Gaussian_params, fp)
            
    # Return a message indicating the completion of the process
    return "Finished calculating Gaussian parameters"
            
    
def get_moments(data, resolution):
    """
    Calculate ND Gaussian parameters of data.

    Args:
        data (ndarray): The input data.
        resolution (tuple): The resolution of the data in each dimension (usually in um).

    Returns:
        tuple: A tuple containing the integral, center, and width of the Gaussian.
               - integral (float): The integral (sum) of the data.
               - center (ndarray): The estimated center of the Gaussian.
               - width (ndarray): The estimated covariate matrix of the Gaussian.
    """
    # Calculate the total sum of the data
    total = data.sum()  
    # Create indices for each dimension
    XYZ = np.indices(data.shape).astype(float)  
    # Get the number of dimensions of the data
    d = len(data.shape)  # Get the number of dimensions of the data

    # Adjust indices for resolution
    for i in range(d):
        XYZ[i] *= float(resolution[i])

    center = np.zeros([d, 1])  # Initialize the center vector
    width = np.zeros([d, d])   # Initialize the covariance matrix

    # Estimate the center
    for i in range(d):
        center[i] = (XYZ[i] * data).sum() / total

    # Estimate the covariance matrix
    for i in range(d):
        for j in range(d):
            width[i, j] = ((XYZ[i] - center[i]) * (XYZ[j] - center[j]) * data).sum() / total

    # Calculate the integral
    integral = data.sum()

    return integral, center, width

def fit_Gaussian_mixture(im, seg, resolution):
    """
    Calculate Gaussian parameters of all segmented regions in an image.

    Args:
        im (ndarray): 3D grayscale image normalized between 0 and 1.
        seg (ndarray): 3D integer label map.
        resolution (tuple): Resolution of the image (usually in um).

    Returns:
        list: A list containing integrals, centers, widths, K, and labels.
              - integrals (list): List of integrals for each segmented region.
              - centers (list): List of estimated centers for each segmented region.
              - widths (list): List of estimated covariance matrices for each segmented region.
              - K (int): Number of segmented regions.
              - labels (ndarray): Array of unique labels.
    """
    # Find all unique labels
    labels = np.unique(seg)
    labels = labels[labels > 0]
    K = len(labels)

    # Get bounding boxes for each label
    label_props = pd.DataFrame(measure.regionprops_table(seg.astype(int), properties=["label", "bbox"]))

    # Initialize lists to store Gaussian parameters
    centers = []
    integrals = []
    widths = []

    # Estimate multivariate Gaussians within each region
    for lab in labels:
        # Crop both images at the bounding box
        box = label_props[["bbox-0", "bbox-1", "bbox-2", "bbox-3", "bbox-4", "bbox-5"]][label_props['label'] == lab].to_numpy()[0]
        image = im[box[0]:box[3], box[1]:box[4], box[2]:box[5]].copy()
        label_image = seg[box[0]:box[3], box[1]:box[4], box[2]:box[5]].copy()
        image[label_image != lab] = 0

        # Calculate Gaussian parameters using the get_moments function
        integral, center, width = get_moments(image, resolution)
        centers.append(np.ravel(center) + np.ravel(box[0:3])*resolution)
        integrals.append(integral)
        widths.append(width)

    return [integrals, centers, widths, K, labels.astype(int)]

# In[]: Calculate track sections between consecutive timepoints
  
def calculate_track_sections(label_files, intensity_files, Gaussian_parameter_files, frames, save_folder_for_affine,
                              save_folder_for_tracks, resolution, max_number_of_cells_per_timepoint):
    """
    Calculate track sections for segmented cells over consecutive time points.

    Parameters:
    - label_files (list of str): List of paths to segmentation label files.
    - intensity_files (list of str): List of paths to intensity image files.
    - Gaussian_parameter_files (list of str): List of paths to Gaussian parameter files.
    - frames (list of int): list of frame numbers
    - save_folder_for_affine (str): Path to the folder where affine transformation results will be saved.
    - save_folder_for_tracks (str): Path to the folder where track section results will be saved.
    - resolution (float): Resolution parameter for affine registration (eg. in um).
    - max_number_of_cells_per_timepoint (int): An upper limit of number of cells segmented in any given frame.

    Returns:
    - str: A message indicating the completion of the process.

    Raises:
    - AssertionError: If the first 7 characters of each filename in `intensity_files` and `label_files`,
      or in `Gaussian_parameter_files` and `intensity_files` are not equal.
    """
    
    # # Read the frame numbers from the label file name
    # frames = [int(os.path.splitext(os.path.basename(label_file))[0][1:5]) for label_file in label_files]

    # Assert that the first 7 characters of each filename in `intensity_files` and `label_files` are equal
    # assert all(Path(input_file_1).name[0:6] == Path(input_file_2).name[0:6] 
    #            for input_file_1, input_file_2 in zip(intensity_files, label_files))

    # Assert that the first 7 characters of each filename in `Gaussian_parameter_files` and `intensity_files` are equal
    # assert all(Path(input_file_1).name[0:6] == Path(input_file_2).name[0:6] 
    #            for input_file_1, input_file_2 in zip(intensity_files, Gaussian_parameter_files))

    # Loop over pairs of consecutive label files
    for file_num in range(len(label_files) - 1):
        
        print(label_files[file_num])
        frame = frames[file_num]
        
        # Check that consecutive files are from consecutive frames
        assert frames[file_num] + 1 == frames[file_num + 1]

        # Load Gaussian parameters for consecutive time points
        with open(Gaussian_parameter_files[file_num], "rb") as fp:
            pi0, mu0, S0, K0, lab_start = pickle.load(fp)
        with open(Gaussian_parameter_files[file_num + 1], "rb") as fp:
            pi1, mu1, S1, K1, lab_target = pickle.load(fp)

        # Adjust labels to differentiate them between consecutive frames
        lab_target += (frame + 1) * max_number_of_cells_per_timepoint
        lab_start += frame * max_number_of_cells_per_timepoint

        # Affine register intensity images
        X_next = imread(intensity_files[file_num + 1],is_ome=False)
        fixed = sitk.GetImageFromArray(X_next.astype(float))
        fixed.SetSpacing(np.flip(resolution))  # Note that the dimensions in SimpleITK are flipped
        affine_file = save_folder_for_affine + '/' + \
                      os.path.splitext(os.path.basename(Gaussian_parameter_files[file_num]))[0] + '.tfm'

        if os.path.isfile(affine_file): # if an affine file already exists, use that
            affine = sitk.ReadTransform(affine_file)
        else:
            X = imread(intensity_files[file_num],is_ome=False)
            moving = sitk.GetImageFromArray(X.astype(float))
            moving.SetSpacing(fixed.GetSpacing())
            affine = affine_registration(fixed, moving)
            sitk.WriteTransform(affine, affine_file)
            
        mu0_corr = np.concatenate([np.reshape(np.flip(
            affine.GetInverse().TransformPoint(np.flip(mu))),
            [1,3]) for mu in mu0], axis=0)

        # # Transform labels of start frame using affine transformation
        # # start_frame = imread(label_files[file_num],is_ome=False)
        # # start_frame[start_frame>0] = start_frame[start_frame>0] + frame * max_number_of_cells_per_timepoint
        # start_frame = np.zeros(X_next.shape)
        # indices = (mu0/resolution).astype(int)
        # start_frame[indices[:,0],indices[:,1],indices[:,2]] = lab_start
        # start_frame[indices[:,0]+1,indices[:,1],indices[:,2]] = lab_start 
        # start_frame[indices[:,0]-1,indices[:,1],indices[:,2]] = lab_start 
        # start_frame[indices[:,0],indices[:,1]+1,indices[:,2]] = lab_start 
        # start_frame[indices[:,0],indices[:,1]-1,indices[:,2]] = lab_start 
        # start_frame[indices[:,0],indices[:,1],indices[:,2]+1] = lab_start 
        # start_frame[indices[:,0],indices[:,1],indices[:,2]-1] = lab_start 

        # start_frame_sitk = sitk.GetImageFromArray(start_frame.astype(float))
        # start_frame_sitk.SetSpacing(fixed.GetSpacing())
        # out = sitk.Resample(start_frame_sitk, fixed, affine, sitk.sitkNearestNeighbor, 0.0,
        #                     start_frame_sitk.GetPixelID())
        # registered = sitk.GetArrayFromImage(out)

        # # Get new coordinates
        # start_frame_props = pd.DataFrame(measure.regionprops_table(registered.astype(int),
        #                                                            properties=["label", "centroid"]))

        # if set(lab_start) == set(start_frame_props['label']):
        #     mu0_corr = np.concatenate(
        #         [start_frame_props[['centroid-0', 'centroid-1', 'centroid-2']][start_frame_props['label'] == lab] *
        #          resolution for lab in lab_start])
        #     print('Affine propagation finished!')
        # else:
        #     print('Affine propagation failed!')

        # Calculate transition matrix and create tracks
        regularisation_parameter = 2  # Regularisation parameter for sinkhorn regularisation of gmmot
        d = 3  # For 3D
        pi0 /= np.sum(pi0)
        pi1 /= np.sum(pi1)
        transition_matrix = GW2_ak(pi0, pi1, mu0_corr, mu1, S0, S1, K0, K1, d,
                                   regularisation_parameter)

        # Check marginals of transition matrix
        print('Marginals of transition matrix (ideally both <0.01) = ' + 
              str(np.linalg.norm(np.sum(transition_matrix[:, :], axis=1) - pi0) / len(pi0)) +
              ' , ' +
              str(np.linalg.norm(np.sum(transition_matrix[:, :], axis=0) - pi1) / len(pi1)))

        # Create tracks and record merges and splits
        include_merge = False  # Allowing merges to be detected could overcomplicate tracking
        # and it's usually easier to add merges manually because they are very rare
        track_section, split_section, merge_section, _ = \
            create_tracks(lab_start, lab_target, transition_matrix, frame + 1, include_merge)

        # Save track section
        with open(save_folder_for_tracks + '/' +
                  os.path.splitext(os.path.basename(Gaussian_parameter_files[file_num]))[0], "wb") as fp:
            pickle.dump([transition_matrix, track_section, split_section, merge_section], fp)

    # Return a message indicating the completion of the process
    return "Finished calculating track sections"

# In[]: Concatenate track sections

def concatenate_track_sections(track_files,Gaussian_parameter_files,save_folder,max_number_of_cells_per_timepoint):
    """
    Concatenate track sections from multiple time points into full tracks.
    
    Parameters:
    - track_files (list of str): List of paths to track section files.
    - Gaussian_parameter_files (list of str): List of paths to Gaussian parameter files.
    - save_folder (str): Path to the folder where the concatenated track data will be saved.
    
    Returns:
    - tuple of pandas.DataFrame: Tuple containing the track, split, and merge DataFrames.
    """
    
    #Initialise dataframes using the first Gaussian parameter file
    with open(Gaussian_parameter_files[0], "rb") as fp:   # Unpickling
        _,_,_,_,lab_start = pickle.load(fp)
    
    split_df = pd.DataFrame(columns=['timepoint','parent', 'child_0', 'child_1'])
    merge_df = pd.DataFrame(columns=['timepoint', 'parent_0', 'parent_1', 'child'])
    track_df = pd.DataFrame({'timepoint': 1, 
                             'label': lab_start + max_number_of_cells_per_timepoint, 
                             'track_id': lab_start + max_number_of_cells_per_timepoint})
    
    # Iterate over all time points to identify tracks, splits, and, merges
    for i in range(len(track_files)):
        
        # Frame number
        frame = i+1
        
        # Load track sections
        with open(track_files[i], "rb") as fp:   
            _, track_section, split_section, merge_section = pickle.load(fp)
        
        # Propagate track_ids
        track_section_prev = track_df[track_df['timepoint']==frame]
        
        # Propagate track_ids in track_df
        updates = np.isin(track_section['track_id'],track_section_prev['label'])
        track_section['track_id'][updates] = [track_section_prev['track_id'][track_section_prev['label']==lab].to_numpy()[0] 
                                              for lab in track_section['track_id'][updates]]
        # Propagate track_ids in split_df
        updates = np.isin(split_section['parent'],track_section_prev['label'])
        split_section['parent'][updates] = [track_section_prev['track_id'][track_section_prev['label']==lab].to_numpy()[0]
                                              for lab in split_section['parent'][updates]]
        # Propagate track_ids in merge_df
        updates = np.isin(merge_section['parent_0'],track_section_prev['label'])
        merge_section['parent_0'][updates] = [track_section_prev['track_id'][track_section_prev['label']==lab].to_numpy()[0]
                                              for lab in merge_section['parent_0'][updates]]
        updates = np.isin(merge_section['parent_1'],track_section_prev['label'])
        merge_section['parent_1'][updates] = [track_section_prev['track_id'][track_section_prev['label']==lab].to_numpy()[0]
                                              for lab in merge_section['parent_1'][updates]]
            
        # Append sections to overall tracks
        track_df = pd.concat([track_df,track_section])
        merge_df = pd.concat([merge_df,merge_section])
        split_df = pd.concat([split_df,split_section])
        
    # Save full tracks
    with open(save_folder + '/Tracks_full', "wb") as fp:   
        pickle.dump([track_df, split_df, merge_df], fp)
        
    # Return track, split, and merge DataFrames
    return track_df, split_df, merge_df

# In[]: Affine registration using Simple ITK
# This improves the tracking by registering consecutive intensity images  
    
def affine_registration(fixed_image, moving_image):    
    """
    Calculates affine transformation between fixed image and moving image.

    Args:
        fixed_image (sitk image): Fixed image.
        moving_image (sitk image): Moving image.

    Returns:
        sitk transformation: Affine transformation between fixed image and moving image.
    """
    
    initial = sitk.CenteredTransformInitializer(fixed_image,moving_image,sitk.AffineTransform(3),sitk.CenteredTransformInitializerFilter.MOMENTS)
        
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=500, 
                                                      estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(initial, inPlace=False)
 
    return registration_method.Execute(fixed_image, moving_image)

# In[]: Optimal transport with sinkhorn regularisation

def GW2_ak(pi0, pi1, mu0, mu1, S0, S1, K0, K1, d, reg_param):
    """
    Compute the GW2 discrete map between two Gaussian Mixture Models (GMMs).
    A modifed version of the function GW2 from the gmmot demo (https://github.com/judelo/gmmot)

    Args:
        pi0 (ndarray): Weights of the GMM components for the first GMM.
        pi1 (ndarray): Weights of the GMM components for the second GMM.
        mu0 (ndarray): Means of the GMM components for the first GMM.
        mu1 (ndarray): Means of the GMM components for the second GMM.
        S0 (ndarray): Covariance matrices of the GMM components for the first GMM.
        S1 (ndarray): Covariance matrices of the GMM components for the second GMM.
        K0 (int): Number of starting labels.
        K1 (int): Number of target labels.
        d (int): Number of dimensions.
        reg_param (float): Regularisation parameter for sinkhorn.

    Returns:
        ndarray: The GW2 discrete map between the two GMMs.
    """
    # Prepare parameters
    pi0 /= np.sum(pi0)
    pi1 /= np.sum(pi1)
    mu0 = np.reshape(mu0,[K0,d])
    mu1 = np.reshape(mu1,[K1,d])
    S0 = np.reshape(S0,[K0,d,d])
    S1 = np.reshape(S1,[K1,d,d])
    
    # Initialize distance matrix
    M = np.zeros((K0, K1))
    
    # Compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            # Compute Gaussian Wasserstein distance between Gaussian k in GMM 0 and Gaussian l in GMM 1
            M[k, l] = GaussianW2(mu0[k, :], mu1[l, :], S0[k, :, :], S1[l, :, :])
    
    # Compute the optimal transport plan using Sinkhorn algorithm (this is the part that was modified by AK)
    wstar = ot.sinkhorn(np.ravel(pi0), np.ravel(pi1), M, reg_param)#, numItermax=10000, stopThr=1e-9)
    
    return wstar

def GaussianW2(m0,m1,Sigma0,Sigma1):
    """
    Compute the quadratic Wasserstein distance between two Gaussians.
    A function from the gmmot demo (https://github.com/judelo/gmmot)

    Args:
        mu0 (ndarray): Means of the GMM components for the first GMM.
        mu1 (ndarray): Means of the GMM components for the second GMM.
        Sigma0 (ndarray): Covariance matrices of the GMM components for the first GMM.
        Sigma1 (ndarray): Covariance matrices of the GMM components for the second GMM.

    Returns:
        ndarray: The quadratic Wasserstein distance between the two GMMs.
    """
    # compute the quadratic Wasserstein distance between two Gaussians with means m0 and m1 and covariances Sigma0 and Sigma1
    Sigma00  = scipy.linalg.sqrtm(Sigma0)
    Sigma010 = scipy.linalg.sqrtm(Sigma00@Sigma1@Sigma00)
    d        = np.linalg.norm(m0-m1)**2+np.trace(Sigma0+Sigma1-2*Sigma010)
    return d
    
# In[]: Create tracks from transition matrices by searching for valid transitions

def create_tracks(start_track_ids,target_labels,transition_matrix,time_point,include_merge):
    """
    Create tracks from the transition_matrix by identifying the maximum probability
    valid transition matrix

    Args:
        start_track_ids (integer array): track_ids of the cells at the start point (i.e. columns).
        target_labels (integer array): Labels of cells at the end point (i.e. rows).
        transition_matrix (ndarray): Transition probability matrix.
        time_point (int): Time point or frame number (of the child cells).
        include_merge (bool): Indicator for whether or not merges are allowed. 

    Returns:
        tuple: A tuple containing the track_section, split_section, merge_section, target_track_ids.
               - track_section (DataFrame): DataFrame containing the tracks from timepoint-1 to timepoint.
                                            Columns: 'label', 'track_id', 'timepoint'.
               - split_section (DataFrame): DataFrame containing cell divisions from timepoint-1 to timepoint.
                                            Columns: 'parent', 'child_0', 'child_1', 'timepoint'.
               - merge_section (DataFrame): DataFrame containing cell merges from timepoint-1 to timepoint.
                                            Columns: 'parent_0', 'parent_1', 'child', 'timepoint'.
               - target_track_ids (integer array): Updated track_ids at timepoint 
                                                   (this will be the start_track_ids array for the next step) 
    """
    
    # initialise split and merge dataframes
    split_section = pd.DataFrame(columns=['timepoint','parent', 'child_0', 'child_1'])
    merge_section = pd.DataFrame(columns=['timepoint', 'parent_0', 'parent_1', 'child'])
    
    # convert transition matrix into a valid matrix
    match include_merge:
        case True:
            matrix = valid_transition_ver_2(transition_matrix)>0
        case False:
            matrix = valid_transition_ver_3(transition_matrix)>0   
    
    # exclude 0 rows
    matrix[start_track_ids==0,:] = 0
    
    # assign track ids to target_labels (this does not take merging or splitting into account so far)
    target_track_ids = np.max(matrix * np.reshape(start_track_ids,[-1,1]),axis = 0)
    
    # for new or "new" cells, assign their target_label
    target_track_ids[target_track_ids==0] = target_labels[target_track_ids==0]
    
    # look for mergers
    merge_targets = np.where(np.sum(matrix, axis = 0) == 2)[0]
    for ind in merge_targets:
        # add merging cells to the merge dataframe
        parents_ids = start_track_ids[np.where(matrix[:,ind]==1)[0]]
        merge_section = pd.concat([merge_section, pd.DataFrame({'timepoint': [time_point], \
                                                                'parent_0': [parents_ids[0]], \
                                                                'parent_1': [parents_ids[1]], \
                                                                'child': [target_labels[ind]]})])
        # children of merged cells should have their own track id
        target_track_ids[ind] = target_labels[ind]
                                 
    # look for splits
    split_sources = np.where(np.sum(matrix, axis = 1) == 2)[0]
    for ind in split_sources:
        # add merging cells to the merge dataframe
        children_ids = target_labels[np.where(matrix[ind,:]==1)[0]]
        split_section = pd.concat([split_section, pd.DataFrame({'timepoint': [time_point], \
                                                                'parent': [start_track_ids[ind]], \
                                                                'child_0': [children_ids[0]], \
                                                                'child_1': [children_ids[1]]})])
        # children of splitting cells should have their own track id
        target_track_ids[np.isin(target_labels,children_ids)] = children_ids
        
    # assign track ids
    track_section = pd.DataFrame({'timepoint': time_point, 'label': target_labels, 'track_id': target_track_ids})
    
    return track_section, split_section, merge_section, target_track_ids
    
def valid_transition_ver_2(transition_matrix):
    """
    Turn transition_matrix into the highest probability valid transition matrix 
    where cells may not split into more than two pieces and no more than two cells can merge at a time

    Args:
        transition_matrix (ndarray): transition probability matrix.

    Returns:
        ndarray: highest probability valid transition matrix 
        
    """
    
    transition = np.copy(transition_matrix)

    # max 2 in each row and column with a ratio at least 1:5
    mask = np.ones(transition.shape)
    top2_per_col = np.sort(transition,axis=0)[-2:,:]
    top2_per_col[-1,:] /= 5
    mask[(transition - np.max(top2_per_col,axis = 0,keepdims=True))<0] = 0
    top2_per_row = np.sort(transition,axis=1)[:,-2:]
    top2_per_row[:,-1] /= 5
    mask[(transition - np.max(top2_per_row,axis = 1,keepdims=True))<0] = 0
    mask[transition==0] = 0
    transition*=mask

    # sort rest into 'connected' components
    connected_colour = 2
    mask_connect = np.copy(mask).astype(int)
    while np.sum(mask_connect==1)>0:
        # pick a connection
        x,y = np.where(mask_connect==1)
        x = x[0]
        y = y[0]
        # colour it
        mask_next = np.copy(mask_connect)
        # propagate colour across the mask
        mask_next[x,y] = connected_colour
        while np.sum(mask_connect-mask_next)!=0:
            x,y = np.where(mask_next==connected_colour)
            mask_connect += mask_next - mask_connect # this is essentially mask_connect = mask_next but I didn't want to use assignment or copy
            mask_next[x,:] = mask[x,:] * connected_colour
            mask_next[:,y] = mask[:,y] * connected_colour

        # increase colour
        connected_colour += 1
    
    # find maximum valid combination per colour

    for colour in range(np.max(mask_connect)):

        coordinates = np.where(mask_connect==colour+1)
        values = transition[mask_connect==colour+1]
        n_nodes = len(values)

        probabilities = []

        for combination in range(2**n_nodes):
            # generate a combination of nodes
            all_nodes = np.zeros(n_nodes).astype(bool)
            select_nodes = np.array(list(str(bin(combination))[2:])).astype(bool)
            all_nodes[-(len(select_nodes)):] = select_nodes

            # check validity of combination
            x = coordinates[0][all_nodes]
            y = coordinates[1][all_nodes]
            validity = np.prod(1-((return_counts(x)==2) * (return_counts(y)==2))) # if any node has both its 
            #coordinates appear twice in the list -> validity = 0

            # add probability of combination to probabilities list
            probabilities.append(validity*np.sum(values[all_nodes]))

        # calculate optimal combination and remove the rest of the values from the transition matrix
        optimal_combination = int(np.where(probabilities == np.max(probabilities))[0])
        all_nodes = np.zeros(n_nodes).astype(bool)
        select_nodes = np.array(list(str(bin(optimal_combination))[2:])).astype(bool)
        all_nodes[-(len(select_nodes)):] = select_nodes

        transition[mask_connect==colour+1] *= all_nodes
        
    return transition

def valid_transition_ver_3(transition_matrix):
    """
    Turn transition_matrix into the highest probability valid transition matrix 
    where cells may not split into more than two pieces and are not allowed to merge 
    
    Args:
        transition_matrix (ndarray): transition probability matrix.

    Returns:
        ndarray: highest probability valid transition matrix 
        
    """

    transition = np.copy(transition_matrix)

    # max 2 in each row with a ratio at least 1:5
    mask = np.ones(transition.shape)
    top2_per_row = np.sort(transition,axis=1)[:,-2:]
    top2_per_row[:,-1] /= 5
    mask[(transition - np.max(top2_per_row,axis = 1,keepdims=True))<0] = 0
    mask[transition==0] = 0
    transition*=mask
    
    # sort rest into 'connected' components
    
    connected_colour = 2
    mask_connect = np.copy(mask).astype(int)
    while np.sum(mask_connect==1)>0:
        # pick a connection
        x,y = np.where(mask_connect==1)
        x = x[0]
        y = y[0]
        # colour it
        mask_next = np.copy(mask_connect)
        # propagate colour across the mask
        mask_next[x,y] = connected_colour
        while np.sum(mask_connect-mask_next)!=0:
            x,y = np.where(mask_next==connected_colour)
            mask_connect += mask_next - mask_connect # this is essentially mask_connect = mask_next but I didn't want to use assignment or copy
            mask_next[x,:] = mask[x,:] * connected_colour
            mask_next[:,y] = mask[:,y] * connected_colour

        # increase colour
        connected_colour += 1
    
    # find maximum valid combination per colour
   
    for colour in range(np.max(mask_connect)):
        
        coordinates = np.where(mask_connect==colour+1)
        values = transition[mask_connect==colour+1]
        n_nodes = len(values)

        probabilities = []

        for combination in range(2**n_nodes):
            # generate a combination of nodes
            all_nodes = np.zeros(n_nodes).astype(bool)
            select_nodes = np.array(list(str(bin(combination))[2:])).astype(bool)
            all_nodes[-(len(select_nodes)):] = select_nodes

            # check validity of combination
            x = coordinates[0][all_nodes]
            y = coordinates[1][all_nodes]
            validity = np.prod(1-((return_counts(y)>1))) # if y 
            #coordinate appears twice in the list -> validity = 0

            # add probability of combination to probabilities list
            probabilities.append(validity*np.sum(values[all_nodes]))

        # calculate optimal combination and remove the rest of the values from the transition matrix
        optimal_combination = int(np.where(probabilities == np.max(probabilities))[0][-1]) # The -1 here is selecting the last option on the list of max probabilities.
                                                                                           # This is because e.g. 1e-6 + 1e-20 = 1e-6 for the computer but the right side is
                                                                                           # actually, mathematically larger. Selecting the last option is selecting the
                                                                                           # option with the most nodes included.
        all_nodes = np.zeros(n_nodes).astype(bool)
        select_nodes = np.array(list(str(bin(optimal_combination))[2:])).astype(bool)
        all_nodes[-(len(select_nodes)):] = select_nodes

        transition[mask_connect==colour+1] *= all_nodes
        
    return transition

def return_counts(numpy_array):
    """
    Count the occurrences of each unique element in the input numpy array.

    Parameters:
    - numpy_array (ndarray):
        The input array for which counts of unique elements are calculated.

    Returns:
    - ndarray:
        An array containing the count of each element in the input numpy array.
    """
    # count the occurrences of each unique element in the input numpy array
    count_unique_elements = np.unique(numpy_array, return_counts=True)
    # create a dictionary mapping each unique element to its count
    count_dict = dict(zip(count_unique_elements[0], count_unique_elements[1]))
    # create an array containing the count of each element in the input numpy array
    counts = np.array([count_dict[i] for i in numpy_array])
    return counts

# In[]

################### Helper functions for track corrections ###################

# In[]: Save intensity image for TrackMate

def save_intensity_for_TrackMate(intensity_files, dimensions, resolution, save_folder):
    """
    Save intensity files for TrackMate.

    Parameters:
        intensity_files (list): List of file paths of intensity images.
        dimensions (int): Dimensionality of the output images (2 for 2D i.e. MIP, 3 for full 3D).
        resolution (tuple): The resolution of the data in each dimension (usually in um).
        save_folder (str): Path to the folder where the intensity file will be saved.

    Returns:
        str: Confirmation message indicating the save location.
    """
    
    # Check if dimensions are either 2 or 3
    assert np.isin(dimensions, [2, 3]) == True

    # Depending on the dimensionality of the output images, concatenate and process them differently
    match dimensions:
        case 2:
            # For 2D images, concatenate along the first axis after taking the maximum intensity along the first axis
            intensity = np.concatenate([np.max(imread(int_file,is_ome=False), axis=0, keepdims=True) 
                                        for int_file in tqdm(intensity_files)], axis=0)
            intensity = np.expand_dims(intensity, axis=1)
        case 3:
            # For 3D images, expand dimensions and concatenate along the first axis
            intensity = np.concatenate([np.expand_dims(imread(int_file,is_ome=False), axis=0) 
                                        for int_file in tqdm(intensity_files)], axis=0)
            
    # Save the intensity file as a TIFF image
    imsave(save_folder + '/intensity.tif', intensity, imagej=True, resolution=1/resolution[1:3],
          metadata={
              'spacing': resolution[0],
              'unit': 'um',
              'axes': 'TZYX'
          })
    
    # Return a confirmation message
    return "Intensity file saved for TrackMate in " + save_folder
    
    
    
# In[]: Save tracks as .xlm readable by TrackMate
    
def save_as_TrackMate(track_df,split_df,merge_df,label_files,frames,max_number_of_cells_per_timepoint,dim,resolution,base_file,save_file):
    """
    Create tracks from the transition_matrix by identifying the maximum probability
    valid transition matrix

    Args:
        track_section (DataFrame): DataFrame containing the tracks from timepoint-1 to timepoint.
                                   Columns: 'label', 'track_id', 'timepoint'.
        split_section (DataFrame): DataFrame containing cell divisions from timepoint-1 to timepoint.
                                   Columns: 'parent', 'child_0', 'child_1', 'timepoint'.
        merge_section (DataFrame): DataFrame containing cell merges from timepoint-1 to timepoint.
                                   Columns: 'parent_0', 'parent_1', 'child', 'timepoint'.
        label_files (list of str): List of paths to segmentation label files.
        frames (list or array of int): List or array of frame numbers (same as in eg. track_df)
        max_number_of_cells_per_timepoint (int): An upper limit of number of cells segmented in any given frame.
        dim (int): Dimensionality of the intensity images (2 for 2D i.e. MIP, 3 for full 3D).
        resolution (tuple): The resolution of the data in each dimension (usually in um).
        base_file (str): Path of the base file to be used for building the .xml file
        save_file (str): Path to the file where the .xml file will be saved.

    Returns:
        str: Confirmation message indicating the save location.
    """
    
    # Number of frames
    n_frames = len(label_files)
    
    # Load base file
    TrackMate = ET.parse(base_file)
    
    # Model
    Model = TrackMate.findall('Model')[0]
    
    Model.set('spatialunits','micron') 
    Model.set('timeunits','frame')
    
    # Fill up AllSpots by loading each frame
    AllSpots = Model.findall('AllSpots')[0]
    AllSpots.clear()
    AllSpots.set('nspots',str(len(track_df)))
    for frame,label_file in zip(frames,label_files):
        # Set up frame
        FrameSpots = ET.SubElement(AllSpots, "SpotsInFrame")
        FrameSpots.set('frame',str(frame-1))
        # Load image and calculate properties
        track_image = imread(label_file,is_ome=False)
        props = pd.DataFrame(measure.regionprops_table(track_image.astype(int),\
                              spacing = resolution,\
                              properties=["label", "centroid", "equivalent_diameter_area"]))
        props['label'] += frame * max_number_of_cells_per_timepoint
        # Slice track_df
        track_section = track_df[track_df['timepoint']==frame]
        for label in track_section['label']:
            # Slice props attributes
            current_spot = props[props['label']==label]
            # Add a spot and set attributes
            Spot = ET.SubElement(FrameSpots, "Spot")
            Spot.set('ID', str(label))
            Spot.set('name', str(label))
            Spot.set('STD_INTENSITY_CH1', '0')
            Spot.set('QUALITY', '0')
            Spot.set('POSITION_T', str(frame-1))
            Spot.set('MIN_INTENSITY_CH1', '0')
            Spot.set('TOTAL_INTENSITY_CH1', '0')
            Spot.set('CONTRAST_CH1', '0')
            Spot.set('SNR_CH1', '0')
            Spot.set('FRAME', str(frame-1))
            Spot.set('MEDIAN_INTENSITY_CH1', '0')
            Spot.set('VISIBILITY', '1')
            Spot.set('RADIUS', str(current_spot['equivalent_diameter_area'].to_numpy()[0]/2))
            Spot.set('POSITION_X', str(current_spot['centroid-2'].to_numpy()[0]))
            Spot.set('POSITION_Y', str(current_spot['centroid-1'].to_numpy()[0]))
            Spot.set('MEAN_INTENSITY_CH1', '0')
            match dim:
                case 3:
                    Spot.set('POSITION_Z', str(current_spot['centroid-0'].to_numpy()[0]))
                case 2:
                    Spot.set('POSITION_Z', str(0))
            Spot.set('MAX_INTENSITY_CH1', '0')
            
    print('All Spots saved')

    # Fill up AllTracks by loading each frame
    AllTracks = Model.findall('AllTracks')[0]
    AllTracks.clear()
    FilteredTracks = Model.findall('FilteredTracks')[0]
    FilteredTracks.clear()
    # create track names
    track_df['track_name'] = track_df['track_id']
    split_df['track_name'] = split_df['parent']
    merge_df['track_name'] = merge_df['child']
    
    for parent,child_1,child_2,track_name in \
        zip(split_df['parent'],split_df['child_0'],\
            split_df['child_1'],split_df['track_name']):
            track_df['track_name'][np.isin(track_df['track_name'],[child_1,child_2])] = \
                track_name
            split_df['track_name'][np.isin(split_df['parent'],[child_1,child_2])] = \
                track_name    

    for parent_1,parent_2,child,track_name in \
        zip(merge_df['parent_0'][::-1],merge_df['parent_1'][::-1],\
            merge_df['child'][::-1],merge_df['track_name'][::-1]):
            track_name_1 = int(track_df['track_name'][track_df['label']==parent_1])
            track_name_2 = int(track_df['track_name'][track_df['label']==parent_2])
            track_df['track_name'][np.isin(track_df['track_name'],[track_name_1,track_name_2])] = \
                track_name
            split_df['track_name'][np.isin(split_df['track_name'],[track_name_1,track_name_2])] = \
                track_name  
    
    # Single cell "tracks" are not counted as tracks
    tracks_names = track_df['track_name'].unique()[track_df['track_name'].sort_values().value_counts(sort=False)>1]
    n_tracks = len(tracks_names)
    
    print('All Tracks identified')
    
    for track_name in tracks_names:
        
        # Update FilteredTracks
        TrackID = ET.SubElement(FilteredTracks, "TrackID")
        TrackID.set('TRACK_ID', str(track_name))
        
        track_section = track_df[track_df['track_name']==track_name]
        split_section = split_df[split_df['track_name']==track_name]
        merge_section = merge_df[merge_df['track_name']==track_name]
        
        Track = ET.SubElement(AllTracks, "Track")
        Track.set('name','Track_' + str(track_name))
        Track.set('TRACK_ID',str(track_name))
        Track.set('TRACK_INDEX',str(track_name))
        Track.set('NUMBER_SPOTS',str(len(track_section)))
        Track.set('NUMBER_GAPS','0')
        Track.set('NUMBER_SPLITS',str(len(split_section)))
        Track.set('NUMBER_MERGES',str(len(merge_section)))
        Track.set('NUMBER_COMPLEX','0')
        Track.set('LONGEST_GAP','0')
        Track.set('TRACK_START',str(np.min(track_section['timepoint'])-1))
        Track.set('TRACK_STOP',str(np.max(track_section['timepoint'])-1))
        Track.set('TRACK_DURATION',str(int(Track.get('TRACK_STOP'))-int(Track.get('TRACK_START'))))
        Track.set('TRACK_DISPLACEMENT','0')
        Track.set('TRACK_X_LOCATION','0')
        Track.set('TRACK_Y_LOCATION','0')
        Track.set('TRACK_Z_LOCATION','0')
        Track.set('TRACK_MEAN_SPEED','0')
        Track.set('TRACK_MAX_SPEED','0')
        Track.set('TRACK_MIN_SPEED','0')
        Track.set('TRACK_MEDIAN_SPEED','0')
        Track.set('TRACK_STD_SPEED','0')
        Track.set('TRACK_MEAN_QUALITY','0')
        Track.set('TOTAL_DISTANCE_TRAVELED','0')
        Track.set('MAX_DISTANCE_TRAVELED','0')
        Track.set('CONFINEMENT_RATIO','0')
        Track.set('MEAN_STRAIGHT_LINE_SPEED','0')
        Track.set('LINEARITY_OF_FORWARD_PROGRESSION','0')
        Track.set('MEAN_DIRECTIONAL_CHANGE_RATE','0')

        # Add track edges
        for label in track_section['label']:
            current = track_section[track_section['label']==label]
            following = track_section[(track_section['track_id']==int(current['track_id']))*\
                                      (track_section['timepoint']>int(current['timepoint']))]
            following = following[following['timepoint']==np.min(following['timepoint'])]

            if not following.empty:             
                # Get positions from AllSpots
                t_current = current['timepoint'].to_numpy()[0]-1
                t_following = following['timepoint'].to_numpy()[0]-1
                i_current = np.where((track_df['label'][track_df['timepoint']==int(current['timepoint'])]==int(current['label'])).to_numpy())[0][0]
                i_following = np.where((track_df['label'][track_df['timepoint']==int(following['timepoint'])]==int(following['label'])).to_numpy())[0][0]
                                
                Edge = ET.SubElement(Track, "Edge")
                Edge.set('SPOT_SOURCE_ID',str(int(current['label'])))
                Edge.set('SPOT_TARGET_ID',str(int(following['label'])))
                Edge.set('LINK_COST','0')
                Edge.set('DIRECTIONAL_CHANGE_RATE','0')
                Edge.set('SPEED','0')
                Edge.set('DISPLACEMENT','0')
                Edge.set('EDGE_TIME',str((t_current+t_following)/2))                
                Edge.set('EDGE_X_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_X'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_X'))/2))
                Edge.set('EDGE_Y_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Y'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Y'))/2))
                Edge.set('EDGE_Z_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Z'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Z'))/2))
     
        # Add split_edges
        for parent,child_1,child_2,timepoint in \
            zip(split_section['parent'],split_section['child_0'],split_section['child_1'],split_section['timepoint']):
                # First child                
                t_current = timepoint-2                
                t_following = int(np.min(track_df['timepoint'][track_df['track_id']==child_1]))-1
                l_current = int(track_df['label'][(track_df['track_id']==parent)*(track_df['timepoint']==(t_current+1))])
                l_following = int(track_df['label'][(track_df['track_id']==child_1)*(track_df['timepoint']==(t_following+1))])
                i_current = np.where((track_df['label'][track_df['timepoint']==(t_current+1)]==l_current).to_numpy())[0][0]
                i_following = np.where((track_df['label'][track_df['timepoint']==(t_following+1)]==l_following).to_numpy())[0][0]
                                
                Edge = ET.SubElement(Track, "Edge")
                Edge.set('SPOT_SOURCE_ID',str(l_current))
                Edge.set('SPOT_TARGET_ID',str(l_following))
                Edge.set('LINK_COST','0')
                Edge.set('DIRECTIONAL_CHANGE_RATE','0')
                Edge.set('SPEED','0')
                Edge.set('DISPLACEMENT','0')
                Edge.set('EDGE_TIME',str((t_current+t_following)/2))
                Edge.set('EDGE_X_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_X'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_X'))/2))
                Edge.set('EDGE_Y_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Y'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Y'))/2))
                Edge.set('EDGE_Z_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Z'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Z'))/2))
                
                # Second child
                t_following = int(np.min(track_df['timepoint'][track_df['track_id']==child_2]))-1
                l_following = int(track_df['label'][(track_df['track_id']==child_2)*(track_df['timepoint']==(t_following+1))])
                i_following = np.where((track_df['label'][track_df['timepoint']==(t_following+1)]==l_following).to_numpy())[0][0]
                
                Edge = ET.SubElement(Track, "Edge")
                Edge.set('SPOT_SOURCE_ID',str(l_current))
                Edge.set('SPOT_TARGET_ID',str(l_following))
                Edge.set('LINK_COST','0')
                Edge.set('DIRECTIONAL_CHANGE_RATE','0')
                Edge.set('SPEED','0')
                Edge.set('DISPLACEMENT','0')
                Edge.set('EDGE_TIME',str((t_current+t_following)/2))
                Edge.set('EDGE_X_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_X'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_X'))/2))
                Edge.set('EDGE_Y_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Y'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Y'))/2))
                Edge.set('EDGE_Z_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Z'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Z'))/2))
                
        # Add merge_edges
        for parent_1,parent_2,child,timepoint in \
            zip(merge_section['parent_0'],merge_section['parent_1'],merge_section['child'],merge_section['timepoint']):
                # First parent
                t_following = timepoint-1
                t_current = int(np.max(track_df['timepoint'][track_df['track_id']==parent_1]))-1
                l_following = int(track_df['label'][(track_df['track_id']==child)*(track_df['timepoint']==(t_following+1))])
                l_current = int(track_df['label'][(track_df['track_id']==parent_1)*(track_df['timepoint']==(t_current+1))])
                i_following = np.where((track_df['label'][track_df['timepoint']==(t_following+1)]==l_following).to_numpy())[0][0]
                i_current = np.where((track_df['label'][track_df['timepoint']==(t_current+1)]==l_current).to_numpy())[0][0]
                                
                Edge = ET.SubElement(Track, "Edge")
                Edge.set('SPOT_SOURCE_ID',str(l_current))
                Edge.set('SPOT_TARGET_ID',str(l_following))
                Edge.set('LINK_COST','0')
                Edge.set('DIRECTIONAL_CHANGE_RATE','0')
                Edge.set('SPEED','0')
                Edge.set('DISPLACEMENT','0')
                Edge.set('EDGE_TIME',str((t_current+t_following)/2))
                Edge.set('EDGE_X_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_X'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_X'))/2))
                Edge.set('EDGE_Y_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Y'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Y'))/2))
                Edge.set('EDGE_Z_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Z'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Z'))/2))
                # Second parent
                t_current = int(np.max(track_df['timepoint'][track_df['track_id']==parent_2]))-1
                l_current = int(track_df['label'][(track_df['track_id']==parent_2)*(track_df['timepoint']==(t_current+1))])
                i_current = np.where((track_df['label'][track_df['timepoint']==(t_current+1)]==l_current).to_numpy())[0][0]
                
                Edge = ET.SubElement(Track, "Edge")
                Edge.set('SPOT_SOURCE_ID',str(l_current))
                Edge.set('SPOT_TARGET_ID',str(l_following))
                Edge.set('LINK_COST','0')
                Edge.set('DIRECTIONAL_CHANGE_RATE','0')
                Edge.set('SPEED','0')
                Edge.set('DISPLACEMENT','0')
                Edge.set('EDGE_TIME',str((t_current+t_following)/2))
                Edge.set('EDGE_X_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_X'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_X'))/2))
                Edge.set('EDGE_Y_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Y'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Y'))/2))
                Edge.set('EDGE_Z_LOCATION',str(float(AllSpots[t_current][i_current].get('POSITION_Z'))/2+ \
                                          float(AllSpots[t_following][i_following].get('POSITION_Z'))/2))
                    
    
    
    
    print('All Tracks saved')

    
    # Settings
    Settings = TrackMate.findall('Settings')[0]
    
        # Image data settings
    ImageData = Settings.findall('ImageData')[0]

    ImageData.set('width',str(track_image.shape[1])) 
    ImageData.set('height',str(track_image.shape[2]))
    match dim:
        case 2:  
            ImageData.set('nslices','1') 
        case 3:  
            ImageData.set('nslices',str(track_image.shape[0]))
    ImageData.set('nframes',str(n_frames)) 
    ImageData.set('pixelwidth',str(resolution[1]))
    ImageData.set('pixelheight',str(resolution[2])) 
    ImageData.set('voxeldepth',str(resolution[0]))
    ImageData.set('filename','intensity.tif')
    ImageData.set('folder','./')
    
        # Basic settings
    BasicSettings  = Settings.findall('BasicSettings')[0]
    BasicSettings.set('xstart',str(0)) 
    BasicSettings.set('xend',str(int(ImageData.get('width'))-1)) 
    BasicSettings.set('ystart',str(0)) 
    BasicSettings.set('yend',str(int(ImageData.get('height'))-1)) 
    BasicSettings.set('zstart',str(0)) 
    BasicSettings.set('zend',str(int(ImageData.get('nslices'))-1)) 
    BasicSettings.set('tstart',str(0)) 
    BasicSettings.set('tend',str(int(ImageData.get('nframes'))-1)) 

    print('Settings done')

    # Set spotMax and trackMax to n_tracks in display settings
    DPS = TrackMate.findall('DisplaySettings')[0]
    trackMax = DPS.text.find('trackMax')
    start = DPS.text[trackMax:trackMax+30].find(':')
    end = DPS.text[trackMax:trackMax+30].find(',')
    DPS.text = DPS.text[:trackMax+start+1] + ' ' + str(n_tracks) + DPS.text[trackMax+end:]
    spotMax = DPS.text.find('spotMax')
    start = DPS.text[spotMax:spotMax+30].find(':')
    end = DPS.text[spotMax:spotMax+30].find(',')
    DPS.text = DPS.text[:spotMax+start+1] + ' ' + str(n_tracks) + DPS.text[spotMax+end:]

    print('Display settings done')


    # Create xml tree and save file
    TrackMate.write(save_file)
    
    return ".xml file successfully saved to " + save_file

# In[]: Load tracks from .xlm 

def load_TrackMate(file_path):
    """
    Load tracks from the .xml file in file_path and translate into DataFrames
    
    Parameters:
    - file_path (str): Path to the .xml file of the final tracks.
    
    Returns:
    - tuple of pandas.DataFrame: Tuple containing the track, split, and merge DataFrames.
    """ 
    
    # Initialise DataFrames
    split_df = pd.DataFrame(columns=['timepoint', 'parent', 'child_0', 'child_1'])
    merge_df = pd.DataFrame(columns=['timepoint', 'parent_0', 'parent_1', 'child'])
    track_df = pd.DataFrame(columns=['timepoint', 'label', 'track_id'])
    
    # Load tracks from TrackMate file
    TrackMate = ET.parse(file_path)
    
    # Model
    Model = TrackMate.findall('Model')[0]
    
    # Load AllSpots into track_df
    AllSpots = Model.findall('AllSpots')[0]
    for FrameSpots in AllSpots:
        timepoint = int(FrameSpots.get('frame')) + 1
        for Spot in FrameSpots:
            label = int(Spot.get('ID'))
            track_df = pd.concat([track_df,
                                  pd.DataFrame({'timepoint': [timepoint],\
                                                'label': [label],\
                                                'track_id': [label]})])
                
    # Load AllTracks into the DataFrames
    AllTracks = Model.findall('AllTracks')[0]
    # Get all edges from all tracks
    edges = pd.DataFrame(columns=['timepoint', 'source', 'target'])
    for Track in AllTracks:
        for Edge in Track:
            edges = pd.concat([edges,
                                pd.DataFrame({'timepoint': [float(Edge.get('EDGE_TIME'))],
                                              'source': [int(Edge.get('SPOT_SOURCE_ID'))],
                                              'target': [int(Edge.get('SPOT_TARGET_ID'))]})])
    
    # Remove all spots from track_df that are not part of any edges
    track_df = track_df[np.isin(track_df['label'],list(set(edges['source']).union(set(edges['target']))))]
    
    # Sort edges by time
    edges = edges.sort_values(by='timepoint')
    
    for source,target in zip(edges['source'],edges['target']):
        
        splits = edges[edges['source']==source] 
        match len(splits):
            case 0:
                # Looks like it was already accounted for
                pass
            case 1:
                # This is not a split
                pass
            case 2:
                # Update split_df
                # Get source track_id 
                source_id = track_df['track_id'][track_df['label']==source].to_numpy()[0]
                # Get timepoint from track_id
                timepoint = track_df['timepoint'][track_df['label']==source].to_numpy()[0] + 1
                # Add to split_df
                split_df = pd.concat([split_df,
                                      pd.DataFrame({'timepoint': [timepoint],\
                                                    'parent': [source_id],\
                                                    'child_0': [splits['target'].to_numpy()[0]],
                                                    'child_1': [splits['target'].to_numpy()[1]]})])
                # Remove splits from edges
                edges = edges[edges['source']!=source]
            case _:
                raise Exception("Length of splits = " + str(len(splits))) 
        
        merges = edges[edges['target']==target]
        match len(merges):
            case 0:
                # Looks like it was already accounted for
                pass
            case 1:
                # This is not a merge
                pass
            case 2:
                # Update merge_df
                # Get source track_ids
                source_ids = track_df['track_id'][np.isin(track_df['label'],merges['source'])].to_numpy()
                # Get timepoint from track_id
                timepoint = track_df['timepoint'][track_df['label']==target].to_numpy()[0]
                # Add to merge_df
                merge_df = pd.concat([merge_df,
                                      pd.DataFrame({'timepoint': [timepoint],\
                                                    'parent_0': [source_ids[0]],\
                                                    'parent_1': [source_ids[1]],
                                                    'child': [target]})])
                # Remove merges from edges
                edges = edges[edges['target']!=target]
            case _:
                raise Exception("Length of merges = " + str(len(merges)))
        
        if (len(splits)==1) and (len(merges)==1):
            # It's a track transition
            track_df['track_id'][track_df['label']==target]= \
                        track_df['track_id'][track_df['label']==source].to_numpy()[0]
            
            
    return track_df,split_df,merge_df