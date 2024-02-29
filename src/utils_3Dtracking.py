"""
Created on Tue Nov 28 09:13:30 2023

@author: Anita Karsa, University of Cambridge, UK
"""

# In[]: Import all necessary tools

import numpy as np
import scipy
import matplotlib.pyplot as plt

from skimage import measure
import pandas as pd

import ot
from gmmot import GaussianW2

# In[]

################### Helper functions for display and outputs ###################


# In[]: Display 3D + time image with segmentations

import ipywidgets as widgets
from ipywidgets import interact


def show_4d_with_contours(im,seg):
    # im: 3D grayscale image normalised between 0 and 1
    # seg: 3D integer label map
    
    n_timepoints = im.shape[0]
    n_slices = im.shape[1]

    
    edges = scipy.ndimage.binary_erosion(seg, np.ones((1,1,5,5), np.uint8), iterations=1)
    edges = seg*(1 - edges)
    max_label = np.max(edges)
    
    def update(timepoint,slice_num):
        timepoint = np.round(timepoint)
        slice_num = np.round(slice_num)
        plt.figure()
        plt.imshow(im[int(timepoint),int(slice_num),:,:],cmap='gray',interpolation='none',vmin=0,vmax=0.95)
        plt.imshow(edges[int(timepoint),int(slice_num),:,:],vmax = max_label,cmap='prism',\
                   alpha=0.5*(edges[int(timepoint),int(slice_num),:,:]>0),interpolation='none')
        # Add label numbers
        df = pd.DataFrame(measure.regionprops_table(edges[int(timepoint),int(slice_num),:,:].astype(int),properties=["label", "centroid"]))
        for i in range(len(df)):
            plt.text(df["centroid-1"][i], df["centroid-0"][i], str(df["label"][i]),color='white',fontsize=12)
            
        plt.show()
    
    timepoint_widget = widgets.FloatSlider(
        value=int(n_timepoints/2),
        min=0,
        max=n_timepoints-1,
        step=1,
        description='Time: ',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='',
        style = {'description_width': 'initial'}
    )
    slice_widget = widgets.FloatSlider(
        value=int(n_slices/2),
        min=0,
        max=n_slices-1,
        step=1,
        description='Slice: ',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='',
        style = {'description_width': 'initial'}
    )
    interact(update, timepoint = timepoint_widget, slice_num = slice_widget)
    
    
# In[]: Plot division tree

def generate_tree(track_df,split_df,merge_df):
    # Function to arrange nodes for better visualisation
    # track_df: dataframe containing the tracks from create_tracks (see example script for usage)
    #           necessary columns = 'track_id', 'timepoint'
    #           This dataframe might contain and additional column called "colours" for specifying 
    #           the colours to be used to plot each cell at each time point
    # split_df: dataframe containing cell divisions from create_tracks (see example script for usage)
    #           necessary columns = 'parent', 'child_0', 'child_1', 'timepoint'
    # merge_df: dataframe containing cell merges from create_tracks (see example script for usage)
    #           necessary columns = 'parent_0', 'parent_1', 'child', 'timepoint'
    
    tree = np.array([]) 
    
    ## Use track_df, split_df and merge_df to arrange nodes in the tree    
    for timepoint in np.unique(track_df['timepoint'].to_numpy()):
    
        # turn dataframe into numpy array
        split_np = split_df[['parent', 'child_0', 'child_1']][split_df['timepoint']==timepoint].to_numpy()
        merge_np = merge_df[['parent_0', 'parent_1', 'child']][merge_df['timepoint']==timepoint].to_numpy()
        track_np = track_df['track_id'][track_df['timepoint']==timepoint].to_numpy()

        # First, add nodes based on split_df
        for parent in range(len(split_np)):

            # find parent and children nodes
            parent_node = split_np[parent,0]
            children = split_np[parent,1:3]
            # find position of parent node in the tree
            parent_id = np.squeeze(np.argwhere(tree==parent_node))
            # insert children on both sides of the parent node
            tree = np.insert(tree,parent_id + 1,children[0])
            tree = np.insert(tree,parent_id,children[1])
            
        # Then, add the child node of the merging nodes
        for parent in range(len(merge_np)):

            # find parents and child nodes
            parent_nodes = merge_np[parent,0:2]
            child = merge_np[parent,2]

            # find position of parent nodes in the tree
            parent_ids = np.squeeze(np.argwhere(np.isin(tree,parent_nodes)))
            # insert child node in between the parent nodes
            tree = np.insert(tree,int(np.floor(np.mean(parent_ids))),child)
            
        # Finally, add nodes from track_df
        track_np = track_np[track_np>0]
        tree = np.insert(tree,len(tree),track_np[~np.isin(track_np,tree)])
    
    ## Find starting and end points for all nodes
    start = np.zeros(tree.shape)
    end = np.zeros(tree.shape)
    all_frames = []
    for n in range(len(tree)):
        all_frames.append(track_df['timepoint'][track_df['track_id']==tree[n]].to_numpy())
        start[n] = all_frames[n][0]
        end[n] = all_frames[n][-1]
        
    ## Draw a graph
    fig =  plt.figure(0)
    fig.set_size_inches(35, 20)
    
    if 'colours' in track_df:
        for n in range(len(tree)):
            plt.scatter(np.repeat(n,len(all_frames[n])),-all_frames[n], 
                        c = np.reshape(np.concatenate(track_df['colours'][track_df['track_id']==tree[n]].to_numpy()),[-1,3]))
            plt.text(n,-start[n],str(int(tree[n])),fontsize=16)
    else:
        for n in range(len(tree)):
            plt.scatter(np.repeat(n,len(all_frames[n])),-all_frames[n])
            plt.text(n,-start[n],str(int(tree[n])),fontsize=16)
        
    #add horizintal lines for cell division events
    for parent_node in split_df['parent'].unique():
        all_children = split_df[['child_0', 'child_1']][split_df['parent']==parent_node].to_numpy()
        all_xs = np.argwhere(np.isin(tree,all_children))
        y = end[np.argwhere(tree==parent_node)]
        plt.hlines(y = -y-1, xmin = np.min(all_xs), xmax = np.max(all_xs), colors = 'black')
    #add horizintal lines for cell merging events
    for child_node in merge_df['child'].unique():
        all_parents = merge_df[['parent_0', 'parent_1']][merge_df['child']==child_node].to_numpy()
        all_xs = np.argwhere(np.isin(tree,all_parents))
        y = start[np.argwhere(tree==child_node)]
        plt.hlines(y = -y+1, xmin = np.min(all_xs), xmax = np.max(all_xs), colors = 'black')
        
    return tree, start, end


# In[]

################### Helper functions for tracking ###################

# In[]: Calculate Gaussian mixture model parameters 
    
def get_moments(data,resolution):
    # Calculate ND Gaussian parameters of data
    
    total = data.sum()
    XYZ = np.indices(data.shape).astype(float)
    d = len(data.shape)
    #adjust for resolution
    for i in range(d):
        XYZ[i] *= float(resolution[i])
    center = np.zeros([d,1])
    width = np.zeros([d,d])
    #estimate center
    for i in range(d):
        center[i] = (XYZ[i]*data).sum()/total
    #estimate covariate matrix
    for i in range(d):
        for j in range(d):
            width[i,j] = ((XYZ[i]-center[i])*(XYZ[j]-center[j])*data).sum()/total
    #calculate integral
    integral = data.sum()

    return integral, center, width

def fit_Gaussian_mixture(im,seg,label_file,resolution):
    # Calculating Gaussian parameters of all segmented regions in an image
    # im: 3D grayscale image normalised between 0 and 1
    # seg: 3D integer label map
    # label_file: file name of the label file; this is just to print out when list(map()) is used
    # resolution: resolution of the image (usually in um)
    
    print(label_file)
    #find all unique labels
    labels = np.unique(seg)
    labels = labels[labels>0]
    K = len(labels)
    #get bounding boxes for each label
    label_props = pd.DataFrame(measure.regionprops_table(seg.astype(int),
                                                         properties=["label","bbox"]))
    #estimate multivariate Gaussians within each region
    centers = []
    integrals = []
    widths = []    
    for lab in labels:
        #crop both images at the bounding box
        box = label_props[["bbox-0","bbox-1","bbox-2","bbox-3","bbox-4","bbox-5"]][label_props['label']==lab].to_numpy()[0]
        image = im[box[0]:box[3],box[1]:box[4],box[2]:box[5]].copy()
        label_image = seg[box[0]:box[3],box[1]:box[4],box[2]:box[5]].copy()
        image[label_image!=lab] = 0
        #calculate Gaussian parameters
        integral, center, width = get_moments(image,resolution)
        centers.append(center)
        integrals.append(integral)
        widths.append(width)
    return [integrals, centers, widths, K, labels.astype(int)]

# In[]: Optimal transport with sinkhorn regularisation

def GW2_ak(pi0,pi1,mu0,mu1,S0,S1):
    # Note from AK: this function is originally from gmmot. I modified it to use 
    # sinkhorn regularisation instead of emd, because that produces better results
    
    # return the GW2 discrete map and the GW2 distance between two GMM
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    d  = mu0.shape[1]
    S0 = S0.reshape(K0,d,d)
    S1 = S1.reshape(K1,d,d)
    M  = np.zeros((K0,K1))
    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianW2(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    # Then we compute the OT distance or OT map thanks to the OT library
    wstar     = ot.sinkhorn(pi0,pi1,M,2,numItermax=10000,stopThr=1e-9)         # discrete transport plan # EDIT: I changed this from emd to sinkhorn (AK) as it was performing much better
    #distGW2   = np.sum(wstar*M)
    return wstar#,distGW2,M
    
# In[]: Create tracks from transition matrices by searching for valid transitions

def create_tracks(start_track_ids,target_labels,transition_matrix,time_point):
    # Creating tracks based on the transition_matrix
    # start_track_ids: track_ids of the cells at the start (i.e. columns)
    # target_labels: labels of cell at the end (i.e. rows)
    # transition_matrix: transition probability matrix
    # time_point: time point
    
    # initialise split and merge dataframes
    split_section = pd.DataFrame(columns=['timepoint','parent', 'child_0', 'child_1'])
    merge_section = pd.DataFrame(columns=['timepoint', 'parent_0', 'parent_1', 'child'])
    
    # convert transition matrix into a valid matrix
    matrix = valid_transition_ver_2(transition_matrix)>0
    
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
    # turn transition_matrix into the highest probability valid transition matrix 
    # where cells may not split into more than two pieces and no more than two cells can merge at a time
    
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

def return_counts(numpy_array):
    count_unique_elements = np.unique(numpy_array,return_counts=True)
    count_dict = dict(zip(count_unique_elements[0],count_unique_elements[1]))
    counts = np.array([count_dict[i] for i in numpy_array])
    return counts