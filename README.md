<p align="center"><img src="docs/images/logo.png" width="500" /></p>

*Copyright (c) 2023-2024 Anita Karsa, University of Cambridge, UK*

*Optimal3dTracks is distributed under the terms of the GNU General Public License*

ABOUT
-------------------------------------------------------------------------------
Optimal3dTracks is a cell tracking tool that turns 3D, segmented cells/nuclei
into cell tracks or lineage trees by identifying cell migration, division and, 
optionally, merging events. It requires the original intensity images and 
the label maps as inputs. Cell tracking is performed by 1. fitting 3D Gaussians 
to each segmented cell or nucleus, 2. calculating transition probability matrices 
using Gaussian Mixture Model Optimal Transport (https://github.com/judelo/gmmot) 
with Kullback-Leibler regularisation, and then 3. turning these matrices into the 
highest-probability valid transition matrices. Affine registration between time 
points is also included for improved accuracy. Step 3 and the use of sinkhorn for 
step 2 are the main innovations of this pipeline. Our transition model allows for 
cells to divide into max. 2 cells between time points and max. 2 cells to merge 
into one between two time points. It does not account for cells going in and out 
of the field of view. The tool also includes the option to save calculated tracks 
as an .xml file for manual track correction/curation using Fiji's TrackMate tool, 
as well as customisable visualisation of the dendogram.  

Our companion method for 3D nucleus segmentation is downloadable from: https://github.com/akarsa/star-3d  

![](docs/images/tracking_example.gif)  

INSTALLATION AND SYSTEM REQUIREMENTS
-------------------------------------------------------------------------------
First, install required packages (see DEPENDENCIES or Requirements.txt). 

Use `pip install optimal3dtracks` to install Optimal3dTracks. This should take up to
a few minutes.

For the manual track correction functionality, you will also need to download 
optimal3dtracks/base_file.xml from https://github.com/akarsa/optimal3dtracks

The software has been extensively tested on a standard computer with Windows 11 OS and
an Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz processor with 128 GB RAM, but it is
likely to work on different systems and with less memory too.
 
The version of each dependency used for testing can be found in DEPENDENCIES and 
Requirements.txt, but the software is likely to work with other versions as well.

DEMO
-------------------------------------------------------------------------------
demo/demo.ipynb is a short demo performing tracking across the five time points in 
demo/intensity and demo/label. This should run under a minute on a regular desktop and 
produce the below dendogram while saving all intermediate results (Gaussian parameters, 
affine transformations, track sections) in demo/. 

<p align="center"><img src="demo/dendogram.png" width="500" /></p>

HOW TO USE
-------------------------------------------------------------------------------

**To perform cell tracking with Optimal3dTracks** (see optimal3dtracks/example.ipynb):
```
# 1. Calculate 3D Gaussian parameters for each segmented region at each time point
calculate_Gaussian_parameters(label_files, intensity_files, save_folder, resolution)

# 2. Create track segments between consecutive time points
calculate_track_sections(label_files, intensity_files, Gaussian_parameter_files, save_folder_for_affine,
                              save_folder_for_tracks, resolution, max_number_of_cells_per_timepoint)

# 3. Concatenate track segments
track_df, split_df, merge_df = concatenate_track_sections(track_files,Gaussian_parameter_files,
                                                          save_folder,max_number_of_cells_per_timepoint)
```

**To save tracks as .xml for correction/curation in Fiji** (see optimal3dtracks/example.ipynb):
```
save_as_TrackMate(track_df,split_df,merge_df,label_files,dimensions,resolution,base_file,save_file)
```
*Note: You can specify whether you'd like to correct tracks in 3D (dimensions = 3) or on the 2D MIPs (dimensions = 2)*

**To plot the dendogram** (see optimal3dtracks/example.ipynb):
```
generate_tree(track_df, split_df, merge_df)
```

HOW TO ACKNOWLEDGE
-------------------------------------------------------------------------------
@software{optimal3dtrack,

  author       = {Anita Karsa},

  title        = {{Optimal3dTracks}},

  month        = feb,

  year         = 2024,

  url 	       = {https://github.com/akarsa/optimal3dtracks}

}

@article{karsa2024optimal3dtracks,

  title={STAR-3D and Optimal3dTracks: Advanced Deep Learning and Optimal Transport Techniques for Automated 3D Segmentation and Tracking in Pre-Implantation Embryos (manuscript under preparation)},

  author={Karsa, Anita and Boulanger, Jerome and Abdelbaki, Ahmed and Niakan, Kathy K. and Muresan, Leila},

}

DEPENDENCIES
-------------------------------------------------------------------------------
Python (3.11.4)

numpy (1.24.3) (https://numpy.org)

scipy (1.10.1) (https://scipy.org)

matplotlib (3.7.1) (https://matplotlib.org)

scikit-image (0.20.0) (https://scikit-image.org)

pandas (1.5.3) (https://pandas.pydata.org)

POT (0.9.1) (https://pythonot.github.io/)

SimpleITK (2.3.0) (https://github.com/SimpleITK/SimpleITK)

csbdeep (0.7.4) (https://github.com/CSBDeep/CSBDeep)

tifffile (2023.2.28) (https://github.com/cgohlke/tifffile)

pathlib (1.0.1) (https://github.com/budlight/pathlib)

ipywidgets (8.0.4) (https://github.com/jupyter-widgets/ipywidgets)


CONTACT INFORMATION
-------------------------------------------------------------------------------
Anita Karsa, Ph.D.

Cambridge Advanced Imaging Centre

Dept. of Physiology, Development, and Neuroscience

University of Cambridge,

Cambridge, UK

ak2557@cam.ac.uk
