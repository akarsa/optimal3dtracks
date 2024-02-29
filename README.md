*Copyright (c) 2023-2024 Anita Karsa, University of Cambridge, UK*

*Cell tracking with Optimal Transport is distributed under the terms of the GNU General Public License*

ABOUT
-------------------------------------------------------------------------------
Cell tracking with optimal transport is a cell tracking tool that turns 3D, segmented cells
into tracks. It requires the original intensity images and the label maps as inputs. Cell
tracking is performed by 1. fitting 3D Gaussians to each segmented cell, 2. calculating 
transition probability matrices using Gaussian Mixture Model Optimal Transport 
(https://github.com/judelo/gmmot) with sinkhorn regularisation, and then 3. turning these 
matrices into the highest-probability valid transition matrices. Step 3 and the use of sinkhorn 
for step 2 are the main innovations of this pipeline. Our transition model allows for cells 
to divide into max. 2 cells between time points and max. 2 cells to merge into one between 
two time points. It does not account for cells going in and out of the field of view. 
The tool also includes customisable visualisation of the division tree.     

HOW TO USE
-------------------------------------------------------------------------------
First, install required packages (see dependencies).

To perform Cell tracking with optimal transport (see src/example.ipynb):

1. Load or define image and label:
image, label = create_dummy()

2. Create simplified tessellation
verts, faces = create_simplified_tessellation(label, num_vertices=30)
* num_vertices: target number of vertices in the simplified tessellation



HOW TO ACKNOWLEDGE
-------------------------------------------------------------------------------
@software{tracking_OT,

  author       = {Anita Karsa},

  title        = {{Cell tracking with Optimal Transport}},

  month        = feb,

  year         = 2024,

  url 	       = {https://github.com/akarsa/cell_tracking_with_optimal_transport}

}

@article{karsa2024cellsegtrack,

  title={Automated 3D cell segmentation and cell tracking in light-sheet images of mouse embryos (manuscript under preparation)},

  author={Karsa, Anita and Boulanger, Jerome and Archer, Matthew and Rapilly, Quentin and Abdelbaki????, Ahmed and Niakan, Kathy K. and Muresan, Leila},

}

DEPENDENCIES
-------------------------------------------------------------------------------
numpy (https://numpy.org)

scipy (https://scipy.org)

matplotlib (https://matplotlib.org)

scikit-image (https://scikit-image.org)

pandas (https://pandas.pydata.org)

ot (https://pythonot.github.io/)

gmmot (https://github.com/judelo/gmmot)


CONTACT INFORMATION
-------------------------------------------------------------------------------
Anita Karsa, Ph.D.

Cambridge Advanced Imaging Centre

Dept. of Physiology, Development, and Neuroscience

University of Cambridge,

Cambridge, UK

ak2557@cam.ac.uk
