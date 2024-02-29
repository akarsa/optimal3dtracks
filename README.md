*Copyright (c) 2023-2024 Anita Karsa, University of Cambridge, UK*

*Cell tracking with optimal transport is distributed under the terms of the GNU General Public License*

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

To perform unfolding (see src/Unfolding.py):

1. Load or define image and label:
image, label = create_dummy()

2. Create simplified tessellation
verts, faces = create_simplified_tessellation(label, num_vertices=30)
* num_vertices: target number of vertices in the simplified tessellation

3. Unfold tessellation
verts_2d, faces_2d, dict_2d_3d = unfold_tessellation(

    verts, faces, base_triangle=0, draw=0

)
* base_triangle: the index of the row in faces that contains the first triangle
to consider (this will be the middle of the unfolded surface)
* draw: 0 or 1 indicating whether the function should plot the unfolded
tessellation or not

4. Unfold and extract layers
layers = unfolded_layers(
    verts, faces, verts_2d, faces_2d, dict_2d_3d, image, n_layers=20
)
* n_layers: number of layers to be exported on both sides of the surface
(i.e. layers will have 2*n_layers+1 slices)

HOW TO ACKNOWLEDGE
-------------------------------------------------------------------------------
@software{anisotropic-stardist3d,

  author       = {Anita Karsa},

  title        = {{Unfolding}},

  month        = feb,

  year         = 2024,

  url 	       = {https://github.com/akarsa/cell_tracking_with_optimal_transport}

}

DEPENDENCIES
-------------------------------------------------------------------------------
numpy (https://numpy.org)

scipy (https://scipy.org)

matplotlib (https://matplotlib.org)

scikit-image (https://scikit-image.org)

tqdm (https://tqdm.github.io)

pymeshlab (https://pymeshlab.readthedocs.io)


CONTACT INFORMATION
-------------------------------------------------------------------------------
Anita Karsa, Ph.D.

Dept. of Physiology, Development, and Neuroscience

University of Cambridge,

Cambridge, UK

ak2557@cam.ac.uk
