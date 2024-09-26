# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 07:45:06 2024

@author: Anita Karsa, University of Cambridge, UK
"""

from setuptools import setup, find_packages

setup(
    name='optimal3dtracks',  # The name of your package
    version='0.0.4',  # Version of your package
    packages=find_packages(),  # Automatically find and include your packages
    # package_data={"optimal3dtracks": ["*.xml"]},
    install_requires=['numpy', 'scipy', 'matplotlib',
                      'scikit-image', 'pandas', 'POT',
                      'SimpleITK', 'csbdeep', 'tifffile',
                      'pathlib', 'ipywidgets'],  # External dependencies can be listed here
    description='Optimal3dTracks is a cell tracking tool that turns 3D, segmented cells/nuclei into cell tracks or \
        lineage trees by identifying cell migration, division and, optionally, merging events.',
    author='Anita Karsa',
    author_email='ak2557@cam.ac.uk',
    url='https://github.com/akarsa/optimal3dtracks/',  # The URL of your GitHub repo
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
)

