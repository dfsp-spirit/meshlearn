"""
Read FreeSurfer brain meshes and computed lgi per-vertex data for them from a directory.
"""

import os
import numpy as np
import nibabel.freesurfer.io as fsio
#from . import nitools as nit
#from . import freesurferdata as fsd
import logging
import brainload as bl


def load_piallgi_morph_data(subjects_dir, subjects_list):
    return bl.group_native("pial_lgi", subjects_dir, subjects_list)

def load_surfaces(subjects_dir, subjects_list):
    surface = "pial"
    for subject in subjects_list:
        
