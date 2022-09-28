# -*- coding: utf-8 -*-

"""
Functions for computing different curvature measure from the two principal curvatures.

This is used to compute vertex-descriptors from the mesh itself. It assumes that you
have some way to compute the principal curvatures, typically the `igl.principal_curvature`
function from the `igl` module, i.e., the libigl Python bindings.

Use `conda install -y -c conda-forge igl` to install igl with conda.
"""

import numpy as np
import pandas as pd
import nibabel.freesurfer.io as fsio
import os

def compute_princial_curvature_for_mesh(vert_coords, faces):
    """
    Compute vertex-wise raw principal curvatures for all vertices of a mesh. Requires the `igl` package.

    Examples:
    ---------
    import nibabel.freesurfer.io as fsio
    vert_coords, faces = fsio.read_geometry("mesh_file_name")
    pv1, pv2 = compute_princial_curvature_for_mesh(vert_coords, faces)
    """

    import igl
    assert isinstance(vert_coords, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert vert_coords.ndim == 2
    assert faces.ndim == 2
    assert vert_coords.shape[1] == 3  # x,y,z float coords.
    assert faces.shape[1] == 3  # triangular face, so 3 integer vertex indices.

    # return values are:
    #  - pd1: direction of 1st principal curvature
    #  - pd2: direction of 2nd principal curvature,
    #  - pv1: 1st principal curvature value
    #  - pv2: 2nd principal curvature value
    pd1, pd2, pv1, pv2 = igl.principal_curvature(vert_coords, faces.astype(np.int64))
    pd1 = None
    pd2 = None
    return pv1, pv2


def separate_pcs(pv1, pv2):
    """
    Compute k1, k2, kmajor and kminor from the raw pv1 and pv2 values.

    Examples:
    ---------
    import nibabel.freesurfer.io as fsio
    vert_coords, faces = fsio.read_geometry("mesh_file_name")
    pv1, pv2 = compute_princial_curvature_for_mesh(vert_coords, faces)
    pc = separate_pcs(pv1, pv2)
    pc['k1']
    """
    pc = dict()
    assert pv1.shape == pv2.shape, "Paramters 'pv1' and 'pv2' must be ndarrays of identical lengths."

    pc['k1'] = pv1
    idx_pv2_larger = np.where(pv2 >= pv1)[0]
    pc['k1'][idx_pv2_larger] = pv2[idx_pv2_larger]

    pc['k2'] = pv1
    idx_pv2_smaller = np.where(pv2 < pv1)[0]
    pc['k2'][idx_pv2_smaller] = pv2[idx_pv2_smaller]

    pv1_abs = np.abs(pv1)
    pv2_abs = np.abs(pv2)

    pc['k_major'] = pv1
    idx_abs_pv2_larger = np.where(pv2_abs >= pv1_abs)[0]
    pc['k_major'][idx_abs_pv2_larger] = pv2[idx_pv2_larger]

    pc['k_minor'] = pv1
    idx_abs_pv2_smaller = np.where(pv2_abs < pv1_abs)[0]
    pc['k_minor'][idx_abs_pv2_smaller] = pv2[idx_pv2_smaller]
    return pc


def shape_descriptor_names():
    """Get list of shape descriptor names."""
    return ['k1', 'k2', 'k_major', 'k_minor', 'mean_curvature', 'gaussian_curvature', 'intrinsic_curvature_index', 'negative_intrinsic_curvature_index', 'gaussian_l2_norm', 'absolute_intrinsic_curvature_index', 'mean_curvature_index', 'negative_mean_curvature_index' ,'mean_l2_norm', 'absolute_mean_curvature_index', 'folding_index', 'curvedness_index', 'shape_index', 'shape_type', 'area_fraction_of_intrinsic_curvature_index', 'area_fraction_of_negative_intrinsic_curvature_index', 'area_fraction_of_mean_curvature_index', 'area_fraction_of_negative_mean_curvature_index', 'sh2sh', 'sk2sk']

class Curvature:

    def __init__(self, pc):
        """
        Create new Curvature computation instance.

        Parameters
        ----------
        pc: dict or str. If a dict, must be the result of calling the `separate_pcs` function. If a string, it will be interpreted as the full path to a valid Freesurfer mesh file, like `lh.white`.

        """
        if isinstance(pc, str):
            if not os.path.isfile(pc):
                raise ValueError(f"Cannot read mesh file '{pc}'.")
            else:
                vert_coords, faces = fsio.read_geometry(pc)
                pv1, pv2 = compute_princial_curvature_for_mesh(vert_coords, faces)
                pc = separate_pcs(pv1, pv2)
        self.k1 = pc['k1']
        self.k2 = pc['k2']
        self.k_major = pc['k_major']
        self.k_minor = pc['k_minor']

    def gaussian_curvature(self):
        return self.k_major * self.k_minor

def compute_shape_descriptors(pc, descriptors=shape_descriptor_names()):
    """Return pandas.DataFrame with computes descriptors."""
    assert isinstance(descriptors, list), "Parameter 'descriptors' must be a list of strings, a subset of the one returned by `shape_descriptor_names`."
    assert isinstance(pc, dict), "Parameter 'pc' must be a dict, as returned by `separate_pcs`."
    for desc in descriptors:
        if not desc in shape_descriptor_names():
            raise ValueError(f"Entry '{desc}' in parameter 'descriptors' is invalid.")

    assert pc['k1'].size == pc['k2'].size
    assert pc['k1'].size == pc['k_major'].size
    assert pc['k1'].size == pc['k_minor'].size

    df = pd.DataFrame({'k1': pc['k1'], 'k2': pc['k2'], 'k_major': pc['k_major'], 'k_minor': pc['k_minor'] })
    curv = Curvature(pc)
    for desc in descriptors:
        if desc in ['k1', 'k2', 'k_major', 'k_minor']:
            continue

        if hasattr(curv, desc):
            print(f"Computing shape descriptor '{desc}' for all {pc['k1'].size} mesh vertices.")
            desc_func = getattr(curv, desc)
            df[desc] = desc_func()
        else:
            print(f"Curvature method for '{desc}' not implemented, skipping.")
    return df



