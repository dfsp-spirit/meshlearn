# -*- coding: utf-8 -*-

"""
Functions for computing different curvature measures from the two principal curvatures.

This is used to compute vertex-descriptors from the mesh itself. It assumes that you
have some way to compute the principal curvatures, typically the `igl.principal_curvature`
function from the `igl` module, i.e., the libigl Python bindings.

Use `conda install -y -c conda-forge igl` to install igl with conda.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
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

    pc['k1'] = np.maximum(pv1, pv2)

    pc['k2'] = np.minimum(pv1, pv2)

    pv1_abs = np.abs(pv1)
    pv2_abs = np.abs(pv2)

    pc['k_major'] = pv1
    idx_abs_pv2_larger = np.where(pv2_abs >= pv1_abs)[0]
    pc['k_major'][idx_abs_pv2_larger] = pv2[idx_abs_pv2_larger]

    pc['k_minor'] = pv1
    idx_abs_pv2_smaller = np.where(pv2_abs < pv1_abs)[0]
    pc['k_minor'][idx_abs_pv2_smaller] = pv2[idx_abs_pv2_smaller]
    return pc


def _shape_descriptor_names():
    """Get list of shape descriptor names."""
    return ['k1', 'k2', 'k_major', 'k_minor', 'mean_curvature', 'gaussian_curvature', 'intrinsic_curvature_index', 'negative_intrinsic_curvature_index', 'gaussian_l2_norm', 'absolute_intrinsic_curvature_index', 'mean_curvature_index', 'negative_mean_curvature_index' ,'mean_l2_norm', 'absolute_mean_curvature_index', 'folding_index', 'curvedness_index', 'shape_index', 'shape_type', 'area_fraction_of_intrinsic_curvature_index', 'area_fraction_of_negative_intrinsic_curvature_index', 'area_fraction_of_mean_curvature_index', 'area_fraction_of_negative_mean_curvature_index', 'sh2sh', 'sk2sk']

class Curvature:
    """Computation of various vertex-wise mesh vertex shape descriptors that are based on curvature."""

    def __init__(self, pc, verbose=False):
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
        self.pc = pc
        self.k1 = pc['k1']
        self.k2 = pc['k2']
        self.k_major = pc['k_major']
        self.k_minor = pc['k_minor']
        self.verbose = verbose

    def compute(self, descriptors):
        """Compute all descriptors listed in 'descriptors', return as pd.DataFrame.

        Parameters
        ----------
        desciptors: list of str, see `available()` for the full list.

        Examples
        --------
        c = Curvature(pc)
        df = c.compute(["gaussian_curvature", "mean_curvature"])
        """
        if not isinstance(descriptors, list) or len(descriptors) == 0:
            raise ValueError(f"Parameter 'descriptor' must contain list of descriptor names, see `Curvature.available()` for available names.")
        return _compute_shape_descriptors(self.pc, descriptors, verbose=self.verbose)

    def compute_all(self):
        """Compute all available descriptors, return as pd.DataFrame."""
        return _compute_shape_descriptors(self.pc, verbose=self.verbose)

    @staticmethod
    def available_descriptors(include_not_implemented = False):
        """
        Get list of available descriptor names.

        Can be used with `compute` function `descriptors` parameter.
        """
        all = _shape_descriptor_names()
        if include_not_implemented:
            return all
        implemented = list()
        for desc in all:
            if hasattr(Curvature, desc):
                implemented.append(desc)
        return implemented


    def gaussian_curvature(self):
        return self.k_major * self.k_minor

    def mean_curvature(self):
        return (self.k_major + self.k_minor) / 2.0

    def intrinsic_curvature_index(self):
        return np.maximum(self.gaussian_curvature(), 0.0)

    def negative_intrinsic_curvature_index(self):
        return np.minimum(self.gaussian_curvature(), 0.0)

    def gaussian_l2_norm(self):
        k = self.gaussian_curvature()
        return k * k

    def mean_l2_norm(self):
        h = self.mean_curvature()
        return h * h

    def absolute_intrinsic_curvature_index(self):
        return np.abs(self.gaussian_curvature())

    def mean_curvature_index(self):
        return np.maximum(self.mean_curvature(), 0.0)

    def negative_mean_curvature_index(self):
        return np.minimum(self.mean_curvature(), 0.0)

    def absolute_mean_curvature_index(self):
        return np.abs(self.mean_curvature())

    def folding_index(self):
        abs_k_maj = np.abs(self.k_major)
        abs_k_min = np.abs(self.k_minor)
        return abs_k_maj * (abs_k_maj - abs_k_min)

    def curvedness_index(self):
        return np.sqrt((self.k_major * self.k_major + self.k_minor * self.k_minor) / 2.0)

    def shape_index(self):
        return (2.0 * np.pi) * np.arctan((self.k1 + self.k2) / (self.k2 - self.k1))

    def shape_type(self):
        shape_type = np.zeros((self.k2.size), dtype=np.float32) # Could be np.int32, but having all as float is more convenient for many use cases.
        border1 = -1.0
        border2 = -0.5
        border3 = 0.0
        border4 = 0.5
        border5 = 1.0
        shape_index = self.shape_index()
        shape_type[np.where((shape_index >= border1) & (shape_index < border2))[0]] = 1
        shape_type[np.where((shape_index >= border2) & (shape_index < border3))[0]] = 2
        shape_type[np.where((shape_index >= border3) & (shape_index < border4))[0]] = 3
        shape_type[np.where((shape_index >= border4) & (shape_index < border5))[0]] = 4
        return shape_type

    # see https://github.com/dfsp-spirit/fsbrain/blob/master/R/curvature.R for some more,
    # the area fraction ones are not implemented here yet.

    def sh2sh(self):
        mln = self.mean_l2_norm()
        amci = self.absolute_mean_curvature_index()
        return mln / amci

    def sk2sk(self):
        gln = self.gaussian_l2_norm()
        aici = self.absolute_intrinsic_curvature_index()
        return gln / aici

    def _save_curv(self, outdir, outfile_prefix="lh.", outfile_suffix=""):
        """
        Compute curvature-based descriptors and save to separate files (one file per descriptor) in FreeSurfer curv format.

        Parameters
        ----------
        outdir : str, the output directory. Must exist and be writable.
        outfile_prefix: str, the prefix to construct the filename in the `outdir`, from `outfile_prefix` + `<descriptor_name>` + `outfile_suffix`.
        outfile_suffix: str, the suffix to construct the filename in the `outdir`, from `outfile_prefix` + `<descriptor_name>` + `outfile_suffix`.

        Returns
        -------
        None, called for side effect of writing to disk.
        """
        if not os.path.isdir(outdir):
            raise ValueError(f"Curvature output directory '{outdir}' does not exist or cannot be read.")
        df = self.compute_all()
        for desc in df.columns:
            outfile = os.path.join(outdir, outfile_prefix + desc + outfile_suffix)
            fsio.write_morph_data(outfile, df[desc])

    def _save_csv(self, output_file):
        """
        Compute curvature-based descriptors and save to a single CSV file using `pandas.DataFrame.to_csv`.

        Parameters
        ----------
        output_file : str, the output file name. The directory must exist and be writable.

        Returns
        -------
        None, called for side effect of writing to disk.
        """
        df = self.compute_all()
        df.to_csv(output_file)



def _compute_shape_descriptors(pc, descriptors=Curvature.available_descriptors(), verbose=False):
    """Return pandas.DataFrame with computes descriptors."""
    assert isinstance(descriptors, list), "Parameter 'descriptors' must be a list of strings, a subset of the one returned by `shape_descriptor_names`."
    assert isinstance(pc, dict), "Parameter 'pc' must be a dict, as returned by `separate_pcs`."
    for desc in descriptors:
        if not desc in _shape_descriptor_names():
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
            if verbose:
                print(f"Computing shape descriptor '{desc}' for all {pc['k1'].size} mesh vertices.")
            desc_func = getattr(curv, desc)
            df[desc] = desc_func()
        else:
            print(f"NOTICE: Curvature method for '{desc}' not implemented, skipping.")
    return df



