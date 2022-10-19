# -*- coding: utf-8 -*-

"""
Model prediction functions.

This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""

from abc import ABC, abstractmethod
import nibabel.freesurfer.io as fsio
from meshlearn.model.persistance import load_model

class MeshPredict():
    """
    Predict per-vertex descriptor values for a mesh.

    Parameters
    ----------
    mesh_file  : str, path to input mesh file.
    descriptor : str, the desciptor name. Currently only 'pial_lgi' is supported.
    """

    model = None
    verbose = True

    def predict(self, mesh_file, descriptor='pial_lgi'):
        vert_coords, faces = fsio.read_geometry(mesh_file)
        self.predict(vert_coords, faces, descriptor)


    @abstractmethod
    def predict(vertices, faces, descriptor):
        pass


class MeshPredictLgi(MeshPredict):
    """
    Pre
    """
    def __init__(self, model_file, verbose=True):
        self.verbose = verbose
        self.model, _ = load_model(model_file, verbose=self.verbose)

    def predict(vertices, faces, descriptor="pial_lgi"):
        self.model.predict()