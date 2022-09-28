#!/usr/bin/env python

import trimesh as tm
from meshlearn.training_data import TrainingData, get_valid_mesh_desc_file_pairs_reconall
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure
import numpy as np
import igl  # Python bindings for ligigl: `conda install -y -c conda-forge igl`
import os
import time


data_dir = "/media/spirit/science/data/abide"
if not os.path.isdir(data_dir):
    data_dir = os.path.expanduser("~/software/freesurfer/subjects")
if not os.path.isdir(data_dir):
    raise ValueError("Could neither find data_dir '{data_dir}' nor FreeSurfer installation at '~/software/freesurfer/'.")
mesh_files, desc_files, cortex_files, files_subject, files_hemi, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_dir, surface="pial", descriptor="thickness", cortex_label=False)

if len(mesh_files) == 0:
    raise ValueError("No valid input files found.")

mesh_file = mesh_files[0]
desc_file = desc_files[0]
print(f"Loading mesh file '{mesh_file}' and pvd descriptor file '{desc_file}'.")
vert_coords, faces, pvd_data = TrainingData.data_from_files(mesh_file, desc_file)
#vert_coords = vert_coords.astype(np.float64)
#faces = faces.astype(np.int64) # required for libigl, e.g., the `principal_curvature` call below. Otherwise you get 'IndexError: vector::_M_range_check: __n ...' then.
mesh = tm.Trimesh(vertices=vert_coords, faces=faces)
print(f"Mesh created. Starting at: {time.ctime()}.")

#print(f"Starting to compute Gaussian curvature using trimesh at: {time.ctime()}.")
#gc = discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=5.0)  # Speed from trimesh is okay, more or less

#print(f"Starting to compute mean curvature using trimesh at: {time.ctime()}.")
# this requires the `rtree` package: `conda install -y -c conda-forge rtree`
#mc = discrete_mean_curvature_measure(mesh, mesh.vertices, radius=5.0)  # Speed from trimesh is horrible, this is unusable actually for FreeSurfer meshes.

print(f"Starting to compute the principal curvature using igl at: {time.ctime()}.")
# ligigl python binding is yet another dependency, but *a lot* faster, so maybe worth it.
# It also allows computation on principal curvatures, not only H and K. This means we can compute extra descriptors, like curvature index or shape index based on them.
pd1, pd2, pv1, pv2 = igl.principal_curvature(vert_coords, faces.astype(np.int64)) # pd1 : #v by 3 maximal curvature direction for each vertex, pd2 : #v by 3 minimal curvature direction for each vertex, pv1 : #v by 1 maximal curvature value for each vertex,  pv2 : #v by 1 minimal curvature value for each vertex
h = 0.5 * (pv1 + pv2) # mean curvature

# See https://github.com/dfsp-spirit/fsbrain/blob/master/R/curvature.R for an implementation of many descriptors.

# The libigl tutorial here also has some ideas for global descriptors in Chapter 6, mesh stats: https://libigl.github.io/libigl-python-bindings/tut-chapter0/
# It is highly recommended in any case.

print(f"Computing all with meshlearn.Curvature at: {time.ctime()}.")
from meshlearn.curvature import Curvature
c = Curvature(mesh_file)
df = c.compute_all()

print(f"All done at: {time.ctime()}.")



do_plot = False
if do_plot:
    from meshplot import plot
    plot(vert_coords, faces)



