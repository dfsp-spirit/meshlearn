

import trimesh as tm
from meshlearn.training_data import TrainingData, get_valid_mesh_desc_file_pairs_reconall
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure
import numpy as np
import igl  # Python bindings for ligigl: `conda install -y -c conda-forge igl`

data_dir = "/media/spirit/science/data/abide"

mesh_files, desc_files, cortex_files, files_subject, files_hemi, miss_subjects = get_valid_mesh_desc_file_pairs_reconall(data_dir, surface="pial", descriptor="pial_lgi", cortex_label=False)

mesh_file = mesh_files[0]
desc_file = desc_files[0]
print(f"Loading mesh file '{mesh_file}' and pvd descriptor file '{desc_file}'.")
vert_coords, faces, pvd_data = TrainingData.data_from_files(mesh_file, desc_file)
vert_coords = vert_coords.astype(np.float64)
faces = faces.astype(np.int64) # required for libigl, e.g., the `principal_curvature` call below. Otherwise you get 'IndexError: vector::_M_range_check: __n ...' then.
mesh = tm.Trimesh(vertices=vert_coords, faces=faces)

#gc = discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=5.0)  # Speed from trimesh is okay, more or less
#mc = discrete_mean_curvature_measure(mesh, mesh.vertices, radius=5.0)  # Speed from trimesh is horrible, this is unusable actually for FreeSurfer meshes.

# ligigl python binding is yet another dependency, but *a lot* faster, so maybe worth it.
# It also allows computation on principal curvatures, not only H and K. This means we can compute extra descriptors, like curvature index or shape index based on them.
pd1, pd2, pv1, pv2 = igl.principal_curvature(vert_coords, faces) # pd1 : #v by 3 maximal curvature direction for each vertex, pd2 : #v by 3 minimal curvature direction for each vertex, pv1 : #v by 1 maximal curvature value for each vertex,  pv2 : #v by 1 minimal curvature value for each vertex
h = 0.5 * (pv1 + pv2) # mean curvature

# See https://github.com/dfsp-spirit/fsbrain/blob/master/R/curvature.R for an implementation of many descriptors.

do_plot = True
if do_plot:
    from meshplot import plot
    plot(vert_coords, faces)



