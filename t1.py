
#!/usr/bin/env python3

import os
import numpy as np
from meshlearn.model.predict import MeshPredictLgi
import nibabel.freesurfer.io as fsio


TEST_DATA_DIR = "./tests/test_data"


mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
descriptor_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial_lgi')


model_pkl_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.pkl')
metadata_json_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.json')  # Metadata file is not needed for predictions, return None if you do not have it.


print(f"Creating MeshPredictLgi instance...")
Mp = MeshPredictLgi(model_pkl_file, metadata_json_file)
print(f"Predicting using MeshPredictLgi instance...")
lgi_predicted = Mp.predict(mesh_file)


print(f"Verifying results...")
num_mesh_vertices = 149244
assert lgi_predicted.size == num_mesh_vertices
assert np.min(lgi_predicted) >= 0.0
assert np.max(lgi_predicted) <= 8.0
lgi_known = fsio.read_morph_data(descriptor_file)
p = np.corrcoef(lgi_predicted, lgi_known)
print(f"Pearson corr: {p[0,1]}")
fsio.write_morph_data("lh.pial_lgi_predicted", lgi_predicted)  # For visualization, e.g. in R with fsbrain.
