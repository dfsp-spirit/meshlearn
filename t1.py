
#!/usr/bin/env python3

import os
import numpy as np
from meshlearn.model.predict import MeshPredictLgi


TEST_DATA_DIR = "./tests/test_data"


mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
descriptor_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial_lgi')


model_pkl_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.pkl')
metadata_json_file = os.path.join(TEST_DATA_DIR, 'models', 'lgbm_lgi', 'ml_model.json')  # Metadata file is not needed for predictions, return None if you do not have it.


print(f"Creating MeshPredictLgi instance...")
Mp = MeshPredictLgi(model_pkl_file, metadata_json_file)
print(f"Predicting using MeshPredictLgi instance...")
pervertex_lgi = Mp.predict(mesh_file)
num_mesh_vertices = 149244
print(f"Verifying results...")
assert pervertex_lgi.size == num_mesh_vertices
assert np.min(pervertex_lgi) >= 0.0
assert np.max(pervertex_lgi) <= 8.0
