import pytest
import meshlearn as ml
import meshlearn.tfdata as tfd
import trimesh
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default:
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)

def _get_test_file_pair():
    mesh_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial')
    descriptor_file = os.path.join(TEST_DATA_DIR, 'tim_only', 'tim_surf_lh.pial_lgi')
    return { mesh_file: descriptor_file }


#def test_load_data():
#    data_files = _get_test_file_pair()
#    vpd = tfd.VertexPropertyDataset(data_files)
#    vpd._data_from_files(data_files)

def test_k_neighborhood():
    data_files = _get_test_file_pair()
    mesh_file_name = data_files.keys()[0] 
    vert_coords, faces = fsio.read_geometry(mesh_file_name)
    tmesh = trimesh.Trimesh(vertices=vert_coords, faces=faces, process=False)
    nh = tfd._k_neighborhoods(tmesh, k=1)

