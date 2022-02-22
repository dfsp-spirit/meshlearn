"""
Read FreeSurfer brain meshes and pre-computed lgi per-vertex data for them from a directory.
"""

import brainload as bl
import trimesh as tm


def load_piallgi_morph_data(subjects_dir, subjects_list):
    return bl.group_native("pial_lgi", subjects_dir, subjects_list)


def load_surfaces(subjects_dir, subjects_list, surf="pial"):
    meshes = {}
    for subject in subjects_list:
        vertices, faces, meta_data = bl.subject_mesh(subject, subjects_dir, surf=surf)
        #meshes[subject] = { "vertices": vertices, "faces" : faces }
        meshes[subject] = tm.Trimesh(vertices=vertices, faces=faces)
    return meshes
        
        
