

__version__ = "0.0.1"
__author__ = 'Tim Schaefer'

"""
Meshlearn high-level API functions.
"""

# The next line makes the listed functions show up in sphinx documentation directly under the package (they also show up under their real sub module, of course)
__all__ = [ 'subject', 'subject_avg', 'group', 'group_native', 'fsaverage_mesh', 'subject_mesh', 'rhi', 'rhv', 'hemi_range', 'annot', 'label', 'stat', 'mesh_to_ply', 'mesh_to_obj', 'read_subjects_file', 'read_subjects_file', 'detect_subjects_in_directory', 'subject_data_native', 'subject_data_standard', 'export_mesh_nocolor_to_file']


from .tf_data import VertexPropertyDataset
