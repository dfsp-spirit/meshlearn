# -*- coding: utf-8 -*-

"""
Functions for computing vertex neighborhoods for a triangular mesh.

The neighborhood of a vertex is a vector of numeric values that describe the spatial mesh region around
this vertex, e.g., by the vertex coordinates and normals of nearby vertices (for some definition of `nearby').

Optionally, global information on the mesh (like total area, vertex and edge count, etc.) can be added to the
local neighborhood, to provide some context.

This basically implements some feature engineering aimed to best describe local mesh structure. The limitation
is that all these descriptors have to be fast to compute: if you want to use them to predict (rather than compute)
some computationally expensive per-vertex descriptor, there is no point in doing that if these features take
(almost) as long to compute as the target feature.


This file is part of meshlearn, see https://github.com/dfsp-spirit/meshlearn for details.
"""

import numpy as np



def _get_mesh_neighborhood_feature_count(neigh_count, with_normals=True, extra_fields=[], with_label=False):
        """
        Compute number of features, i.e., length of an observation or number of columns in a data row (without the final label column).

        Parameters
        ----------
        neigh_count: int, number of neighbors considered max (3 floats per neighbor, the x,y,z coords)
        with_normals: bool, whether the neighborhood includes the vertex normals (3 floats per neighbor)
        extra_fields: list, each extra field is one column
        with_label: bool, whether to consider an extra label column
        """
        num_per_vertex_features = 3  # For x,y,z coords
        if with_normals:
            num_per_vertex_features += 3
        return neigh_count * num_per_vertex_features + len(extra_fields) + int(with_label)


def neighborhoods_euclid_around_points(query_vert_coords, query_vert_indices, kdtree, neighborhood_radius, mesh, pvd_data, max_num_neighbors=0, add_desc_vertex_index=True, add_desc_neigh_size=True, verbose=True, filter_smaller_neighborhoods=False, extra_columns = {}):
    """
    Compute the vertex neighborhood of the Tmesh for a given vertex using Euclidean distance (ball point).

    This uses a kdtree to compute all vertices in a certain radius around the central query vertex.

    Parameters
    ----------
    query_vert_coords   : `nx3` numpy.ndarray for n 3D points, typically a subset of the vertices of the mesh represented in the `kdtree`
    query_vert_indices  : 1D `np.ndarray` of ints, the vertex indices in the mesh for the `query_vert_coords`. required to assign proper `pvd_data` value. Set to `None` to assume that the coords are the coords of all mesh vertices.
    kdtree              : `scipy.spatial.KDTree` instance of all mesh vertex coordinates (not just the `query_vert_coords`)
    neighborhood_radius : the radius of the sphere used in the `kdtree` query, in mesh spatial units. use 25 for 25mm with freesurfer meshes. must be changed together with `max_num_neighbors`.
    mesh                : `tmesh.Trimesh` instance representing the full mesh.
    pvd_data            : None or vector of length `vert_coords.shape[0]` (number of vertices in mesh), assigning a descriptor value (cortical thoickness, lgi, ...) to each vertex.
    max_num_neighbors   : number of neighbors max to consider per neighborhood. must be changed together with `neighborhood_radius`. Set to `None` or `0` to auto-determine from min size over all neighborhoods (will differ across mesh files then!).
    add_desc_vertex_index: bool, whether to add descriptor: vertex index in mesh
    add_desc_neigh_size: bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
    verbose: bool, whether to print output (or be silent)
    filter_smaller_neighborhoods: bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
    extra_columns: dict, the keys are strings and define the column name, the values are 1D float np.ndarrays with one value per mesh vertex (size equal to that of `pvd_data`).


    Returns
    -------
    neighborhoods            : `nxm` numpy.ndarray, where `n` is the number of query_vert_coords (`query_vert_coords.shape[0]`, to be precise), and `m` is the number of features and depends on the parameters including `max_num_neighbors`, `add_desc_vertex_index`, `add_desc_neigh_size`, and `extra_columns`.
    col_names                : list of `m` strings, the column names for the `neighborhoods` matrix
    kept_vertex_indices_mesh : numpy 1D array of ints, the vertex indices of the returned neighborhoods, in the same order as the neighborhoods. Only relevant if `filter_smaller_neighborhoods` is True, otherwise this will be identical to the passed `query_vert_indices`.
    """
    if kdtree is None:
        raise ValueError("No kdtree initialized yet.")
    if not isinstance(query_vert_coords, np.ndarray):
        raise ValueError("Expected np.ndarray as 'query_vert_coords' input.")
    if not isinstance(query_vert_indices, np.ndarray):
        raise ValueError("Expected np.ndarray as 'query_vert_indices' input.")
    if not query_vert_coords.shape[1] == 3:
        raise ValueError("Expected np.ndarray with 2nd dimension of length 3 as input.")
    mesh_num_verts = int(mesh.vertices.size / 3)
    if pvd_data is not None:
        assert np.array(pvd_data).size == mesh_num_verts, f"Expected {mesh_num_verts} per-vertex data values for mesh with {mesh_num_verts} verts, but got {np.array(pvd_data).size} pvd values."

    num_query_verts = query_vert_coords.shape[0]
    if query_vert_indices is None:
        query_vert_indices = np.arange(num_query_verts) # Assume we were passed coords of all vertices, so we can compute the indices.

    if not num_query_verts == query_vert_indices.size:
        raise ValueError("Expected number of rows in 'query_vert_coords' to match length of 'query_vert_indices'.")

    neighbor_indices = kdtree.query_ball_point(x=query_vert_coords, r=neighborhood_radius) # list of arrays
    assert neighbor_indices.shape[0] == num_query_verts
    assert isinstance(neighbor_indices, np.ndarray)
    assert neighbor_indices.ndim == 1  # It is a 1D numpy array of lists.
    assert neighbor_indices.dtype == object # Because the inner lists are variable length.

    ## Atm, the number of neighbors differs between the source vertices, as we simply found all within a fixed Euclidean radius (and vertex density obvisouly differs).
    ## So we need to fix the lengths to max_num_neighbors.
    neigh_lengths = [len(neigh) for neigh in neighbor_indices]
    min_neigh_size = np.min(neigh_lengths)


    if max_num_neighbors == 0 or max_num_neighbors is None:
        max_num_neighbors = min_neigh_size # set to minimum to avoid NANs
        print(f"[neig]   - Auto-determinded max_num_neighbors to be {min_neigh_size} for mesh.")

    if verbose:
        max_neigh_size = np.max(neigh_lengths)
        mean_neigh_size = np.mean(neigh_lengths)
        median_neigh_size = np.median(neigh_lengths)
        print(f"[neig]   - Min neigh size across {len(neighbor_indices)} neighborhoods is {min_neigh_size}, max is {max_neigh_size}, mean is {mean_neigh_size}, median is {median_neigh_size}")

    too_small_action = "filter" if filter_smaller_neighborhoods else "fill"

    if too_small_action == "filter":
        ## Filter neighborhoods which are too small.
        kept_vertex_indices_rel = np.where([len(neigh) >= max_num_neighbors for neigh in neighbor_indices])[0] # These are indices into the query_vert_coords, but that may not be all vertices in the mesh.
        assert isinstance(kept_vertex_indices_rel, np.ndarray)
        assert kept_vertex_indices_rel.ndim == 1
        kept_vertex_indices_mesh = query_vert_indices[kept_vertex_indices_rel]
        neighbor_indices_filtered = [neigh[0:max_num_neighbors] for neigh in neighbor_indices if len(neigh) >= max_num_neighbors]
        if verbose:
            print(f"[neig]   - Filtered neighborhoods, {len(neighbor_indices_filtered)} of {len(neighbor_indices)} left after removing all smaller than {max_num_neighbors} verts")

        num_query_verts_after_filtering = len(neighbor_indices_filtered)
        assert len(kept_vertex_indices_rel) == len(kept_vertex_indices_mesh)
        assert num_query_verts_after_filtering == len(kept_vertex_indices_rel), f"Expected {len(kept_vertex_indices_rel)} neighborhoods to be left after size filtering (relative indices), but found {num_query_verts_after_filtering}."
        assert num_query_verts_after_filtering == len(kept_vertex_indices_mesh), f"Expected {len(kept_vertex_indices_mesh)} neighborhoods to be left after size filtering (absolute/mesh indices), but found {num_query_verts_after_filtering}."

        neighbor_indices = neighbor_indices_filtered
        neigh_lengths_full_filtered_row_subset = np.array(neigh_lengths)[kept_vertex_indices_rel]  # These are the full lengths (before limiting to max_num_neighbors), but only for the subset of vertices that were kept.
        neigh_lengths_nonnan_after_filtering = [len(neigh) for neigh in neighbor_indices_filtered] # These are all identical (their length is 'max_num_neighbors') in the 'filter' case.
    elif too_small_action == "fill":
        neighbor_indices_filtered = [neigh[0:max_num_neighbors] for neigh in neighbor_indices] # Do not filter them, but restrict all rows to length 'max_num_neighbors'.
        neighbor_indices = neighbor_indices_filtered
        num_query_verts_after_filtering = len(neighbor_indices)     # We kept all of them.
        kept_vertex_indices_rel = np.arange(len(neighbor_indices))  # We kept all of them.
        kept_vertex_indices_mesh = query_vert_indices               # No changes.
        neigh_lengths_full_filtered_row_subset = np.array(neigh_lengths)  # These are the full lengths (before limiting to max_num_neighbors). We did not filter anything, really.
        neigh_lengths_nonnan_after_filtering = [min(len(neigh), max_num_neighbors) for neigh in neighbor_indices]
        if verbose:
            vertex_indices_to_fill_rel = np.where([len(neigh) < max_num_neighbors for neigh in neighbor_indices])[0] # These are indices into the query_vert_coords.
            print(f"[neig]   - Filled neighborhood vertex coords and normals with 'np.nan' for the {vertex_indices_to_fill_rel.size} neighborhoods smaller than {max_num_neighbors} verts.")

        # We do not need to do anything special to fill with NANs: later, we create a matrix of NAN values,
        # and the places which are not filled stay NAN.
    else:
        raise ValueError(f"Invalid 'too_small_action' to apply to neighborhoods smaller than {max_num_neighbors} vertices.")

    neigh_lengths_after_preproc = [len(neigh) for neigh in neighbor_indices]
    assert np.unique(neigh_lengths_after_preproc).size == 1, f"Expected same size for all pre-processed neighborhoods, but found {np.unique(neigh_lengths_after_preproc).size} different sizes."  # They should all have the same size now.
    neigh_length = np.unique(neigh_lengths_after_preproc)[0]

    extra_fields = []
    if add_desc_vertex_index:
        extra_fields.append("vertex_index")
    if add_desc_neigh_size:
        extra_fields.append("neigh_size")
    for ec_key in extra_columns.keys():
        extra_fields.append(ec_key)
        assert np.array(extra_columns[ec_key]).size == mesh_num_verts, f"Expected {mesh_num_verts} per-vertex data values in extra_column '{ec_key}' for mesh with {mesh_num_verts} verts, but found {np.array(extra_columns[ec_key]).size} pvd values."

    with_label = pvd_data is not None  # Add label if pvd_data is available.
    neighborhood_col_num_values = _get_mesh_neighborhood_feature_count(max_num_neighbors, with_normals=True, extra_fields=extra_fields, with_label=with_label)
    # 3 (x,y,z) coord entries per neighbor, 3 (x,y,z) vertex normal entries per neighbor, 1 pvd label value per neighborhood (or None)
    label_tag = "no label added, because no per-vertex descriptor data available"
    if with_label:
        label_tag = "the last 1 of them is the label"
    if verbose:
        print(f"[neig]   - Current settings with max_num_neighbors={max_num_neighbors} and {len(extra_fields)} extra columns lead to {neighborhood_col_num_values} columns ({label_tag}) per observation.")

    ## Full matrix for all neighborhoods
    neighborhoods = np.empty((num_query_verts_after_filtering, neighborhood_col_num_values), dtype=np.float32)
    neighborhoods[:] = np.nan

    col_names = []
    for n_idx in range(max_num_neighbors):
        for coord in ["x", "y", "z"]:
            col_names.append("nc" + str(n_idx) + coord) # "coord) # nc for neighbor coord
    for n_idx in range(max_num_neighbors):
        for coord in ["x", "y", "z"]:
            col_names.append("nn" + str(n_idx) + coord) # nn for neighbor normal
    if add_desc_vertex_index:
        col_names.append("svidx") # 'svidx' for source vertex index (in mesh)
    if add_desc_neigh_size:
        col_names.append("nsize") # 'nsize' for neighborhood size
    for ec_key in extra_columns.keys():
        col_names.append(ec_key)
    if with_label:
        col_names.append("label")

    assert mesh.vertices.ndim == 2
    assert mesh.vertices.shape[1] == 3 #x,y,z

    very_verbose = True

    do_insert_by_column = False

    if do_insert_by_column:
        central_vert_rel_indices = [i for i in range(len(neighbor_indices))]
        central_vert_mesh_indices = kept_vertex_indices_mesh[central_vert_rel_indices]
    else:
        for central_vert_rel_idx, neigh_vert_indices in enumerate(neighbor_indices):
            if very_verbose and central_vert_rel_idx % 10000 == 0:
                print(f"[neig]     * At vertex {central_vert_rel_idx} of {len(neighbor_indices)}.")

            central_vert_idx_mesh = kept_vertex_indices_mesh[central_vert_rel_idx]
            col_start_idx = 0
            this_vertex_num_neighbors = neigh_lengths_nonnan_after_filtering[central_vert_rel_idx] # We need to know this for each vertex so we only fill the correct length in case (and leave the rest at np.nan) for neighborhoods smaller than 'max_num_neighbors'.

            col_end_idx = col_start_idx+(this_vertex_num_neighbors*3)
            neighborhoods[central_vert_rel_idx, col_start_idx:col_end_idx] = np.ravel(mesh.vertices[neigh_vert_indices]) # Add absolute vertex coords

            col_start_idx = col_end_idx
            col_end_idx = col_start_idx+(this_vertex_num_neighbors*3)

            neighborhoods[central_vert_rel_idx, col_start_idx:col_end_idx] = np.ravel(mesh.vertex_normals[neigh_vert_indices]) # Add vertex normals
            col_idx = col_end_idx  # The end is not included.


            if add_desc_vertex_index:
                neighborhoods[central_vert_rel_idx, col_idx] = central_vert_idx_mesh   # Add index of central vertex
                col_idx += 1

            if add_desc_neigh_size:
                neighborhoods[central_vert_rel_idx, col_idx] = neigh_lengths_full_filtered_row_subset[central_vert_rel_idx]   # Add neighborhood size
                col_idx += 1

            for ec_key in extra_columns.keys():
                neighborhoods[central_vert_rel_idx, col_idx] = extra_columns[ec_key][central_vert_rel_idx]   # Add extra_column pvd-value for the vertex.
                col_idx += 1

            if with_label:
                neighborhoods[central_vert_rel_idx, col_idx] = pvd_data[central_vert_idx_mesh] # Add label (lgi, thickness, or whatever)

    assert neighborhoods.shape[0] == len(kept_vertex_indices_mesh), f"Expected {len(kept_vertex_indices_mesh)} neighborhoods, but found {neighborhoods.shape[0]}."
    assert neighborhoods.shape[1] == len(col_names)

    return neighborhoods, col_names, kept_vertex_indices_mesh
