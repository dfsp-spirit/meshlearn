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
import psutil



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


def neighborhoods_euclid_around_points(query_vert_coords, query_vert_indices, kdtree, neighborhood_radius, mesh, pvd_data, max_num_neighbors=0, add_desc_vertex_index=True, add_desc_neigh_size=True, verbose=True, filter_smaller_neighborhoods=False, neighborhood_radius_factors=[], extra_columns = {}):
    """
    Compute the vertex neighborhood of the Tmesh for a given vertex using Euclidean distance (ball point).

    This uses a kdtree to compute all vertices in a certain radius around the central query vertex.

    Parameters
    ----------
    query_vert_coords            : `nx3` numpy.ndarray of float for n 3D points, typically a subset of the vertices of the mesh represented in the `kdtree`
    query_vert_indices           : 1D `np.ndarray` of ints, the vertex indices in the mesh for the `query_vert_coords`. required to assign proper `pvd_data` value. Set to `None` to assume that the coords are the coords of all mesh vertices.
    kdtree                       : `scipy.spatial.KDTree` instance of all mesh vertex coordinates (not just the `query_vert_coords`)
    neighborhood_radius          : float, the radius of the sphere used in the `kdtree` query, in mesh spatial units. use 25 for 25mm with freesurfer meshes. must be changed together with `max_num_neighbors`.
    mesh                         : `tmesh.Trimesh` instance representing the full mesh.
    pvd_data                     : None or vector of length `vert_coords.shape[0]` (number of vertices in mesh), assigning a descriptor value (cortical thoickness, lgi, ...) to each vertex.
    max_num_neighbors            : number of neighbors max to consider per neighborhood. must be changed together with `neighborhood_radius`. Set to `None` or `0` to auto-determine from min size over all neighborhoods (will differ across mesh files then!).
    add_desc_vertex_index        : bool, whether to add descriptor: vertex index in mesh
    add_desc_neigh_size          : bool, whether to add descriptor: number of neighbors in ball query radius (before any filtering due to `mesh_neighborhood_count`)
    verbose                      : bool, whether to print output (or be silent)
    filter_smaller_neighborhoods : bool, whether to skip neighborhoods smaller than `mesh_neighborhood_count`. If false, missing vertex values are filled with NAN.
    extra_columns                : dict, the keys are strings and define the column name, the values are 1D float np.ndarrays with one value per mesh vertex (size equal to that of `pvd_data`).
    neighborhood_radius_factors  : list of float, extra factors to create additional radii based on `neighborhood_radius`, in which the number of vertices in the radius will be computed for each query vertex, and the resulting data will be added as an extra descriptor column.


    Returns
    -------
    neighborhoods            : `nxm` numpy.ndarray, where `n` is the number of query_vert_coords (`query_vert_coords.shape[0]`, to be precise), and `m` is the number of features and depends on the parameters including `max_num_neighbors`, `add_desc_vertex_index`, `add_desc_neigh_size`, and `extra_columns`.
    col_names                : list of `m` strings, the column names for the `neighborhoods` matrix
    kept_vertex_indices_mesh : numpy 1D array of ints, the vertex indices of the returned neighborhoods, in the same order as the neighborhoods. Only relevant if `filter_smaller_neighborhoods` is True, otherwise this will be identical to the passed `query_vert_indices`.
    """

    do_insert_by_column = True  # bool, internal, leave alone. Leaving this at the default of True is much faster, but does not alter results.



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

    too_small_action = "filter" if filter_smaller_neighborhoods else "fill"

    if verbose:
        max_neigh_size = np.max(neigh_lengths)
        mean_neigh_size = np.mean(neigh_lengths)
        median_neigh_size = np.median(neigh_lengths)
        print(f"[neig]  - Min neigh size across {len(neighbor_indices)} neighborhoods is {min_neigh_size}, max is {max_neigh_size}, mean is {mean_neigh_size}, median is {median_neigh_size}")

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
            print(f"[neig]  - Filled neighborhood vertex coords and normals with 'np.nan' for the {vertex_indices_to_fill_rel.size} neighborhoods smaller than {max_num_neighbors} verts.")

        # We do not need to do anything special to fill with NANs: later, we create a matrix of NAN values,
        # and the places which are not filled stay NAN.
    else:
        raise ValueError(f"Invalid 'too_small_action' to apply to neighborhoods smaller than {max_num_neighbors} vertices.")

    del neighbor_indices_filtered

    if verbose:
        print(f"[load] Neighborhood vertices and normals computed, RAM available is about {int(psutil.virtual_memory().available / 1024. / 1024.)} MB")

    assert any([any(np.isnan(x)) for x in neighbor_indices]) == False, f"Expected no NaN-values in neighbor_indices, but found some."
    neigh_lengths_after_preproc = [len(neigh) for neigh in neighbor_indices]
    if too_small_action == "filter":
        assert np.unique(neigh_lengths_after_preproc).size == 1, f"Expected same size for all pre-processed neighborhoods with filtering, but found {np.unique(neigh_lengths_after_preproc).size} different sizes."  # They should all have the same size now.

    extra_fields = []
    if add_desc_vertex_index:
        extra_fields.append("vertex_index")
    if add_desc_neigh_size:
        extra_fields.append("neigh_size")
    for ec_key in extra_columns.keys():
        extra_fields.append(ec_key)
        assert np.array(extra_columns[ec_key]).size == mesh_num_verts, f"Expected {mesh_num_verts} per-vertex data values in extra_column '{ec_key}' for mesh with {mesh_num_verts} verts, but found {np.array(extra_columns[ec_key]).size} pvd values."

    rfactor_colums = {}
    for rf_idx, rfactor in enumerate(neighborhood_radius_factors):
        rfactor_colname = "nrf_idx" + str(rf_idx) + "_nn"
        extra_fields.append(rfactor_colname)
        if verbose:
            print(f"[load] Computing extra ball radius {neighborhood_radius * rfactor} descriptor column ({rf_idx+1} of {len(neighborhood_radius_factors)}), RAM available is about {int(psutil.virtual_memory().available / 1024. / 1024.)} MB")
        rfactor_colums[rfactor_colname] = np.array([len(neigh) for neigh in kdtree.query_ball_point(x=query_vert_coords, r=neighborhood_radius * rfactor)])[kept_vertex_indices_rel]

    if verbose:
        print(f"[load] Added {len(neighborhood_radius_factors)} extra ball radius descriptor columns, RAM available is about {int(psutil.virtual_memory().available / 1024. / 1024.)} MB")

    with_label = pvd_data is not None  # Add label if pvd_data is available.
    neighborhood_col_num_values = _get_mesh_neighborhood_feature_count(max_num_neighbors, with_normals=True, extra_fields=extra_fields, with_label=with_label)
    # 3 (x,y,z) coord entries per neighbor, 3 (x,y,z) vertex normal entries per neighbor, 1 pvd label value per neighborhood (or None)
    label_tag = "no label added, because no per-vertex descriptor data available"
    if with_label:
        label_tag = "the last 1 of them is the label"
    if verbose:
        print(f"[neig]  - Current settings with max_num_neighbors={max_num_neighbors} and {len(extra_fields)} extra columns lead to {neighborhood_col_num_values} columns ({label_tag}) per observation.")

    ## Full matrix for all neighborhoods
    neighborhoods = np.empty((num_query_verts_after_filtering, neighborhood_col_num_values), dtype=np.float32)
    neighborhoods[:] = np.nan

    col_names = []
    for n_idx in range(max_num_neighbors):
        for coord in ["x", "y", "z"]:
            col_names.append("nc" + str(n_idx) + coord)  # "coord) # nc for neighbor coord
    for n_idx in range(max_num_neighbors):
        for coord in ["x", "y", "z"]:
            col_names.append("nn" + str(n_idx) + coord)  # nn for neighbor normal
    if add_desc_vertex_index:
        col_names.append("svidx")  # 'svidx' for source vertex index (in mesh)
    if add_desc_neigh_size:
        col_names.append("nsize")  # 'nsize' for neighborhood size
    for ec_key in extra_columns.keys():  # Extra columns passed to function as param.
        col_names.append(ec_key)
    for rfactor_key in rfactor_colums.keys():  # Extra columns resulting from 'neighborhood_radius_factors' param. (Dict may be empty.)
        col_names.append(rfactor_key)
    if with_label:
        col_names.append("label")

    assert mesh.vertices.ndim == 2
    assert mesh.vertices.shape[1] == 3 #x,y,z
    assert num_query_verts_after_filtering == len(kept_vertex_indices_mesh), f"Expected num_query_verts_after_filtering {num_query_verts_after_filtering} to be equal to len(kept_vertex_indices_mesh) {len(kept_vertex_indices_mesh)}."

    very_verbose = False

    if do_insert_by_column:
        if very_verbose:
            print(f"[neig]   - Inserting by column.")

        ## Turn the 'neighbor_indices' list of lists into a 2D np.ndarray
        #print(f"[neig]   - Turning into 2D matrix starts.")
        neighbor_indices_mat = np.array([xi+[mesh_num_verts]*(max_num_neighbors-len(xi)) for xi in neighbor_indices])  # Set default value to number of mesh verts, i.e., 1 *more* than available (0-based) vertex indices in mesh, to hit extra NaN row later. All empty (NaN) indices will hit the last row later, resulting in NaN values (instead of out-of-bounds errors).
        assert neighbor_indices_mat.shape == (len(neighbor_indices), max_num_neighbors, )
        #print(f"[neig]   - Turning into 2D matrix done.")

        current_col_idx = 0

        # Add an extra, final row to the mesh vertices, that is all NaN. We will later
        # map all invalid (NaN) indices to this row index (maxindex +1), so we get NaN values (instead of errors)
        # when accessing them.
        nan_row = np.empty((1, 3), dtype=mesh.vertices.dtype)
        nan_row[:] = np.NaN
        mesh_verts_ext = np.r_[ mesh.vertices, nan_row ]  # Append the NaN-row.
        assert mesh_verts_ext.shape == (mesh_num_verts + 1, 3,), f"Expected mesh_verts_ext shape {(mesh_num_verts + 1, 3,)} but found {mesh_verts_ext.shape}."  # x,y,z
        #mesh_verts_ext = np.r_[ mesh.vertices[kept_vertex_indices_mesh, :], nan_row ]  # Append the NaN-row.
        #assert mesh_verts_ext.shape == (len(kept_vertex_indices_mesh) + 1, 3,), f"Expected mesh_verts_ext shape {(len(kept_vertex_indices_mesh) + 1, 3,)} but found {mesh_verts_ext.shape}."  # x,y,z

        # Add vertex coords.
        for neigh_rel_idx in np.arange(max_num_neighbors):  # neighbor_indices_mat.shape == (len(neighbor_indices), max_num_neighbors, )
            col_start_idx = current_col_idx
            col_end_idx = current_col_idx+3
            #print(f"[neigh]      * At coords neigh_rel_idx {neigh_rel_idx}, assigning to columns (inclusive) {col_start_idx} to (inclusice) {col_end_idx}.")
            col_verts_indices = neighbor_indices_mat[:, neigh_rel_idx]
            assert col_verts_indices.shape == (num_query_verts_after_filtering, ), f"Expected col_verts_indices.shape to be {(num_query_verts_after_filtering, )} but found {col_verts_indices.shape}."
            col_verts_xyz = mesh_verts_ext[col_verts_indices, :]
            assert col_verts_xyz.shape == (num_query_verts_after_filtering, 3,), f"Expected col_verts_xyz.shape to be {(num_query_verts_after_filtering, 3,)} but found {col_verts_xyz.shape}."
            neighborhoods[:, col_start_idx:col_end_idx] = col_verts_xyz
            current_col_idx = col_end_idx

        # Add vertex normals.
        mesh_normals_ext = np.r_[ mesh.vertices, nan_row ]
        assert mesh_normals_ext.shape == (mesh_num_verts + 1, 3,), f"Expected mesh_normals_ext shape {(mesh_num_verts + 1, 3,)} but found {mesh_normals_ext.shape}."  # x,y,z
        #assert mesh_normals_ext.shape == (len(kept_vertex_indices_mesh) + 1, 3,), f"Expected mesh_normals_ext shape {(len(kept_vertex_indices_mesh) + 1, 3,)} but found {mesh_normals_ext.shape}."  # x,y,z
        for neigh_rel_idx in np.arange(max_num_neighbors):
            col_start_idx = current_col_idx
            col_end_idx = current_col_idx+3
            #(f"[neigh]      * At normals neigh_rel_idx {neigh_rel_idx}, assigning to columns (inclusive) {col_start_idx} to (inclusice) {col_end_idx}.")
            col_verts_indices = neighbor_indices_mat[:, neigh_rel_idx]
            assert col_verts_indices.shape == (num_query_verts_after_filtering, ), f"Expected col_verts_indices.shape to be {(num_query_verts_after_filtering, )} but found {col_verts_indices.shape}."
            col_normals_xyz = mesh_normals_ext[col_verts_indices, :]
            assert col_normals_xyz.shape == (num_query_verts_after_filtering, 3,), f"Expected col_normals_xyz.shape to be {(num_query_verts_after_filtering, 3,)} but found {col_normals_xyz.shape}."
            neighborhoods[:, col_start_idx:col_end_idx] = col_normals_xyz
            current_col_idx = col_end_idx

        if add_desc_vertex_index:
            neighborhoods[:, current_col_idx] = kept_vertex_indices_mesh  # The vertex index column.
            current_col_idx += 1
        if add_desc_neigh_size:
            neighborhoods[:, current_col_idx] = neigh_lengths_full_filtered_row_subset   # Add neighborhood size column.
            current_col_idx += 1
        for ec_key in extra_columns.keys():
            neighborhoods[:, current_col_idx] = np.squeeze(extra_columns[ec_key][kept_vertex_indices_mesh])   # Add extra_column pvd-value for the vertex.
            current_col_idx += 1
        for rfactor_key in rfactor_colums.keys():
            neighborhoods[:, current_col_idx] = rfactor_colums[rfactor_key]   # Add neighbor counts with extra radii (derived from neighborhood_radius_factors)
            current_col_idx += 1
        if with_label:
            neighborhoods[:, current_col_idx] = pvd_data[kept_vertex_indices_mesh] # Add label (lgi, thickness, or whatever)
            current_col_idx += 1

        current_col_idx -= 1  # We added 1 too much above.
        assert current_col_idx + 1 == neighborhood_col_num_values, f"Inserted {current_col_idx + 1} columns, but neighborhoods matrix expects {neighborhood_col_num_values} columns."

    else:
        if very_verbose:
            print(f"[neig]   - Inserting by row.")
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

            for rfactor_key in rfactor_colums.keys():
                neighborhoods[central_vert_rel_idx, col_idx] = rfactor_colums[rfactor_key][central_vert_rel_idx]   # Add neighbor counts with extra radii (derived from neighborhood_radius_factors)
                col_idx += 1

            if with_label:
                neighborhoods[central_vert_rel_idx, col_idx] = pvd_data[central_vert_idx_mesh] # Add label (lgi, thickness, or whatever)
                col_idx += 1
        assert central_vert_rel_idx+1 == neighborhoods.shape[0], f"Done after inserting value in row with index {central_vert_rel_idx}, but neighborhood matrix has {neighborhoods.shape[0]} rows. Should end with last one."
        col_idx -= 1  # We added too much above.
        assert col_idx+1 == neighborhoods.shape[1], f"Done after inserting value in column with index {col_idx}, but neighborhood matrix has {neighborhoods.shape[1]} columns. Should end with last one."

    assert neighborhoods.shape[0] == len(kept_vertex_indices_mesh), f"Expected {len(kept_vertex_indices_mesh)} neighborhoods, but found {neighborhoods.shape[0]}."
    assert neighborhoods.shape[1] == len(col_names)

    return neighborhoods, col_names, kept_vertex_indices_mesh
