
import numpy as np
import trimesh as tm



def _get_mesh_neighborhood_feature_count(neigh_count, with_normals=True, extra_fields=[], with_label=False):
        """
        Compute number of features, i.e., length of an observation or number of columns in a data row (without the final label column).
        """
        num_per_vertex_features = 3  # For x,y,z coords
        if with_normals:
            num_per_vertex_features += 3
        return neigh_count * num_per_vertex_features + len(extra_fields) + int(with_label)

def neighborhoods_euclid_around_points(query_vert_coords, query_vert_indices, kdtree, neighborhood_radius, mesh, pvd_data, max_num_neighbors=0, add_desc_vertex_index=False, add_desc_neigh_size=False, verbose=True):
    """
    Compute the vertex neighborhood of the Tmesh for a given vertex using Euclidean distance (ball point).

    This uses a kdtree to compute all vertices in a certain radius. This is an alternative approach to the
    _k_neighborhoods() function below, which computes the k-neighborhoods on the mesh instead
    of simple Euclidean distance.

    Parameters
    ----------
    query_vert_coords nx3 numpy.ndarray for n 3D points
    query_vert_indices np.array of ints, the vertex indices in the mesh for the vert_coords. required to assign proper pvd_data value. Set to None to assume that the coords are the coords of all mesh vertices.
    kdtree scipy.spatial.KDTree instance
    neighborhood_radius the radius for sphere used in kdtree query, in mesh spatial units. use 25 for 25mm with freesurfer meshes. must be changed together with num_neighbors.
    mesh tmesh.Trimesh instance (the one from which vert_coords also come, currently duplicated)
    max_num_neighbors number of neighbors max to consider per neighborhood. must be changed together with neighborhood_radius. Set to None or 0 to auto-determine from min size over all neighborhoods (will differ across mesh files then!).
    pvd_data vector of length vert_coords.shape[0] (number of vertes in mesh), assigning a descriptor value (cortical thoickness, lgi, ...) to each vertex.

    Returns
    -------
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
    neigh_lengths_filtered = np.array(neigh_lengths)[kept_vertex_indices_rel]  # These are the full lengths (before limiting to max_num_neighbors), but only for the subset of vertices that were kept.

    extra_fields = []
    if add_desc_vertex_index:
        extra_fields.append("vertex_index")
    if add_desc_neigh_size:
        extra_fields.append("neigh_size")

    neighborhood_col_num_values = _get_mesh_neighborhood_feature_count(max_num_neighbors, with_normals=True, extra_fields=extra_fields, with_label=True)
    # 3 (x,y,z) coord entries per neighbor, 3 (x,y,z) vertex normal entries per neighbor, 1 pvd label value per neighborhood
    if verbose:
        print(f"[neigh]   - Current settings with max_num_neighbors={max_num_neighbors} and {len(extra_fields)} extra columns lead to {neighborhood_col_num_values} columns (the last 1 of them is the label) per observation.")

    ## Full matrix for all neighborhoods
    neighborhoods = np.zeros((num_query_verts_after_filtering, neighborhood_col_num_values), dtype=np.float)

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
    col_names.append("label")

    assert mesh.vertices.ndim == 2
    assert mesh.vertices.shape[1] == 3 #x,y,z

    for central_vert_rel_idx, neigh_vert_indices in enumerate(neighbor_indices):
        central_vert_idx_mesh = kept_vertex_indices_mesh[central_vert_rel_idx]
        col_start_idx = 0

        col_end_idx = col_start_idx+(max_num_neighbors*3)
        #print(f"Add coords for {len(neigh_indices)} neighbors into col positions {col_start_idx} to {col_end_idx} (num columns is {neighborhood_col_num_values}).")

        neighborhoods[central_vert_rel_idx, col_start_idx:col_end_idx] = np.ravel(mesh.vertices[neigh_vert_indices]) # Add absolute vertex coords

        col_start_idx = col_end_idx
        col_end_idx = col_start_idx+(max_num_neighbors*3)
        #print(f"Add normals for {max_num_neighbors} neighbors into col positions {col_start_idx} to {col_end_idx}")

        neighborhoods[central_vert_rel_idx, col_start_idx:col_end_idx] = np.ravel(mesh.vertex_normals[neigh_vert_indices]) # Add vertex normals
        col_idx = col_end_idx  # The end is not included.


        if add_desc_vertex_index:
            #print(f"Add source vertex index in mesh at col position {col_idx}")
            neighborhoods[central_vert_rel_idx, col_idx] = central_vert_idx_mesh   # Add index of central vertex
            col_idx += 1

        if add_desc_neigh_size:
            #print(f"Add neighborhood size at col position {col_idx}")
            neighborhoods[central_vert_rel_idx, col_idx] = neigh_lengths_filtered[central_vert_rel_idx]   # Add index of central vertex
            col_idx += 1

        #print(f"Adding pvd value at row {central_vert_rel_idx}, column {col_idx}, value is from mesh vertex idx {central_vert_idx_mesh}.")
        neighborhoods[central_vert_rel_idx, col_idx] = pvd_data[central_vert_idx_mesh] # Add label (lgi, thickness, or whatever)

    #if verbose:
    #    neighborhoods_size_bytes = getsizeof(neighborhoods)
    #    print(f"Neighborhood size in RAM is about {neighborhoods_size_bytes} bytes, or {neighborhoods_size_bytes / 1024. / 1024.} MB.")

    assert neighborhoods.shape[0] == len(kept_vertex_indices_mesh), f"Expected {len(kept_vertex_indices_mesh)} neighborhoods, but found {neighborhoods.shape[0]}."
    assert neighborhoods.shape[1] == len(col_names)

    return neighborhoods, col_names, kept_vertex_indices_mesh




# ------------------------------------------- Neighborhoods based on vertex adjacency in meshes -------------------------



def mesh_k_neighborhoods(tmesh, k=1):
    """
    Compute k-neighborhood for all mesh vertices. This is quite slow for larger k.

    Parameters:
    -----------
    tmesh: tmesh.Tmesh instance, the mesh for which vertex neighborhoods are to be computed
    k: positive integer, the hop distance (number of mesh esges to travel) to define neighborhoods

    Returns
    -------
    dictionary, keys are integer vertex indices in the mesh. values are 1D numpy.ndarrays of vertex indices making up the neighborhood for the key vertex.
    """
    print("TODO: mesh_k_neighborhoods: this should accept the max number of neighbors per neighboorhood to include and return a matrix of shape (num_verts, neigh_data_len). Also normals should be part of neigh_data_len.")
    if not isinstance(tmesh, tm.Trimesh):
        raise ValueError("Parameter 'tmesh' must be a trimesh.Trimesh instance.")
    neighborhoods = dict()
    print("Mesh has {nv} vertices, coords are in {d}d space.".format(nv=tmesh.vertices.shape[0], d=tmesh.vertices.shape[1]))
    print("Computing k-neighborhoods for k={step_idx}, will compute up to k={k}.".format(step_idx=1, k=k))
    for vert_idx in range(tmesh.vertices.shape[0]):
        neighborhoods[vert_idx] = np.array(tmesh.vertex_neighbors[vert_idx])
    if k == 1:
        return neighborhoods
    else:
        for step_idx in range(2, k+1):
            print("Computing k-neighborhoods for k={step_idx}, will compute up to k={k}.".format(step_idx=step_idx, k=k))
            for vert_idx in neighborhoods.keys():
                cur_neighbors = neighborhoods[vert_idx]
                neighborhoods[vert_idx] = np.unique(np.concatenate([neighborhoods.get(key) for key in cur_neighbors]))
    nsizes = np.array([len(v) for k,v in neighborhoods.items()])
    print("Neighborhood sizes are min={min}, max={max}, mean={mean}.".format(min=nsizes.min(), max=nsizes.max(), mean=nsizes.mean()))
    return neighborhoods

# ------------------------------------------- extraction and centering of mesh coords ------------------------------------------

def mesh_neighborhoods_coords(neighborhoods, tmesh, num_neighbors_max=None, do_center=True):
    """
    Extract coordinates of neighborhood vertex indices from the mesh, optionally setting the central query vertex to the origin ```(0,0,0)```.

    Parameters:
    -----------
    neighborhoods: dictionary, keys are integer vertex indices in the mesh. values are 1D numpy.ndarrays of vertex indices making up the neighborhood for the key vertex. Typically obtained by calling `_k_neighborhoods()`.
    tmesh: tmesh.Tmesh instance, the mesh for which vertex neighborhoods are to be computed
    num_neighbors: positive integer, how many neighbors to return per vertex. Serves to limit/fix the neighborhood size. All vertices must have at least this number of neighbors, otherwise an error is raised. Set to None to return all, which may lead to different neighborhood sizes for the individual vertices.


    Returns
    -------
    list of num_neighbors x 3 numpy.ndarrays, each 2D array contains the centered neighborhood coordinates for a single vertex
    """
    if not isinstance(neighborhoods, dict):
        raise ValueError("Parameter 'neighborhoods' must be a dict.")
    if not isinstance(tmesh, tm.Trimesh):
        raise ValueError("Parameter 'tmesh' must be a trimesh.Trimesh instance.")
    all_neigh_coords = list()

    vert_idx = 0
    for central_vertex, neighbors in neighborhoods.items(): # TODO: Check whether central vertex is part of the neighborhood, add it if not.
        if num_neighbors_max is None:
            neigh_coords = tmesh.vertices[neighbors, :]
        else:
            neigh_coords = tmesh.vertices[neighbors[0:num_neighbors_max], :]

        if do_center:
            central_coords = tmesh.vertices[central_vertex, :]
            all_neigh_coords.append(np.subtract(neigh_coords, central_coords))
        else:
            all_neigh_coords.append(central_coords)
        vert_idx += 1
    return all_neigh_coords





