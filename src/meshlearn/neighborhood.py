
import numpy as np
import trimesh as tm

def neighborhoods_euclid_around_points(vert_coords, kdtree, neighborhood_radius, mesh, pvd_data, max_num_neighbors=0):
    """
    Compute the vertex neighborhood of the Tmesh for a given vertex using Euclidean distance (ball point).

    This uses a kdtree to compute all vertices in a certain radius. This is an alternative approach to the
    _k_neighborhoods() function below, which computes the k-neighborhoods on the mesh instead
    of simple Euclidean distance.

    Parameters
    ----------
    vert_coords nx3 numpy.ndarray for n 3D points
    kdtree scipy.spatial.KDTree instance
    neighborhood_radius the radius for sphere used in kdtree query, in mesh spatial units. use 25 for 25mm with freesurfer meshes. must be changed together with num_neighbors.
    mesh tmesh.Trimesh instance (the one from which vert_coords also come, currently duplicated)
    max_num_neighbors number of neighbors max to consider per neighborhood. must be changed together with neighborhood_radius.
    pvd_data vector of length vert_coords.shape[0] (number of vertes in mesh), assigning a descriptor value (cortical thoickness, lgi, ...) to each vertex.

    Returns
    -------
    """
    if kdtree is None:
        raise ValueError("No kdtree initialized yet.")
    if not isinstance(vert_coords, np.ndarray):
        raise ValueError("Expected np.ndarray as input.")
    if not vert_coords.shape[1] == 3:
        raise ValueError("Expected np.ndarray with 2nd dimension of length 3 as input.")
    assert vert_coords.shape[1] == 3  # 3 coords for the x,y,z. Just to make sure it is not transposed.
    num_verts_in_mesh = vert_coords.shape[0]
    neighbor_indices = kdtree.query_ball_point(x=vert_coords, r=neighborhood_radius) # list of arrays
    assert neighbor_indices.shape[0] == num_verts_in_mesh

    ## Atm, the number of neighbors differs between the source vertices, as we simply found all within a fixed Euclidean radius (and vertex density obvisouly differs).
    ## So we need to fix the lengths to max_num_neighbors.
    neigh_lengths = [len(neigh) for neigh in neighbor_indices]
    min_neigh_size = np.min(neigh_lengths)
    max_neigh_size = np.max(neigh_lengths)
    mean_neigh_size = np.mean(neigh_lengths)
    median_neigh_size = np.median(neigh_lengths)


    if max_num_neighbors == 0:
        max_num_neighbors = min_neigh_size # set to minimum to avoid NANs
        print(f"Auto-determinded max_num_neighbors to be {min_neigh_size} for mesh.")
    print(f"min neigh size across {len(neighbor_indices)} neighborhoods is {min_neigh_size}, max is {max_neigh_size}, mean is {mean_neigh_size}, median is {median_neigh_size}")

    ## filter neighborhoods which are too small
    neighbor_indices_filtered = [neigh[0:max_num_neighbors] for neigh in neighbor_indices if len(neigh) >= max_num_neighbors]
    print(f"Filtered neighborhoods, {len(neighbor_indices_filtered)} of {len(neighbor_indices)} left after removing all smaller than {max_num_neighbors} verts")

    neighbor_indices = neighbor_indices_filtered

    neighborhood_col_num_values = max_num_neighbors * (3 + 3) + 1 # 3 (x,y,z) coord entries per neighbor, 3 (x,y,z) vertex normal entries per neighbor, 1 pvd value per neighborhood

    ## Full matrix for all neighborhoods
    neighborhoods = np.zeros((num_verts_in_mesh, neighborhood_col_num_values), dtype=np.float)

    col_names = []
    for n_idx in range(max_num_neighbors):
        for coord in ["x", "y", "z"]:
            col_names.append("nc" + str(n_idx)) # "coord) # nc for neighbor coord
    for n_idx in range(max_num_neighbors):
        for coord in ["x", "y", "z"]:
            col_names.append("nn" + str(n_idx)) # nn for neighbor normal
    col_names.append("label")


    for central_vert_idx, neigh_vert_indices in enumerate(neighbor_indices):
        col_start_idx = 0

        col_end_idx = col_start_idx+(max_num_neighbors*3)
        #print(f"Add coords for {len(neigh_indices)} neighbors into col positions {col_start_idx} to {col_end_idx} (num columns is {neighborhood_col_num_values}).")
        neighborhoods[central_vert_idx, col_start_idx:col_end_idx] = np.ravel(mesh.vertices[neigh_vert_indices] - mesh.vertices[central_vert_idx]) # Add vertex coords relative to central vertex

        col_start_idx = col_end_idx
        col_end_idx = col_start_idx+(max_num_neighbors*3)
        #print(f"Add normals for {max_num_neighbors} neighbors into col positions {col_start_idx} to {col_end_idx}")

        neighborhoods[central_vert_idx, col_start_idx:col_end_idx] = np.ravel(mesh.vertex_normals[neigh_vert_indices]) # Add vertex normals
        col_start_idx = col_end_idx

        #print(f"Add pvd value at col position {col_start_idx}")

        neighborhoods[central_vert_idx, col_start_idx] = pvd_data[central_vert_idx] # Add label (lgi, thickness, or whatever)

    print("Neighborhoods filled.")

    assert neighborhoods.shape[1] == len(col_names)
    return neighborhoods, col_names




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





