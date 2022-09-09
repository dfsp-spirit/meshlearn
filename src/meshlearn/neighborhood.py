
import numpy as np
import trimesh as tm

def neighborhoods_euclid_around_points(vert_coords, kdtree, neighborhood_radius):
    """
    Compute the vertex neighborhood of the Tmesh for a given vertex using Euclidean distance (ball point).

    This uses a kdtree to compute all vertices in a certain radius. This is an alternative approach to the
    _k_neighborhoods() function below, which computes the k-neighborhoods on the mesh instead
    of simple Euclidean distance.

    Parameters
    ----------
    vert_coords nx3 numpy.ndarray for n 3D points
    kdtree scipy.spatial.KDTree instance

    Returns
    -------

    """
    if kdtree is None:
        raise ValueError("No kdtree initialized yet.")
    if not isinstance(vert_coords, np.ndarray):
        raise ValueError("Expected np.ndarray as input.")
    if not vert_coords.shape[1] == 3:
        raise ValueError("Expected np.ndarray with 2nd dimension of length 3 as input.")
    neighborhoods = kdtree.query_ball_point(x=vert_coords, r=neighborhood_radius)
    return neighborhoods




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





