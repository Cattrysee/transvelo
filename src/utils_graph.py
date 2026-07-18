import numpy as np
import scipy.sparse as sp

def velocity_graph_smoothing(velocity, adj):

    """
    velocity: (cells, genes)
    adj: scipy sparse connectivity
    """

    if sp.issparse(adj):
        adj = adj.tocsr()

    smoothed = adj @ velocity

    return smoothed