import networkx as nx
import numpy as np


# degrees
def get_list_in_degree(G):
    return [G.in_degree(n) for n in G.nodes()]


def get_list_out_degree(G):
    return [G.out_degree(n) for n in G.nodes()]


def get_list_degree(G):
    return [G.out_degree(n) + G.in_degree(n) for n in G.nodes()]


# clustering coef, seems wrong
def get_clust_list(G):
    return list(nx.clustering(nx.Graph(G)).values())


# Node activity
def node_activity(G, dims):
    vectors = degree_vectors(G, dims)
    result = []
    for n in G:
        vector = vectors[n]
        na = float(np.sum([v != 0 for v in vector])) / float(len(dims))
        result.append(na)
    return result


# dimensional degree
def dimensional_degree(G, reference, dims):
    vectors = degree_vectors(G, dims)
    result = []
    for n in G:
        vector = vectors[n]
        result.append(vector[dims.index(reference)])
    return result


# node type distribution
# def nodeTypeDistribution(G, types):
#    result = []
#    for n in G:
#        typee = G.nodes['type']
#        result[types.index(typee)] += 1
#    return result

# MPC Varro et al.
def degree_vectors(G, dims):
    vectors = {}
    for n in G.nodes():
        vectors[n] = ([0] * len(dims))
    for e in G.edges:
        dim_e = G[e[0]][e[1]][e[2]]['type']
        if (not dim_e in dims):
            continue
        vectors[e[0]][dims.index(dim_e)] += 1
        vectors[e[1]][dims.index(dim_e)] += 1
    return vectors


def mpc(G, dims):
    degreevecs = degree_vectors(G, dims)
    mpc = {}
    for n, vs in degreevecs.items():
        dim = len(vs)
        deg = np.sum(vs)
        summ = 0.
        for d in vs:
            summ = summ + ((d / deg) ** 2)
        mpc_v = (1 - summ) * dim / (dim - 1)
        mpc[n] = mpc_v
    return mpc
