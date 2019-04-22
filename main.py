from itertools import combinations, product
from collections import OrderedDict
import numpy as np

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #


def my_pdist(X, metric):
    return np.array([np.linalg.norm(a - b) for a, b in combinations(X, 2)])

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #


def my_squareform(vector):
    dim = int(np.ceil((vector.size * 2) ** 0.5))
    mat = np.zeros((dim, dim))
    start, n = 0, dim - 1
    left_indexes = list(range(1, dim))
    for i in range(dim - 1):
        mat[i, left_indexes[i:]] = vector[range(start, start + n)]
        start += n
        n -= 1
    mat.T[np.triu_indices(dim)] = mat[np.triu_indices(dim)]
    return mat

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #


class Cluster:
    def __init__(self, name, leaves, lookup):
        self.name = name
        self.leaves = leaves
        self.lookup = lookup

    def distance(self, other):
        products = product(self.leaves, other.leaves)
        return min([self.lookup[tuple(sorted(product))] for product in products])


def create_clusters(n, lookup):
    clusters = OrderedDict()
    for c in range(n):
        clusters[c] = Cluster(c, [c], lookup)
    return clusters


def merge_clusters(clusters, a, b, name, lookup):
    cluster = Cluster(name, clusters[a].leaves + clusters[b].leaves, lookup)
    clusters[name] = cluster
    del clusters[a]
    del clusters[b]
    return clusters


def my_linkage(matrix, method, metric):
    vector = my_pdist(matrix, metric=metric)
    n = int(np.ceil((vector.size * 2) ** 0.5))
    indexes = combinations(range(n), 2)
    lookup = {idx: dis for dis, idx in zip(vector, indexes)}

    clusters = create_clusters(n, lookup)
    output = []

    while len(clusters) > 1:
        distances = [(c1.distance(c2), c1.name, c2.name) for c1, c2 in combinations(clusters.values(), 2)]
        d, a, b = min(distances, key=lambda tup: tup[0])
        clusters = merge_clusters(clusters, a, b, n, lookup)
        output.append([a, b, d, len(clusters[n].leaves)])
        n += 1

    return np.array(output)

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
