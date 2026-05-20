"""TSP ordering using 2-opt on a precomputed SE(3) distance matrix.

For K <= 50 viewpoints, 2-opt converges in well under a second.
"""
import numpy as np
import se3


def build_distance_matrix(poses, w_t=1.0, w_r=0.1):
    K = len(poses)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            d = se3.se3_distance(poses[i], poses[j], w_t=w_t, w_r=w_r)
            D[i, j] = d
            D[j, i] = d
    return D


def tour_length(order, D):
    return sum(D[order[i], order[i + 1]] for i in range(len(order) - 1))


def two_opt(D, order=None, max_iter=2000):
    """2-opt local search. Open tour (path, not cycle)."""
    n = len(D)
    if order is None:
        order = list(range(n))
    best_len = tour_length(order, D)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_order = order[:i] + order[i:j][::-1] + order[j:]
                new_len = tour_length(new_order, D)
                if new_len < best_len - 1e-9:
                    order = new_order
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break
        it += 1
    return order, best_len


def nearest_neighbor_order(D, start=0):
    n = len(D)
    visited = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        last = visited[-1]
        nxt = min(remaining, key=lambda j: D[last, j])
        visited.append(nxt)
        remaining.remove(nxt)
    return visited


def solve_tsp(poses, w_t=1.0, w_r=0.1):
    """Nearest-neighbor + 2-opt. Returns ordered list of pose indices."""
    D = build_distance_matrix(poses, w_t=w_t, w_r=w_r)
    init = nearest_neighbor_order(D)
    order, length = two_opt(D, init)
    return order, length, D
