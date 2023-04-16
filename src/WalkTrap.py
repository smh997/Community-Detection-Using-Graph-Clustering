import numpy as np
from utils.Community import Community
import networkx as nx
from heapq import heappush, heappop
import copy


def walktrap(G, t, self_loops=True, log=False):
    """
    WalkTrap algorithm based on Random Walk for finding Communities
    :param G: Network graph
    :param t: Number of steps in Random Walk
    :param self_loops: Necessity of self-loops
    :param log: Availability of logs of the process
    :return: Partitions, Communities, Delta-Sigmas, and Modularity of communities
    """
    # Adding self-loops
    if self_loops:
        for v in G.nodes:
            G.add_edge(v, v)

    # N: Number of vertices in the graph
    N = G.number_of_nodes()
    # A: Adjacency matrix
    A = np.array(nx.to_numpy_array(G))
    # Dx: Diagonal degree matrix
    Dx = np.zeros((N, N))
    # P: Transition matrix
    P = np.zeros((N, N))

    for i, A_r in enumerate(A):
        d_i = np.sum(A_r)
        P[i] = A_r / d_i
        Dx[i, i] = d_i ** (-0.5)

    # P_t: Probability vector of a random walk of t steps
    P_t = np.linalg.matrix_power(P, t)

    # Total weight of all the edges excluding self-edges
    total_weight = G.number_of_edges() - N

    # Total number of all initial communities
    community_count = N
    # Dictionary of all initial communities, indexed by community ID
    communities = {}
    for C_id in range(N):
        communities[C_id] = Community(A, P_t, total_weight, C_id)

    # Minheap to store delta sigmas between communities: <deltaSigma(C1,C2), C1_id, C2_id>
    min_sigma_heap = []
    for e in G.edges:
        C1_id = e[0]
        C2_id = e[1]
        if C1_id != C2_id:
            # Apply Definition 1 and Theorem 3
            delta_sigma = (0.5 / N) * np.sum(np.square(np.matmul(Dx, P_t[C1_id]) - np.matmul(Dx, P_t[C2_id])))
            heappush(min_sigma_heap, (delta_sigma, C1_id, C2_id))
            # Update each community with its neighboring communities
            communities[C1_id].Neighbor_comms[C2_id] = delta_sigma
            communities[C2_id].Neighbor_comms[C1_id] = delta_sigma
    # Delta sigmas of partitions merged at each step
    delta_sigmas = []
    # Partitions: a list of length k that stores IDs of communities for each partitioning
    partitions = [set(np.arange(N))]  # Make first partition, single-vertex communities. At every step active communities are in the last entry of "partitions"
    # Calculate modularity Q for this partition
    modularities = [np.sum([communities[C_id].Q() for C_id in partitions[0]])]
    if log:
        print("Partition 0: ", partitions[0])
        print("Q(0) = ", modularities[0])
    for k in range(1, N):
        # partitions[k-1]: Current partition
        # partitions[k]: Next partition to be created in this iteration
        # Choose communities C1, C2 to merge, according to minimum delta sigma
        # Need to also check if C1_id and C2_id are communities at the current partition partitions[k-1]
        C1C2_ds, C1_id, C2_id = None, None, None
        while min_sigma_heap:
            C1C2_ds, C1_id, C2_id = heappop(min_sigma_heap)
            if C1_id in partitions[k - 1] and C2_id in partitions[k - 1]:
                break
        if not min_sigma_heap and (C1_id in partitions[k - 1] and C2_id in partitions[k - 1]):
            break
        # Delta sigma of this step
        delta_sigmas.append(C1C2_ds)

        # Merge C1, C2 into C3, assign to it next possible ID, that is C3_ID = Total number of communities so far
        C3_id = community_count
        community_count += 1
        communities[C3_id] = Community(A, P_t, total_weight, C3_id, communities[C1_id], communities[C2_id])

        # Add new partition (k-th)
        partitions.append(copy.deepcopy(partitions[k - 1]))
        partitions[k].add(C3_id)
        partitions[k].remove(C1_id)
        partitions[k].remove(C2_id)

        # Update delta_sigma_heap with entries concerning community C3 and communities adjacent to C1, C2
        # Check all new neighbours of community C3
        for C_id in communities[C3_id].Neighbor_comms.keys():
            # If C is neighbour of both C1 and C2 then we can apply Theorem 4
            if (C_id in communities[C1_id].Neighbor_comms) and (C_id in communities[C2_id].Neighbor_comms):
                C1C_ds = communities[C1_id].Neighbor_comms[C_id]
                C2C_ds = communities[C2_id].Neighbor_comms[C_id]
                # Apply Theorem 4 to (C, C3)
                delta_sigma = ((communities[C1_id].size + communities[C_id].size) * C1C_ds + (
                            communities[C2_id].size + communities[C_id].size) * C2C_ds - communities[
                          C_id].size * C1C2_ds) / (communities[C3_id].size + communities[C_id].size)

            # Otherwise apply Theorem 3 to (C, C3)
            else:
                delta_sigma = np.sum(np.square(np.matmul(Dx, communities[C_id].P_c) - np.matmul(Dx, communities[C3_id].P_c))) * \
                     communities[C_id].size * communities[C3_id].size / (
                                 (communities[C_id].size + communities[C3_id].size) * N)

            # Update min_sigma_heap and update delta sigmas between C3 and C
            heappush(min_sigma_heap, (delta_sigma, C3_id, C_id))
            communities[C3_id].Neighbor_comms[C_id] = delta_sigma
            communities[C_id].Neighbor_comms[C3_id] = delta_sigma

            # Calculate and store modularity Q for this partition
        modularities.append(np.sum([communities[C_id].Q() for C_id in partitions[k]]))

        if log:
            print("Partition ", k, ": ", partitions[k])
            print("\tMerging ", C1_id, " + ", C2_id, " --> ", C3_id)
            print("\tQ(", k, ") = ", modularities[k])
            print("\tdelta_sigma = ", delta_sigmas[k - 1])
    if self_loops:
        for v in G.nodes:
            G.remove_edge(v, v)

    return np.array(partitions), communities, np.array(delta_sigmas), np.array(modularities)

