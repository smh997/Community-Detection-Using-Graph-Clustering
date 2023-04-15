import networkx as nx
from matplotlib import pyplot as plt
from WalkTrap import walktrap
import numpy as np
import time
import community
from communities.algorithms import louvain_method, girvan_newman, spectral_clustering
from utils.tools import partition_to_plot, comm2part


def compare_algorithms(G, K_truth, dataset=None, plot=False):
    """
    Running and comparing algorithms on the given network regarding the ground truth
    :param G: Network graph
    :param K_truth: Ground truth
    :return: None
    """
    G = nx.convert_node_labels_to_integers(G)
    pos = nx.spring_layout(G)
    print("Ground truth: ", K_truth, " communities")

    # Q vs k for various t
    plt.figure()
    for t in list(range(2, 9)) + [20, 50, 100]:
        parts, coms, _, Qs = walktrap(G, t)
        ks = np.arange(len(Qs))
        # K: Best number of communities found by algorithm
        K = len(Qs) - np.argmax(Qs)
        plt.plot(ks, Qs, label='t = ' + str(t) + ", K = " + str(K))
    plt.xlabel('iteration k')
    plt.ylabel('Modularity Q')
    plt.title('Modularity Q vs iteration')
    plt.legend()
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/Qvsk of WalkTrap on {dataset}.png')

    # Eta vs k for various t
    plt.figure()
    for t in list(range(2, 9)) + [20, 50, 100]:
        parts, coms, ds, _ = walktrap(G, t)
        etas = ds[1:] / ds[0:-1]
        ks = np.arange(len(etas)) + 1
        # K: Best number of communities found by algorithm
        K = 1 + len(etas) - np.argmax(etas)
        plt.plot(ks, etas, label='t = ' + str(t) + ", K = " + str(K))
    plt.xlabel('iteration k')
    plt.ylabel('$\eta$')
    plt.title('Quality $\eta$ of partition vs iteration')
    plt.legend()
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/Evsk of WalkTrap on {dataset}.png')

    # Comparison

    # WalkTrap Algorithm:
    t = 2
    start_time = time.time()
    parts, coms, _, Qs = walktrap(G, t)
    wt_time = time.time() - start_time
    Qmax_index = np.argmax(Qs)
    print("Walktrap ( t =", str(t), ") algorithm:")
    print("\tOptimal number of communities: K = ", len(Qs) - Qmax_index)
    print("\tBest modularity: Q = ", Qs[Qmax_index])
    print("\tRuntime: ", wt_time, " seconds")
    my_best_part = partition_to_plot(coms, parts[Qmax_index])
    nx.draw(G, pos, node_color=list(my_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/WalkTrap t2 on {dataset}.png')

    t = 5
    start_time = time.time()
    parts, coms, _, Qs = walktrap(G, t)
    wt_time = time.time() - start_time
    Qmax_index = np.argmax(Qs)
    print("Walktrap ( t =", str(t), ") algorithm:")
    print("\tOptimal number of communities: K = ", len(Qs) - Qmax_index)
    print("\tBest modularity: Q = ", Qs[Qmax_index])
    print("\tRuntime: ", wt_time, " seconds")
    my_best_part = partition_to_plot(coms, parts[Qmax_index])
    nx.draw(G, pos, node_color=list(my_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/WalkTrap t5 on {dataset}.png')

    t = 8
    start_time = time.time()
    parts, coms, _, Qs = walktrap(G, t)
    wt_time = time.time() - start_time
    Qmax_index = np.argmax(Qs)
    print("Walktrap ( t =", str(t), ") algorithm:")
    print("\tOptimal number of communities: K = ", len(Qs) - Qmax_index)
    print("\tBest modularity: Q = ", Qs[Qmax_index])
    print("\tRuntime: ", wt_time, " seconds")
    my_best_part = partition_to_plot(coms, parts[Qmax_index])
    nx.draw(G, pos, node_color=list(my_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/WalkTrap t8 on {dataset}.png')

    # Louvain Algorithm:
    adj_matrix = np.array(nx.to_numpy_array(G))
    start_time = time.time()
    comms, _ = louvain_method(adj_matrix)
    part = comm2part(comms)
    Q = community.modularity(part, G)
    print("Louvain algorithm:")
    print("\tOptimal number of communities: K = ", len(set(part.values())))
    print("\tBest modularity: Q = ", Q)
    print("\tRuntime: ", time.time() - start_time, " seconds")
    louvain_best_part = part
    nx.draw(G, pos, node_color=list(louvain_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/Louvain on {dataset}.png')

    # Girvan-Newman Algorithm:
    start_time = time.time()
    comms, _ = girvan_newman(adj_matrix)
    part = comm2part(comms)
    Q = community.modularity(part, G)
    print("Girvan-Newman algorithm:")
    print("\tOptimal number of communities: K = ", len(set(part.values())))
    print("\tBest modularity: Q = ", Q)
    print("\tRuntime: ", time.time() - start_time, " seconds")
    girvan_best_part = part
    nx.draw(G, pos, node_color=list(girvan_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/GirvanNewman on {dataset}.png')

    # Spectral Clustering Algorithm
    k = K_truth-1
    start_time = time.time()
    comms = spectral_clustering(adj_matrix, k=k)
    part = comm2part(comms)
    Q = community.modularity(part, G)
    print("Spectral Clustering algorithm:")
    print("\tOptimal number of communities: K = ", len(set(part.values())))
    print("\tBest modularity: Q = ", Q)
    print("\tRuntime: ", time.time() - start_time, " seconds")
    spectral_best_part = part
    nx.draw(G, pos, node_color=list(spectral_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/Spectral truth-1 on {dataset}.png')

    k = K_truth
    start_time = time.time()
    comms = spectral_clustering(adj_matrix, k=k)
    part = comm2part(comms)
    Q = community.modularity(part, G)
    print("Spectral Clustering:")
    print("\tOptimal number of communities: K = ", len(set(part.values())))
    print("\tBest modularity: Q = ", Q)
    print("\tRuntime: ", time.time() - start_time, " seconds")
    spectral_best_part = part
    nx.draw(G, pos, node_color=list(spectral_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/Spectral truth on {dataset}.png')

    k = K_truth + 1
    start_time = time.time()
    comms = spectral_clustering(adj_matrix, k=k)
    part = comm2part(comms)
    Q = community.modularity(part, G)
    print("Spectral Clustering:")
    print("\tOptimal number of communities: K = ", len(set(part.values())))
    print("\tBest modularity: Q = ", Q)
    print("\tRuntime: ", time.time() - start_time, " seconds")
    spectral_best_part = part
    nx.draw(G, pos, node_color=list(spectral_best_part.values()))
    if plot:
        plt.show()
    else:
        plt.savefig(f'../output/Spectral truth plus 1 on {dataset}.png')


# Experiments on Zachary’s Karate Club graph (Unweighted)
G = nx.read_edgelist('../dataset/zachary/zachary3.txt', nodetype=int)
compare_algorithms(G, 4, dataset='KarateClub(unweighted)')


# Experiments on Zachary’s Karate Club graph (Weighted)
G = nx.karate_club_graph()
compare_algorithms(G, 4, dataset='KarateClub(weighted)')

# Experiments on the college football network
G = nx.read_gml('../dataset/football/football.gml', label='id')
compare_algorithms(G, 12, 'CollegeFootball')


