import copy
import networkx as nx


######################################################################
# Convert partition representation: set -> list of sets
######################################################################
# def partition_set_to_sets(comms, partition):
#     list_of_sets = []
#     for C_id in partition:
#         list_of_sets.append(copy.deepcopy(comms[C_id].vertices))
#     return list_of_sets


######################################################################
# Convert partition representation: dictionary -> list of sets
######################################################################
# def partition_dict_to_sets(d):
#     inverse_dict = {}
#     for k, v in d.items():
#         if v not in inverse_dict:
#             inverse_dict[v] = set()
#         inverse_dict[v].add(k)
#
#     return list(inverse_dict.values())


######################################################################
# Convert edge list to Networkx Graph
######################################################################
# def edge_list_to_graph(edges, verbose=False):
#     G = nx.Graph()
#     G.add_edges_from(edges)
#     if verbose:
#         print(G.number_of_edges(), " edges, ", G.number_of_nodes(), " nodes")
#     return G


######################################################################
# Prepare partition for color plotting: as dictionary
######################################################################
def partition_to_plot(coms, partition):
    p_dict = {}
    for i, C_id in enumerate(partition):
        for v in coms[C_id].V:
            p_dict[v] = i
    return p_dict


def comm2part(comms):
    part = dict()
    for c_id, c in enumerate(comms):
        for v in c:
            part[v] = c_id
    return part


def part2comm(part):
    comms = [set() for i in range(len(set(part.values())))]
    for v in part:
        comms[part[v]].add(v)
    return comms