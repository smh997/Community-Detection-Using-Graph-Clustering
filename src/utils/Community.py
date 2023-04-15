class Community:
    def __init__(self, A, P_t, G_w, id, C1=None, C2=None):
        """
        Community is a class representing communities in the network
        :param A: Adjacency matrix
        :param P_t: Probability vector
        :param G_w: Total weight of the edges in the whole network
        :param id: ID of the community
        :param C1: Community1 that this new community will be made of
        :param C2: Community2 that this new community will be made of
        """
        self.id = id
        self.A = A
        self.P_t = P_t
        self.G_w = G_w

        # Merging 2 communities to create a new one
        if C1:
            self.size = C1.size + C2.size
            self.P_c = (C1.size * C1.P_c + C2.size * C2.P_c) / self.size
            # Merge info of Neighboring communities and remove C1, C2
            self.Neighbor_comms = {k: C1.Neighbor_comms.get(k, 0) + C2.Neighbor_comms.get(k, 0) for k in set(C1.Neighbor_comms) | set(C2.Neighbor_comms)}
            del self.Neighbor_comms[C1.id]
            del self.Neighbor_comms[C2.id]
            self.V = C1.V.union(C2.V)  # V: Vertices of this new community
            C1C2_I_c = 0.  # C1C2_I_c: Total weights of edges between C1 and C2
            for v1 in C1.V:
                for id, x in enumerate(A[v1]):
                    if x == 1. and id in C2.V:
                        C1C2_I_c += 1.
            self.I_c = C1.I_c + C2.I_c + C1C2_I_c  # I_c: Internal weights of edges in this new community
            self.T_c = C1.T_c + C2.T_c  # T_c: Total weights of edges of this new community including links to other communities
        # Creating a new community from a single vertex
        else:
            self.size = 1
            self.P_c = P_t[self.id]
            self.Neighbor_comms = {}
            self.V = set([self.id])
            self.I_c = 0.
            self.T_c = self.I_c + (len([id for id, x in enumerate(A[self.id]) if x == 1. and id != self.id]) / 2.)  # External edges have 0.5 weight, ignore edge to itself

    def Q(self):
        return (self.I_c - (self.T_c * self.T_c / self.G_w)) / self.G_w
