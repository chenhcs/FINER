import numpy as np
import random
import networkx as nx
import math
from multiprocessing import Pool
from functools import partial


def read_net(file_name, node_size, gene_list):
    gene_iso_dict = {}
    for gene in set(gene_list):
        idx = np.where(gene_list == gene)[0]
        gene_iso_dict[gene] = idx

    fr = open(file_name)
    G = nx.Graph()
    nodes = np.arange(node_size)
    G.add_nodes_from(nodes)
    edges = []
    while True:
        line = fr.readline()
        if not line:
            break
        node1, node2 = line.split('\n')[0].split(',')
        isoid1 = gene_iso_dict[node1]
        isoid2 = gene_iso_dict[node2]
        for i in isoid1:
            for j in isoid2:
                edges.append([i,j])
    fr.close()

    edges = np.array(edges)
    G.add_edges_from(edges)

    nodes_with_edges = []
    genes_with_edges = []
    for node in G.nodes():
        if G.degree(node) > 0:
            nodes_with_edges.append(node)
            genes_with_edges.append(gene_list[node])

    print("# of nodes:", G.number_of_nodes())
    print("# of nodes with edges:", len(nodes_with_edges))
    print("# of edges:", G.number_of_edges())
    return G, list(set(genes_with_edges))


def adjacency(G):
    adjacency_list = {}
    for node in G.nodes():
        neighbors = nx.neighbors(G, node)
        adjacency_list[node] = list(neighbors)
    return adjacency_list


def get_degree(iter, nodes, pos_nodes_set, iii_net):
    degree_map = {}
    for nd in nodes:
        degree_map[nd] = len(set(iii_net.neighbors(nd)).intersection(pos_nodes_set))
    return degree_map


def create_neg_table(G, non_functional_set, gene_list):
    print('Creating negative sampling table...')
    non_functional_nodes = []
    for node in G.nodes():
        if node in non_functional_set:
            non_functional_nodes.append(node)
    power = 0.75
    norm = sum([math.pow(G.degree(node), power) for node in non_functional_nodes]) # Normalizing constant
    table_size = int(1e8) # Length of the unigram table
    table = np.zeros(table_size, dtype=np.uint32)
    p = 0 # Cumulative probability
    i = 0
    for node in non_functional_nodes:
        p += float(math.pow(G.degree(node), power))/norm
        while i < table_size and float(i) / table_size < p:
            table[i] = node
            i += 1
    print('Finish')

    return table


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-
    method-efficient-sampling-with-many-discrete-outcomes for details.
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class Walks():
    def __init__(self, nx_G, p, q, initial_flag, instance_to_bag=None, neighbor_changed=None, alias_edges_old=None):
        """
        Construct network neighborhoods for each node in every layer.
        Refer to: http://github.com/aditya-grover/node2vec for details.
        """
        self.G = nx_G
        self.p = p
        self.q = q

        print("Preprocessing random walk transition probs...")
        self.preprocess_transition_probs(initial_flag, instance_to_bag, neighbor_changed, alias_edges_old)

    def node2vec_walk(self, walk_length, start_node):
        """Simulate a random walk starting from start node."""
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    idx = alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])
                    walk.append(cur_nbrs[idx])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """Repeatedly simulate random walks from each node."""
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print('%3d/%3d' % (walk_iter + 1, num_walks))
            random.shuffle(nodes)
            for node in nodes:
                if G.degree(node) == 0:
                    continue
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        """Get the alias edge setup lists for a given edge."""
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1.0/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(1.0)
            else:
                unnormalized_probs.append(1.0/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self, initial_flag, instance_to_bag, neighbor_changed, alias_edges_old):
        """Preprocessing of transition probabilities for guiding the random walks."""
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            out_degree = len(list(G.neighbors(node)))
            if out_degree == 0:
                continue
            else:
                normalized_probs = [1.0 / out_degree] * out_degree

            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        cnt = 0
        if initial_flag:
            alias_edges_bags = {}
            for edge in G.edges():
                src = edge[0]
                dst = edge[1]
                src_bag = instance_to_bag[src]
                dst_bag = instance_to_bag[dst]
                if (src_bag, dst_bag) in alias_edges_bags:
                    alias_edges[(src, dst)] = alias_edges_bags[(src_bag, dst_bag)]
                    alias_edges[(dst, src)] = alias_edges_bags[(dst_bag, src_bag)]
                else:
                    alias_edges[(src, dst)] = self.get_alias_edge(src, dst)
                    alias_edges[(dst, src)] = self.get_alias_edge(dst, src)
                    alias_edges_bags[(src_bag, dst_bag)] = alias_edges[(src, dst)]
                    alias_edges_bags[(dst_bag, src_bag)] = alias_edges[(dst, src)]

                cnt += 1
                if cnt%10000 == 0:
                    print("Finished:", cnt / 10000)

        else:
            for edge in G.edges():
                src = edge[0]
                dst = edge[1]
                if neighbor_changed[src] or neighbor_changed[dst]:
                    alias_edges[(src, dst)] = self.get_alias_edge(src, dst)
                    alias_edges[(dst, src)] = self.get_alias_edge(dst, src)
                else:
                    alias_edges[(src, dst)] = alias_edges_old[(src, dst)]
                    alias_edges[(dst, src)] = alias_edges_old[(dst, src)]

                cnt += 1
                if cnt%10000 == 0:
                    print("Finished:", cnt / 10000)

        print("Finish calculating transition probs for alias edges...")

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return
