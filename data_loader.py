import torch
import numpy as np
import random
from torch_geometric.data import Data


def data_loader():

    labels = []
    node_features = []
    edge_features = []

    counter = 0
    nr_nodes = 100
    nodes = range(0, nr_nodes)

    for node in nodes:
        # spammer
        if random.random() > 0.5:
            # more likely to have many connections (a maximum of 1/4 of the nodes in the graph)
            nr_nbrs = int(random.random() * (nr_nodes / 4))
            counter = counter + nr_nbrs
            # more likely to have sent many bytes
            node_features.append((random.random() + 1))
            # if the node is more likely to be a spammer then
            # the value will be closer to 1. The same value is
            # used for all the edges to the node's neighbours
            edge_features += [(random.random() + 3) / 4.] * nr_nbrs
            # associate a label
            labels.append(1)

        # non-spammer
        else:
            # at most connected to 10 nbrs
            nr_nbrs = int(random.random() * 10)
            counter = counter + nr_nbrs
            print(node, nr_nbrs)
            # associate more bytes and random bytes
            node_features.append(random.random())
            edge_features += [random.random()] * nr_nbrs
            labels.append(0)

        # randomly sample nr_nbrs out of the 1D array [1, 2, ..., nr_nodes]
        nbrs = np.random.choice(nodes, size=nr_nbrs)
        nbrs = nbrs.reshape((1, nr_nbrs))

        # add the edges of nbrs
        node_edges = np.concatenate([np.ones((1, nr_nbrs), dtype=np.int32) * node, nbrs], axis=0)

        #  add the overall edges
        if node == 0:
            edges = node_edges
        else:
            edges = np.concatenate([edges, node_edges], axis=1)

    print(edges)
    assert counter == len(edges[0]) == len(edges[1]) == len(edge_features)

    x = torch.tensor(np.expand_dims(node_features, 1), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_attr = torch.tensor(np.expand_dims(edge_features, 1), dtype=torch.float)

    # x:
    #       the node features
    # edge_index:
    #       a 2D array, with the directed edges from the nodes in the
    #       first row to the nodes in the second row
    # y:
    #       the labels (0 or 1) depending on if it is a spammer or not
    # edge_attr:
    #       the attributes associated to each edge, in this case byte
    #       transfer simulation
    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    print(data)

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # train only on the 80% nodes
    data.train_mask[:int(0.8 * data.num_nodes)] = 1
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # test on 20 % nodes
    data.test_mask[- int(0.2 * data.num_nodes):] = 1
    return data
