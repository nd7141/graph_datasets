import functools
import operator

import networkx as nx
import os
import ast
import networkx.algorithms.isomorphism as iso

def writeg(graph, fn):
    n = len(graph)
    m = len(graph.edges())
    with open(fn, 'w+') as f:
        f.write(f"{n} {m}\n")
        for e in graph.edges():
            f.write(f"{e[0]} {e[1]}\n")


def formatg(graph):
    "Format graph for correct writing"
    nodes = graph.nodes()
    mapping = dict(zip(nodes, range(len(nodes))))
    return nx.relabel_nodes(graph, mapping)

def write_node_labels(fn, node_labels):
    sorted_labels = sorted(enumerate(node_labels), key=lambda x: x[1])
    with open(fn, 'w+') as f:
        for i in range(len(sorted_labels)):
            node, label = sorted_labels[i]
            if i < len(sorted_labels) - 1:
                if label != sorted_labels[i+1][1]:
                    f.write(f"{node} 0\n")
                else:
                    f.write(f"{node} 1\n")
            else:
                f.write(f"{node} 0\n")

# def write_edge_labels(edges, fn, edge_labels):
#     print(edges)
#     print(edge_labels)
#     assert len(edges) == len(edge_labels), f"Got wrong number of labels {len(edges)} {len(edge_labels)}"
#     with open(fn, 'w') as f:
#         for i in range(len(edges)):
#             e = edges[i]
#             lab = edge_labels[i]
#             f.write(f"{e[0]} {e[1]} {lab}\n")

def convert_dortmund_to_graphml(input_folder, output_folder):
    fns = os.listdir(input_folder)
    graphs_fn = indicator_fn = graph_labels_fn = \
        node_labels_fn = edge_labels_fn = None
    for fn in fns:
        if 'A.txt' in fn:
            graphs_fn = input_folder + fn
        elif '_graph_indicator.txt' in fn:
            indicator_fn = input_folder + fn
        elif '_graph_labels.txt' in fn:
            graph_labels_fn = input_folder + fn
        elif '_node_labels.txt' in fn:
            node_labels_fn = input_folder + fn
        elif '_edge_labels.txt' in fn:
            edge_labels_fn = input_folder + fn

    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with open(indicator_fn) as f:
        nodes2graph = dict()
        for i, line in enumerate(f):
            nodes2graph[i + 1] = int(line.strip())

    if node_labels_fn:
        node_labels_f = open(node_labels_fn)
    if edge_labels_fn:
        edge_labels_f = open(edge_labels_fn)
    with open(graphs_fn) as f:
        current_graph = 1
        edges = []
        for i, line in enumerate(f):
            l = line.strip().split(',')
            u, v = int(l[0]), int(l[1])
            g1, g2 = nodes2graph[u], nodes2graph[v]
            assert g1 == g2, 'Nodes should be connected in the same graph. Line {}, graphs {} {}'. \
                format(i, g1, g2)

            if g1 != current_graph:  # assumes indicators are sorted
                # print(g1, current_graph, edges)
                G = nx.Graph()
                G.add_edges_from(edges)
                G = formatg(G)
                if node_labels_fn:
                    node_labels = [next(node_labels_f) for _ in range(len(G))]
                # if edge_labels_fn:
                #     edge_labels = [int(next(edge_labels_f)) for _ in range(2 * len(G.edges()))]
                # print(len(G.edges()), len(G.nodes()), len(set(edges)), len(G)*(len(G)-1)/2)
                writeg(G, output_folder + 'graph_{}.adj'.format(current_graph))
                if node_labels_fn:
                    write_node_labels(output_folder + '{}.node_labels'.format(current_graph), node_labels)
                # if edge_labels_fn:
                #     write_edge_labels(edges, output_folder + '{}.edge_labels'.format(current_graph), edge_labels)
                edges = []
                current_graph += 1
                if current_graph % 1000 == 0:
                    print('Finished {} dataset'.format(current_graph - 1))

            edges.append((u, v))

    # write last graph
    G = nx.Graph()
    G.add_edges_from(edges)
    G = formatg(G)
    if node_labels_fn:
        node_labels = [next(node_labels_f) for _ in range(len(G))]
    # if edge_labels_fn:
    #     edge_labels = [int(next(edge_labels_f)) for _ in range(2 * len(G.edges()))]
    # print(len(G.edges()), len(G.nodes()), len(set(edges)), len(G)*(len(G)-1)/2)
    writeg(G, output_folder + 'graph_{}.adj'.format(current_graph))
    if node_labels_fn:
        write_node_labels(output_folder + '{}.node_labels'.format(current_graph), node_labels)
    # if edge_labels_fn:
    #     write_edge_labels(edges, output_folder + '{}.edge_labels'.format(current_graph), edge_labels)

    if node_labels_fn:
        node_labels_f.close()
    if edge_labels_fn:
        edge_labels_f.close()


def get_clean_graph_idx(graph_labels, path_to_orbits):
    '''
    Return indices of the dataset that should be included for training.
    It gets graph orbits and keep one graph from each orbit if orbit contains the same labels,
    or else removes entirely the orbit.
    :param dataset_name:
    :param path_to_orbits:
    :return:
    '''
    # get a list of lists, with graphs that belong to orbits
    with open(path_to_orbits) as f:
        true_orbits = [list(map(int, ast.literal_eval(''.join(line.split()[2:])))) for line in f]

    # get labels in each orbit
    orbit_labels = [[graph_labels[graph] for graph in orbit] for orbit in true_orbits]

    # keep a representative of the orbit
    orbit_graphs = []
    for i, orbit in enumerate(true_orbits):
        assert len(orbit) == len(orbit_labels[i])
        if len(set(orbit_labels[i])) == 1:  # there is only one label in the orbit
            orbit_graphs.append(orbit[0])  # keep only first graph from the orbit

    # calculate all graphs needed to be removed
    iso_graphs = set()
    for orbit in true_orbits:
        iso_graphs = iso_graphs.union(orbit)

    iso_graphs = iso_graphs.difference(orbit_graphs)

    clean_graph_idx = [idx for idx in range(len(graph_labels)) if idx + 1 not in iso_graphs]

    return clean_graph_idx


class GraphStruct():
    def __init__(self, edges, node_labels, node_attributes,
                 edge_labels, edge_attributes):
        self.edges = edges
        self.edge_labels = edge_labels
        self.edge_attributes = edge_attributes
        self.nodes = set(functools.reduce(operator.iconcat, self.edges, []))
        self.node_labels = {node: node_labels[node] for node in self.nodes} if len(node_labels) else dict()
        self.node_attributes = {node: node_attributes[node] for node in self.nodes} if len(node_attributes) else dict()

    def is_edge_label_directed(self):
        el = self.edge_labels
        for e in el:
            if el[e] != el[(e[1], e[0])]:
                return True

        return False

    def is_edge_attribute_directed(self):
        ea = self.edge_attributes
        for e in ea:
            if ea[e] != ea[(e[1], e[0])]:
                return True

        return False

    def convert_to_nx(self):
        G = nx.Graph()
        if self.is_edge_label_directed() or self.is_edge_attribute_directed():
            G = nx.DiGraph()
        G.add_edges_from(self.edges)
        nx.set_edge_attributes(G, self.edge_labels, 'edge_label') if len(self.edge_labels) else None
        nx.set_edge_attributes(G, self.edge_attributes, 'edge_attribute') if len(self.edge_attributes) else None
        nx.set_node_attributes(G, self.node_labels, 'node_label') if len(self.node_labels) else None
        nx.set_node_attributes(G, self.node_attributes, 'node_attribute') if len(self.node_attributes) else None
        return G


def clean_dataset(dataset, input_folder, output_folder, path_to_orbits, save_new_dataset=True):
    fns = os.listdir(input_folder)
    graphs_fn = indicator_fn = graph_labels_fn = \
        node_labels_fn = edge_labels_fn = \
        edge_attributes_fn = node_attributes_fn = graph_attributes_fn = None
    for fn in fns:
        if 'A.txt' in fn:
            graphs_fn = input_folder + fn
        elif '_graph_indicator.txt' in fn:
            indicator_fn = input_folder + fn
        elif '_graph_labels.txt' in fn:
            graph_labels_fn = input_folder + fn
        elif '_node_labels.txt' in fn:
            node_labels_fn = input_folder + fn
        elif '_edge_labels.txt' in fn:
            edge_labels_fn = input_folder + fn
        elif '_node_attributes.txt' in fn:
            node_attributes_fn = input_folder + fn
        elif '_edge_attributes.txt' in fn:
            edge_attributes_fn = input_folder + fn
        elif '_graph_attributes.txt' in fn:
            graph_attributes_fn = input_folder + fn

    if edge_labels_fn:
        edge_labels_f = open(edge_labels_fn)
    if edge_attributes_fn:
        edge_attributes_f = open(edge_attributes_fn)

    with open(indicator_fn) as f:
        nodes2graph = dict()
        for i, line in enumerate(f):
            nodes2graph[i + 1] = int(line.strip())

    node_labels = dict()
    if node_labels_fn:
        with open(node_labels_fn) as f:
            for i, line in enumerate(f):
                node_labels[i + 1] = line.strip()

    node_attributes = dict()
    if node_attributes_fn:
        with open(node_attributes_fn) as f:
            for i, line in enumerate(f):
                node_attributes[i + 1] = line.strip()

    if graph_attributes_fn:
        graph_attributes = dict()
        with open(graph_attributes_fn) as f:
            for i, line in enumerate(f):
                graph_attributes[i + 1] = line.strip()

    new_graphs = []
    with open(graphs_fn) as f:
        current_graph = 1
        edges = []
        edge_labels = dict()
        edge_attributes = dict()
        for i, line in enumerate(f):
            l = line.strip().split(',')
            u, v = int(l[0]), int(l[1])
            g1, g2 = nodes2graph[u], nodes2graph[v]
            assert g1 == g2, 'Nodes should be connected in the same graph. Line {}, graphs {} {}'. \
                format(i, g1, g2)

            if g1 != current_graph:  # assumes indicators are sorted
                # print(g1, current_graph, edges)
                G = GraphStruct(edges, node_labels, node_attributes, edge_labels, edge_attributes)

                new_graphs.append(G)

                edges = []
                edge_labels = dict()
                edge_attributes = dict()
                current_graph += 1
                if current_graph % 1000 == 0:
                    print('Finished {} dataset'.format(current_graph - 1))

            edges.append((u, v))
            if edge_labels_fn:
                edge_labels[(u, v)] = next(edge_labels_f).strip()
            if edge_attributes_fn:
                edge_attributes[(u, v)] = next(edge_attributes_f).strip()

    # last graph
    G = GraphStruct(edges, node_labels, node_attributes, edge_labels, edge_attributes)

    new_graphs.append(G)

    if edge_labels_fn:
        edge_labels_f.close()
    if edge_attributes_fn:
        edge_attributes_f.close()

    if not save_new_dataset:
        return new_graphs

    # output phase
    total_nodes = 1
    os.makedirs(output_folder, exist_ok=True)

    clean_graph_edges_f = open(output_folder + f'{dataset}_A.txt', 'w+')
    clean_graph_indicator_f = open(output_folder + f'{dataset}_graph_indicator.txt', 'w+')
    clean_graph_labels_f = open(output_folder + f'{dataset}_graph_labels.txt', 'w+')

    if node_labels_fn:
        clean_node_labels_f = open(output_folder + f'{dataset}_node_labels.txt', 'w+')
    if edge_labels_fn:
        clean_edge_labels_f = open(output_folder + f'{dataset}_edge_labels.txt', 'w+')
    if node_attributes_fn:
        clean_node_attributes_f = open(output_folder + f'{dataset}_node_attributes.txt', 'w+')
    if edge_attributes_fn:
        clean_edge_attributes_f = open(output_folder + f'{dataset}_edge_attributes.txt', 'w+')
    if graph_attributes_fn:
        clean_graph_attributes_f = open(output_folder + f'{dataset}_graph_attributes.txt', 'w+')

    graph_labels = dict()
    with open(graph_labels_fn) as f:
        for i, label in enumerate(f):
            graph_labels[i + 1] = label.strip()

    clean_graph_idx = get_clean_graph_idx(graph_labels, path_to_orbits + dataset + '_orbits.txt')

    current_graph_idx = 1
    for graph_idx, graph in enumerate(new_graphs):
        if graph_idx in clean_graph_idx:
            # print('Saving', graph_idx, len(graph), len(graph.edges()))
            # relabel nodes in the graph to be consequently distributed among all graphs
            # N = len(graph)
            nodes = graph.nodes
            N = len(nodes)

            mapping = dict(zip(nodes, range(total_nodes, total_nodes + N)))
            inv_mapping = dict(zip(range(total_nodes, total_nodes + N), nodes))
            # graph_formatted = nx.relabel_nodes(graph, mapping)

            # write graph edges
            for edge in sorted(graph.edges):
                u = mapping[edge[0]]
                v = mapping[edge[1]]
                clean_graph_edges_f.write('{}, {}\n'.format(u, v))

            # write node-to-graph correspondance
            for _ in range(total_nodes, total_nodes + N):
                clean_graph_indicator_f.write(f'{current_graph_idx}\n')

            # write graph labels
            clean_graph_labels_f.write('{}\n'.format(graph_labels[graph_idx + 1]))

            # write node labels
            if node_labels_fn:
                for node in range(total_nodes, total_nodes + N):
                    node_label = graph.node_labels[inv_mapping[node]]
                    # node_label = graph_formatted.nodes(data=True)[node]['node_label']
                    clean_node_labels_f.write(f'{node_label}\n')

            # write node attributes
            if node_attributes_fn:
                for node in range(total_nodes, total_nodes + N):
                    node_attribute = graph.node_attributes[inv_mapping[node]]
                    # node_attribute = graph_formatted.nodes(data=True)[node]['node_attribute']
                    clean_node_attributes_f.write(f'{node_attribute}\n')

            # write edge labels
            if edge_labels_fn:
                for edge in sorted(graph.edges):
                    clean_edge_labels_f.write('{}\n'.format(graph.edge_labels[(edge[0], edge[1])]))

            # write edge attributes
            if edge_attributes_fn:
                for edge in sorted(graph.edges):
                    clean_edge_attributes_f.write('{}\n'.format(graph.edge_attributes[(edge[0], edge[1])]))

            # write graph attributes
            if graph_attributes_fn:
                clean_graph_attributes_f.write('{}\n'.format(graph_attributes[graph_idx + 1]))

            current_graph_idx += 1
            total_nodes += N
        else:
            pass
            # print('Missing', current_graph_idx, graph_idx, len(graph), len(graph.edges()))

    clean_graph_edges_f.close()
    clean_graph_indicator_f.close()
    clean_graph_labels_f.close()

    if node_labels_fn:
        clean_node_labels_f.close()
    if edge_labels_fn:
        clean_edge_labels_f.close()
    if node_attributes_fn:
        clean_node_attributes_f.close()
    if edge_attributes_fn:
        clean_edge_attributes_f.close()
    if graph_attributes_fn:
        clean_graph_attributes_f.close()

    return new_graphs


def node_label_match(x1, x2):
    return x1['node_label'] == x2['node_label']


def edge_label_match(x1, x2):
    return x1['edge_label'] == x2['edge_label']


def node_attribute_match(x1, x2):
    return x1['node_attribute'] == x2['node_attribute']


def edge_attribute_match(x1, x2):
    return x1['edge_attribute'] == x2['edge_attribute']


def node_label_attribute_match(x1, x2):
    return node_label_match(x1, x2) and node_attribute_match(x1, x2)


def edge_label_attribute_match(x1, x2):
    return edge_label_match(x1, x2) and edge_attribute_match(x1, x2)


def read_graphs(dataset, dataset_path):
    input_folder = f'{dataset_path}/{dataset}/'
    assert os.path.exists(input_folder), f'Path to dataset should contain folder {dataset}'
    graphs = clean_dataset(dataset, input_folder, '', path_to_orbits, save_new_dataset=False)

    graph_labels = dict()
    with open(input_folder + dataset + '_graph_labels.txt') as f:
        for i, label in enumerate(f):
            graph_labels[i] = label.strip()

    return graphs, graph_labels


def get_filenames(dataset, dataset_path):
    input_folder = f'{dataset_path}/{dataset}/'
    fns = os.listdir(input_folder)
    graphs_fn = indicator_fn = graph_labels_fn = \
        node_labels_fn = edge_labels_fn = \
        edge_attributes_fn = node_attributes_fn = graph_attributes_fn = None
    for fn in fns:
        if 'A.txt' in fn:
            graphs_fn = input_folder + fn
        elif '_graph_indicator.txt' in fn:
            indicator_fn = input_folder + fn
        elif '_graph_labels.txt' in fn:
            graph_labels_fn = input_folder + fn
        elif '_node_labels.txt' in fn:
            node_labels_fn = input_folder + fn
        elif '_edge_labels.txt' in fn:
            edge_labels_fn = input_folder + fn
        elif '_node_attributes.txt' in fn:
            node_attributes_fn = input_folder + fn
        elif '_edge_attributes.txt' in fn:
            edge_attributes_fn = input_folder + fn
        elif '_graph_attributes.txt' in fn:
            graph_attributes_fn = input_folder + fn
    return graphs_fn, indicator_fn, graph_labels_fn, node_labels_fn, edge_labels_fn, \
           edge_attributes_fn, node_attributes_fn, graph_attributes_fn


def get_node_match(node_labels_fn, node_attributes_fn):
    node_match = None
    if node_labels_fn is not None and node_attributes_fn is not None:
        print('Using node labels & attributes')
        node_match = node_label_attribute_match
    elif node_labels_fn is not None:
        print('Using node labels')
        node_match = node_label_match
    elif node_attributes_fn is not None:
        print('Using node attributes')
        node_match = node_attribute_match
    else:
        print('Using no node information')
    return node_match


def get_edge_match(edge_labels_fn, edge_attributes_fn):
    edge_match = None
    if edge_labels_fn is not None and edge_attributes_fn is not None:
        print('Using edge labels & attributes')
        edge_match = edge_label_attribute_match
    elif edge_labels_fn is not None:
        print('Using edge labels')
        edge_match = edge_label_match
    elif edge_attributes_fn is not None:
        print('Using edge attributes')
        edge_match = edge_attribute_match
    else:
        print('Using no edge information')
    return edge_match


def get_graph_labels(dataset, path_to_old_datasets):
    graph_labels_fn = f'{path_to_old_datasets}/{dataset}/{dataset}_graph_labels.txt'
    graph_labels = dict()
    with open(graph_labels_fn) as f:
        for i, label in enumerate(f):
            graph_labels[i + 1] = label.strip()
    return graph_labels


def verify_correctness_of_clean_datasets(dataset, path_to_old_datasets, path_to_new_datasets, path_to_orbits):
    '''

    :param dataset: e.g. MUTAG
    :param path_to_old_datasets: e.g. datasets/
    :param path_to_new_datasets: e.g. clean_datasets/
    :param path_to_orbits: e.g. orbits/no_labels/
    :return:
    '''

    new_graphs, new_graph_labels = read_graphs(dataset, path_to_new_datasets)
    old_graphs, old_graph_labels = read_graphs(dataset, path_to_old_datasets)

    graphs_fn, indicator_fn, graph_labels_fn, node_labels_fn, edge_labels_fn, \
    edge_attributes_fn, node_attributes_fn, graph_attributes_fn = get_filenames(dataset, path_to_old_datasets)

    # these are used to account for node/edge label/attribute during graph isomorphism check
    node_match = get_node_match(node_labels_fn, node_attributes_fn)
    edge_match = get_edge_match(edge_labels_fn, edge_attributes_fn)

    graph_labels = get_graph_labels(dataset, path_to_old_datasets)

    clean_graph_idx = get_clean_graph_idx(graph_labels, path_to_orbits + dataset + '_orbits.txt')

    for idx1, idx2 in enumerate(clean_graph_idx):

        nodes1, edges1 = new_graphs[idx1].nodes, new_graphs[idx1].edges
        nodes2, edges2 = old_graphs[idx2].nodes, old_graphs[idx2].edges

        if len(edges1) > 100 or len(edges2) > 100:
        #         graph is too big for isomorphism test
            if len(nodes1) != len(nodes2) or len(edges1) != len(edges2):
                print('not-isomorphic')
                print('Discrepancy detected. Something wrong. Debug.', idx1, idx2)
                print(edges1)
                print(edges2)
                print(len(nodes1), len(edges1))
                print(len(nodes2), len(edges2))
            else:
                continue

        # G1 is the survived graph from the original dataset, hence it should be identical to G2.
        G1 = new_graphs[idx1].convert_to_nx()
        G2 = old_graphs[idx2].convert_to_nx()

        gm = iso.GraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
        matcher_gen = gm.match()

        try:
            mapping = next(matcher_gen)  # check that graphs G1 and G2 are isomorphic
        except StopIteration:
            print('not-isomorphic')
            print('Discrepancy detected. Something wrong. Debug.', idx1, idx2)
            print(G1.edges())
            print(G2.edges())
            print(len(G1), len(G1.edges()))
            print(len(G2), len(G2.edges()))
            break
        else:
            # check that graph labels are the same
            assert new_graph_labels[idx1] == old_graph_labels[idx2]

            print('Passed', idx1, idx2)
    else:
        print('Clean dataset {} is correct'.format(dataset))


if __name__ == "__main__":

    dataset = 'BZR_MD'

    ds = [
        'FIRSTMM_DB',
          'OHSU',
          'KKI',
          'Peking_1',
          'MUTAG',
          'MSRC_21C',
          'MSRC_9',
          'Cuneiform',
          'SYNTHETIC',
          'COX2_MD',
          'BZR_MD',
          'PTC_MM',
          'PTC_MR',
          'PTC_FM',
          'PTC_FR',
          'DHFR_MD',
          'Synthie',
          'BZR',
          'ER_MD',
          'COX2',
          'MSRC_21',
          'ENZYMES',
          'DHFR',
          'IMDB-BINARY',
          'PROTEINS',
          'DD',
          'IMDB-MULTI',
          'AIDS',
          'REDDIT-BINARY',
          'Letter-high',
          'Letter-low',
          'Letter-med',
          'Fingerprint',
          'COIL-DEL',
          'COIL-RAG',
          'NCI1',
          'NCI109',
          'FRANKENSTEIN',
          'Mutagenicity',
          'REDDIT-MULTI-5K',
          'COLLAB',
          'Tox21_ARE',
          'Tox21_aromatase',
          'Tox21_MMP',
          'Tox21_ER',
          'Tox21_HSE',
          'Tox21_AHR',
          'Tox21_PPAR-gamma',
          'Tox21_AR-LBD',
          'Tox21_p53',
          'Tox21_ER_LBD',
          'Tox21_ATAD5',
          'Tox21_AR',
          'REDDIT-MULTI-12K'
    ]

    ds = [
        'MUTAG',
          'IMDB-MULTI',
    ]
    for dataset in ds:

        try:
            print(dataset)
            output_folder = f'datasets_clean/{dataset}/'
            input_folder = f'datasets_old/{dataset}/'
            path_to_orbits = 'orbits/no_labels/'
            clean_dataset(dataset, input_folder, output_folder, path_to_orbits)

            path_to_old_datasets = 'datasets_old/'
            path_to_new_datasets = 'datasets_clean/'
            path_to_orbits = 'orbits/no_labels/'
            verify_correctness_of_clean_datasets(dataset, path_to_old_datasets, path_to_new_datasets, path_to_orbits)
        except Exception as e:
            print('Failed on {} dataset'.format(e))
            print('Exception:', e)
            raise e
            break

    # dataset = 'BZR'
    # input_folder = f"datasets/{dataset}/"
    # output_folder = f"datasets/data_adj/{dataset}_adj/"
    # convert_dortmund_to_graphml(input_folder, output_folder)

    # dir = "datasets/data_graphml/data_graphml/NCI109/"
    # out = "datasets/NCI109_adj/"
    # os.makedirs(out, exist_ok=True)
    # fns = os.listdir(dir)
    # for fn in fns:
    #     if fn.endswith('.graphml'):
    #         G = formatg(nx.read_graphml(dir + fn))
    #         writeg(G, out + fn.split('.')[0] + '.adj')

    # dataset = 'COLLAB'
    # ds = [
    #     'FIRSTMM_DB',
    #       'OHSU',
    #       'KKI',
    #       'Peking_1',
    #       'MUTAG',
    #       'MSRC_21C',
    #       'MSRC_9',
    #       'Cuneiform',
    #       'SYNTHETIC',
    #       'COX2_MD',
    #       'BZR_MD',
    #       'PTC_MM',
    #       'PTC_MR',
    #       'PTC_FM',
    #       'PTC_FR',
    #       'DHFR_MD',
    #       'Synthie',
    #       'BZR',
    #       'ER_MD',
    #       'COX2',
    #       'MSRC_21',
    #       'ENZYMES',
    #       'DHFR',
    #       'IMDB-BINARY',
    #       'PROTEINS',
    #       'DD',
    #       'IMDB-MULTI',
    #       'AIDS',
    #       'REDDIT-BINARY',
    #       'Letter-high',
    #       'Letter-low',
    #       'Letter-med',
    #       'Fingerprint',
    #       'COIL-DEL',
    #       'COIL-RAG',
    #       'NCI1',
    #       'NCI109',
    #       'FRANKENSTEIN',
    #       'Mutagenicity',
    #       'REDDIT-MULTI-5K',
    #       'COLLAB',
    #       'Tox21_ARE',
    #       'Tox21_aromatase',
    #       'Tox21_MMP',
    #       'Tox21_ER',
    #       'Tox21_HSE',
    #       'Tox21_AHR',
    #       'Tox21_PPAR-gamma',
    #       'Tox21_AR-LBD',
    #       'Tox21_p53',
    #       'Tox21_ER_LBD',
    #       'Tox21_ATAD5',
    #       'Tox21_AR',
    #       'REDDIT-MULTI-12K',
    #       'DBLP_v1'
    # ]
    # # ds = ['MUTAG']
    # for dataset in ds:
    #     # try:
    #         print(dataset)
    #         convert_dortmund_to_graphml(f'datasets/{dataset}/')
    #     # except Exception as e:
    #     #     print('Failed with', dataset, e)
    #
    # from pprint import pprint
    #
    # l = ['FIRSTMM_DB',
    #      'OHSU',
    #      'KKI',
    #      'Peking_1',
    #      'MUTAG',
    #      'MSRC_21C',
    #      'MSRC_9',
    #      'Cuneiform',
    #      'SYNTHETIC',
    #      'COX2_MD',
    #      'BZR_MD',
    #      'PTC_MM',
    #      'PTC_MR',
    #      'PTC_FM',
    #      'PTC_FR',
    #      'DHFR_MD',
    #      'Synthie',
    #      'BZR',
    #      'ER_MD',
    #      'COX2',
    #      'MSRC_21',
    #      'ENZYMES',
    #      'DHFR',
    #      'IMDB-BINARY',
    #      'PROTEINS',
    #      'DD',
    #      'IMDB-MULTI',
    #      'AIDS',
    #      'REDDIT-BINARY',
    #      'Letter-high',
    #      'Letter-low',
    #      'Letter-med',
    #      'Fingerprint',
    #      'COIL-DEL',
    #      'COIL-RAG',
    #      'NCI1',
    #      'NCI109',
    #      'FRANKENSTEIN',
    #      'Mutagenicity',
    #      'REDDIT-MULTI-5K',
    #      'COLLAB',
    #      'Tox21_ARE',
    #      'Tox21_aromatase',
    #      'Tox21_MMP',
    #      'Tox21_ER',
    #      'Tox21_HSE',
    #      'Tox21_AHR',
    #      'Tox21_PPAR-gamma',
    #      'Tox21_AR-LBD',
    #      'Tox21_p53',
    #      'Tox21_ER_LBD',
    #      'Tox21_ATAD5',
    #      'Tox21_AR',
    #      'REDDIT-MULTI-12K',
    #      'DBLP_v1']
    # import os
    #
    # os.listdir()
    #
    # ['AIDS',
    #  'BZR',
    #  'BZR_MD',
    #  'COIL-DEL',
    #  'COIL-RAG',
    #  'COLLAB',
    #  'COX2',
    #  'COX2_MD',
    #  'Cuneiform',
    #  'DBLP_v1',
    #  'DD',
    #  'DHFR',
    #  'DHFR_MD',
    #  'ENZYMES',
    #  'ER_MD',
    #  'Fingerprint',
    #  'FIRSTMM_DB',
    #  'FRANKENSTEIN',
    #  'IMDB-BINARY',
    #  'IMDB-MULTI',
    #  'KKI',
    #  'Letter-high',
    #  'Letter-low',
    #  'Letter-med',
    #  'MSRC_21',
    #  'MSRC_21C',
    #  'MSRC_9',
    #  'MUTAG',
    #  'Mutagenicity',
    #  'NCI1',
    #  'NCI109',
    #  'OHSU',
    #  'Peking_1',
    #  'PROTEINS',
    #  'PTC_FM',
    #  'PTC_FR',
    #  'PTC_MM',
    #  'PTC_MR',
    #  'REDDIT-BINARY',
    #  'REDDIT-MULTI-12K',
    #  'REDDIT-MULTI-5K',
    #  'SYNTHETIC',
    #  'Synthie',
    #  'Tox21_AHR',
    #  'Tox21_AR',
    #  'Tox21_ARE',
    #  'Tox21_AR-LBD',
    #  'Tox21_aromatase',
    #  'Tox21_ATAD5',
    #  'Tox21_ER',
    #  'Tox21_ER_LBD',
    #  'Tox21_HSE',
    #  'Tox21_MMP',
    #  'Tox21_p53',
    #  'Tox21_PPAR-gamma',
    #  'TRIANGLES',
    #  'TWITTER-Real-Graph-Partia']
