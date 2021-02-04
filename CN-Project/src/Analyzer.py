import networkx as nx
import numpy as np
import warnings
import os
from pprint import pprint
from WordNetVinculation import wordnet_enrichement
from matplotlib import pyplot as plt
from textwrap import wrap
from multiprocessing.pool import ThreadPool
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from time import time
import pickle

# Define the posible measures
MAX, MIN, MEAN, STD, AVERAGE, WEIGHTED_STD = 'max', 'min', 'mean', 'std', 'weighted average', 'weighted std'
# Dictionary with the numpy function which corresponds to each possible descriptor value
func = {MAX: np.max, MIN: np.min, MEAN: np.mean, AVERAGE: np.average, STD:np.std,
        WEIGHTED_STD: lambda values, weights: np.sqrt(np.average((values- np.average(values, weights=weights))**2, weights=weights))
        }

# Define the files for saving some precomputed values
BETWEENNEESS_ARRAY, EIGENVECTOR_ARRAY, PAGERANK_ARRAY, ECCENTRICITY_DICT = 'Betweenneess.npy', 'EigenvectorCentrality.npy', 'PageRank.npy', 'Eccentricities.dict'
# Keys of the edge and node descriptors
DEPTH = 'WordNetDepth'
SYNONYMS, ANTONYMS, PATH_SIMILARITY, WUP_SIMILARITY, EDIT = 'synonyms','antonyms','path_similarity','wup_similarity', 'edit_distance'
WEIGHT = 'weight'
# Define the parameters for the RIM experimentation
RANDOM_WALK_STEPS = (4, 7, 10, 25, 50, 100, 500)
RIM_REPETITIONS = 100

# Information for generating the clustering plots
RELEVANT_INFO = (('Average path similarity', PATH_SIMILARITY), ('Average node depth', 'Wordnet Depth'),
                 ('Average synonym connections', 'Synonyms'), ('Average antonym connections', 'Antonyms'),
                 ('Average edit distances', EDIT),
                 ('Parts of Speech', 'PoS'), ('Relevance', 'Relevance'))
WORD_TAGS = {0:'None', 1: 'N', 2:'V', 3:'AJ', 4:'AD', 5:'P', 6:'PP', 7:'I', 8:'C', 9: 'A'}

# Threads for computing RIM in parallel
THREADS = 8

# Information about the clusterization
PERCENT_OF_RELEVANCE = 0.5
KMEANS_CLUSTERS = (2,4,8)
RIM_REDUCED_COMPONENTS = 'auto'
MIN_VARIANCE_FOR_COMPONENTS = 1.

# Paths
RIM_PATH = os.path.join('Results','RIMAnalysis')
CLU_PATH = os.path.join('Network','pofs.clu')

class Analyzer:
    """
    The purpose of this class is to obtain an analysis of the structural descriptors of the provided network.
    """
    def __init__(self, graph=None, root='networks', file='PairsP.net', calculate_other_relevances = False):
        """
        If a graph in NetworkX format is provided, initialize the Analyzer object with it, else, read
        the graph from the Pajek format file 'file' from the directory 'root/dir/'.
        :param graph: NetworkX graph. Graph for initialize the Analyzer
        :param root: String. Root directory where graphs directories are located.
        :param calculate_other_relevances: boolean. Calculate or not some betweenneess relevances
        :param file: String. Name of the network file to open.
        """
        # If reading the graph from pajek
        if graph is None:
            # Read the graph
            self.graph = nx.read_pajek(os.path.join(root, file))
            # Remove a bug of the pajek when reading
            if '*Arcs' in self.graph:
                self.graph.remove_node('*Arcs'), self.graph.remove_node(':2')

            # Convert in ordered graphs, for ensuring the consistency when indexing
            if type(self.graph) is nx.classes.multidigraph.MultiDiGraph:
                self.graph = nx.OrderedMultiDiGraph(self.graph)
            elif type(self.graph) is nx.classes.graph.Graph:
                self.graph = nx.OrderedGraph(self.graph)
        # Else use the given graph
        else:
            self.graph = graph
        # Save a non directed version of the graph for using it in some algorithms
        self.free_graph = nx.OrderedGraph(self.graph)
        print('Average Clustering of the network: ' + str(np.round(nx.average_clustering(self.free_graph), decimals=4)))

        # Calculate the other relevances and save them into a file for no repeating calculations in the future
        if calculate_other_relevances:
            # Calculate the eccentricity and extract the radius and diameter
            if ECCENTRICITY_DICT not in os.listdir():
                self.eccentricities = {node : nx.eccentricity(self.free_graph, node) for node in self.free_graph.nodes}
                pickle.dump(self.eccentricities, open(ECCENTRICITY_DICT,'wb'))
            else:
                self.eccentricities = pickle.load(open(ECCENTRICITY_DICT,'rb'))
            self.radius = nx.radius(self.free_graph, e=self.eccentricities)
            self.diameter = nx.diameter(self.free_graph, e=self.eccentricities)
            print("Radius: "+str(self.radius))
            print("Diameter: "+str(self.diameter))

            # Calulate the Betweenneess centrality
            if BETWEENNEESS_ARRAY not in os.listdir():
                self.betweenneess = np.array([betweenneess for betweenneess in nx.betweenness_centrality(nx.OrderedDiGraph(self.graph), weight=WEIGHT).values()])
                self.betweenneess = self.betweenneess/np.max(self.betweenneess)
                np.save(BETWEENNEESS_ARRAY, self.betweenneess)
            else:
                self.betweenneess = np.load(BETWEENNEESS_ARRAY)

            # Calculate the Eigenvector Centrality
            if EIGENVECTOR_ARRAY not in os.listdir():
                self.eigenvector_cent = np.array([cent for cent in nx.eigenvector_centrality_numpy(nx.OrderedDiGraph(self.graph), weight=WEIGHT).values()])
                self.eigenvector_cent = self.eigenvector_cent/np.max(self.eigenvector_cent)
                np.save(EIGENVECTOR_ARRAY, self.eigenvector_cent)
            else:
                self.eigenvector_cent = np.load(EIGENVECTOR_ARRAY)

            # Calculate the PageRank score
            if PAGERANK_ARRAY not in os.listdir():
                self.pagerank = np.array([rank for rank in nx.pagerank(nx.OrderedDiGraph(self.graph), weight=WEIGHT).values()])
                self.pagerank = self.pagerank/np.max(self.pagerank)
                np.save(PAGERANK_ARRAY, self.pagerank)
            else:
                self.pagerank = np.load(PAGERANK_ARRAY)




    def get_numeric_descriptors(self):
        """
        Generates a dictionary with the following network descriptors:
        Number of nodes, Number of edges, Max degree, Min degree,
        Average degree, Average clustering coefficient,
        Degree assortativity, Average path length and Diameter.

        :return Dictionary with all the named network numeric descriptors.
        """

        # Define the 'descriptors' dictionary containing all the previously mentioned network descriptors.
        # Each key of the dictionary is a different descriptor obtained by a NetworkX function
        # except 'Max degree', 'Min degree' and 'Average degree', which uses a function from ours
        descriptors = {'Number of nodes': self.graph.number_of_nodes(),
                       'Number of edges': self.graph.number_of_edges(),
                       'Max degree': self.degree_descriptors(descriptor=MAX),
                       'Min degree': self.degree_descriptors(descriptor=MIN),
                       'Average degree': self.degree_descriptors(descriptor=MEAN),
                       'Max weight of an edge': self.degree_descriptors(descriptor=MAX, weight=WEIGHT),
                       'Min weight of an edge': self.degree_descriptors(descriptor=MIN, weight=WEIGHT),
                       'Average edges weight': self.degree_descriptors(descriptor=MEAN, weight=WEIGHT),
                       'Std of weight': self.degree_descriptors(descriptor=STD, weight=WEIGHT),
                       'Average neighbor degree': np.mean(list(nx.average_degree_connectivity(G=self.graph, source='out', target='out').values())),
                       'Average neighbor weight': np.mean(list(nx.average_degree_connectivity(G=self.graph, source='out', target='out', weight=WEIGHT).values())),
                       'Average clustering coefficient': nx.average_clustering(G=nx.Graph(self.graph))
                        }

        return descriptors  # Return the descriptor dictionary

    def get_wordnet_related_descriptors(self):
        """
        Generates a dictionary with the following network descriptors:
        Number of nodes, Number of edges, Max degree, Min degree,
        Average degree, Average clustering coefficient,
        Degree assortativity, Average path length and Diameter.

        :return Dictionary with all the named network numeric descriptors.
        """

        # Define the 'descriptors' dictionary containing all the previously mentioned network descriptors.
        # Each key of the dictionary is a different descriptor obtained by a NetworkX function
        # except 'Max degree', 'Min degree' and 'Average degree', which uses a function from ours
        descriptors = {'Max node depth': self.node_descriptors(descriptor=MAX, data=DEPTH),
                       'Min node depth': self.node_descriptors(descriptor=MIN, data=DEPTH),
                       'Average node depth': self.node_descriptors(descriptor=MEAN, data=DEPTH),
                       'Std of node depth': self.node_descriptors(descriptor=STD, data=DEPTH),
                       'Average path similarity': self.edge_descriptors(descriptor=AVERAGE, data=PATH_SIMILARITY),
                       'Std of path similarity': self.edge_descriptors(descriptor=WEIGHTED_STD, data=PATH_SIMILARITY),
                       'Average wup similarity': self.edge_descriptors(descriptor=AVERAGE, data=WUP_SIMILARITY),
                       'Std of wup similarity': self.edge_descriptors(descriptor=WEIGHTED_STD, data=WUP_SIMILARITY),
                       'Average synonym connections': self.edge_descriptors(descriptor=AVERAGE, data=SYNONYMS),
                       'Std of synonym connections': self.edge_descriptors(descriptor=WEIGHTED_STD, data=SYNONYMS),
                       'Average antonym connections': self.edge_descriptors(descriptor=AVERAGE, data=ANTONYMS),
                       'Std of antonym connections': self.edge_descriptors(descriptor=WEIGHTED_STD, data=ANTONYMS),
                       'Average edit distances': self.edge_descriptors(descriptor=AVERAGE, data=EDIT),
                       'Std of edit distances': self.edge_descriptors(descriptor=WEIGHTED_STD, data=EDIT)
                        }

        return descriptors  # Return the descriptor dictionary

    def tag_nodes_from_clu(self, clu_path = CLU_PATH):
        """
        Read the clu file
        :param clu_path: str. Path of the clu file
        :return: Tags of the clu file
        """
        with open(clu_path, 'r') as f:
            lines = f.readlines()
            # Skip the first comment lines
            offset = 0
            while lines[offset][0] in ['%', '*']: offset+=1
            lines = lines[offset:]
            # Read all the tags
            tags = [int(line) for line in lines]
        # Return them as numpy array
        return np.array(tags, dtype=np.int8)

    def get_network_with_restrictions(self, nodes = 20, include_neighbors=True):
        """
        Generate subgraphs with the 'nodes' nodes each all the wordnet descriptor
        :param nodes: int. Amount of nodes for including in the subgraph.
        :param include_neighbors: boolean. Include in the subgraph the neighbors of the selected nodes
        :return: Dictionary with the descriptor as key and the subgraphs as values.
        """
        subnets = {
            'Maximizing Average path similarity of responses': self.subgraph_maximizing_descriptor(descriptor=AVERAGE, nodes=nodes, data=PATH_SIMILARITY,
                                                                                                   include_neighbors=include_neighbors),
            'Maximizing Average wup similarity of responses': self.subgraph_maximizing_descriptor(descriptor=AVERAGE,  nodes=nodes, data=WUP_SIMILARITY,
                                                                                                  include_neighbors=include_neighbors),
            'Maximizing Average antonyms in responses': self.subgraph_maximizing_descriptor(descriptor=AVERAGE, nodes=nodes, data=ANTONYMS,
                                                                                            include_neighbors=include_neighbors),
            'Maximizing Average synonyms in responses': self.subgraph_maximizing_descriptor(descriptor=AVERAGE, nodes=nodes, data=SYNONYMS,
                                                                                            include_neighbors=include_neighbors),
            'Maximizing Average edit distances in responses': self.subgraph_maximizing_descriptor(descriptor=AVERAGE,
                                                                                            nodes=nodes, data=EDIT,
                                                                                            include_neighbors=include_neighbors)
        }
        # Return de dictionary
        return subnets

    def save_restricted_plots(self, nodes=20, path='SubGraphs'):
        """
        Generate subgraphs with the 'nodes' nodes each all the wordnet descriptor and plot them
        :param nodes:  int. Amount of nodes for including in the subgraph.
        :param path: str. Path were saving the plots
        """
        # Enrich the name of the file with the amount of nodes
        path+='_'+str(nodes)+'_nodes'
        # Save the version with and without neighbors
        for include_neighbors in (True, False):
            save_at = os.path.join(path, 'With'+('out' if not include_neighbors else '')+' Neighbors')
            # For each subgraph using a different descriptor
            for name, subgraph in self.get_network_with_restrictions(nodes=nodes,include_neighbors=include_neighbors).items():
                #Plot the subgraph
                Analyzer(subgraph).plot_graph(save_at=save_at, file_name=name)

    def subgraph_maximizing_descriptor(self, descriptor, nodes, data, weight = WEIGHT, include_neighbors=True):
        """
        Return a subgraph maximizing the given descriptor for a concrete data attribute
        :param descriptor: str. Function to apply 'max', 'min', 'average', 'mean', 'std' or 'weighted std'
        :param nodes: int. Amount of nodes to maximize
        :param data: str. Descriptor to maximize
        :param weight: str. Name of the attribute containing the weight
        :param include_neighbors: boolean. Include the neighbors within the subgraph or not
        :return:
        NetworkX subgraph. The subgraph which maximizes the given descriptor for the concrete data attribute
        """
        # Calculate the descriptor for all nodes
        descriptors = np.array([self.edge_descriptors(descriptor=descriptor, data=data, nodes=node,weight=weight)
                                                 for node in self.graph.nodes])
        # Take the ones which maximize it
        best_nodes = np.array(self.graph.nodes)[np.argpartition(-descriptors, kth=nodes)[:nodes]]
        # Return the subgraph with these nodes, including or not the neighbors
        return self.get_subgraph(nodes=best_nodes, include_neighbors=include_neighbors)

    def get_subgraph(self, nodes, include_neighbors=True):
        """
        Return a subgraph containing the given nodes and its neighbors or not
        :param nodes: List of str. List of nodes to include on the subgraph
        :param include_neighbors: boolean. Include the neighbors on the subgraph or not
        :return:
        NetworkX. The subgraph containing the given nodes and its neighbors or not
        """
        # If include neighbors include them in the node list
        if include_neighbors:
            nodes = np.unique(list(self.graph.edges(nodes)))
        # Return the subgraph containing those nodes
        return nx.subgraph(G=self.graph, nbunch=nodes)

    def degree_descriptors(self, descriptor, weight = None):
        """
        Gets the max, min or average degree of a network.

        :param descriptor: str. 'max', 'min', 'mean', 'average', 'std' or 'weighted std'. Descriptor to extract
        :return Integer value for min or max. Float value for average.
        """

        # Read the degree of the nodes in the graph as a dtype named array (first col: node, second col: degree)
        degrees = np.array(list(self.graph.degree(weight=weight)), dtype=[('node', '<U128'), ('degree', np.int)])

        # Apply the requested function to the 'degree' column of the array
        return func[descriptor](degrees['degree'])

    def edge_descriptors(self, descriptor, data = None, nodes=None, weight=WEIGHT):
        """
        Gets the max, min, mean, average, std or weighted std of the network attribute given by data.

        :param descriptor: 'max', 'min', 'mean', 'average', 'std' or 'weighted std'. Descriptor to extract
        :param data: str. Attribute to extract
        :param nodes: List of str. List of nodes from which extract the descriptors (if None extract from all nodes)
        :param weight: str. Name of the weight attribute
        :return Float value of the descriptor
        """

        # Read the degree of the nodes in the graph as a dtype named array (first col: node, second col: degree)
        edge_descriptors = np.array([data for _,_, data in self.graph.edges(nbunch=nodes, data=data)], dtype=np.float)
        # If the descriptor is weighted extract the weights
        if descriptor == AVERAGE or descriptor == WEIGHTED_STD:
            weights = np.array([weight for _,_, weight in self.graph.edges(nbunch=nodes, data=weight)], dtype=np.float)[edge_descriptors>=0]

        # Erases the invalid data, which uses the descriptor -1
        edge_descriptors = edge_descriptors[edge_descriptors >= 0]
        # Apply the requested function to the edge descriptors
        if len(edge_descriptors) > 0:
            if descriptor == AVERAGE or descriptor == WEIGHTED_STD:
                return func[descriptor](edge_descriptors, weights=weights)
            else:
                return func[descriptor](edge_descriptors)
        # In no edge had information return -1
        else:
            return -1

    def node_descriptors(self, descriptor, data = None):
        """
        Gets the max, min or average degree of a network.

        :param descriptor: 'max', 'min', 'mean', 'average', 'std' or 'weighted std'. Descriptor to extract
        :param data: str. Attribute to extract
        :return Float value of the descriptors.
        """

        # Read the degree of the nodes in the graph as a dtype named array (first col: node, second col: degree)
        nodes = np.array([data for _, data in self.graph.nodes(data=data)], dtype=np.float)
        # Erases the invalid data, which uses the descriptor -1
        nodes = nodes[nodes >= 0]
        # Apply the requested function to the 'degree' column of the array
        if len(nodes)>0:
            return func[descriptor](nodes)
        else:
            return -1


    def get_node_wordnet_descriptors(self):
        """
        Generate a dictionary with all the descriptors related with wordnet or NLP
        :return: Dictionary with all the descriptors
        """

        node_descriptors = {
        'Max path similarity of responses': [self.edge_descriptors(descriptor=MAX, data=PATH_SIMILARITY,nodes=node) for node in self.graph.nodes],
        'Average path similarity of responses': [self.edge_descriptors(descriptor=AVERAGE, data=PATH_SIMILARITY, nodes=node) for node in self.graph.nodes],
        'Std path similarity':[self.edge_descriptors(descriptor=STD, data=PATH_SIMILARITY, nodes=node) for node in self.graph.nodes],

        'Max wup similarity of responses': [self.edge_descriptors(descriptor=MAX, data=PATH_SIMILARITY, nodes=node) for node in self.graph.nodes],
        'Average wup similarity of responses': [self.edge_descriptors(descriptor=AVERAGE, data=PATH_SIMILARITY, nodes=node) for node in self.graph.nodes],
        'Std wup similarity': [self.edge_descriptors(descriptor=STD, data=PATH_SIMILARITY, nodes=node) for node in self.graph.nodes],

        'Amount of different responses which were synonyms': [self.edge_descriptors(descriptor=MEAN, data=SYNONYMS,nodes=node) for node in self.graph.nodes],
        'One Response was a synonim': [self.edge_descriptors(descriptor=MAX, data=SYNONYMS,nodes=node) for node in self.graph.nodes],
        'Average synonim responses': [self.edge_descriptors(descriptor=AVERAGE, data=SYNONYMS, nodes=node) for node in self.graph.nodes],

        'Amount of different responses which were antonyms': [self.edge_descriptors(descriptor=MEAN, data=ANTONYMS, nodes=node) for node in self.graph.nodes],
        'One response was an antonym': [self.edge_descriptors(descriptor=MAX, data=ANTONYMS, nodes=node) for node in self.graph.nodes],
        'Average antonym responses': [self.edge_descriptors(descriptor=AVERAGE, data=ANTONYMS, nodes=node) for node  in self.graph.nodes],

        'Average edit distance': [self.edge_descriptors(descriptor=AVERAGE, data=EDIT, nodes=node) for node
                                          in self.graph.nodes]
                            }
        return node_descriptors


    def enrich_network(self):
        """
        Enrich the model with the wordnet information
        """
        self.graph = wordnet_enrichement(graph=self.graph)

    def plot_distributions(self, results_path):
        """
        Plot the distributions of each wordnet descriptor
        :param results_path: str. Path where saving the distributions
        """
        # Extract the descriptors
        node_descriptors = self.get_node_wordnet_descriptors()
        node_descriptors['Degree'] = None
        node_descriptors['Strenght'] = 'weight'

        # Take each descriptor
        for x_label, node_attribute  in node_descriptors.items():
            # Plot it with multiple or 10 bins
            for bins in (None, 10):
                # Plot them also in log-log scale or not
                for log_log_scale in (True, False):
                    file = x_label+(' (log-log)' if log_log_scale else '')+(' ' if bins is None else '')
                    if not (log_log_scale and bins is None):
                        # Save the plot
                        self.plot_distribution(path=results_path, file=file, x_label=x_label, log_log=log_log_scale, descriptor=node_attribute, bins=bins)


    def RIM_based_analysis(self, weight=WEIGHT, steps = RANDOM_WALK_STEPS, repetitions = RIM_REPETITIONS,
                           n_clusters_to_try = KMEANS_CLUSTERS, reduced_components=RIM_REDUCED_COMPONENTS, path=RIM_PATH):
        """
        Execute the complete RIM analysis saving all the concerning plots
        :param weight: str. Attribute containing the weight
        :param steps: List of int. Steps to try with the RIM algorithm
        :param repetitions: int. Monte Carlo executions of the RIM
        :param n_clusters_to_try: List of int. List with the cluster with which experiment
        :param reduced_components: int or str. Number of components to use for clustering. If 'auto' select
        enough components for maintaining the MIN_VARIANCE_FOR_COMPONENTS information (0.98)
        :param path: str. Path where saving the RIM results
        :return:
        """


        # Get the tags including the Parts of Speech
        tags = self.tag_nodes_from_clu()[:len(self.graph.nodes)]
        # Repeat the process for each S parameter
        for s in steps:
            steps_path = os.path.join(path, str(s)+' Steps')
            # Execute the RIM algorithm repetition times for N steps
            word_association_space, nodes = self.RIM(steps=s, weight=weight,repetitions = repetitions)
            # Extract the relevance index of each node
            node_relevance = np.sum(word_association_space, axis=0)
            node_relevance = node_relevance/np.max(node_relevance)
            # Extract the uncertainty index of each node
            node_uncertainty = np.std(word_association_space, axis=0)
            node_uncertainty = node_uncertainty/np.max(node_uncertainty)
            #Reduce the componen through PCA for better clustering
            if reduced_components == 'auto':
                pca_components = np.sum(np.cumsum(PCA().fit(X=word_association_space).explained_variance_ratio_) < MIN_VARIANCE_FOR_COMPONENTS)
            else:
                pca_components = reduced_components
            word_association_space = PCA(n_components=pca_components).fit_transform(X=word_association_space)

            # Extract the subgraphs maximizing relevance and uncertainty measures
            for measure in ('Normalized Betweenness', 'Normalized Eigenvector Centrality', 'Relevance', 'PageRank', 'Uncertainty'):
                if measure == 'Relevance': measure_array = node_relevance
                elif measure == 'Uncertainty': measure_array = node_uncertainty
                elif measure == 'Normalized Betweenness': measure_array = self.betweenneess
                elif measure == 'Normalized Eigenvector Centrality': measure_array = self.eigenvector_cent
                elif measure == 'PageRank': measure_array = self.pagerank
                # Generate the with and without neighbors
                for include_neighbors in (False, True):
                    graphs_path = os.path.join(steps_path, measure+' Graphs', 'With'+('out' if not include_neighbors else '')+' Neighbors')
                    # With N nodes or nodes above a certain threshold
                    for plot in ('%', 'n'):
                        iterator = (.90, .80, .70, .60, .50, .40) if plot == '%' else (2,4,8,16,32)
                        for relevance in iterator:
                            # Extract the relevant nodes
                            relevance_mask = measure_array > relevance if plot == '%' else np.argpartition(-measure_array, kth=relevance)[:relevance]
                            # Stablish the file name
                            file_name = 'Nodes with more than ' + str(int(relevance * 100)) + '% of '+measure if plot == '%' else \
                                ('Most Relevant Hubs' if measure == 'Relevance' else ('Nodes with High Uncertainty' if measure == 'Uncertainty' else 'Most central nodes'))
                            # Generate the subgraph
                            subgraph = self.get_subgraph(nodes=nodes[relevance_mask], include_neighbors=include_neighbors)
                            # Reorder the nodes for indexing correctly
                            relevants_relevance = [measure_array[nodes==node][0] for node in subgraph.nodes()]
                            # Calculate its eccentricities for representing it as the size of the node
                            eccentrities = np.array([(((self.diameter -self.eccentricities[node])/(self.diameter-self.radius))*0.9+0.1)
                                                            for node in subgraph.nodes()])
                            # Plot them
                            Analyzer(subgraph).plot_graph(save_at=graphs_path, node_color=relevants_relevance,
                                                          node_color_label = 'Node '+measure,
                                                          file_name=file_name, node_size=eccentrities*100)

            # For each clustering or community detection algorithm detect the communities or clusters
            for clustering in ('Kmeans', 'Async Fluid Modularity', 'Greedy Modularity'):
                # Take a number of clusters
                for n_clusters in n_clusters_to_try:
                    if clustering == 'Greedy Modularity':
                        # If greedy modularity only execute one time
                        if n_clusters > n_clusters_to_try[0]:
                            continue
                        else:
                            n_clusters = 'N'
                    # Select the path where saving and create it
                    current_path = os.path.join(steps_path, clustering, str(n_clusters)+' Clusters')
                    if not os.path.isdir(current_path):
                        os.makedirs(current_path)
                    # If is Kmeans extract the clusters
                    if clustering == 'Kmeans':
                        clusters = KMeans(n_clusters=n_clusters).fit(X=word_association_space).predict(
                            word_association_space)
                    # Else extract the communities
                    else:
                        if clustering == 'Async Fluid Modularity':
                            cluster_sets = nx.algorithms.community.asyn_fluid.asyn_fluidc(
                                G=nx.convert_node_labels_to_integers(G=nx.OrderedGraph(self.graph)), k=n_clusters)
                        elif clustering == 'Greedy Modularity':
                            cluster_sets = nx.algorithms.community.modularity_max.greedy_modularity_communities(
                                G=nx.convert_node_labels_to_integers(G=nx.OrderedGraph(self.graph)), weight=weight)
                        # Transform the output for corresponding with the Kmeans output format
                        clusters = np.empty(len(self.graph), dtype=np.uint8)
                        for i, cluster in enumerate(cluster_sets):
                            clusters[list(cluster)] = i

                    # Write the legend for indicating the intra cluster and inter cluster information
                    legend = []
                    # For each cluster
                    for cluster in np.unique(clusters):
                        legend.append([])
                        # Take the nodes within the cluster
                        clustered_nodes_mask = clusters == cluster
                        clustered_nodes_id = nodes[clustered_nodes_mask]
                        # Generate a subgraph with them
                        subgraph = Analyzer(nx.subgraph(self.graph, nbunch=clustered_nodes_id))
                        # Take the edges with goes from the cluster to outside
                        non_clustered_edges = list(nx.edge_boundary(G=self.graph, nbunch1=clustered_nodes_id))
                        # Get all the edge descriptors
                        edge_descriptors = subgraph.get_numeric_descriptors()
                        wordnet_descriptors = subgraph.get_wordnet_related_descriptors()
                        # For each analysis selected
                        for name, info in RELEVANT_INFO:
                            # Write the legend with its analysis for the current cluster
                            descriptor = 'C'+str(cluster)+'. '+info+': '
                            # If is the Part of Speech transform it to a readable format
                            if info == 'PoS':
                                descriptor += get_mean_tags_of_graph_as_str(subgraph_tags=tags[clustered_nodes_mask])
                            # If is the relevance, calculate also the amount of nodes with high relevance
                            elif info == 'Relevance':
                                descriptor += str(np.round(np.mean(node_relevance[clustered_nodes_mask]), decimals=2)) +\
                                              '±'+str(np.round(np.std(node_relevance[clustered_nodes_mask]), decimals=2))
                                relevant_nodes = nodes[np.bitwise_and(clustered_nodes_mask, node_relevance>PERCENT_OF_RELEVANCE)]

                                descriptor += ' (Nodes with relevance > '+str(int(PERCENT_OF_RELEVANCE*100))+'%: '+\
                                                str(np.round(100*len(relevant_nodes)/len(clustered_nodes_id), decimals=2))+'% ('+str(len(relevant_nodes))+' Nodes))'
                            # If is any other calculate also the standard deviation
                            else:
                                descriptor += str(np.round(wordnet_descriptors[name], decimals=3))
                                std_name = name.replace('Average','Std of')
                                if std_name in wordnet_descriptors:
                                    descriptor += '±'+str(np.round(wordnet_descriptors[std_name], decimals=2))
                            # If is Path similarity of Edit distance add also the inter cluster information
                            if info == PATH_SIMILARITY or info == EDIT:
                                outside_connections_subgraph = Analyzer(nx.DiGraph(self.graph).edge_subgraph(edges=non_clustered_edges))
                                avg = outside_connections_subgraph.edge_descriptors(descriptor=AVERAGE, data=info)
                                std = outside_connections_subgraph.edge_descriptors(descriptor=WEIGHTED_STD, data=info)
                                descriptor+=' (With outside: '+str(np.round(avg, decimals=3))+'±'+\
                                            str(np.round(std, decimals=2))+')'
                            # Append this cluster info to its legend
                            legend[-1].append(descriptor)

                        # Save the complete info of each cluster in a txt
                        with open(os.path.join(current_path, 'Cluster '+str(cluster)+'.txt'), 'w') as f:
                            pprint("-"*20+' NUMERIC DESCRIPTORS '+"-"*20, f)
                            pprint(edge_descriptors, f)
                            pprint("-"*(20+20+len(' NUMERIC DESCRIPTORS ')),f)
                            pprint("-" * 20 + ' WORDNET DESCRIPTORS ' + "-" * 20,f)
                            pprint(wordnet_descriptors, f)
                            pprint("-" * (20 + 20 + len(' WORDNET DESCRIPTORS ')),f)

                    # For ploting it transfor to two dimensions, through PCA or TSNE
                    pca_space = PCA(n_components=2).fit_transform(X=word_association_space)
                    tsne_space = TSNE(n_components=2).fit_transform(X=word_association_space)
                    # For each legend plot an independent figure
                    for leg, (name, info) in zip(np.array(legend).T, RELEVANT_INFO):
                        # With PCA
                        save_plot(path=os.path.join(current_path, 'PCA'),
                                  title='PCA - RIM '+str(s)+' Steps - ' + (str(pca_components) + ' PCA Dims ('+str(int(MIN_VARIANCE_FOR_COMPONENTS*100))+'% of info) ' if clustering == 'Kmeans' else '')+ clustering +' Clustering\n'+ name,
                                  X=pca_space[:,0], Y=pca_space[:,1],
                                  colors=clusters, legend = leg)
                        # With TSNE
                        save_plot(path=os.path.join(current_path, 'TSNE'), X=tsne_space[:,0], Y=tsne_space[:,1],
                                  title='TSNE -  RIM '+str(s)+' Steps - ' + (str(pca_components) + ' PCA Dims ('+str(int(MIN_VARIANCE_FOR_COMPONENTS*100))+'% of info) ' if clustering == 'Kmeans' else '')+ clustering +' Clustering\n'+ name,
                                  colors=clusters, legend = leg)





    def RIM(self, steps = RANDOM_WALK_STEPS, weight = WEIGHT, repetitions = RIM_REPETITIONS, threads = THREADS):
        """
        Executes the RIM algorithm as proposed by Javier Borge-Holthoefer and Alex Arenas in
        "Navigating word association norms to extract semantic information" (2009)

        :param steps: int. Amount of random walk steps to perform
        :param weight: str. Label of the attribute to use as weights
        :param repetitions: int. Monte Carlo simulations for averaging the estimation
        :param threads: int. Number of threads for computing it in parallel
        :return: Tuple with a float matrix representing the word association space and
        a str list, representing the how the words corresponds to that indexes
        """


        # To non directed ordered graph
        graph = nx.OrderedGraph(self.graph)
        # To numeric ids, for fast indexing of the vector matrices
        numeric_id_graph = nx.convert_node_labels_to_integers(G=graph)
        # Save the nodes correspondences (idx = numerical_id, value = node word)
        nodes = np.array(graph.nodes())
        # Build the initial matrix and the output vector matrix
        ortogonal_matrix = np.identity(len(nodes), dtype=np.float)
        # Compute it step by step, for avoiding memory overflows and take profit of CPU caches
        word_association_space = np.zeros_like(ortogonal_matrix)
        print("Computing RIM")
        # Compute the RIM n times
        for rep in range(repetitions):
            # Compute in parallel a random walk from each node
            paths = []
            with ThreadPool(processes=None) as pool:
                for i in range((len(nodes)//threads)+1):
                    start_time = time()
                    # Create thread jobs
                    async_results = [pool.apply_async(random_walk, (numeric_id_graph, node, steps, weight)) for node in
                                         range(min(i*threads, len(nodes)), min((i+1)*threads, len(nodes)))]
                    # Execute them
                    paths.extend([async_result.get() for async_result in async_results])
                    print("Rep " + str(rep) + ": Computing from " + str(min((i) * threads, len(nodes))) + " to " + str(
                        min((i + 1) * threads, len(nodes)))+". Done in: "+str(np.round(time()-start_time, decimals=4))+" seconds")

                pool.close()
                pool.terminate()
                pool.join()

            print("All paths computed")
            for i, path in enumerate(paths):
                word_association_space[i] += np.sum(ortogonal_matrix[path], axis=0)
            print("Association Space Built")
        # Return the matrix of vectors and the correspondences between nodes and indexes
        return word_association_space/repetitions, nodes




    def plot_distribution(self, bins=10, descriptor = None, log_log=False, file=None, path=None, plot_histogram=None,
                          calculate_gamma=True, x_label='Degree', erase_non_valids = True):
        """
        Plots (or saves) the Probability Degree Distribution (PDF) and the Complementary Cumulative Degree Distribution
        (CCDF) histograms in linear or log-log scale of the graph.

        :param bins: int. Number of bins to use in the histogram.
        :param log_log: bool. Apply log_log scale?
        :param file: String. Name of the file (If given, add it to the title of the plot).
        :param path: String. Location to store the plot generated by this function.
        :param plot_histogram: Histogram already computed to be plotted (for the theoretical results).
        """
        non_computable_nodes = 0
        path = os.path.join(path, x_label)
        if not os.path.isdir(path):
            os.makedirs(path)

        # Get the descriptor of all the nodes in the graph
        if descriptor is None or type(descriptor) == str:
            descriptor = np.array([d for _, d in self.graph.degree(weight=descriptor)])
        else:
            descriptor = np.array(descriptor)
            if erase_non_valids:
                non_computable_nodes = np.sum(descriptor<0)/len(descriptor)
                descriptor = descriptor[descriptor>=0]
        # Calculate the gamma exponent of the degree distribution (using the MLE formula)
        if calculate_gamma:
            gamma = 1 + len(descriptor)*((np.sum(np.log(descriptor/(np.min(descriptor)-0.5))))**-1)
            if np.isnan(gamma):
                calculate_gamma = False
        # Divide the bins differently depending on the scale used
        if log_log:  # 10 bins for log-log scale
            try:
                bins = np.logspace(np.log10(np.min(descriptor)), np.log10(np.max(descriptor)), min(bins, np.max(descriptor)))
            except:
                return None
        elif bins is None:  # Same number of bins as the maximum number of descriptor for linear scale
            if np.all(np.unique(descriptor) == [0., 1.]):
                bins = np.array([0, 0.5, 1])
            else:
                bins = np.arange(np.min(descriptor), np.max(descriptor))
                if len(bins)<=1:
                    return None
        else:
            if np.all(np.unique(descriptor) == [0.,1.]):
                return None
            else:
                bins = np.linspace(np.min(descriptor), np.max(descriptor),num=bins)

        # Calculate the histogram with the given bins
        histogram, bins = np.histogram(descriptor, bins=bins)
        # Transform the histogram into a probability distribution (so that it adds 1)

        # In case the 'theoretical' histogram is also provided (excluding Dirac delta function case)
        if plot_histogram is not None and np.min(descriptor) != np.max(descriptor):
            # Arrange it so that it contains the same bins that the equivalent histogram obtained during the
            # experimentation and therefore, both can be comparable
            histogram = [np.sum(plot_histogram[np.bitwise_and(plot_histogram[:, 0] > bins[bin-1],
                        plot_histogram[:, 0] <= bins[bin]), 1]) for bin in range(1, len(bins))]
        # With this, 'histogram' represents now the PDF plot
        histogram = histogram/np.sum(histogram)
        try:
            # Plot the PDF and the CCDF
            # CCDF is the complementary of CDD (1-CDD). CDD is the cumulative sum of the PDF
            for title in ('Probability Distribution', 'Complementary Cumulative Distribution'):
                if calculate_gamma:
                    title += ". Gamma - " + str(np.round(gamma, decimals=3))  # Add the gamma value to the title

                # If we compute the CCDF, we have to modify the histogram to get the cumulative sum
                if 'Cumulative' in title:
                    histogram = 1-np.cumsum(histogram)

                # Choose the scale for the axes
                # Set the width of the bins equal for all of them
                width = [bins[i]-bins[i-1] for i in range(1, len(bins))]
                plt.bar(bins[1:], histogram, width=width)
                if log_log:  # Log-log scale
                    # Set the scale of x and y axes to log
                    plt.xscale('log')
                    plt.yscale('log')

                if file is not None:  # Add the name of the network to the plot title if given
                    title += ' (' + file + ')'
                save_path = os.path.join(path, title + '.png')
                title = "\n".join(wrap(title, 60))
                if non_computable_nodes > 0:
                    title += '\nNon computable nodes: '+str(np.round(non_computable_nodes*100,decimals=2))+'%'

                # Plot titles and name of the axis
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel('Probability')
                if np.all(np.unique(descriptor) == [0.,1.]):
                    x_ticks = ('No', 'Yes') if 'Cumulative' not in title else ('1-No', '1-(No+Yes)')
                    plt.xticks(bins[1:], x_ticks)
                # Save plot in path
                plt.savefig(save_path, dpi=440)

                plt.close()

        except ValueError:
            # Catch the same-degree network exception (it would generate an histogram with only one bin)
            warnings.warn("Nonsensical distribution: all the nodes with k>0 have the same degree.")
            plt.close()

    def plot_graph(self, layout=nx.planar_layout, alg_name='', save_at=None, file_name='Graph', print_labels=True, node_color='pink',
                   node_color_label = 'Node Relevance', node_size = None):
        """
        Plots (or saves) the network stored in self.graph.

        :param layout: Layout of the nodes of the represented network.
        :param alg_name: String. Name of the model used to generate the network.
        :param save_at: String. Location to store the plot generated by this function.
        """
        if len(self.graph.edges) == 0:
            return None
        elif layout == nx.planar_layout and not nx.check_planarity(self.graph)[0]:
            layout = nx.spring_layout

        # Define the plot of the network and its characteristics (pink nodes with 95% of opacity and without labels)
        nodes_degree = np.array([self.graph.degree[node] for node in self.graph.nodes])
        if node_size is None:
            node_size = (nodes_degree/np.max(nodes_degree))*80+20
        edge_intensity = np.array([weight for u, v, weight in self.graph.edges(data=WEIGHT)])
        edge_color = [plt.cm.Blues(color) for color in edge_intensity/np.max(edge_intensity)]
        if type(node_color) is not str:
            node_color = [plt.cm.Reds(color) for color in node_color]
    	
        nx.draw_networkx(self.graph, pos=layout(G=self.graph), with_labels=print_labels, node_color=node_color, alpha=0.8, node_size=node_size, width=1.,
                         edge_color=edge_color, font_size=3)
        # Remove the black borders of the plot of the network
        [plt.axes().spines[side].set_color('white') for side in plt.axes().spines.keys()]
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin = np.min(edge_intensity), vmax=np.max(edge_intensity)))
        sm._A = []
        plt.colorbar(sm).set_label ('Edge Weight')
        if type(node_color) is not str:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
            sm._A = []
            plt.colorbar(sm).set_label(node_color_label)

        plt.title(file_name+'. ('+str(len(self.graph))+' Nodes)')

        plt.tight_layout()  # Avoid the lengthy titles to be cut

        # Show or save the plot
        if save_at is None:  # Show the plot if the title of the file is not given
            plt.show()
        else:
            if not os.path.isdir(save_at):
                os.makedirs(save_at)
            plt.savefig(os.path.join(save_at, file_name.replace('%','')+'.png'), dpi=440)

        plt.close()

def random_walk(graph, node, steps=RANDOM_WALK_STEPS, weight=WEIGHT):
    """
    Makes n steps of a random walk within the graph, starting by node start_node
    :param graph: networkx graph. Graph were performing the random walk
    :param node: Node where start the random walking
    :param steps: int. amount of steps to perform
    :param weight: str. Label of the attribute to use as weight
    :return: List of ids. List of the nodes where the random walker passed through
    """
    # Initialize the path
    path = [node]
    for _ in range(steps):
        # Take the connected nodes and its weights
        _, nodes, weights = np.array(list(graph.edges(nbunch=node, data = weight, default=1)), dtype=np.float).T
        # Select one of them at random (having each one a probability equal to its weight)
        probs = np.cumsum(weights)
        node = int(nodes[np.argmax(probs >= np.random.uniform(low=0., high=probs[-1]))])
        # Extend the path with the new node
        path.append(node)
    return path

def save_plot(path, X,Y, title, colors=None, legend=None):
    """
    Saves a plot at the given path
    :param path: str. Path where saving the plot
    :param X: Float array. X values
    :param Y: Float array. Y Values
    :param colors: List of ids. Colors
    """
    # Plot each cluster with it color
    for color in np.unique(colors):
        color_mask = colors == color
        x, y = X[color_mask], Y[color_mask]
        plt.plot(x,y,'o')
    # Generate the legend if not given
    if legend is None:
        legend = ['Cluster '+str(color) for color in np.unique(colors)]
    # Put the legend at the correct place
    plt.legend(legend, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
    # Plot the title
    plt.title(title)
    # Erase the ticks
    plt.tick_params(which='both', bottom=False, top=False, right=False, left=False,
                    labelbottom=False, labelleft=False)
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(path, title.replace('\n','').replace('%','')), dpi=660)
    plt.close()



def get_mean_tags_of_graph_as_str(subgraph_tags):
    """
    Get as string the information referent to the part of speech given tags
    :param subgraph_tags: List of str. List including the part of speech of each node within a cluster
    :return: str. information referent to the part of speech given tags
    """
    # Count the occurrences of each one
    type, counts = np.unique(subgraph_tags, return_counts=True)
    # Normalize it
    counts = counts/len(subgraph_tags)
    # Built the string
    txt = ', '.join([WORD_TAGS[tag]+': '+str(np.round(prob, decimals=2)) for tag, prob in zip(type, counts)])
    # Return it
    return txt