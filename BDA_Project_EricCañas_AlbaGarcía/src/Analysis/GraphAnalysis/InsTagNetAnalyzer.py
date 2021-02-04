import networkx as nx
import Database.Constants as FIELDS
import numpy as np
from itertools import combinations
from Analysis.GraphAnalysis.GraphAnalyzer import GraphAnalyzer, WEIGHT, FREQ
import os
from Analysis.Constants import *
from Analysis.Commons import *

SIMILARITY = 'similarity'

class InsTagNetAnalyzer(GraphAnalyzer):
    def __init__(self, collection, results_dir = os.path.join(RESULTS_DIR, GRAPHS_FOLDER_NAME),
                 base_field = FIELDS.COMPLETE_PATH[FIELDS.OBJECT], restriction = {}):
        self.results_dir = os.path.join(results_dir, base_field)
        self.base_field = base_field
        self.restriction = restriction
        super().__init__(graph=self._build_instagnet(collection=collection, field=base_field, restriction=restriction),
                         results_dir=self.results_dir)


    def _build_instagnet(self, collection, field, restriction = {}, minimum_usage_threshold = 2, non_relevant_words = NON_RELEVANT_WORDS):
        restriction = {**restriction, field : {'$exists': True, '$ne': None}}
        unwind_op = get_unwind_operation(collection=collection, field=field, restriction=restriction)
        results = collection.aggregate(unwind_op + [{'$match' : restriction}, {'$project' : {field : True}}])
        data = get_list_for_var_from_mongo_output(mongo_output=results, var=field)
        words, freq = np.unique(np.concatenate(data), return_counts = True)
        valid_words_mask = np.bitwise_and(freq >= minimum_usage_threshold, np.isin(words, list(non_relevant_words)) == False)
        erased_words = set(words[valid_words_mask == False]) # sets have search complexity ('in') of O(log(N)).
        # Frequency information is added before for avoiding future searches to slow down the graph initialization
        node_info = [(word, {FREQ : freq}) for word, freq in zip(words[valid_words_mask], freq[valid_words_mask])]
        instagnet = nx.OrderedGraph()
        instagnet.add_nodes_from(nodes_for_adding=node_info)
        # Erase non valid words from their sub-lists
        if len(erased_words) > 0:
            data = [[word for word in words if word not in erased_words] for words in data]
        # Calculate weight
        for sample in data:
            for (u, v) in combinations(sample, r=2):
                if (u, v) in instagnet.edges:
                    instagnet[u][v][WEIGHT] += 1
                else:
                    instagnet.add_edge(u_of_edge=u, v_of_edge=v, weight=1)
        nodes_freq = nx.get_node_attributes(G=instagnet, name=FREQ)
        # Calculate similarity
        for u, v, data in instagnet.edges(data=True):
            instagnet[u][v][SIMILARITY] = data[WEIGHT]/min(nodes_freq[u], nodes_freq[v])
        return instagnet

    def plot_graph(self, layout=nx.planar_layout, minimum_ploteable_weight_percentage=0.2,
                   file_name='InsTagNet', weight=SIMILARITY):
        field_name = self.base_field.split('.')[-1].replace('_', ' ').title()
        node_color_name = field_name+' Frequency'
        file_name = file_name + ' ({field})'.format(field=field_name)
        super().plot_graph(layout=layout, minimum_ploteable_weight_percentage=minimum_ploteable_weight_percentage,
                           file_name=file_name, print_labels=True, node_color=FREQ, weight=weight,
                           node_color_label=node_color_name, node_size=800*minimum_ploteable_weight_percentage,
                           font_size=5)