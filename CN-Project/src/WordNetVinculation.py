import networkx as nx
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

def wordnet_enrichement(graph):
	"""
	Returns a neighborhood dict with all wordnet related info for each edge
	:param graph: Networkx graph with semantic labels for each node
	:return:
	"""
	# As the synset is not defined in the graph, all possibilities for the given word will be taken into account.
	# For similarities, max similarities among the combinatory of its synsets will be taken. For depths, the minimum one.

	# Set the default values on all edges and nodes:
	for label in ('synonyms','antonyms','path_similarity','wup_similarity', 'edit_distance'):
		nx.set_edge_attributes(G=graph, values='-1', name=label)

	nx.set_node_attributes(G=graph, values='-1', name='WordNetDepth')
	for u in graph.nodes():
		# ---------------------------------------- U INFORMATION -------------------------------------
		u_all_synsets = wordnet.synsets(u)
		# If u is not a word in WordNet fill all its output edges with default values
		if len(u_all_synsets) == 0: continue
		# Take all synonyms and all antonyms of these synonyms
		u_synonyms = [synonym.name() for synset in u_all_synsets for synonym in synset.lemmas()]
		u_antonyms = [antonym.name() for synset in u_all_synsets for synonym in synset.lemmas() for antonym in
					  synonym.antonyms()]

		# Set as node featurethe minimum depth of u on the WordNet hierarchy
		nx.set_node_attributes(G=graph,
							   values={u: {'WordNetDepth': str(min([u_synset.min_depth() for u_synset in u_all_synsets]))}})

		for _, v in graph.edges(u):
			# ---------------------------------------- V INFORMATION -------------------------------------
			v_all_synsets = wordnet.synsets(v)
			if len(v_all_synsets) == 0: continue
			# Take all synonyms and all antonyms of these synonyms
			v_synonyms = [synonym.name() for synset in wordnet.synsets(v) for synonym in synset.lemmas()]
			v_antonyms = [antonym.name() for synset in v_all_synsets for synonym in synset.lemmas() for antonym in synonym.antonyms()]

			# Calculate if u and v have shares synonyms of if they are antonyms
			are_synonyms = len([synonym for synonym in u_synonyms if synonym in v_synonyms]) > 0
			are_antonyms = len([antonym for antonym in u_antonyms if antonym in v_synonyms]) > 0 or \
						   len([antonym for antonym in v_antonyms if antonym in u_synonyms]) > 0

			# Calculate the maximum path similarities among all possibles synsets that u and v can have
			max_path_similarity = max([similarity for u_synset in u_all_synsets for v_synset in v_all_synsets
									   for similarity in [u_synset.path_similarity(v_synset)] if similarity is not None]+[0])
			max_wup_similarity = max([similarity for u_synset in u_all_synsets for v_synset in v_all_synsets
									   for similarity in [u_synset.wup_similarity(v_synset)] if similarity is not None]+[0])


			# Set as node featurethe minimum depth of v on the WordNet hierarchy
			nx.set_node_attributes(G=graph,
								   values={v: {'WordNetDepth': str(min([v_synset.min_depth() for v_synset in v_all_synsets]))}})
			# Fill the edge
			fill_edge(graph=graph, u=u, v=v, are_synonyms=are_synonyms, are_antonyms=are_antonyms,
					  path_similarity=max_path_similarity, wup_similarity=max_wup_similarity)

	return graph



def fill_edge(graph, u, v, are_synonyms=None, are_antonyms=None, path_similarity=None, wup_similarity=None):
	"""
	Add the wordnet information to the given edge
	:param graph: NetworkX Graph. Graph where adding the information
	:param u: str. Node origin
	:param v: str. Node destination
	:param are_synonyms: int. 1 if u and v are synonyms 0 if not -1 if is not computable
	:param are_antonyms: int. 1 if u and v are antonyms 0 if not -1 if is not computable
	:param path_similarity: float. WordNet path similarity between u and v
	:param wup_similarity: float. WordNet wup similarity between u and v
	"""
	# Generate the dictionary
	attributes = {(u,v,0):
					  {'synonyms':str(int(are_synonyms)),
					   'antonyms':str(int(are_antonyms)),
					   'path_similarity':str(round(path_similarity, ndigits=3)),
					   'wup_similarity':str(round(wup_similarity, ndigits=3)),
					   'edit_distance':str(round(nltk.edit_distance(u,v), ndigits=3))}}
	# Add the information to the graph
	nx.set_edge_attributes(graph, attributes)