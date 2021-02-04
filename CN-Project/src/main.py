import os

from Analyzer import Analyzer

# Constants
results_path = 'Results'
networks_folder = 'Network'
network = 'PairsP.net'

if __name__ == '__main__':

    # Initialize the graph as the network obtained by the current model
    model = Analyzer(root=networks_folder, file=network, calculate_other_relevances=True)
    # Enrich it with the wordnet information
    model.enrich_network()
    # Save all the distribution histograms
    model.plot_distributions(results_path=results_path)
    # Save all the plots maximizing all wordnet measures
    model.save_restricted_plots(path=os.path.join(results_path, 'Subgraphs'))
    # Save all the plots of the RIM analysis
    model.RIM_based_analysis()


