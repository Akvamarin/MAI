import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from RPKM import RPKM, compute_error

batch_size = [100, 500, 1000]
partitions = np.arange(1, 7)

# Artificial datasets
repetitions = 5
clusters = [3, 9]
dimensions = [2, 4, 8]
instances = [100, 1000, 10000, 100000, 1000000]

# Real dataset
repetitions_ = 5
clusters_ = [3, 9]
dimensions_ = [2, 4, 8]
instances_ = [100, 1000, 10000, 100000, 1000000]


# PRINT THE RESULTS (STORED IN A DICTIONARY)

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


# PLOTTING FUNCTIONS

def plot_grid(X, results, index, path, i, distances, error):
    for cluster in range(results.cluster_centers_.shape[0]):
        cluster_mask = results.labels_ == cluster
        plt.plot(X[cluster_mask, 0], X[cluster_mask, 1], '.', color='C' + str(cluster))

    for cluster in range(results.cluster_centers_.shape[0]):
        plt.plot(results.cluster_centers_[cluster, 0], results.cluster_centers_[cluster, 1], 'o', color='k')

    if index is not None:
        lower_bounds = np.amin(X, axis=0)
        upper_bounds = np.amax(X, axis=0)

        major_ticks_x = np.arange(lower_bounds[0], upper_bounds[0] + 0.1,
                                  (upper_bounds[0] - lower_bounds[0]) / (2 ** index))
        major_ticks_y = np.arange(lower_bounds[1], upper_bounds[1] + 0.1,
                                  (upper_bounds[1] - lower_bounds[1]) / (2 ** index))
        minor_ticks_x = np.arange(lower_bounds[0], upper_bounds[0], 1)
        minor_ticks_y = np.arange(lower_bounds[1], upper_bounds[1], 1)

        ax = plt.gca()
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)

        ax.grid(which='major', color='#999999', alpha=1)
        ax.grid(which='minor', alpha=0.5, linestyle=':')

        plt.savefig(os.path.join(path, str(i) + '_distances=' + str(results.dist_comp) + '_error=' +
                                 str(round(results.error, 2)) + '.png'))

    else:
        plt.savefig(os.path.join(path, str(i) + '_distances=' + str(distances) + '_error=' +
                                 str(round(error, 2)) + '.png'))


def plot_general_1(results, path):

    fig, ax = plt.subplots(len(dimensions), len(clusters), sharex='col', sharey='row')
    legend = ['KM++', 'MB 100', 'MB 500', 'MB 1000', 'RPKM 1', 'RPKM 2', 'RPKM 3', 'RPKM 4', 'RPKM 5', 'RPKM 6']

    for i, d in enumerate(dimensions):
        for j, K in enumerate(clusters):
            for experiment in legend:
                results_ = [results[K][d][n][experiment]['distances'] for n in instances[1:]]
                ax[i, j].plot(instances[1:], np.mean(results_, axis=1), marker='.')

                ax[i, j].set_xscale('log')
                ax[i, j].set_yscale('log')

    [ax[0, j].title.set_text('K:' + str(clusters[j])) for j in range(len(clusters))]
    [(ax[i, -1].yaxis.set_label_position("right"), ax[i, -1].set_ylabel('d:' + str(dimensions[i])))
     for i in range(len(dimensions))]

    fig.text(0.5, 0, 'Dataset size', ha='center')
    fig.text(0, 0.5, 'Distance computations', va='center', rotation='vertical')

    ax[0, 0].legend(legend, bbox_to_anchor=(0, 1.22, 2.1, 0.102), bbox_transform=ax[0, 0].transAxes,
                    loc="lower left", mode="expand", ncol=5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'plot1.png'))


def plot_general_2(results, path, data):

    fig, ax = plt.subplots(len(dimensions), len(clusters), sharex='col', sharey='row')

    if data == 'artificial':
        legend = [str(instance) for instance in instances]
    if data == 'real':
        legend = [str(instance) for instance in instances_]

    for i, d in enumerate(dimensions):
        for j, K in enumerate(clusters):
            for experiment in legend:
                results_ = [np.array(results[K][d][int(experiment)]['RPKM ' + str(i)]['partitions'])/int(experiment)
                            for i in partitions]
                ax[i, j].plot(partitions, np.mean(results_, axis=1), marker='.')

    [ax[0, j].title.set_text('K:' + str(clusters[j])) for j in range(len(clusters))]
    [(ax[i, -1].yaxis.set_label_position("right"), ax[i, -1].set_ylabel('d:' + str(dimensions[i])))
     for i in range(len(dimensions))]

    fig.text(0.5, 0, 'Partition index', ha='center')
    fig.text(0, 0.5, 'Number of representatives/Data size', va='center', rotation='vertical')

    ax[0, 0].legend(legend, bbox_to_anchor=(0, 1.22, 2.1, 0.102), bbox_transform=ax[0, 0].transAxes,
                    loc="lower left", mode="expand", ncol=5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'plot2.png'))


def compute_std_error(rpkm_error, km_error):

    return (km_error - rpkm_error)/km_error


def plot_general_3(results, path, data):

    fig, ax = plt.subplots(len(dimensions), len(clusters), sharex='col', sharey='row')

    if data == 'artificial':
        legend = [str(instance) for instance in instances]
    if data == 'real':
        legend = [str(instance) for instance in instances_]

    for i, d in enumerate(dimensions):
        for j, K in enumerate(clusters):
            for experiment in legend:
                if data == 'artificial':
                    X, y = make_blobs(n_samples=int(experiment), centers=K, n_features=d, random_state=0)
                if data == 'real':
                    X = np.load(os.path.join('data', 'real_data.npy'))
                    X = X[..., np.random.choice(range(X.shape[1]), size=d, replace=False)]
                    X = X[np.random.choice(range(X.shape[0]), size=int(experiment), replace=False)]

                km = np.array([KMeans(n_clusters=K, n_init=1, algorithm='full',
                               init=np.array(results[K][d][int(experiment)]['RPKM ' + str(p)]['clusters'][0])).fit(X=X)
                               for p in partitions])
                results_ = [compute_std_error(results[K][d][int(experiment)]['RPKM ' + str(p)]['error'][0],
                            compute_error(X, km[n].labels_, km[n].cluster_centers_)) for n, p in enumerate(partitions)]

                ax[i, j].plot(partitions, results_, -0.5, 0, marker='.')

    [ax[0, j].title.set_text('K:' + str(clusters[j])) for j in range(len(clusters))]
    [(ax[i, -1].yaxis.set_label_position("right"), ax[i, -1].set_ylabel('d:' + str(dimensions[i])))
     for i in range(len(dimensions))]

    fig.text(0.5, 0, 'Partition index', ha='center')
    fig.text(0, 0.5, 'Standard error', va='center', rotation='vertical')

    ax[0, 0].legend(legend, bbox_to_anchor=(0, 1.22, 2.1, 0.102), bbox_transform=ax[0, 0].transAxes,
                    loc="lower left", mode="expand", ncol=5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'plot3.png'))


def plot_general_4(results, path):

    fig, ax = plt.subplots(len(instances[1:]), len(dimensions), sharex='col', sharey='row')
    legend = ['KM++', 'MB 100', 'MB 500', 'MB 1000', 'RPKM 1', 'RPKM 2', 'RPKM 3', 'RPKM 4', 'RPKM 5', 'RPKM 6']

    for i, n in enumerate(instances[1:]):
        for j, d in enumerate(dimensions):
            for experiment in legend:
                results_ = results[9][d][n][experiment]['error']
                ax[i, j].plot(np.mean(results[9][d][n][experiment]['distances']), np.mean(results_),
                              linestyle="None", marker='.')

            ax[i, j].set_xscale('log')
            ax[i, j].set_yscale('log')

    [ax[0, j].title.set_text('d:' + str(dimensions[j])) for j in range(len(dimensions))]
    [(ax[i, -1].yaxis.set_label_position("right"), ax[i, -1].set_ylabel('n:' + str(instances[i + 1])))
     for i in range(len(instances[1:]))]

    fig.text(0.5, 0, 'Distance computations', ha='center')
    fig.text(0, 0.5, 'Error', va='center', rotation='vertical')

    ax[0, 0].legend(legend, bbox_to_anchor=(0, 1.22, 3.2, 0.102), bbox_transform=ax[0, 0].transAxes,
                    loc="lower left", mode="expand", ncol=5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'plot4.png'))


if __name__ == '__main__':

    # FIRST EXPERIMENT. GRID PLOTS
    print("First batch of experiments. Grid plots.")

    X, y = make_blobs(n_samples=10000, centers=3, n_features=2, random_state=0)

    # K-MEANS++
    distances, error = [], []
    for iteration in range(repetitions):

        results = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=iteration * 100,
                         algorithm='full').fit(X=X)

        distances.append(results.n_iter_ * results.cluster_centers_.shape[0] * X.shape[0])
        error.append(compute_error(X, results.labels_, results.cluster_centers_))

        path = os.path.join('results', 'grid-plots', 'K-MEANS++')

        if not os.path.isdir(path):
            os.makedirs(path)

        plot_grid(X, results, None, path, iteration, distances[-1], error[-1])

        print('K-MEANS++. Average distance computations=' + str(np.mean(distances)) +
              ' Average error=' + str(np.mean(error)))

    # RPKM
    for partition in partitions:

        distances, error = [], []
        for iteration in range(repetitions):

            results = RPKM(n_clusters=3, max_part=partition, random_state=iteration).fit(X=X)

            distances.append(results.dist_comp)
            error.append(results.error)

            path = os.path.join('results', 'grid-plots', 'RPKM', 'index=' + str(partition) +
                                ' partitions=' + str(results.partitions))

            if not os.path.isdir(path):
                os.makedirs(path)

            plot_grid(X, results, partition, path, iteration, None, None)

        print('RPKM. Partition index ' + str(partition) + '. Average distance computations='
              + str(np.mean(distances)) + ' Average error=' + str(np.mean(error)))

    # SECOND SET OF EXPERIMENTS. ARTIFICIAL DATASETS
    print("\nSecond batch of experiments. Artificial datasets.")
    results = {}
    for cluster in clusters:
        results[cluster] = {}
        for dimension in dimensions:
            results[cluster][dimension] = {}
            for instance in instances:
                results[cluster][dimension][instance] = {}
                for iteration in range(repetitions):

                    print('Iteration ' + str(iteration) + '.')

                    X, y = make_blobs(n_samples=instance, centers=cluster, n_features=dimension, random_state=iteration)

                    if instance > 100:

                        # K-MEANS++
                        results_km = KMeans(n_clusters=cluster, init='k-means++', n_init=1, random_state=iteration,
                                            algorithm='full').fit(X=X)

                        print('K-MEANS++ finished.')

                        if iteration == 0:
                            results[cluster][dimension][instance]['KM++'] = {'distances': [], 'error': []}

                        results[cluster][dimension][instance]['KM++']['distances'].append(
                            results_km.n_iter_ * results_km.cluster_centers_.shape[0] * X.shape[0])
                        results[cluster][dimension][instance]['KM++']['error'].append(
                            compute_error(X, results_km.labels_, results_km.cluster_centers_))

                        # MINI-BATCH K-MEANS
                        for batch in batch_size:

                            results_mb = MiniBatchKMeans(n_clusters=cluster, init='k-means++', max_iter=300, tol=1e-4,
                                                         max_no_improvement=None, batch_size=batch, n_init=1,
                                                         random_state=iteration).fit(X=X)

                            print('MINI-BATCH K-MEANS ' + str(batch) + ' finished.')

                            if iteration == 0:
                                results[cluster][dimension][instance]['MB ' + str(batch)] = {'distances': [],
                                                                                             'error': []}

                            results[cluster][dimension][instance]['MB ' + str(batch)]['distances'].append(
                                results_mb.n_iter_ * results_mb.cluster_centers_.shape[0] * batch)
                            results[cluster][dimension][instance]['MB ' + str(batch)]['error'].append(
                                compute_error(X, results_mb.labels_, results_mb.cluster_centers_))

                    # RPKM
                    for partition in partitions:
                        try:
                            if iteration == 0:
                                results[cluster][dimension][instance]['RPKM ' + str(partition)] = \
                                    {'distances': [], 'partitions': [], 'clusters': [], 'error': []}

                            results_rpkm = RPKM(n_clusters=cluster, max_part=partition, random_state=iteration).fit(X=X)

                            print('RPKM ' + str(partition) + ' finished.')

                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['distances'].append(
                                results_rpkm.dist_comp)
                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['partitions'].append(
                                results_rpkm.partitions)
                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['clusters'].append(
                                results_rpkm.cluster_centers_)
                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['error'].append(
                                results_rpkm.error)
                        except:
                            print("RPKM CRASHED")

                    print('ARTIFICIAL DATASETS. Experiment with parameters K=' + str(cluster) + ' d=' + str(dimension)
                          + ' n=' + str(instance) + ' finished.')

    path = os.path.join('results', 'general-plots', 'artificial-datasets')

    if not os.path.isdir(path):
        os.makedirs(path)

    pretty(results)
    # Plot 1: Distance computations / Dataset size
    plot_general_1(results, path)

    # Plot 2: Partition size per dataset size / Partition index
    plot_general_2(results, path, 'artificial')

    # Plot 3: Standard error / Partition index
    plot_general_3(results, path, 'artificial')

    # Plot 4: Error / Distance computations
    plot_general_4(results, path)

    # THIRD SET OF EXPERIMENTS. REAL DATASET
    print("\nThird batch of experiments. Real datasets.")

    results = {}
    for cluster in clusters:
        results[cluster] = {}
        for dimension in dimensions_:
            results[cluster][dimension] = {}
            for instance in instances_:
                results[cluster][dimension][instance] = {}
                for iteration in range(repetitions_):

                    print('Iteration ' + str(iteration) + '.')

                    X = np.load(os.path.join('data', 'real_data.npy'))
                    X = X[..., np.random.choice(range(X.shape[1]), size=dimension, replace=False)]
                    X = X[np.random.choice(range(X.shape[0]), size=instance, replace=False)]

                    if instance > 100:

                        if iteration == 0:
                            results[cluster][dimension][instance]['KM++'] = {'distances': [], 'error': []}

                        # K-MEANS++
                        results_km = KMeans(n_clusters=cluster, init='k-means++', n_init=1, random_state=iteration,
                                            algorithm='full').fit(X=X)

                        print('K-MEANS++ finished.')

                        results[cluster][dimension][instance]['KM++']['distances'].append(
                            results_km.n_iter_ * results_km.cluster_centers_.shape[0] * X.shape[0])
                        results[cluster][dimension][instance]['KM++']['error'].append(
                            compute_error(X, results_km.labels_, results_km.cluster_centers_))

                        # MINI-BATCH K-MEANS
                        for batch in batch_size:

                            results_mb = MiniBatchKMeans(n_clusters=cluster, init='k-means++', max_iter=300, tol=1e-4,
                                                         max_no_improvement=None, batch_size=batch, n_init=1,
                                                         random_state=iteration).fit(X=X)

                            print('MINI-BATCH K-MEANS ' + str(batch) + ' finished.')

                            if iteration == 0:
                                results[cluster][dimension][instance]['MB ' + str(batch)] = {'distances': [],
                                                                                             'error': []}

                            results[cluster][dimension][instance]['MB ' + str(batch)]['distances'].append(
                                results_mb.n_iter_ * results_mb.cluster_centers_.shape[0] * batch)
                            results[cluster][dimension][instance]['MB ' + str(batch)]['error'].append(
                                compute_error(X, results_mb.labels_, results_mb.cluster_centers_))

                    # RPKM
                    for partition in partitions:
                        try:
                            if iteration == 0:
                                results[cluster][dimension][instance]['RPKM ' + str(partition)] = \
                                    {'distances': [], 'partitions': [], 'clusters': [], 'error': []}
                            results_rpkm = RPKM(n_clusters=cluster, max_part=partition, random_state=iteration).fit(X=X)

                            print('RPKM ' + str(partition) + ' finished.')

                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['distances'].append(
                                results_rpkm.dist_comp)
                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['partitions'].append(
                                results_rpkm.partitions)
                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['clusters'].append(
                                results_rpkm.cluster_centers_)
                            results[cluster][dimension][instance]['RPKM ' + str(partition)]['error'].append(
                                results_rpkm.error)
                        except:
                            print("RPKM CRASHED")

                    print('REAL DATASET. Experiment with parameters K=' + str(cluster) + ' d=' + str(dimension)
                          + ' n=' + str(instance) + ' finished.')

    path = os.path.join('results', 'general-plots', 'real-dataset')

    if not os.path.isdir(path):
        os.makedirs(path)
    pretty(results)
    # Plot 1: Distance computations / Dataset size
    plot_general_1(results, path)

    # Plot 2: Partition size per dataset size / Partition index
    plot_general_2(results, path, 'real')

    # Plot 3: Standard error / Partition index
    plot_general_3(results, path, 'real')

    # Plot 4: Error / Distance computations
    plot_general_4(results, path)
