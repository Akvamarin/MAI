"""Recursive Partition based K-Means clustering"""

# Author: Alba María García García <alba.maria.4@hotmail.com>

import numpy as np
from math import ceil


def compute_min_partition(dimensions, n_clusters):

    """Compute the minimum partitions per dimension for given a number of clusters.
        Parameters
        ----------
        dimensions : int
            Number of dimensions of the dataset.
        n_clusters : int
            Number of clusters.
        Returns
        -------
        m : int
            Number of partitions per dimension.
            The total number of partitions is computed as m**dimensions.
    """

    return 2**(ceil(n_clusters**(1/dimensions)) - 1).bit_length()


def create_grid_partitions(X, m, lower_bounds, upper_bounds):

    """Compute the grid partitions over a dataset X.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        m : int
            Index of the grid partitions. The total number of partitions
            is computed as m**n_features.
        lower_bounds : array, shape [n_samples,]
            Lower bounds of every dimension in the dataset X. Needed to
            compute the bounding box of the dataset within the feature space.
        upper_bounds: array, shape [n_samples,]
            Upper bounds of every dimension in the dataset X. Needed to
            compute the bounding box of the dataset within the feature space.
        Returns
        -------
        representatives : array, shape [m**n_features(*), n_features]
            Ndarray of representatives (points in the feature space that represent
            the instances within a grid partition).
        cardinality : array, shape [m**n_features(*),]
            Number of elements within each partition.

        (*) Maximum value, it decreases if empty partitions appear.
    """

    # First obtain the partition ID each instance of X belongs to
    partition_asg = get_partition_assigment(X, m, lower_bounds, upper_bounds)

    # Get the different partition identifiers, indexes (ID of the first instance that belongs to the partition)
    # and their cardinality (number of instances)
    partitions, indexes, cardinality = np.unique(partition_asg, return_counts=True, return_index=True)

    # Divide the partitions into those with only one instance ('soft_partitions') and those with more than one
    # ('heavy_partitions') using their cardinality as discriminant
    heavy_mask, soft_mask = cardinality > 1, cardinality == 1
    heavy_partitions, soft_partitions = partitions[heavy_mask], partitions[soft_mask]

    # The representative of the soft partitions is the only instance inside
    soft_representatives = X[indexes[soft_mask]]

    # However, partitions with more than one instance require to compute the mean. To do so, we have to locate which
    # are those partitions ('partition_mask') and obtain their cardinality ('heavy_cardinality).
    partition_mask = [np.where(partition_asg == heavy_partitions[i])[0] for i in range(heavy_partitions.shape[0])]
    heavy_cardinality = cardinality[heavy_mask]
    heavy_representatives = [np.sum(X[partition_mask[partition]], axis=0) / heavy_cardinality[partition]
                       for partition in range(heavy_partitions.shape[0])]

    # Finally, we put together both set of representatives in a single array
    representatives = np.zeros((len(cardinality), X.shape[-1]), dtype=np.float)
    if len(soft_representatives) == 0:
        representatives = np.array(heavy_representatives)
    elif len(heavy_representatives) == 0:
        representatives = soft_representatives
    else:
        representatives[heavy_mask], representatives[soft_mask] = heavy_representatives, soft_representatives

    return representatives, cardinality


def get_partition_assigment(X, m, lower_bounds, upper_bounds):

    """Compute the partition asssignation to every instance of the dataset X.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        m : int
            Index of the grid partitions. The total number of partitions
            is computed as m**n_features.
        lower_bounds : array, shape [n_samples,]
            Lower bounds of every dimension in the dataset X. Needed to
            compute the bounding box of the dataset within the feature space.
        upper_bounds: array, shape [n_samples,]
            Upper bounds of every dimension in the dataset X. Needed to
            compute the bounding box of the dataset within the feature space.
        Returns
        -------
        partition_asg : array, shape [n_samples,]
            Partition assignation to every instance.
    """

    # First, compute the size of the partitions for every dimension of the feature space.
    # Then, assign a partition (in array-like format) to every instance of the dataset
    partition_size_dim = (upper_bounds - lower_bounds) / m
    partition_asg = np.array([np.clip((X[sample] - lower_bounds) // partition_size_dim, a_min=None, a_max=m - 1)
                              for sample in range(X.shape[0])], dtype=np.ulonglong)

    # Transform the array-like format to ID format for an easier handling
    partition_asg = np.sum(partition_asg * np.array([m ** d for d in range(X.shape[1])])[::-1], axis=1)

    return partition_asg


def initialize_centroids(representatives, n_clusters):

    """Get the initial centroids by means of random initialization.
        Parameters
        ----------
        representatives : array, shape [m**n_features*, n_features]
            Ndarray of representatives (points in the feature space that represent
            the instances within a grid partition).
        n_clusters : int
            Number of clusters.
        Returns
        -------
        cluster_centers_ : array, shape [n_clusters, n_features]
            Ndarray of centroids (points in the feature space that represent
            the centers of the clusters).

        (*) Maximum value, it decreases if empty partitions appear.
    """

    # Return the centroids using a random sampling of the representatives without replacement
    centroids_index = np.random.choice(range(representatives.shape[0]), n_clusters, replace=False)

    return representatives[centroids_index]


def check_stopping_criterion(prev_cluster_centers_, cluster_centers_):

    """Compute the tolerance for the current iteration.
        Parameters
        ----------
        prev_cluster_centers_ : array, shape [n_clusters, n_features]
            Ndarray of representatives (points in the feature space that represent
            the instances within a grid partition) of the previous iteration.
        cluster_centers_ : array, shape [n_clusters, n_features]
            Ndarray of centroids (points in the feature space that represent
            the centers of the clusters) of the current iteration.
        Returns
        -------
        tolerance : float
            Maximum displacement of the centroids among two consecutive iterations
            (using L2-norm squared).
    """

    return np.max(np.square(np.linalg.norm(prev_cluster_centers_ - cluster_centers_)))


def compute_inertia(representatives, cluster_centers_):

    """Compute the inertia for the current iteration.
        Parameters
        ----------
        representatives : array, shape [m**n_features(*), n_features]
            Ndarray of representatives (points in the feature space that represent
            the instances within a grid partition).
        cluster_centers_ : array, shape [n_clusters, n_features]
            Ndarray of centroids (points in the feature space that represent
            the centers of the clusters).
        Returns
        -------
        inertia : int
            Sum of distances (using L2-norm squared) of each instance to their
            corresponding cluster.

        (*) Maximum value, it decreases if empty partitions appear.
    """

    # Compute the inertia, first with decimals
    inertia = np.sum(np.square(np.min(np.array([np.linalg.norm(representatives - cluster_centers_[cluster], axis=1)
                          for cluster in range(cluster_centers_.shape[0])]), axis=0)))

    return int(round(inertia))  # Round to integer


def assign_clusters(representatives, cluster_centers_):

    """Assign a cluster to every representative of the grid partition.
        Parameters
        ----------
        representatives : array, shape [m**n_features(*), n_features]
            Ndarray of representatives (points in the feature space that represent
            the instances within a grid partition).
        cluster_centers_ : array, shape [n_clusters, n_features]
            Ndarray of centroids (points in the feature space that represent
            the centers of the clusters).
        Returns
        -------
        labels_ : array, shape [m**n_features(*),]
            Cluster ID assigned to each representative.

        (*) Maximum value, it decreases if empty partitions appear.
    """

    # Compute the distances of each instance to each cluster center
    distances = np.array([np.linalg.norm(representatives - cluster_centers_[cluster], axis=1)
                          for cluster in range(cluster_centers_.shape[0])])

    return np.argmin(distances, axis=0)  # Return the cluster ID assigned to each representative


def update_centroids(representatives, cardinality, labels_, n_clusters):

    """Update the position of the cluster centers.
        Parameters
        ----------
        representatives : array, shape [m**n_features(*), n_features]
            Ndarray of representatives (points in the feature space that represent
            the instances within a grid partition).
        cardinality : array, shape [m**n_features(*),]
            Number of elements within each partition.
        labels_ : array, shape [m**n_features(*),]
            Index of the cluster each sample belongs to.
        n_clusters : int
            Number of clusters.
        Returns
        -------
        cluster_centers_ : array, shape [n_clusters, n_features]
            Ndarray of centroids (points in the feature space that represent
            the centers of the clusters).

        (*) Maximum value, it decreases if empty partitions appear.
    """

    # Initialize the cluster_centers_ array
    cluster_centers_ = np.empty([n_clusters, representatives.shape[1]])

    # Iterate over all clusters
    for cluster in range(cluster_centers_.shape[0]):
        # Obtain to which cluster belongs each representative
        # Then, for each cluster, add the cardinality of the partitions they allocate
        cluster_asg = labels_ == cluster
        card_sum = np.sum(cardinality[cluster_asg])
        if card_sum > 0:  # Avoiding empty clusters
            # Obtain the new centroid of the current cluster as the weighted mean of the representatives
            cluster_centers_[cluster] = np.sum(representatives[cluster_asg].T*cardinality[cluster_asg], axis=1)/card_sum

    return cluster_centers_


def compute_error(X, labels_, cluster_centers_):

    """Compute the error for the final set of cluster centers.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        labels_ : array, shape [m**n_features(*),]
            Index of the cluster each sample belongs to.
        cluster_centers_ : array, shape [n_clusters, n_features]
            Ndarray of centroids (points in the feature space that represent
            the centers of the clusters).
        Returns
        -------
        error : float
            Sum of distances from each instance of X to its closest
        cluster center (using L2-norm).
    """
    return np.sum(np.linalg.norm(X - cluster_centers_[labels_], axis=1))


def get_labels(representatives, m, lower_bounds, upper_bounds, X, labels_, cluster_centers_):

    """Assign the labels to each instance of the dataset X according to the labels got for each representative.

        Parameters
        ----------
        representatives : array, shape [m**n_features(*), n_features]
            Ndarray of representatives (points in the feature space that represent
            the instances within a grid partition).
        m : int
            Index of the grid partitions. The total number of partitions
            is computed as m**n_features.
        lower_bounds : array, shape [n_samples,]
            Lower bounds of every dimension in the dataset X. Needed to
            compute the bounding box of the dataset within the feature space.
        upper_bounds: array, shape [n_samples,]
            Upper bounds of every dimension in the dataset X. Needed to
            compute the bounding box of the dataset within the feature space.
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        labels_ : ndarray of shape (representatives,)
            Labels of each representative.
        cluster_centers_ : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers. If the algorithm stops before fully
            converging (see ``tol`` and ``max_iter``), these will not be
            consistent with ``labels_``.

        Returns
        -------
        labels_ : ndarray of shape (n_samples,)
            Labels of each point
    """

    # Compute the labels for every instance of the dataset. 'representative_asg' contains the partition each
    # representative belongs to and 'instance_asg' to which partition belongs each instance.
    representative_asg = get_partition_assigment(representatives, m, lower_bounds, upper_bounds)
    instance_asg = get_partition_assigment(X, m, lower_bounds, upper_bounds)

    # Since we know the label of each representative and clusters must be created using the partition as
    # basic unit - not the instance -, we have to match the label to each instance according its partition
    final_labels = np.empty(X.shape[0], dtype=np.int)

    for i, representative in enumerate(representative_asg):
        final_labels[instance_asg == representative] = labels_[i]

    return np.clip(final_labels, 0, len(cluster_centers_)-1)  # Finally, we assign the labels obtained


class RPKM():

    """Recursive Partition based K-Means clustering.

        Parameters
        ----------
        n_clusters : int, default=8
            The number of clusters to form as well as the number of
            centroids to generate.
        max_part : int, default=2
            Maximum number of partition computations and thus iterations
            of the RPKM algorithm for a single run.
        max_iter : int, default=300
            Maximum number of iterations of the Lloyd's algorithm for a
            single run.
        random_state : int, default=None
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.
        tol : float, default=1e-4
            Relative tolerance with regards to inertia to declare convergence.
        verbose : int, default=0
            Verbosity mode.
        Attributes
        ----------
        cluster_centers_ : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers. If the algorithm stops before fully
            converging (see ``tol`` and ``max_iter``), these will not be
            consistent with ``labels_``.
        labels_ : ndarray of shape (n_samples,)
            Labels of each point.
        inertia_ : float
            Sum of squared distances of samples to their closest cluster center.
        n_part_ : int
            Maximum number of RPKM iterations.
        n_iter_ : int
            Maximum number of Weighted Loyd iterations.

        dist_comp : int
            Number of distance computations performed throughout the whole algorithm.
        partitions : int
            Number of final non-empty partitions.
        error: float
            Sum of distances from each instance of X to its closest
            cluster center (using L2-norm).

        """

    def __init__(self, n_clusters=8, max_part=2, max_iter=300, random_state=None, tol=1e-4, verbose=0):

        self.n_clusters = n_clusters
        self.max_part = max_part
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = 0
        self.n_part_ = 0
        self.n_iter_ = 0

        self.dist_comp = 0
        self.partitions = 0
        self.error = 0

    def fit(self, X, y=None):

        """Compute Recursive Partition based K-Means clustering.
            Parameters
            ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
                Training instances to cluster. It must be noted that the data
                will be converted to C ordering, which will cause a memory
                copy if the given data is not C-contiguous.
            y : Ignored
                Not used, present here for API consistency by convention.
            Returns
            -------
            self
                Fitted estimator.
        """

        # Check values passed by argument

        if self.n_clusters <= 0:
            raise ValueError('Number of clusters should be a positive number, got %d instead' % self.n_clusters)

        if self.max_part <= 0:
            raise ValueError('Number of external iterations should be a positive number, got %d instead' % self.max_part)

        if self.max_iter <= 0:
            raise ValueError('Number of internal iterations should be a positive number, got %d instead' % self.max_iter)

        if self.random_state is not None and self.random_state < 0:
            raise ValueError('Seed should be a positive number or zero, got %d instead' % self.random_state)

        if self.tol < 0:
            raise ValueError('Tolerance should be a positive number or zero, got %d instead' % self.tol)

        if not(self.verbose == 0 or self.verbose == 1):
            raise ValueError('Verbose should be a 0 or 1, got %d instead' % self.verbose)

        if X.ndim != 2:
            raise ValueError('X should be a two-dimensional ndarray, got %d dimensions instead' % X.ndim)

        # Verify that the number of samples given is larger than k
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (X.shape[0], self.n_clusters))

        np.random.seed(self.random_state)  # Initialize the seed for centroid initialization

        # Get the lower and upper bounds of every dimension
        lower_bounds = np.amin(X, axis=0)
        upper_bounds = np.amax(X, axis=0)

        # Compute the minimum number of partitions needed for the given number of clusters
        # Then, obtain the representatives and cardinality of each partition from the initial grid partitions
        m = compute_min_partition(X.shape[1], self.n_clusters)
        representatives, cardinality = create_grid_partitions(X, m, lower_bounds, upper_bounds)

        # Keep updating the partitions until there are enough representatives
        # (at least equal to the number of clusters)
        while representatives.shape[0] < self.n_clusters:
            m = m * X.shape[1]
            representatives, cardinality = create_grid_partitions(X, m, lower_bounds, upper_bounds)

        self.partitions = representatives.shape[0]  # Update the number of partitions

        # Initialize the centroids in a random fashion (no K-Means++ available)
        self.cluster_centers_ = initialize_centroids(representatives, self.n_clusters)
        if self.verbose == 1:
            print("Initialization complete")

        # Iterate over a maximum of self.n_part_ iterations
        for self.n_part_ in range(int(np.log2(m)), int(np.log2(m)) + self.max_part):

            if self.verbose == 1:
                print("Partition index " + str(self.n_part_) + "\n")

            # Save the clusters at the beginning of the iteration to check convergence later
            prev_cluster_centers_partition = self.cluster_centers_

            # Iterate over a maximum of self.max_iter iterations
            for self.n_iter_ in range(self.max_iter):

                # Store the centroids of the previous iteration. Then, assign a cluster to every instance in the dataset
                # Finally, update the centroids of every cluster taking into account the representatives assigned
                prev_cluster_centers_ = self.cluster_centers_
                self.labels_ = assign_clusters(representatives, self.cluster_centers_)
                self.cluster_centers_ = update_centroids(representatives, cardinality, self.labels_, self.n_clusters)
                # Update the number of distances computed to date
                self.dist_comp += representatives.shape[0] * self.n_clusters

                if self.verbose == 1:
                    # Compute the inertia at current iteration
                    self.inertia_ = compute_inertia(representatives, self.cluster_centers_)
                    print("Iteration " + str(self.n_iter_) + ", inertia " + str(self.inertia_))

                # Early stopping: check the centroids' displacement with regard to the tolerance given
                if check_stopping_criterion(prev_cluster_centers_, self.cluster_centers_) < self.tol:
                    # Early stopping criterion met, compute final inertia
                    self.inertia_ = compute_inertia(representatives, self.cluster_centers_)
                    if self.verbose == 1:
                        print("Converged at iteration " + str(self.n_iter_) + ": center shift " + str(self.inertia_) +
                              " within tolerance " + str(self.tol))
                    break

            # In case the algorithm stopped after reaching the maximum number of iterations
            if check_stopping_criterion(prev_cluster_centers_partition, self.cluster_centers_) < self.tol:
                # Compute final inertia
                self.inertia_ = compute_inertia(representatives, self.cluster_centers_)
                if self.verbose == 1:
                    print("Converged at iteration " + str(self.n_iter_) + ": center shift " + str(self.inertia_) +
                          " within tolerance " + str(self.tol))
                break

            # Define the new partitions if it not the last iteration
            if (self.max_part + 1 - self.n_part_) > 1:
                m = m * X.shape[1]
                representatives, cardinality = create_grid_partitions(X, m, lower_bounds, upper_bounds)

        # Obtain the labels for every instance in the dataset
        self.labels_ = get_labels(representatives, m, lower_bounds, upper_bounds, X, self.labels_, self.cluster_centers_)

        # Obtain the number of partitions and the error of the final set of clusters
        self.partitions = representatives.shape[0]
        self.error = compute_error(X, self.labels_, self.cluster_centers_)

        return self

    def fit_predict(self, X, y=None):

        """Compute cluster centers and predict cluster index for each sample.
            Convenience method; equivalent to calling fit(X) followed by
            predict(X).
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                New data to transform.
            y : Ignored
                Not used, present here for API consistency by convention.
            Returns
            -------
            labels : array, shape [n_samples,]
                Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
            In the vector quantization literature, `cluster_centers_` is called
            the code book and each value returned by `predict` is the index of
            the closest code in the code book.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                New data to predict.
            Returns
            -------
            labels : array, shape [n_samples,]
                Index of the cluster each sample belongs to.
        """

        self.labels_ = assign_clusters(X, self.cluster_centers_)

        return self
