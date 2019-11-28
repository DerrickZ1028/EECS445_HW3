"""
EECS 445 - Introduction to Machine Learning
Fall 2019 - Homework 3
Clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from operator import methodcaller
from numpy import random

from clustering_classes import Cluster, ClusterSet, Point

def random_init(points, k):
    """
    Arguments:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    # TODO: Implement this function
    n = len(points)
    prob = np.ones((n))/n
    index = range(n)
    rand = random.choice(index, k, p = prob, replace = False)
    cs = []
    for i in range(k):
        cs.append(points[rand[i]])
    return cs

def k_means_pp_init(points, k):
    """
    Arguments:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points selected from points
    """
    # TODO: Implement this function
    cs = []
    n = len(points)
    prob = np.ones((n))/n
    index = range(n)
    while len(cs) < k:
        #distance = []
        i = random.choice(index, 1, p = prob, replace = False)
        #print(i)
        cs.append(points[i[0]])
        count = 0
        for point in points:
            dis = []
            for j in range(len(cs)):
                dis.append(point.distance(cs[j]))
            d = np.amin(dis)
            prob[count] = d**2
            count += 1
        prob /= sum(prob)
    return cs
        


def k_means(points, k, init='random'):
    """
    Clusters points into k clusters using k_means clustering.
    Arguments:
        points: a list of Point objects
        k: the number of clusters
        init: The method of initialization. One of ['random', 'kpp'].
              If init='kpp', use k_means_pp_init to initialize clusters.
              If init='random', use random_init to initialize clusters.
              Default value 'random'.
    Returns:
        Instance of ClusterSet with k clusters
    """
    # TODO: Implement this function
    # c_set = ClusterSet()
    # assert(len(c_set.get_clusters()) == 0)
    cs = []
    if init == 'random':
        cs = random_init(points, k)
    if init == 'kpp':
        cs = k_means_pp_init(points, k)
    # cluster = []
    # for i in range(k):
    #     cluster.append([])
    # for point in points:
    #     index = 0
    #     dis = 1e300
    #     for i in range(k):
    #         if point.distance(cs[i]) < dis:
    #             dis = point.distance(cs[i])
    #             index = i
    #     cluster[index].append(point)
    # for i in range(k):
    #     c_set.add(Cluster(cluster[i]))
    c_set = ClusterSet()
    score = 0
    next_score,c_temp, cs = cal_next_score(points, k, cs)
    while(not c_temp.equivalent(c_set)):
        c_set = c_temp
        score = next_score
        next_score, c_temp, cs = cal_next_score(points, k, cs)
    return c_set

def cal_next_score(points, k, cs):
    #print('----')
    #cs = c_set.get_centroids()
    c_set = ClusterSet()
    #c_set.get_clusters() = []
    assert(len(c_set.get_clusters()) == 0)
    cluster = []
    for i in range(k):
        cluster.append([])
    for point in points:
        index = 0
        dis = []
        for i in range(k):
           # print(cs[i].get_features())
            dis.append(point.distance(cs[i]))   
        index = dis.index(min(dis))
        cluster[index].append(point)
    for i in range(k):
        assert(len(cluster[i]) > 0)
        c_set.add(Cluster(cluster[i]))
    score = c_set.get_score()
    cs = c_set.get_centroids()
    return score, c_set, cs

def spectral_clustering(points, k):
    """
    Uses sklearn's spectral clustering implementation to cluster the input
    data into k clusters
    Arguments:
        points: a list of Points objects
        k: the number of clusters
    Returns:
        Instance of ClusterSet with k clusters
    """
    X = np.array([point.get_features() for point in points])
    spectral = SpectralClustering(
        n_clusters=k, n_init=1, affinity='nearest_neighbors', n_neighbors=50)
    y_pred = spectral.fit_predict(X)
    clusters = ClusterSet()
    for i in range(k):
        cluster_members = [p for j, p in enumerate(points) if y_pred[j] == i]
        clusters.add(Cluster(cluster_members))
    return clusters

def plot_performance(k_means_scores, kpp_scores, spec_scores, k_vals):
    """
    Uses matplotlib to generate a graph of performance vs. k
    Arguments:
        k_means_scores: A list of len(k_vals) average purity scores from
            running the k-means algorithm with random initialization
        kpp_scores: A list of len(k_vals) average purity scores from running
            the k-means algorithm with k_means++ initialization
        spec_scores: A list of len(k_vals) average purity scores from running
            the spectral clustering algorithm
        k_vals: A list of integer k values used to calculate the above scores
    """
    # TODO: Implement this function
    k = range(1,k_vals+1)
    plt.plot(k, k_means_scores, label = 'k_mean')
    plt.plot(k, kpp_scores, label = 'kpp')
    plt.plot(k, spec_scores, label = 'spectral')
    plt.legend()
    plt.show()

def get_data():
    """
    Retrieves the data to be used for the k-means clustering as a list of
    Point objects
    """
    data = np.load('kmeans_data.npz')
    X, y = data['X'], data['y']
    X = X.reshape((len(X), -1))
    return [Point(image, label) for image, label in zip(X, y)]

def main():
    points = get_data()
    # TODO: Implement this function
    # for 3.h and 3.i
    # k_mean_score = []
    # kpp_score = []
    # spe_score = []
    # for k in range(1,11):
    #     k_s = []
    #     kpp_s = []
    #     s_s = []
    #     for _ in range(10):
    #         k_cs = k_means(points, k, 'random')
    #         k_s.append(k_cs.get_score())
    #         kpp_cs = k_means(points, k, 'kpp')
    #         kpp_s.append(kpp_cs.get_score())
    #         s_cs = spectral_clustering(points, k)
    #         s_s.append(s_cs.get_score())
    #     k_mean_score.append(np.mean(k_s))
    #     kpp_score.append(np.mean(kpp_s))
    #     spe_score.append(np.mean(s_s))
    #     #print(spe_score[k-1])
    # plot_performance(k_mean_score, kpp_score, spe_score,10)
    """ 3.j """
    # Display representative examples of each cluster for clustering algorithms
    np.random.seed(42)
    k_s = []
    kpp_s = []
    s_s = []
    for _ in range(10):

        kmeans = k_means(points, 2, 'random')
        kpp = k_means(points, 2, 'kpp')
        spectral = spectral_clustering(points, 2)
        k_s.append(kmeans.get_score())
        kpp_s.append(kpp.get_score())
        s_s.append(spectral.get_score())
    #visualize_clusters(kmeans, kpp, spectral)
    print('kmean: max:{}, min:{}, avg{}'.format(np.amax(k_s), np.amin(k_s), np.mean(k_s)))
    print('kpp: max:{}, min:{}, avg{}'.format(np.amax(kpp_s), np.amin(kpp_s), np.mean(kpp_s)))
    print('spec: max:{}, min:{}, avg{}'.format(np.amax(s_s), np.amin(s_s), np.mean(s_s)))

if __name__ == '__main__':
    main()
