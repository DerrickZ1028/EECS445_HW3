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
    rand = np.random.randint(len(points),size = k)
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
    rand = np.random.randint(len(points),size = 1)
    first_cen = points[rand[0]]
    cs.append(first_cen)
    assert(len(cs) == 1)
    while len(cs) < k:
        distance = []
        for point in points:
            dis = 100000
            for c in cs:
                new_dis = point.distance(c)
                if new_dis < dis:
                    dis = new_dis
            distance.append(dis**2)
        distance /= np.sum(distance)
        for i in range(1,len(points)):
            distance[i] += distance[i-1]
        a = random.uniform(0,1)
        for i in range(len(points)):
            if distance[i] >= a :
                cs.append(points[i])
                break
    assert(len(cs) == k)
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
    c_set = ClusterSet()
    cs = []
    if init == 'random':
        cs = random_init(points, k)
    if init == 'kpp':
        cs = k_means_pp_init(points, k)
    clu_list = []
    for i in range(k):
        clus = Cluster(points= [])
        assert(len(clus.points) == 0)
        #clus.points.append(cs[i])
        clu_list.append(clus)
    # print('clu_list length:')
    # print(len(clu_list))
    # print(len(points))
    count = 0
    assert(len(clu_list[0].points) == 0)
    for point in points:
        # print('count')
        # print(count)
        count += 1
        index = 0
        dis = 100000
        for i in range(k):
            if point.distance(cs[i]) < dis:
                dis = point.distance(cs[i])
                index = i
        clu_list[index].points.append(point)
        # print('appending to : {},l'.format(index))
        # print(len(clu_list[index].points))
    for clus in clu_list:
        # print('----')
        # print(len(clus.points))
        # print('----')
        c_set.add(clus)
    score = c_set.get_score()
    print('k:{}, score:{}'.format(k,score))
    old_score = score
    diff = 1
    while diff > 1e-3:
        cs = c_set.get_centroids()
        #print(cs)
        clu_list = []
        for i in range(k):
            clus = Cluster(points=[])
            assert(len(clus.points)==0)
            #clus.points.append(cs[i])
            clu_list.append(clus)
        for point in points:
            index = 0
            dis = 100000
            for i in range(k):
                #print(type(cs[i]))
                if point.distance(cs[i]) < dis:
                    dis = point.distance(cs[i])
                    index = i
            clu_list[index].points.append(point)
        c_set = ClusterSet()
        for clus in clu_list:
            c_set.add(clus)
        score = c_set.get_score()
        print('k:{}, score:{}'.format(k,score))
        diff = score - old_score
        old_score = score
    return c_set

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
    plt.plot(k, k_mean_scores, label = 'k_mean')
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
    k_mean_score = []
    kpp_score = []
    spe_score = []
    for k in range(1,11):
        print('k = {}'.format(k))
        print('k_mean:')
        k_cs = k_means(points, k, 'random')
        k_mean_score.append(k_cs.get_score())
        print(k_mean_score[k-1])
        print('kpp:')
        kpp_cs = k_means(points, k, 'kpp')
        kpp_score.append(kpp_cs.get_score())
        print(kpp_score[k-1])
        print('spec:')
        s_cs = spectral_clustering(points, k)
        spe_score.append(s_cs.get_score())
        print(spe_score[k-1])
    plot_performance(k_mean_score, kpp_score, spe_score,10)
    """ 3.j """
    return
    # Display representative examples of each cluster for clustering algorithms
    np.random.seed(42)
    kmeans = k_means(points, 1, 'random')
    kpp = k_means(points, 1, 'kpp')
    spectral = spectral_clustering(points, 1)
    visualize_clusters(kmeans, kpp, spectral)

if __name__ == '__main__':
    main()
