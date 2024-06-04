# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

# Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class KMeans:
    n_clusters: int        # number of clusters (k)
    plus: bool             # whether to use k-means++
    centroids: np.ndarray  # centroids of clusters
    labels: np.ndarray     # labels of data points
    iter_converged: int    # number of iterations to converge
    
    def __init__(self, n_clusters: int, plus=True) -> None:
        self.n_clusters = n_clusters
        self.plus = plus
        self.centroids = None
        self.labels = None
        self.iter_converged = 0
    
    def fit(self, X: np.ndarray) -> None:
        if self.n_clusters > len(X):
            raise ValueError('Number of clusters should not exceed the number of data points')
        
        # initialize centroids
        if self.plus:
            self.centroids = self.kmeans_plus_plus(X)
        else:
            self.centroids = X[np.random.choice(len(X), size=self.n_clusters, replace=False)]
        
        self.iter_converged = 0
        while True:
            # assign labels
            self.labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
            
            # update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # check convergence
            # print(np.linalg.norm(self.centroids - new_centroids))
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
            self.iter_converged += 1
        
    def kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using k-means++
        """
        # randomly choose the first centroid
        centroids = [X[np.random.choice(len(X))]]
        
        # choose the remaining centroids
        for _ in range(1, self.n_clusters):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            probabilities = distances**2 / np.sum(distances**2)
            centroids.append(X[np.random.choice(len(X), p=probabilities)])
        
        return np.array(centroids)
            
    def plot(self, attr_names: list = None, block=True) -> None:
        """
        Visualize the clustering result
        """
        plt.figure(figsize=(8, 6))
        for i in range(self.n_clusters):
            plt.scatter(X[self.labels == i, 0], X[self.labels == i, 1], label=f'簇{i + 1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='red', marker='+', s=150, label='质心')
        
        try:
            plt.xlabel(attr_names[0])
            plt.ylabel(attr_names[1])
        except:
            print('Missing attribute names')
            
        plt.legend()
        plt.show(block=block)


if __name__ == '__main__':
    columns = ['编号', '密度', '含糖率']
    data = [
        [ 1, 0.697, 0.460],
        [ 2, 0.774, 0.376],
        [ 3, 0.634, 0.264],
        [ 4, 0.608, 0.318],
        [ 5, 0.556, 0.215],
        [ 6, 0.403, 0.237],
        [ 7, 0.481, 0.149],
        [ 8, 0.437, 0.211],
        [ 9, 0.666, 0.091],
        [10, 0.243, 0.267],
        
        [11, 0.245, 0.057],
        [12, 0.343, 0.099],
        [13, 0.639, 0.161],
        [14, 0.657, 0.198],
        [15, 0.360, 0.370],
        [16, 0.593, 0.042],
        [17, 0.719, 0.103],
        [18, 0.359, 0.188],
        [19, 0.339, 0.241],
        [20, 0.282, 0.257],
        
        [21, 0.748, 0.232],
        [22, 0.714, 0.346],
        [23, 0.483, 0.312],
        [24, 0.478, 0.437],
        [25, 0.525, 0.369],
        [26, 0.751, 0.489],
        [27, 0.532, 0.472],
        [28, 0.473, 0.376],
        [29, 0.725, 0.445],
        [30, 0.446, 0.459]
    ]
    
    X = np.array(data)[:, 1:]
    
    k = 2
    
    # k-means and visualization
    kmeans = KMeans(n_clusters=k, plus=True)
    kmeans.fit(X)
    print(f'Converged in {kmeans.iter_converged} iterations')
    kmeans.plot(columns[1:], block=False)
    
    # ground truth
    y = np.ones(len(X))
    y[8:21] = 0    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='坏瓜')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='好瓜')
    plt.xlabel(columns[1])
    plt.ylabel(columns[2])
    plt.title('Ground Truth')
    plt.legend()
    plt.show(block=False)
    
    # Compare random initialization and k-means++
    iters, iters_plus = [], []
    loop = 1000
    
    start = time.time()
    for i in range(loop):
        kmeans = KMeans(n_clusters=k, plus=False)
        kmeans.fit(X)
        iters.append(kmeans.iter_converged)
    end = time.time()
    print(f'random: {end - start:.2f}s')
    
    start = time.time()
    for i in range(loop):
        kmeans = KMeans(n_clusters=k, plus=True)
        kmeans.fit(X)
        iters_plus.append(kmeans.iter_converged)
    end = time.time()
    print(f'k-means++: {end - start:.2f}s')
    
    print()
    print(f'Average iterations (random): {np.mean(iters):.2f}')
    print(f'Average iterations (k-means++): {np.mean(iters_plus):.2f}')
    
    # Histogram for distribution of iterations
    y_top = loop * 0.35  # set the upper limit of y-axis
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(iters, bins=np.arange(0.5, 11, 1), edgecolor='black')
    plt.ylim(0, y_top)
    plt.title('random')
    plt.xlabel('iterations')
    plt.ylabel('frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(iters_plus, bins=np.arange(0.5, 11, 1), edgecolor='black')
    plt.ylim(0, y_top)
    plt.title('k-means++')
    plt.xlabel('iterations')
    plt.ylabel('frequency')
    
    plt.tight_layout()
    plt.show()
    