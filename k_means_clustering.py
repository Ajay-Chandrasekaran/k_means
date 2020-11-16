import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initialize_centroids(data, k_centroids):
    '''Randomly picks k elements from data as centroids'''
    
    index = np.random.choice(data.shape[0], k_centroids)
    return data[index]

def find_closest_centroid(data, centroids):
    '''Assign each data element the closest centroid'''
    
    closest_centroid_index = np.zeros(data.shape[0])
    for ind in range(data.shape[0]):
        closest_centroid_index[ind] = np.argmin(np.sum(np.square(data[ind] - centroids), axis=1))
    
    return closest_centroid_index

def compute_centroids(data, closest_centroid_index, n_centroids):
    '''Recompute the centroids from all the data elements assigned it'''
    new_centroids = np.zeros((n_centroids, data.shape[1]))
    n_neighbours = [0] * n_centroids
    
    for ind in range(data.shape[0]):
        centroid_index = int(closest_centroid_index[ind])
        new_centroids[centroid_index] += data[ind]
        n_neighbours[centroid_index] += 1
    
    for ind in range(n_centroids):
        new_centroids[ind] /= (n_neighbours[ind], n_neighbours[ind])
    
    return new_centroids

def k_means_2d(data, k_centroids, iterations):
    centroids = initialize_centroids(data, 3)
    centroids_history = list()
    centroids_history.append(centroids)
    
    for i in range(iterations):
        closest_centroid = find_closest_centroid(data, centroids)
        centroids = compute_centroids(data, closest_centroid, k_centroids)
        centroids_history.append(centroids)
    
    return centroids_history


if __name__ == "__main__":
    pass
