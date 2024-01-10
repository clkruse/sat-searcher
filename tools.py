import faiss
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import shapely
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt

import gee.utils as utils


def get_neighbors(search_vec, vectors, metric='cosine', n=5):
    """
    Find the n nearest neighbors to a search vector in a set of vectors.
    """
    # compute the similarity between the search vector and all vectors
    if metric == 'cosine':
        sims = cosine_similarity(search_vec.reshape(1, -1), vectors)[0]
        sorted_sims = np.argsort(sims)[::-1]
    elif metric == 'euclid':
        sims = euclidean_distances(search_vec.reshape(1, -1), vectors)[0]
        # sort the similarities in descending order
        sorted_sims = np.argsort(sims)
    scores = sims[sorted_sims]
    # return the indices of the n most similar vectors
    return sorted_sims[:n], scores[:n]

def get_neighbors_faiss(search_vec, index, n=5):
    distances, indices = index.search(np.expand_dims(search_vec, axis=0), n)
    return indices[0], distances[0]


def tile_from_point(x, y, size=32):
    # create a tile
    tile_geom = utils.Tile(y, x, size).create_geometry()
    # add to map
    return shapely.geometry.mapping(tile_geom)


def retrieve_neighbors(search_vec,
                       index,
                       map_data,
                       centroids,
                       threshold = 100,
                       n=100):
    neighbors, distances = get_neighbors_faiss(search_vec, index, n=n+1)
    neighbors = neighbors[distances < threshold][1:]
    print(f"Found {len(neighbors)} neighbors beneath threshold. Min distance: {distances[1]}, max distance: {distances[-1]}", end='\r')
    result_fc = {"type": "FeatureCollection", "features": []}
    # add the matching neighbors to the map
    for i, index in enumerate(neighbors):
        neighbor_geom = tile_from_point(centroids[index][0], centroids[index][1])
        result_fc['features'].append(neighbor_geom)
    map_data.data = result_fc
    #print(f"{len(result_fc['features'])} of {n} tiles added to the map", end='\r')


def normalize_and_clip(embeddings):
    # Step 1: Calculate mean and standard deviation
    mean = np.mean(embeddings)
    std_dev = np.std(embeddings)
    print(f"Mean: {np.mean(embeddings)}, std: {np.std(embeddings)}")
    # Step 2: Normalize using min-max scaling
    normalized_embeddings = (embeddings - mean) / std_dev

    # Step 3: Clip values beyond 4 standard deviations
    clipped_embeddings = np.clip(normalized_embeddings, -4, 4)

    # Map values to [0, 1] range
    min_value = clipped_embeddings.min()
    max_value = clipped_embeddings.max()
    print(f"Min value: {min_value}, max value: {max_value}")
    normalized_and_clipped_embeddings = (clipped_embeddings - min_value) / (max_value - min_value)

    return normalized_and_clipped_embeddings

def load_embeddings(centroid_dir='./outputs/centroids/', embedding_dir='./outputs/embeddings_8bit/'):
    
    file_names = [f.split('.npy')[0] for f in os.listdir(centroid_dir) if f.endswith('.npy')]

    centroids = []
    embeddings = []
    # load the centroids
    for f in file_names:
        centroids.append(np.load(centroid_dir + f + '.npy'))
        embeddings.append(np.load(embedding_dir + f + '.npy'))
    centroids = np.concatenate(centroids, axis=0)
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Loaded {len(embeddings):,} embeddings")
    return centroids, embeddings

def index_embeddings(embeddings):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index