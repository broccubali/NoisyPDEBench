import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Open the HDF5 file
f = h5py.File("../a.h5", "r")

# Get all keys in the HDF5 file
keys = list(f.keys())

# Generate all possible pairs of keys
pairs = combinations(keys, 2)

# Iterate over each pair and compute cosine similarity
for key1, key2 in pairs:
    data_1 = np.squeeze(f[key1]["data"][:, :])
    data_2 = np.squeeze(f[key2]["data"][:, :])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(data_1, data_2)

    # Compute the overall cosine similarity as the mean of the similarity matrix
    overall_similarity = np.mean(similarity_matrix)

    print(f"Overall Cosine Similarity between {key1} and {key2}: {overall_similarity}")
