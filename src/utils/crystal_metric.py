import numpy as np
import torch

from itertools import combinations
from sklearn.decomposition import PCA

def crystal_metric(reps, data_id, aux_info):
    """
    Compute the crystal metric for the given representations and data_id.
    """
    if data_id == "lattice":
        return lattice_metric(reps, aux_info)
    elif data_id == "greater":
        return greater_metric(reps, aux_info)
    elif data_id == "family_tree":
        return family_tree_metric(reps, aux_info)
    elif data_id == "equivalence":
        return equivalence_metric(reps, aux_info)
    elif data_id == "circle":
        return circle_metric(reps, aux_info)
    elif data_id == "permutation":
        return permutation_metric(reps, aux_info)
    else:
        raise ValueError(f"Unknown data_id: {data_id}")
    
def lattice_metric(reps, aux_info):
    lattice_size = aux_info['lattice_size']
    deviation_arr = []
    points = [(i, j) for i in range(lattice_size) for j in range(lattice_size)]
    
    def side_length_deviation(a, b, c, d):
        a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
        
        # Compute lengths of opposite sides
        length_ab = np.linalg.norm(b - a)
        length_cd = np.linalg.norm(d - c)
        length_ac = np.linalg.norm(c - a)
        length_bd = np.linalg.norm(b - d)
        length_bc = np.linalg.norm(c - b)
        length_ad = np.linalg.norm(d - a)
        
        # Calculate side length deviation
        side_deviation = np.sqrt((length_ab - length_cd)**2 + (length_ac - length_bd)**2) / np.sqrt((length_ab ** 2 + length_bc ** 2 + length_cd ** 2 + length_ad ** 2)/2)
        
        return side_deviation

    # Compute the deviation from a perfect parallelogram for all quadrilaterals
    for quad in combinations(points, 3):
        a, b, c = quad
        d = (c[0] + b[0] - a[0], c[1] + b[1] - a[1])
        if d[0] < 0 or d[0] >= lattice_size or d[1] < 0 or d[1] >= lattice_size:
            continue

        if a[0] == b[0] and b[0] == c[0]:
            continue
        if a[1] == b[1] and b[1] == c[1]:
            continue

        a = lattice_size * a[0] + a[1]
        b = lattice_size * b[0] + b[1]
        c = lattice_size * c[0] + c[1]
        d = lattice_size * d[0] + d[1]

        a = reps[a]
        b = reps[b]
        c = reps[c]
        d = reps[d]
        deviation = side_length_deviation(a, b, c, d)
        deviation_arr.append(deviation)

    # Obtatin explained variance ratios
    pca = PCA(n_components=min(reps.shape[0], reps.shape[1]))
    emb_pca = pca.fit_transform(reps)
    variances = pca.explained_variance_ratio_

    metric_dict = {
        'metric': float(np.mean(deviation_arr)),
        'variances': variances.tolist(),
    }

    return metric_dict
    

def greater_metric(reps, aux_info):
    diff_arr = []

    # Compute the difference between consecutive representations
    # We expect the perfect representation to be equidistant
    for i in range(reps.shape[0]-1):
        diff_arr.append(np.linalg.norm(reps[i] - reps[i+1]))

    pca = PCA(n_components=min(reps.shape[0], reps.shape[1]))
    emb_pca = pca.fit_transform(reps)
    variances = pca.explained_variance_ratio_

    metric_dict = {
        'metric': float(np.std(diff_arr) / np.mean(diff_arr)),
        'variances': variances.tolist(),
    }
    return metric_dict

def family_tree_metric(reps, aux_info):

    dict_level = aux_info['dict_level']
    reps = reps[1:(max(dict_level.keys()) + 1)]

    pca = PCA(n_components=min(reps.shape[0], reps.shape[1]))
    reps = pca.fit_transform(reps)
    reps = reps[:, :2]


    # Group embeddings by generation
    levels = {}
    for node, generation in dict_level.items():
        if generation not in levels:
            levels[generation] = []
        levels[generation].append(reps[node-1])
    
    # Compute one-dimensionality for each generation
    level_scores = {}
    for generation, points in levels.items():
        if len(points) < 5:
            continue
        
        points_array = np.stack(points)  # Convert to NumPy array
        pca_sub = PCA(n_components=min(points_array.shape[0], points_array.shape[1]))
        pca_sub.fit(points_array)
        one_dimensionality = pca_sub.explained_variance_ratio_[0]  # Ratio of variance explained by the first PC
        level_scores[generation] = one_dimensionality
            

#    pca.fit_transform(reps)
    variances = pca.explained_variance_ratio_

    metric_dict = {
        'metric': float(1 - np.mean(list(level_scores.values()))),
        'variances': variances.tolist(),
    }
    return metric_dict

def equivalence_metric(reps, aux_info):
    mod = aux_info['mod']
    n = reps.shape[0]

    # Compute the difference between representations within the same equivalence class
    diff_arr = []
    cross_diff_arr = []
    for i in range(n):
        for j in range(n):
            if i % mod != j % mod:
                cross_diff_arr.append(np.linalg.norm(reps[i] - reps[j]))
            else:
                diff_arr.append(np.linalg.norm(reps[i] - reps[j]))

    # Filter Outliers
    diff_arr = np.array(diff_arr)
    diff_arr = diff_arr[diff_arr < np.mean(cross_diff_arr)]

    pca = PCA(n_components=min(reps.shape[0], reps.shape[1]))
    emb_pca = pca.fit_transform(reps)
    variances = pca.explained_variance_ratio_

    print(np.mean(diff_arr) , np.mean(cross_diff_arr))
    metric_dict = {
        'metric': float(np.mean(diff_arr) / np.mean(cross_diff_arr)),
        'variances': variances.tolist(),
    }
    return metric_dict


def circle_metric(reps, aux_info):

    pca = PCA(n_components=min(reps.shape[0], reps.shape[1]))
    emb_pca = pca.fit_transform(reps)
    variances = pca.explained_variance_ratio_

    points = emb_pca[:, :2]

    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Normalize points to [0, 1] in both dimensions
    normalized_points = (points - [min_x, min_y]) / [width, height]

    # Compute the centroid of the points
    centroid = np.mean(normalized_points, axis=0)
    
    # Compute distances of points from the centroid
    distances = np.linalg.norm(normalized_points - centroid, axis=1)
    
    # Mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Circularity score
    circularity_score = (std_distance / mean_distance)


    metric_dict = {
        'metric': float(circularity_score),
        'variances': variances.tolist(),
    }
    return metric_dict


def permutation_metric(reps, aux_info): # average distance between permutations in the same coset

    pca = PCA(n_components=min(reps.shape[0], reps.shape[1]))
    emb_pca = pca.fit_transform(reps)
    variances = pca.explained_variance_ratio_

    points = emb_pca[:, :2]

    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Normalize points to [0, 1] in both dimensions
    normalized_points = (points - [min_x, min_y]) / [width, height]

    scatter = np.array(normalized_points)
    
    scatter -= scatter.mean(axis=0)

    angles = np.linspace(0, 2 * np.pi, aux_info['p'], endpoint=False)

    distances = []
    for angle in angles[1:]: 
        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        # rotate scatterplot
        rotated_scatter = scatter @ rotation_matrix.T
        
        # nearest-neighbor distances
        total_distance = 0
        for point in scatter:
            distances_to_rotated = np.linalg.norm(rotated_scatter - point, axis=1)
            total_distance += np.min(distances_to_rotated)
        distances.append(total_distance / len(scatter))
    
    # Symmetry score (inverse of average distance)
    symmetry_score = 1 / (1 + np.mean(distances))

    metric_dict = {
        'metric': float(symmetry_score),
        'variances': variances.tolist(),
    }
    return metric_dict
