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

    pca = PCA(n_components=min(reps.shape[0], reps.shape[1]))
    reps = pca.fit_transform(reps)
    reps = reps[:, :2]

    dict_level = aux_info['dict_level']

    # Group individuals by generation
    generation_groups = {}
    for individual, generation in dict_level.items():
        if generation not in generation_groups:
            generation_groups[generation] = []
        generation_groups[generation].append(individual)


    # Compute the collinearity of representations for individuals within the same generation
    collinearity_by_generation = {}

    for generation, individuals in generation_groups.items():
        # Get the indices of individuals in this generation
        indices = [individual for individual in individuals]
        # Extract their representations
        gen_representations = reps[indices]

        # Compute collinearity by fixing one vector as a pivot
        if gen_representations.shape[0] > 2:  # Ensure there are at least three individuals
            pivot = gen_representations[1] - gen_representations[0]  # Difference between first two vectors
            dot_products = (gen_representations[2:] - gen_representations[0]) @ pivot
            norms = np.linalg.norm((gen_representations[2:] - gen_representations[0]), axis=1) * np.linalg.norm(pivot)
            
            norms = np.where(norms == 0, np.nan, norms)
            collinearity = np.abs(dot_products / norms)  # Cosine similarity with the pivot
            collinearity = np.nan_to_num(collinearity, nan=1.0)
            collinearity_by_generation[generation] = collinearity.mean()
            

    variances = pca.explained_variance_ratio_

    metric_dict = {
        'metric': float(1 - np.mean([collinearity for collinearity in collinearity_by_generation.values() if not np.isnan(collinearity)])),
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

    # Compute the centroid of the points
    centroid = np.mean(emb_pca, axis=0)
    
    # Compute distances of points from the centroid
    distances = np.linalg.norm(emb_pca - centroid, axis=1)
    
    # Mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Circularity score
    circularity_score = 1 - (std_distance / mean_distance)


    metric_dict = {
        'metric': float(circularity_score),
        'variances': variances.tolist(),
    }
    return metric_dict
