from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
import random
import itertools
import numpy as np
from collections import defaultdict

colors = [ "#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(1000)]

def visualize_embedding(emb, title="", save_path=None, dict_level = None, color_dict=True, adjust_overlapping_text=False):
    # Riya: adjustText is a library that cleans up overlapping text in the figure, which is helpful for permutation. Feel free to comment it out.

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb.detach().numpy())
    print("Explained Variance Ratio", pca.explained_variance_ratio_)
    dim1 = 0
    dim2 = 1
    plt.rcParams.update({'font.size': 12})
    plt.title(title)
    if adjust_overlapping_text:
        texts = []
        x = []
        y = []
    for i in range(len(emb_pca)):
        if dict_level:
            if i in dict_level:
                plt.scatter(emb_pca[i, dim1], emb_pca[i, dim2], c=colors[dict_level[i]] if color_dict else 'k')
                if adjust_overlapping_text:
                    texts.append(plt.text(emb_pca[i, dim1], emb_pca[i, dim2], str(dict_level[i]), fontsize=12))
                else:
                    plt.text(emb_pca[i, dim1], emb_pca[i, dim2], str(dict_level[i]), fontsize=12)
        else:
            plt.scatter(emb_pca[i, dim1], emb_pca[i, dim2], c='k')
            if adjust_overlapping_text:
                texts.append(plt.text(emb_pca[i, dim1], emb_pca[i, dim2], str(i), fontsize=12))
            else:
                plt.text(emb_pca[i, dim1], emb_pca[i, dim2], str(i), fontsize=12)

        if adjust_overlapping_text:
            x.append(emb_pca[i,dim1])
            y.append(emb_pca[i,dim2])

    if adjust_overlapping_text:
        print("Adjusting text")
        adjust_text(texts, x=x, y=y, autoalign='xy', force_points=0.5, only_move = {'text':'xy'})
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
    #plt.show()
    #plt.close()

def visualize_embedding_permutations(emb, right_coset_list, left_coset_list, title="", save_path=None, dict_level=None, adjust_overlapping_text=False, text=False):
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb.detach().numpy())
    print("Explained Variance Ratio", pca.explained_variance_ratio_)

    # generate all permutations of [0, 1, 2, 3]
    all_permutations = list(itertools.permutations(range(4)))
    perm_to_index = {perm: idx for idx, perm in enumerate(all_permutations)}

    fig, axs = plt.subplots(14, 4, figsize=(10, 28))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    texts = []

    # plot right cosets with colors based on permutation arrays
    for idx, coset in enumerate(right_coset_list):
        ax = axs[idx // 4, idx % 4]

        ax.set_xticks([])  
        ax.set_yticks([]) 

        ax.set_title(f"Right Coset {idx}")
        coset_cnt = len(coset)
        colors = plt.cm.viridis(np.linspace(0, 1, coset_cnt))

        for pidx, perm_array in enumerate(coset):
            color = colors[pidx]
            for perm in perm_array:
                perm_tuple = tuple(perm)
                perm_index = perm_to_index[perm_tuple]
                ax.scatter(emb_pca[perm_index, 0], emb_pca[perm_index, 1], c=[color])
                
                # Add text based on dict_level
                if dict_level and perm_index in dict_level:
                    label = str(dict_level[perm_index])
                    if text:
                        if adjust_overlapping_text:
                            texts.append(ax.text(emb_pca[perm_index, 0], emb_pca[perm_index, 1], label, fontsize=12))
                        else:
                            ax.text(emb_pca[perm_index, 0], emb_pca[perm_index, 1], label, fontsize=8)

    # plog left cosets
    for idx, coset in enumerate(left_coset_list):
        ax = axs[(idx + len(right_coset_list)) // 4, (idx + len(right_coset_list)) % 4]
        ax.set_xticks([]) 
        ax.set_yticks([]) 

        ax.set_title(f"Left Coset {idx}")
        coset_cnt = len(coset)
        colors = plt.cm.viridis(np.linspace(0, 1, coset_cnt))

        for pidx, perm_array in enumerate(coset):
            color = colors[pidx]
            for perm in perm_array:
                perm_tuple = tuple(perm)
                perm_index = perm_to_index[perm_tuple]
                ax.scatter(emb_pca[perm_index, 0], emb_pca[perm_index, 1], c=[color])
                
                if dict_level and perm_index in dict_level:
                    label = str(dict_level[perm_index])
                    if text:
                        if adjust_overlapping_text:
                            texts.append(ax.text(emb_pca[perm_index, 0], emb_pca[perm_index, 1], label, fontsize=12))
                        else:
                            ax.text(emb_pca[perm_index, 0], emb_pca[perm_index, 1], label, fontsize=8)

    if adjust_overlapping_text:
        adjust_text(texts, autoalign='xy', force_points=0.5, only_move={'text': 'xy'})

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')

def silhouette_score(points, labels, penalty_weight=0):
    points = np.array(points)
    labels = np.array(labels)

    clusters = defaultdict(list)
    for point, label in zip(points, labels):
        clusters[label].append(point)

    silhouette_scores = []
    for _, (point, label) in enumerate(zip(points, labels)):
        same_cluster = np.array(clusters[label])

        # compute a(i): mean distance within the same cluster (excluding the point itself)
        a = np.mean([np.linalg.norm(point - other) for other in same_cluster if not np.array_equal(point, other)])

        # compute b(i): mean distance to the nearest different cluster
        b = float('inf')
        for other_label, other_cluster in clusters.items():
            if other_label != label:
                other_cluster = np.array(other_cluster)
                mean_distance = np.mean([np.linalg.norm(point - other) for other in other_cluster])
                b = min(b, mean_distance)

        # edge case when a cluster has only one point (in case, shouldn't be an issue)
        if len(same_cluster) == 1:
            silhouette_scores.append(0) 
        else:
            silhouette_scores.append((b - a) / max(a, b))

    # average
    avg_silhouette_score = np.mean(silhouette_scores)

    # a penalty for the number of unique labels, set to 0
    num_clusters = len(clusters)
    penalty = penalty_weight * num_clusters
    adjusted_score = avg_silhouette_score - penalty

    return adjusted_score

def plot_single_coset(array_list, emb_pca, perm_to_index, title=None, save_path=None, dict_level=None, adjust_overlapping_text=False):
    plt.rcParams.update({'font.size': 12})
    
    if title:
        plt.title(title)
    
    if adjust_overlapping_text:
        texts = []

    array_count = len(array_list)
    colors = plt.cm.viridis(np.linspace(0, 1, array_count)) # plt.cm.tab20(range(array_count%20)) 

    for array_idx, perm_array in enumerate(array_list):
        color = colors[array_idx]
        for perm in perm_array:
            perm_tuple = tuple(perm)
            if perm_tuple in perm_to_index:
                perm_index = perm_to_index[perm_tuple]
                plt.scatter(emb_pca[perm_index, 0], emb_pca[perm_index, 1], c=[color])
                
                if dict_level and perm_index in dict_level:
                    label = str(dict_level[perm_index])
                    if adjust_overlapping_text:
                        texts.append(plt.text(emb_pca[perm_index, 0], emb_pca[perm_index, 1], label, fontsize=12))
                    else:
                        plt.text(emb_pca[perm_index, 0], emb_pca[perm_index, 1], label, fontsize=8)
    
    if adjust_overlapping_text:
        adjust_text(texts, autoalign='xy', force_points=0.5, only_move={'text': 'xy'})
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    plt.close()

def visualize_best_embedding(emb, right_coset_list, left_coset_list, title="", save_name=None, dict_level=None, adjust_overlapping_text=False, penalty_weight=0, input_best=None):
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb.detach().numpy())
    print("Explained Variance Ratio", pca.explained_variance_ratio_)

    total_ev = np.sum(pca.explained_variance_ratio_)

    save_path = None
    if save_name:
        save_path = f"{save_name}_ev_{total_ev:.4f}.png"

    # generate all permutations of [0, 1, 2, 3]
    all_permutations = list(itertools.permutations(range(4)))
    perm_to_index = {perm: idx for idx, perm in enumerate(all_permutations)}

    full_coset_list = right_coset_list + left_coset_list

    coset_scores = []
    for idx, coset in enumerate(full_coset_list):
        X_arr = []
        label_arr = []
        
        labels=np.arange(len(coset))

        for pidx, perm_array in enumerate(coset):
            for perm in perm_array:
                perm_tuple = tuple(perm)
                perm_index = perm_to_index[perm_tuple]
                X_arr.append([emb_pca[perm_index, 0], emb_pca[perm_index, 1]])
                label_arr.append(labels[pidx])
        coset_scores.append(silhouette_score(X_arr, label_arr, penalty_weight))
        
    coset_name = 'Right'
    best_coset = np.argmax(coset_scores)

    if input_best:
        best_coset = input_best

    if best_coset > len(right_coset_list) - 1:
        best_coset = best_coset - len(right_coset_list)
        coset_name = 'Left'
        coset = left_coset_list[best_coset]
    else: 
        coset=right_coset_list[best_coset]
    
    print(f"Best coset found, {coset_name} {best_coset}")

    plot_single_coset(coset, emb_pca, perm_to_index, title=title, save_path=save_path, dict_level=dict_level, adjust_overlapping_text=adjust_overlapping_text)


def visualize_embedding_3d(emb, title="", save_path=None, dict_level = None, color_dict=True):
    pca = PCA(n_components=3)
    emb_pca = pca.fit_transform(emb.detach().numpy())
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(len(emb_pca)):
        if dict_level:
            if i in dict_level:
                ax.scatter(emb_pca[i, 0], emb_pca[i, 1], emb_pca[i, 2], 
                           c=colors[dict_level[i]] if color_dict else 'k')
                ax.text(emb_pca[i, 0], emb_pca[i, 1], emb_pca[i, 2], 
                        str(dict_level[i]), fontsize=12)
        else:
            ax.scatter(emb_pca[i, 0], emb_pca[i, 1], emb_pca[i, 2], c='k')
            ax.text(emb_pca[i, 0], emb_pca[i, 1], emb_pca[i, 2], str(i), fontsize=12)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
