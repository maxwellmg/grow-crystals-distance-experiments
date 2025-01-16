from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
import random

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
        plt.savefig(save_path)


