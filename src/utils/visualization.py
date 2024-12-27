from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embedding(emb, title="", save_path=None, dict_level = None):

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb.detach().numpy())
    print("Explained Variance Ratio", pca.explained_variance_ratio_)
    dim1 = 0
    dim2 = 1
    plt.rcParams.update({'font.size': 12})
    plt.title(title)
    for i in range(len(emb_pca)):
        if dict_level:
            plt.text(emb_pca[i, dim1], emb_pca[i, dim2], str(dict_level[i]), fontsize=12)
        else:
            plt.text(emb_pca[i, dim1], emb_pca[i, dim2], str(i), fontsize=12)
        plt.scatter(emb_pca[:, dim1], emb_pca[:, dim2])
    if save_path:
        plt.savefig(save_path)