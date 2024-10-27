import sys, re
import platform
import sys, re
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.spatial.distance import squareform
from scipy.spatial import distance
import scipy.cluster.hierarchy as shc
import library as seqlib
from kmodes.kmodes import KModes
import DashML.data_fx as dfx


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Deconvolution/Out/"
    save_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Dendrogram/Clusters/"
else:
    data_path = sys.path[1] + "/DashML/Deconvolution/Out/"
    save_path = sys.path[1] + "/DashML/Deconvolution/Dendrogram/Clusters/"

#reactivity format for RNAcofold
def scale_reactivities(reactivities):
    min = reactivities.min()
    max = reactivities.max()
    smin = 0
    smax = 2

    reactivities = ((reactivities - min) / (max - min)) * (smax - smin) + smin
    return reactivities

# Create linkage matrix and then plot the dendrogram
def distancematrix(seq='hcv'):
    #df = pd.read_csv(data_path + seq + "_reactivity_full.csv")
    #Users / timshel / NanoporeAnalysis / DashML / Deconvolution / Out / HCV_read_depth_full2.csv
    df = pd.read_csv(data_path + seq + "_read_depth_full.csv")
    df = df[['position', 'contig', 'read_index', 'Reactivity', 'Predict']]
    df = df.groupby(by=['position','contig','read_index']).mean().reset_index()
    df.fillna(value=0, inplace=True)
    print(len(df))
    df = df.sort_values(by=['position', 'read_index'])

    ##### Compute Distance Between Reads #####
    reads = df['read_index'].unique()
    n = len(reads)
    print(n)
    #reactivity kmeans distance
    dx = df.sort_values(by=['read_index', 'position'])
    dx = (dx.pivot(index=["position"], columns="read_index", values="Reactivity")
         .rename_axis(columns=None)
         .reset_index())
    dx.fillna(value=0, inplace=True)
    dx = dx.drop(columns=['position'])
    dcorr = dx #correlation matrix data
    mm = dx.to_numpy()
    mm = np.divide(mm, np.linalg.norm(mm))
    #clustermap
    km = distance.cdist(mm.T, mm.T, 'euclidean')
    km = np.divide(km,np.linalg.norm(km))


    # reactivity kmode hamming
    dx = df.sort_values(by=['read_index', 'position'])
    dx = (dx.pivot(index=["position"], columns="read_index", values="Predict")
          .rename_axis(columns=None)
          .reset_index())
    dx.fillna(value=0, inplace=True)
    dx = dx.drop(columns=['position'])
    mm2 = dx.to_numpy()

    #clustermap
    ham = distance.cdist(mm2.T, mm2.T, 'hamming')
    ham = np.divide(ham,np.linalg.norm(ham))

    return km, ham, df, reads, mm.T, mm2.T

#### method ultimately decided by which method most closely approximates the desired function
def get_kmeans_clusters(X):
    # Dendogram for Heirarchical Clustering
    #only yields top nodes not all nodes
    plt.figure(figsize=(10, 7))
    plt.title("KMeans Hierarchical Clustering Dendrogram")
    #perform clustering
    agg_cluster = shc.linkage(X, metric='euclidean', method='complete', optimal_ordering=True)
    #p default 30
    # TODO: return ax and set ymin to 6
    dend = shc.dendrogram(agg_cluster, show_leaf_counts=True, get_leaves=True, truncate_mode='lastp', color_threshold=.05)
    ax=plt.gca()
    #ax.set_ylim(ymin=8.6, ymax=10)
    clusters = {}
    for c, n in zip(dend['leaves'], dend['ivl']):
        clusters[c] = int(re.sub('\(|\)', '', n))
    #print(clusters)
    print("kmeans num clusters: ", len(clusters))
    plt.show()
    return agg_cluster, clusters

def get_kmodes_clusters(X):
    # Dendogram for Heirarchical Clustering
    # only yields top nodes not all nodes
    plt.figure(figsize=(10, 7))
    plt.title("Hamming Hierarchical Clustering Dendrogram")
    #perform clustering
    agg_cluster = shc.linkage(X, metric='hamming', method='complete', optimal_ordering=True)
    dend = shc.dendrogram(agg_cluster, show_leaf_counts=True, truncate_mode='lastp', color_threshold=.1)
    ax = plt.gca()
    #ax.set_ylim(ymin=.83, ymax=.95)
    clusters = {}
    for c, n in zip(dend['leaves'], dend['ivl']):
        clusters[c] = int(re.sub('\(|\)', '', n))
    #print(clusters)
    print("hamming num clusters: ", np.unique(clusters))
    plt.show()
    return agg_cluster, clusters

#hybrid correlation matrix using distance instance of correlation
# TODO: clustermap and corr matrix are read not cluster based???
# currently passing in distance matrix and performing clustering
# need to rewrite to remove error with RNASE_p as well
# removes some levels of associated dendrogram, maybe pass agg_cluster

def clustermap(X, seq='HCV', tit='Kmeans'):
    # cannot contain nan
    X = np.nan_to_num(X)
    z_col = linkage(np.transpose(squareform(X)), method='complete')
    print(z_col.shape)
    z_row = linkage(squareform(X), method='complete')
    print(z_row.shape)
    g = sns.clustermap(data=X, row_linkage=z_row, col_linkage=z_col, figsize = (8,8))
    #g.ax_heatmap.set_xlabel("Cluster Labels", labelpad=1, fontweight='extra bold')
    #g.ax_heatmap.set_ylabel("Clusters Labels", labelpad=1, fontweight='extra bold')
    #plt.title(tit + " Inter-Cluster Distances ")
    plt.ylabel("Distance (Min-Max)")
    g.savefig(save_path + seq +'_'+ tit + '_clustermap.png', dpi=600)
    plt.show(block=False)
    print(dir(g))
    #print(g.data2d)


def corrmatrix(seq='HCV', data=None, tit="Kmeans"):
    g = sns.heatmap(np.corrcoef(data))
    g.set_xlabel("Reads")
    g.set_ylabel("Reads")
    g.set_title( tit + " Read Correlations")
    g.get_figure().savefig(save_path + seq +'_'+ tit + '_corr_matrix.png', dpi=600)
    plt.show(block=False)
    print(np.corrcoef(data))

#### kmeans to get centroids ####
def get_centroids(df, clusters, reads, seq):
    #unique, counts = np.unique(clusters, return_counts=True)
    seq_len = seqlib.get_seqlen(seq)
    clust_count = clusters
    max_clusters = dict(sorted(clust_count.items(), key=lambda item: item[1], reverse=True))
    print('kmeans clusters size ', clust_count)
    # create output file
    f = open(save_path + seq + '_kmeans_cluster_counts.csv', "w")
    for k, v in clust_count.items():
        f.write(seq + ', kmeans, ' + str(k) + ',' + str(v) + '\n')
    f.close()
    df_centroid = pd.DataFrame(columns=['cluster', 'position', 'centroid'])
    # set cluster labels to reads
    df['cluster'] = 0
    for r, c in zip(reads, max_clusters.keys()):
        df.loc[df['read_index'] == r, ['cluster']] = c
    # get arrays of predictions for all reads in each cluster
    for i in max_clusters.keys():
        dt = df.loc[df['cluster']==i]
        cluster_reads = dt['read_index'].unique()
        X = []
        for cr in cluster_reads:
            #pad to full sequence length for each read
            x = df.loc[df['read_index']==cr, ['Reactivity']].to_numpy().flatten()
            #scale
            x = scale_reactivities(x)
            pad = np.zeros(seqlib.get_seqlen(seq)-(len(x)))
            #print("x", len(x))
            x = np.concatenate([x,pad])
            X.append(x)
        X = np.array(X)
        #fit dendrogram cluster with kmeans
        km = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(X)
        df_centroid = pd.concat([df_centroid, pd.DataFrame({'cluster': i, 'position': np.arange(0, seq_len), 'centroid': np.array(km.cluster_centers_).flatten()})])

    #print(df_centroid.head(500))
    df_centroid.to_csv(save_path + seq + "_centroids.csv", index=False)

def get_mode_centroids(df, clusters, reads, seq):
    #unique, counts = np.unique(clusters, return_counts=True)
    seq_len = seqlib.get_seqlen(seq)
    clust_count = clusters
    max_clusters = dict(sorted(clust_count.items(), key=lambda item: item[1], reverse=True))
    print('clusters size ', clust_count)
    # create output file
    f = open(save_path + seq + '_hamming_cluster_counts.csv', "w")
    for k, v in clust_count.items():
        f.write(seq + ', hamming, ' + str(k) + ',' + str(v) + '\n')
    f.close()
    df_centroid = pd.DataFrame(columns=['cluster', 'position', 'centroid'])

    # set cluster labels to reads
    df['cluster'] = 0
    for r, c in zip(reads, max_clusters.keys()):
        df.loc[df['read_index'] == r, ['cluster']] = c

    # get reactivities of all reads in each cluster
    # calculate centroid
    for i in max_clusters.keys():
        dt = df.loc[df['cluster']==i]
        cluster_reads = dt['read_index'].unique()
        X = []
        for cr in cluster_reads:
            #pad to full sequence length for each read
            x = df.loc[df['read_index']==cr, ['Predict']].to_numpy().flatten()
            X.append(x)
        #print(np.array(X))
        #print("X", len(x))
        X = np.array(X)
        # get centroid
        km = KModes(n_clusters=1, init='Huang', n_init=5, verbose=0).fit(X)
        df_centroid = pd.concat([df_centroid, pd.DataFrame({'cluster': i, 'position': np.arange(0, seq_len), 'centroid': np.array(km.cluster_centroids_).flatten()})])

    df_centroid.to_csv(save_path + seq + "_mode_centroids.csv", index=False)


def den_array(seq):
    dist_k, dist_ham, df, reads, euclid_reads, ham_reads = distancematrix(seq=seq)
    #dendrograms
    Xk, clusters = get_kmeans_clusters(euclid_reads)
    Xh, clusters_ham = get_kmodes_clusters(ham_reads)
    #centroids
    get_centroids(df, clusters, reads, seq)
    get_mode_centroids(df, clusters_ham, reads, seq)
    #save max clusters
    #get ordered similarity from most to least similar for clusters
    #TODO: should pass agg linkage to cluster map, use same cluster ideally
    clustermap(dist_ham, seq=seq, tit='hamming')
    clustermap(dist_k, seq=seq, tit='kmeans')
    # TODO: correlation between reads or clusters???
    corrmatrix(seq=seq, data=ham_reads, tit="Hamming")
    corrmatrix(seq=seq, data=euclid_reads, tit="Kmeans")



# if __name__ == "__main__":
#     print("Starting main......")
#     i = int(sys.argv[1])  # get the value of the $SLURM_ARRAY_TASK_ID
#     #'RNAse_P' run separately
#     sequences = ['RNAse_P',"cen_3'utr", "cen_3'utr_complex", 'cen_FL', 'cen_FL_complex',
#                  "ik2_3'utr_complex", 'ik2_FL_complex', 'T_thermophila', 'ik2_FL', 'HCV',
#                  "ik2_3'utr"]
#     print(sequences[i])
#     den_array(sequences[i])




#den_array(seq="HCV")
den_array(seq='HCV')


sys.exit(0)
