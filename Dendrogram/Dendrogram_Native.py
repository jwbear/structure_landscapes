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
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/dendrogram/Clusters/"
    save_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/dendrogram/Native/"
else:
    data_path = sys.path[1] + "/DashML/Deconvolution/Dendrogram/Clusters/"
    save_path = sys.path[1] + "/DashML/Deconvolution/Dendrogram/Native/"


#reactivity format for RNAcofold
def scale_reactivities(reactivities):
    min = reactivities.min()
    max = reactivities.max()
    smin = 0
    smax = 2

    reactivities = ((reactivities - min) / (max - min)) * (smax - smin) + smin
    return reactivities

def get_centroids(seq="HCV"):
    df_kmeans = pd.read_csv(data_path + seq + "_centroids.csv", names=['cluster', 'position', 'reactivity'],
                            header=0)
    df_hamming = pd.read_csv(data_path + seq + "_mode_centroids.csv" , names=['cluster', 'position', 'reactivity'],
                             header=0)
    return df_kmeans, df_hamming

# Create distance matrixes between native sequence and all sequences using hamming and kmeans
# TODO not yet used
def distance_matrix_native_sequences(seq='HCV'):
    native = get_native_hamming_rx(seq=seq)
    df = pd.read_csv(data_path + seq + "_read_depth_full.csv")
    df = df[['position', 'contig', 'read_index', 'Reactivity', 'Predict']]
    df = df.groupby(by=['position','contig','read_index']).mean().reset_index()
    df.fillna(value=0, inplace=True)
    print(len(df))
    df = df.sort_values(by=['position', 'read_index'])
    ##### Compute Hamming Distance Between Reads #####
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
    #clustermap
    km = distance.cdist(native, mm.T, 'euclidean')

    # reactivity kmode hamming
    # dx = df.sort_values(by=['read_index', 'position'])
    # dx = (dx.pivot(index=["position"], columns="read_index", values="Predict")
    #       .rename_axis(columns=None)
    #       .reset_index())
    # dx.fillna(value=0, inplace=True)
    # dx = dx.drop(columns=['position'])
    # mm2 = dx.to_numpy()
    #clustermap
    ham = distance.cdist(native, mm.T, 'hamming')
    return km, ham, df, reads, mm.T, mm.T

# Create distance matrixes between native sequence and cluster centroids using hamming and kmeans
# todo: bug padding incorrect in dendrogram
def distance_matrix(df_kmeans, df_hamming, seq="HCV"):
    native = get_native_hamming_rx(seq=seq)

    ##### Compute Distance Between centroids and native #####

    #reactivity kmeans distance
    dx = df_kmeans.sort_values(by=['cluster', 'position'])
    clusters = dx['cluster'].unique()
    dx = (dx.pivot(index=["position"], columns="cluster", values="reactivity")
         .rename_axis(columns=None)
         .reset_index())
    dx.fillna(value=0, inplace=True)
    dx.drop(columns=['position'], inplace=True)
    mm = dx.to_numpy()
    # kmeans distance matrix
    mm = mm.T
    km = distance.cdist(native, mm, 'euclidean')
    km = np.divide(km, np.linalg.norm(km))
    df = pd.DataFrame(np.transpose(km), columns=['Distance'])
    df['Cluster'] = clusters
    df.to_csv(save_path + seq + '_kmeans_structure_distance.csv', index=False)
    # correlation between clusters matrix
    k_map = distance.cdist(mm, mm, 'hamming')
    k_map = np.divide(k_map, np.linalg.norm(k_map))


    # reactivity kmode distance
    dx = df_hamming.sort_values(by=['cluster', 'position'])
    clusters = dx['cluster'].unique()
    dx = (dx.pivot(index=["position"], columns="cluster", values="reactivity")
          .rename_axis(columns=None)
          .reset_index())
    dx.fillna(value=0, inplace=True)
    dx = dx.drop(columns=['position'])
    mm = dx.to_numpy()
    # distance matrix
    mm = mm.T
    # hamming distance matrix
    ham = distance.cdist(native, mm, 'hamming')
    ham = np.divide(ham, np.linalg.norm(ham))
    df = pd.DataFrame(np.transpose(ham), columns=['Distance'])
    df['Cluster'] = clusters
    df.to_csv(save_path + seq + '_hamming_structure_distance.csv', index=False)
    #correlation between clusters matrix
    ham_map = distance.cdist(mm, mm, 'hamming')
    ham_map = np.divide(ham_map, np.linalg.norm(ham_map))

    return k_map, ham_map, km, ham

# get reactivities for native structures using hamming binary reactivities
# remove sections with no data for each cluster, only compare non-zero positions
def get_native_hamming_rx(seq="HCV"):
    df_native = dfx.get_structure()
    df_native = df_native.loc[df_native['Sequence_Name']==seq]
    #print(df_native.head())
    df_native = df_native[['BaseType']]
    df_native['BaseType'] = np.where(df_native['BaseType']=='S', 1, 0)
    mm = df_native.to_numpy()
    #print(mm.shape)
    return mm.T

# TODO: get reactivities for native structures using native reactivities calculated in Native_Nanopore
# remove sections with no data for each cluster, only compare non-zero positions
def get_native_kmeans_rx(seq="HCV"):
    df_native = dfx.get_structure()
    df_native = df_native.loc[df_native['Sequence_Name']==seq]
    #print(df_native.head())
    df_native = df_native[['BaseType']]
    df_native['read_index'] = -1
    df_native['BaseType'] = np.where(df_native['BaseType']=='S', 1, 0)
    #print(df_native.head())
    mm = df_native.to_numpy()
    #print(mm.shape)
    return mm.T

###### Correlation between Representatives Centroids ######
def corrmatrix(seq='HCV', data=None, tit="Kmeans"):
    plt.clf()
    plt.figure(figsize=(20, 20))
    g = sns.heatmap(np.corrcoef(data), annot=True)
    g.set_xlabel("Clusters")
    g.set_ylabel("Clusters")
    g.set_title( tit + " Cluster Correlations")
    g.get_figure().savefig(save_path + seq +'_'+ tit + '_corr_matrix.png', dpi=600)
    plt.show()
    print(np.corrcoef(data))


###### Heatmap of Cluster Representatives Against native structure ######
def heatmap(X, seq="HCV", method="Kmeans"):
    fig = plt.gcf()
    plt.figure(figsize=(20, 8))
    plt.title( seq + " " + method + " Centroid Distance from Native Structure.\n")
    ax = sns.heatmap(X, cmap='crest', annot=True, square=True)
    ax.set(xlabel="Clusters", ylabel=seq)
    plt.show()
    plt.savefig(save_path + seq + '_heatmap.png', dpi=600)


def den_array(seq):
    df_kmeans, df_hamming = get_centroids(seq)
    k_corr, ham_corr, dist_k, dist_ham = distance_matrix(df_kmeans, df_hamming, seq=seq)
    # heatmap of cluster distances from native sequences
    heatmap(dist_ham, method="Hamming")
    heatmap(dist_k, method="Kmeans")
    # todo: get ordered similarity from most to least similar for clusters
    # TODO: correlation between reads or clusters???
    corrmatrix(seq=seq, data=ham_corr, tit="Hamming")
    corrmatrix(seq=seq, data=k_corr, tit="Kmeans")


# if __name__ == "__main__":
#     print("Starting main......")
#     i = int(sys.argv[1])  # get the value of the $SLURM_ARRAY_TASK_ID
#     #'RNAse_P' run separately
#     sequences = ['RNAse_P',"cen_3'utr", "cen_3'utr_complex", 'cen_FL', 'cen_FL_complex',
#                  "ik2_3'utr_complex", 'ik2_FL_complex', 'T_thermophila', 'ik2_FL', 'HCV',
#                  "ik2_3'utr"]
#     print(sequences[i])
#     den_array(sequences[i])



den_array(seq='T_thermophila')
sys.exit(0)
