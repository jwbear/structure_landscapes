import os
import sys, re
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
import numpy as np
from scipy import linalg
from sklearn import mixture
from sklearn.mixture import BayesianGaussianMixture
import Metric as mx

data_path = None
save_path = None

#### dmso ####
#### dmso ####
dmso_sequences = None
acim_sequences = None

def get_metric(dfr):
    #### predict modification based on percent modified read depth ####
    mean = dfr['percent_modified'].mean()
    dfr['Predict'] = np.where(dfr['percent_modified'] > mean, -1, 1)
    print(dfr.columns)
    return dfr

#### GMM for each position across all reads ###
# predict clusters [mod vs unmod] for each position
# plot clusters for a few positions, for paper
# count percent modified by gmm and he see how it compares to
# TODO: have to determine if 0 or 1 is modified by comparing to dmso prediction
# 0 is unmodified, 1 is modified
# when percent_modified of 1 increases in acim, then a modification has occurred
# increase has range, higher increase likely modification
def positional_gmm():
    ### acim sequence file ###
    for seq, fpath in acim_sequences.items():
        print(seq)
        dmso = pd.read_csv(dmso_sequences.get(seq))
        data = pd.read_csv(fpath)
        a = data[['read_index', 'position', 'contig', 'event_level_mean', 'event_length']]
        a = a.groupby(by=['read_index', 'position', 'contig']).mean().reset_index()
        a = a.pivot(index='read_index', columns=['position'], values=['event_level_mean', 'event_length'])
        a.fillna(0, inplace=True)

        d = dmso[['read_index', 'position', 'contig', 'event_level_mean', 'event_length']]
        d = d.groupby(by=['read_index', 'position', 'contig']).mean().reset_index()
        d = d.pivot(index='read_index', columns=['position'], values=['event_level_mean', 'event_length'])
        d.fillna(0, inplace=True)

        #iterate over positions
        dfn = pd.DataFrame(columns=['position','contig', 'read_depth', 'percent_modified', 'Predict'])
        for i in a.columns.get_level_values(1).unique():
            #print(i)
            ### acim data for all reads at position i ###
            b = pd.concat([a['event_level_mean'][i],a['event_length'][i]], axis=1)
            b = b.to_numpy()

            ### dmso data for all reads at position i training ###
            c = pd.concat([d['event_level_mean'][i], d['event_length'][i]], axis=1)
            c = c.to_numpy()
            # training gaussian mixture model, separate into modified vs unmodified
            num_components = 2
            #{‘full’, ‘tied’, ‘diag’, ‘spherical’}
            gmm = BayesianGaussianMixture(n_components = num_components, max_iter=10000,
                                          random_state=1)
            model = gmm.fit(c) #train on dmso
            dmso_labels = model.predict(c)
            unique, counts = np.unique(dmso_labels, return_counts=True)
            #print("dmso")
            dml = dict(zip(unique, counts))
            dmso_percent_modified = (dml.get(1) if dml.get(1) is not None else 0)/ len(dmso_labels)

            #n_components = gmm.covariances_
            #num_components = n_components
            #print(num_components)

            #gplot.plot_sigma_vector(gmm.means_, gmm.covariances_)
            #gplot.plot_gaussian_mixture(gmm.means_, gmm.covariances_, gmm.weights_)

            #predictions from gmm
            #print("acim")
            labels = model.predict(b)
            unique, counts = np.unique(labels, return_counts=True)
            al = dict(zip(unique, counts))
            acim_percent_modified = (al.get(1) if al.get(1) is not None else 0) / len(labels)
            percent_modified = acim_percent_modified - dmso_percent_modified
            #print("dmso_percent_modified: ", dmso_percent_modified)
            #print("acim_percent_modified: ", acim_percent_modified)
            if percent_modified > .02:
                predict = -1
            else: predict = 1
            dfn.loc[len(dfn.index)] = [i, seq, len(labels), percent_modified, predict]

        print(dfn.head())
        print("based on mean")
        dfn = get_metric(dfn)
        mx.get_Metric(dfn)
        dfn.to_csv(save_path + seq + "_gmm.csv")
            #percent_modified =
            # plot_results(a, gmm.predict(a), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")
            # sys.exit(0)
            #a['Predict'] = labels
            #print(labels)
            # frame = a
            # frame['cluster'] = labels.astype(int)
            # title = "GMM"
            #
            # #x = np.arange(0, 1, 0.1)
            # color=['blue','green','cyan', 'orange', 'pink','yellow']
            # for k in range(0,num_components):
            #     data = frame[frame["cluster"]==k]
            #     plt.scatter(data.index, data['cluster'], c=color[k])
            # plt.xticks(a.index, a.index)
            # plt.yticks(np.unique(labels), np.unique(labels) )
            # plt.title(title)
            # #save_plot(plt=plt, mod=mod, fig_name=title)
            # plt.show()
