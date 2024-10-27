import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import data_fx as dfx
from sklearn.neighbors import LocalOutlierFactor
import Metric as mx

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = None
save_path = None


#### dmso ####
dmso_sequences = None
acim_sequences = None

def get_metric(dfr):
    #### aggregate counts of Predict, read_depth,
    #### another decider based on the percentage of modified reads ###
    dfr = dfr[['position', 'contig', 'Predict']]
    dfr['Predict'] = dfr['Predict'].astype('category')
    dfr = dfr.groupby(['position', 'contig', 'Predict'], observed=False).size().unstack(fill_value=0)
    dfr.reset_index(inplace=True)
    dfr['read_depth'] = dfr[-1] + dfr[1]
    dfr['percent_modified'] = dfr[-1] / dfr['read_depth']
    #### predict modification based on percent modified read depth ####
    mean = dfr['percent_modified'].mean()
    dfr['Predict'] = np.where(dfr['percent_modified'] > mean, -1, 1)
    return dfr
def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])



def novelty_signal():
    for seq, fpath in acim_sequences.items():
        print(seq)
        dmso = pd.read_csv(dmso_sequences.get(seq))
        acim = pd.read_csv(acim_sequences.get(seq))

        ### train on all dmso ####
        dftrain = dmso[['event_level_mean']]
        X_train = dftrain.to_numpy()
        dfx = acim[['event_level_mean']]
        X = dfx.to_numpy()

        # fit the model for novelty detection (novelty=True)
        clf = LocalOutlierFactor(n_neighbors=5, novelty=True, contamination=.4)
        clf.fit(X_train)
        # DO NOT use predict, decision_function and score_samples on X_train as this
        # would give wrong results but only on new unseen data (not used in X_train),
        # e.g. X_test, X_outliers or the meshgrid
        y_pred_test = clf.predict(X)
        # n_errors = (y_pred_test != ground_truth).sum()
        X_scores = clf.negative_outlier_factor_

        #df = pd.DataFrame({"Shape_Map": ground_truth, "Lof_Novelty": y_pred_test, "BaseType":np.array(dft["BaseType"])})
        dft = acim[['position', 'contig', 'read_index']]
        dft['Predict'] = y_pred_test
        dft['VARNA'] = np.where(y_pred_test == -1, 1, 0)
        dft.to_csv(save_path + seq + "_lof_signal.csv")
        dft = get_metric(dft)
        dft.to_csv(save_path + seq + "_lof_signal_metric.csv")
        dft = mx.get_Metric(dft)



def novelty_dwell():
    for seq, fpath in acim_sequences.items():
        print(seq)
        dmso = pd.read_csv(dmso_sequences.get(seq))
        acim = pd.read_csv(acim_sequences.get(seq))

        ### train on all dmso ####
        dftrain = dmso[['event_length']]
        X_train = dftrain.to_numpy()
        dfx = acim[['event_length']]
        X = dfx.to_numpy()

        # fit the model for novelty detection (novelty=True)
        clf = LocalOutlierFactor(n_neighbors=5, novelty=True, contamination=.4)
        clf.fit(X_train)
        # DO NOT use predict, decision_function and score_samples on X_train as this
        # would give wrong results but only on new unseen data (not used in X_train),
        # e.g. X_test, X_outliers or the meshgrid
        y_pred_test = clf.predict(X)

        # df = pd.DataFrame({"Shape_Map": ground_truth, "Lof_Novelty": y_pred_test, "BaseType":np.array(dft["BaseType"])})
        dft = acim[['position', 'contig', 'read_index']]
        dft['Predict'] = y_pred_test
        dft['VARNA'] = np.where(y_pred_test == -1, 1, 0)
        dft.to_csv(save_path + seq + "_lof_dwell.csv")
        dft = get_metric(dft)
        dft.to_csv(save_path + seq + "_lof_dwell_metric.csv")
        dft = mx.get_Metric(dft)


def plot_lof(X, X_train, X_test, clf, y_pred_test, ground_truth):
    X_scores = clf.negative_outlier_factor_
    n_errors = (y_pred_test != ground_truth).sum()
    # print("LOF Novelty errors ", n_errors)
    # unique, counts = np.unique(y_pred_test, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print("SHAPE MAP Values:")
    # unique, counts = np.unique(ground_truth, return_counts=True)
    # print(dict(zip(unique, counts)))
    plt.scatter(X[:, 0], X[:, 0], color="k", label="Data points")
    # plot circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    # print(len(radius[0:len(X)]))
    # print(len(X))
    scatter = plt.scatter(
        X[:, 0],
        X[:, 0],
        s=(1000 * radius[0:len(X)]),
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    plt.xlim((-2, 3))
    plt.ylim((-.4, .7))
    #plt.xlabel("prediction errors: %d" % (n_errors))
    plt.legend(
        handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
    )
    plt.title("Local Outlier Factor (LOF) Novelty")
    plt.show()

    xx, yy, zz = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))


    # plot the learned frontier, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection with LOF")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")
    plt.axis("tight")
    plt.xlim((50, 150))
    plt.ylim((-.02, .1))
    plt.legend(
        [a.collections[0], b1, b2],
        [
            "learned frontier",
            "training observations",
            "new regular observations",
            "new abnormal observations",
        ],
        loc="upper left",
        prop=matplotlib.font_manager.FontProperties(size=11),
    )
    # plt.xlabel(
    #     "errors novel regular: %d/40"
    #     % (n_error_test)
    # )
    plt.show()
