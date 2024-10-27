import sys, re
import os.path
import traceback
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import Metric as mx

data_path = None
save_path = None

#### dmso ####
dmso = None
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
    print(dfr.columns)
    return dfr

def ksm(delta):
    delta_ecdf = delta.to_numpy().flatten()
    end = len(delta_ecdf)

    log_dens = []
    n = 0
    width = 2
    for i in range(0,len(delta_ecdf)):
        m = n
        n = n + width
        if n < len(delta_ecdf) and m < len(delta_ecdf):
            kde = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(delta_ecdf.reshape(-1, 1)[m:n])
            l= kde.score_samples(delta_ecdf.reshape(-1,1)[m:n])
            log_dens.append(l)
        else:
            kde = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(delta_ecdf.reshape(-1, 1)[m-width:])
            l = kde.score_samples(delta_ecdf.reshape(-1, 1)[m:])
            log_dens.append(l)
            break;

    tmp = []
    for l in log_dens:
        for v in l:
            tmp.append(v)
    #print(tmp)
    #kde peaks
    #delta_ecdf = tmp

    #sklearn signal processing peaks
    #delta_ecdf = dfx.unit_vector_norm(delta_ecdf)
    #print(delta_ecdf)
    #sys.exit(0)

    p = find_peaks(delta_ecdf, plateau_size=[0, 2])


    #set peaks in vector of all bases
    peaks = np.ones(len(delta_ecdf))
    peaks[p[0]] = -1
    #peaks[p[1]['left_edges']] = -1
    #peaks[p[1]['right_edges']] = -1
    # print(p[1]['left_edges'])
    # print(f"{np.abs(tmp[60:80])} {shape_outliers[60:80]} \n{s1[60:80]}")
    #sys.exit()

    pred_outliers = np.where((peaks == -1), True, False).sum()
    #print("Total Predicted Outlier Sites: ", pred_outliers)

    delta_ecdf = delta_ecdf
    return peaks, delta_ecdf


def get_dwell_reactivity_peaks():
    for seq, fpath in acim_sequences.items():
        print(seq)
        dmsot = dmso[dmso['contig']==seq.replace('_complex', '')]
        dmsot['contig'] = seq
        acim = pd.read_csv(acim_sequences.get(seq))

        #print(np.mean(height))
        dfn = pd.DataFrame()
        for read in acim['read_index'].unique():
            print("Read: ", read)
            acimt = acim.loc[acim['read_index']==read]
            df = acimt.merge(dmsot, on=['position', 'contig'], how='left', suffixes=['_acim', '_dmso'])
            df = df.fillna(value=0)
            df['delta'] = np.subtract(df['event_length_acim'], df['event_length_dmso'])
            df = df[['read_index', 'contig', 'position', 'delta']]
            peaks, height = ksm(df['delta'])
            # print(np.mean(height))
            if peaks is not None:
                df.loc[(df.read_index == read), 'Predict'] = peaks
                df.loc[(df.read_index == read), 'VARNA'] = np.where(peaks == -1, 1, 0)

            df.sort_values(by=['read_index', 'contig', 'position'])
            dfn = pd.concat([dfn,df])

        dfn[['read_index', 'position', 'contig', 'Predict', 'VARNA']].to_csv(save_path +seq+ "_dwell_peaks.csv", index=False)
        dfn = get_metric(dfn)
        dfn.to_csv(save_path +seq+ "_dwell_peaks_metric.csv", index=False)
        mx.get_Metric(dfn)
    return
