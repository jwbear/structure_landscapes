import sys, re
import platform
import os.path
import traceback
import numpy as np
import pandas as pd
import math
import scipy.signal
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
import data_fx as dfx
import Metric as mx



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/DashML/Deconvolution/Decon/"
    f_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Deconvolution/"
else:
    data_path = sys.path[1] + "/DashML/Basecall/"
    save_path = sys.path[1] + "/DashML/Deconvolution/BC/"

def ksm(df, sequence="T_thermophila"):
    bc_acim = df[(df['contig'] == sequence) & (df['Modification'].astype(str) == "ACIM")]
    bc_dmso = df[(df['contig'] == sequence) & (df['Modification'].astype(str) == "DMSO")]

    if len(bc_dmso) < 1 or len(bc_acim) < 1:
        print(f"\n {sequence} data i nsufficient: DMSO: {len(bc_dmso)}, ACIM: {len(bc_acim)}")
        return None


    c = ['Basecall_Reactivity']
    xr = bc_dmso[c].to_numpy().flatten()  # unit_vector_norm(bc_dmso[c].to_numpy()).flatten()
    yr = bc_acim[c].to_numpy().flatten()  # unit_vector_norm(bc_acim[c].to_numpy()).flatten()
    end = len(xr)
    ##### manual cdf to measure ks #####
    # CDF
    # CDF(x) = "number of samples <= x"/"number of samples"
    x1 = np.sort(xr)
    y1 = np.sort(yr)

    def ecdf(x, v):
        res = np.searchsorted(x, v, side='right') / x.size
        return res

    delta_ecdf = np.subtract(yr, xr)


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

    p = find_peaks(delta_ecdf, plateau_size=[0, 10])

    #real values of delta_ecdf normalized
    peaks_value = dfx.unit_vector_norm(delta_ecdf)

    #set peaks in vector of all bases
    s1 = np.ones(len(delta_ecdf))
    s1[p[0]] = -1
    # s2 = np.zeros(len(delta_ecdf))
    # for i,n in enumerate(p[1].get('peak_heights')):
    #     if i in p[0]:
    #         s2[i] = n

    #print(sequence)
    dfc = pd.DataFrame()
    #print(p[1].get('peak_heights'))
    #dfc['peak_weight'] = s2
    dfc['position'] = bc_dmso['position']
    dfc['contig'] = sequence
    dfc['is_peak'] = s1
    dfc['peak_height'] = delta_ecdf
    dfc['ins'] = bc_dmso['Insertion'].astype('float32').to_numpy() - bc_acim['Insertion'].astype('float32').to_numpy()
    dfc['mis'] = bc_dmso['Mismatch'].astype('float32').to_numpy() - bc_acim['Mismatch'].astype('float32').to_numpy()
    #dfc['is_peak2'] = np.where(dfc['peak_height'] > .008, 1, 0)
    #dfc.to_csv(sequence + "_weightcompare.csv")
    #print(dfc.head(100))
    return dfc

#### compare all sequences in dataframe ####
### TODO: specify sequence
### TODO: dmso_complex should only contain complex sequences and not other sequences,
# works now because insufficienct data check
def get_bc_reactivity_peaks():
    # basecall error files
    # TODO: create single file, wo header in bash
    bc_files = {"dmso": data_path + "DMSO_mod_rates.csv",
                "acim":  data_path + "ACIM_mod_rates.csv",
                ### create copy of dmso data for complex comparison
                "dmso_complex":  data_path + "DMSO_mod_rates.csv",
                "complex":  data_path + "ACIM_mod_rates_complex.csv"
                }

    # ,position,contig,Basecall_Reactivity,Quality,Mismatch,Deletion,Insertion,Aligned_Reads,Sequence
    cols = ['position', 'contig', 'Basecall_Reactivity', 'Quality',
            'Mismatch', 'Deletion', 'Insertion', 'Aligned_Reads', 'Sequence', 'Modification']
    csv_cols = ['position', 'contig', 'Basecall_Reactivity', 'Quality',
                'Mismatch', 'Deletion', 'Insertion', 'Aligned_Reads', 'Sequence']
    grp_cols = ['position', 'contig', 'Sequence', 'Modification']
    mean_cols = ['Basecall_Reactivity', 'Quality', 'Mismatch',
                 'Deletion', 'Insertion', 'Aligned_Reads']
    df = pd.DataFrame(columns=cols)
    for f, f_path in bc_files.items():
        if "dmso" in f:
            mod = "DMSO"
        else:
            mod = "ACIM"

        dfn = pd.read_csv(f_path,
                          names=csv_cols,
                          header=0)
        dfn['Modification'] = mod
        #### data set for complex
        if "complex" in f:
            dfn['contig'] = dfn['contig'] + '_complex'

        df = pd.DataFrame(np.concatenate([df.values, dfn.values]), columns=cols)

    #average duplicate entries
    df = df.groupby(grp_cols, as_index=False)[mean_cols].mean()

    #print(df.groupby(by=['contig', 'Modification', 'Complex'], group_keys=True)['contig'].count())

    #df = pd.concat([df, dfn], axis=0, ignore_index=True)
    #df['Peak_Weight'] = 0
    dft = pd.DataFrame()
    for sequence in df['contig'].unique():
        #print("Sequence: ", sequence)
        #print()
        peak_weight = ksm(df=df, sequence=sequence)
        if peak_weight is not None:
            dft = pd.concat([dft, peak_weight], ignore_index=True)
            # if 'complex' in sequence:
            #     df.loc[(df.contig == sequence.replace('_complex', '')) & (df.Modification == 'DMSO'), 'Peak_Weight'] = peak_weight
            #     df.loc[(df.contig == sequence) & (df.Modification =='ACIM'), 'Peak_Weight'] = peak_weight
            # else:
            #     df.loc[(df.contig == sequence) & (df.Modification == 'DMSO'), 'Peak_Weight'] = peak_weight
            #     df.loc[(df.contig == sequence) & (df.Modification == 'ACIM'), 'Peak_Weight'] = peak_weight

    #print(dft.columns)
    #temper with including negatives or not, with other predictions
    df = pd.DataFrame()
    for s in dft['contig'].unique():
        print(s)
        dfmetric = dft[dft['contig']==s]
        dfmetric['Predict'] = np.where(dfmetric['is_peak'] == -1, -1, 1)
        dfmetric['VARNA'] = np.where(dfmetric['is_peak'] == -1, 1, 0)
        df = pd.concat([df,dfmetric])
        #dfmetric['Predict'] = np.where(np.abs(dfmetric['peak_height']) > .05, 1, dfmetric['Predict'])
        dfmetric.to_csv(save_path + s + "_weightcompare.csv")
        mx.get_Metric(dfmetric)

    df.to_csv(save_path + "bc_weightcompare.csv")
    mx.get_Metric(df)
    df.to_csv(save_path + "bc_weightcompare_metric.csv")
    return dft
