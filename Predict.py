import sys, re
import platform
import os.path
import traceback
import numpy as np
import pandas as pd
import math
import sklearn.preprocessing
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import BpProbabilities.BasePairProbabilities as bpp
import DashML.Deconvolution.data_fx as dfx
import Metric as mx
import seaborn as sns

### Note: Predict > 3 is noisier with more coverage, better for average predictions
### Predict > 4 stabilizes in the deconvolution offers improved predictions
#TODO: get all non reactive signle molecule base pair probabilities
# and get intermolecular non reactive bp probabilities for caomparison

#### TODO ## could use dmso reactivities ???, need to calculate dmso reactivities and integrate

#### Paper GMM positional induced clusters reflect predictions, nice
# TODO: add to Predict, + for indicate
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Deconvolution/"
    save_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Deconvolution/Out/"
else:
    data_path = sys.path[1] + "/DashML/Deconvolution/"
    save_path = sys.path[1] + "/DashML/Deconvolution/Out/"


def scale_reactivities(reactivities):
    min = reactivities.min()
    max = reactivities.max()
    smin = 0
    smax = 2

    reactivities = ((reactivities - min) / (max - min)) * (smax - smin) + smin
    return reactivities

def get_mods():
    putative_sequences = ['HSP70_HSPA1A', "cen_3'utr", "cen_3'utr_complex", 'cen_FL', 'cen_FL_complex',
                 "ik2_3'utr_complex", 'ik2_FL_complex', 'ik2_FL', "ik2_3'utr"]
    # sequences = ['RNAse_P', "cen_3'utr", "cen_3'utr_complex", 'cen_FL', 'cen_FL_complex',
    #              "ik2_3'utr_complex", 'ik2_FL_complex', 'T_thermophila', 'ik2_FL', 'HCV',
    #              "ik2_3'utr"]

    sequences = ['HSP70_HSPA1A']

    for seq in sequences:
        print(seq)
        df_bc = pd.read_csv(data_path + "BC/" +seq+ "_weightcompare.csv")
        df_signal = pd.read_csv(data_path + "Signal/" +seq+ "_signal_peaks.csv")
        df_dwell = pd.read_csv(data_path + "Dwell/" +seq+ "_dwell_peaks.csv")
        if seq not in putative_sequences:
            dfs = dfx.get_structure_ext()
            dfs.rename(columns={'Position':'position', 'Sequence_Name': 'contig'}, inplace=True)
            dfs.drop(columns=['BaseType', 'StructureType'], inplace=True)
        else:
            dfs = dfx.structuresforputativeseqs()
        # correction for missing positions in source files, merge does not resolve issue
        dfs = dfs.loc[dfs['contig']==seq]
        dfs['read_index'] = 0
        reads = df_dwell['read_index'].unique()
        df_tmp = pd.DataFrame()
        for r in reads:
            dt = dfs
            dt['read_index'] = r
            df_tmp = pd.concat([df_tmp, dt])
        dfs = df_tmp
        df_dwell = dfs.merge(df_dwell, on=['read_index', 'position','contig'], how="left")
        df_lofd = pd.read_csv(data_path + "LOF/" +seq+ "_lof_dwell.csv")
        df_lofs = pd.read_csv(data_path + "LOF/" +seq+ "_lof_signal.csv")
        df_gmm = pd.read_csv(data_path + "GMM/" + seq + "_gmm.csv")

        # -1 is outlier, 1 is inlier
        df_bc = df_bc[['position', 'contig', 'Predict']]
        df_bc['Predict'] = np.where(df_bc['Predict'] == -1, 1, 0)
        df_bc.rename(columns={'Predict': 'Predict_BC'}, inplace=True)
        df_signal = df_signal[['position', 'contig','read_index', 'Predict']]
        df_signal['Predict'] = np.where(df_signal['Predict'] == -1, 1, 0)
        df_signal.rename(columns={'Predict': 'Predict_Signal'}, inplace=True)
        df_dwell = df_dwell[['position', 'contig', 'read_index', 'Sequence', 'Predict']]
        df_dwell['Predict'] = np.where(df_dwell['Predict'] == -1, 1, 0)
        df_dwell.loc[(df_dwell['Predict'] == 1) & (df_dwell['Sequence'] != 'G'), "Predict"] = 3
        df_dwell.loc[(df_dwell['Predict'] == 1) & (df_dwell['Sequence'] == 'G'), "Predict"] = 2
        df_dwell.loc[df_dwell['Predict'] == 0, "Predict"] = -2
        df_dwell.rename(columns={'Predict': 'Predict_Dwell'}, inplace=True)
        df_lofd = df_lofd[['position', 'contig', 'read_index', 'Predict']]
        df_lofd['Predict'] = np.where(df_lofd['Predict'] == -1, 1, 0)
        df_lofd.rename(columns={'Predict': 'Predict_Lofs'}, inplace=True)
        df_lofs = df_lofs[['position', 'contig', 'read_index', 'Predict']]
        df_lofs['Predict'] = np.where(df_lofs['Predict'] == -1, 1, 0)
        df_lofs.rename(columns={'Predict': 'Predict_Lofd'}, inplace=True)
        df_gmm['Predict'] = np.where(df_gmm['Predict'] == 1, 1, 0)
        df_gmm.rename(columns={'Predict': 'Predict_Gmm'}, inplace=True)

        df = df_signal.merge(df_dwell, on=['position', 'contig', 'read_index'], how='right')
        df = df.merge(df_lofd, on=['position', 'contig', 'read_index'], how='left')
        df = df.merge(df_lofs, on=['position', 'contig', 'read_index'], how='left')
        df = df.merge(df_gmm, on=['position', 'contig'], how='left')
        df = df.merge(df_bc, on=['position', 'contig'], how='left')
        df.fillna(value=0, inplace=True)


        def mean_read(df):
            df['Reactivity'] = df['Predict_BC'] + df['Predict_Signal'] + df['Predict_Dwell'] + \
                                df['Predict_Lofs'] + df['Predict_Lofd'] + df['Predict_Gmm']
            dfr = df[['position', 'contig', 'read_index', 'Reactivity']]
            dfr['VARNA'] = np.where(dfr['Reactivity'] >=3, 1, 0)
            dfr['Predict'] = np.where(dfr['VARNA'] == 1, -1, 1)
            print("Reactivity Alone")
            mx.get_Metric(dfr, seq + "_mean_")
            df['VARNA'] = np.where(df['Reactivity'] >= 3, 1, 0)
            df['Predict'] = np.where(df['VARNA'] == 1, -1, 1)
            dt = dfs[dfs['contig']== seq]
            df = dt.merge(df, on=['position', 'contig'], how="left")
            dfr.to_csv(save_path + seq + "_reactivity_ranking.csv", index=False)
            df.sort_values(by=['read_index', 'position', 'contig'])
            df.to_csv(save_path + seq + "_reactivity_full.csv", index=False)

        mean_read(df)

        def read_depth(df):
            print("Read_Depth")
            #### aggregate counts of Predict, read_depth,
            #### another decider based on the percentage of modified reads ###
            #print(df['Predict_Gmm'].head(100))
            #print(df['Predict_Lofd'].head(100))
            df['Reactivity'] = df['Predict_BC'] + df['Predict_Signal'] + df['Predict_Dwell'] + \
                               df['Predict_Lofs'] + df['Predict_Lofd'] + df['Predict_Gmm']
            dfr = df[['position', 'contig', 'read_index', 'Reactivity']]
            dfr['VARNA'] = np.where(dfr['Reactivity'] >= 6, 1, 0)
            dfr['Predict'] = np.where(dfr['VARNA'] == 1, -1, 1)
            # decimal reactivities for shape in full set
            dfr_full = dfr
            dfr_full['Reactivity'] = dfr_full['Reactivity']/8
            dfr_full.to_csv(save_path + seq + "_read_depth_full.csv", index=False)
            dfr['Predict'] = dfr['Predict'].astype('category')
            dfr = (dfr.groupby(['position', 'contig', 'Predict'], observed=False).size().
                   unstack(fill_value=0).reset_index())
            dfr['read_depth']  = dfr[-1] + dfr[1]
            dfr['percent_modified'] = dfr[-1]/dfr['read_depth']
            dfr['RNAFold_Shape_Reactivity'] = scale_reactivities(dfr['percent_modified'])
            mean = dfr['percent_modified'].mean()
            std = dfr['percent_modified'].std()
            print("mean ", mean)
            #### predict modification based on percent modified read depth ####
            dfr['Predict'] = np.where(dfr['percent_modified'] > mean + std/3, -1, 1)
            ### adjust prediction with base pairing probabilities
            dfr['position'] = dfr['position'] + 1 #remove 0 indexing for other apps
            dfr = bpp.get_adjusted_probabilities(dfr, seq)
            #print(dfr.head(100))
            dvarna = pd.DataFrame()
            dvarna['position'] = dfr['position']
            #dvarna = dfr[['position', 'Predict']]
            dvarna['VARNA'] = np.where(dfr['Predict'] == -1, 1, 0)
            # pandas BUG can't drop predict index
            # dvarna = dvarna.drop(columns=['Predict'])
            # dvarna = dvarna.reset_index(drop=True)
            # dvarna.set_index('position', inplace=True)
            #correct so varna indexing works properly
            p0 = dvarna['position'].iloc[0]
            if p0 != 0:
                for i in range(0, p0):
                    print(i)
                    dvarna.loc[len(df.index)] = [i, 0]  # adding a row
            dvarna.sort_values(by=['position'], inplace=True)
            dvarna['position'] = dvarna['position'] + 1
            dvarna.to_csv(save_path + seq + "_VARNA.csv", index=False, header=False)
            print("Deconvolution")
            dfr['position'] = dfr['position'] - 1
            mx.get_Metric(dfr, seq)
            dfr.to_csv(save_path + seq + "_read_depth.csv", index=False)
            dvienna = dfr[['position', 'RNAFold_Shape_Reactivity']]
            dvienna['position'] = dvienna['position'] + 1
            dvienna.to_csv(save_path + seq + "_rnafold.dat", index=False, sep='\t', header=False)

        read_depth(df)

get_mods()
sys.exit(0)
