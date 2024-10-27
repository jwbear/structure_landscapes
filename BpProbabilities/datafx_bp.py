import itertools
import sys, re
import os.path
import platform
import traceback
import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
import DashML.Deconvolution.BpProbabilities.BasePairProbabilities as bp
import DashML.Deconvolution.BpProbabilities.library as lib

#### TODO: readd kmodes columns
#### needed to remove missing data and much easier to calculate interactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# get sequence lengths
alen = lib.get_seqlen("cen_3'utr")
blen = lib.get_seqlen("ik2_3'utr")

#### modified reactivities
def get_centroids():
    #TODO df_a, df_b missing from server run
    df_A = pd.read_csv("/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/BpProbabilities/"
                        "Centroids/cen_3'utr_centroids.csv")
    df_B = pd.read_csv("/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/BpProbabilities/"
                        "Centroids/ik2_3'utr_centroids.csv")
    df_AB = pd.read_csv("/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/BpProbabilities/"
                       "Centroids/cen_3'utr_complex_centroids.csv")
    df_BA = pd.read_csv("/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/BpProbabilities/"
                       "Centroids/ik2_3'utr_complex_centroids.csv")

    ### create position numbers in clusters
    def get_positions(df, seqlen):
        position = np.array(np.arange(1,seqlen+1, dtype=int))
        clust_num= len(df['cluster'].unique())

        i = 0
        p1 = []
        while (i < clust_num):
            p1.append(position)
            i = i + 1
        return np.array(p1).flatten()

    #format for existing function, remove excess clusters
    df_A['position'] = get_positions(df_A, alen)
    df_A.rename(columns={'centroid':'Predict'}, inplace=True)
    #df_A['Predict'] = np.where(df_A['Predict'] >= .75, -1, 1)
    df_A = df_A.loc[df_A['cluster']<3]
    #print(df_A.tail())
    #print(df_A['cluster'].unique())
    df_B['position'] = get_positions(df_B, blen)
    df_B.rename(columns={'centroid': 'Predict'}, inplace=True)
    #df_B['Predict'] = np.where(df_B['Predict'] >= .75, -1, 1)
    df_B = df_B.loc[df_B['cluster'] < 3]
    #print(df_B.tail())
    df_AB['position'] = get_positions(df_AB, alen)
    df_AB.rename(columns={'centroid': 'Predict'}, inplace=True)
    df_AB['Predict'] = np.where(df_AB['Predict'] >= .75, -1, 1)
    df_AB = df_AB.loc[df_AB['cluster'] < 3]
    #print(df_A.tail())
    df_BA['position'] = get_positions(df_BA, blen)
    df_BA.rename(columns={'centroid': 'Predict'}, inplace=True)
    df_BA['Predict'] = np.where(df_BA['Predict'] >= .75, -1, 1)
    df_BA = df_BA.loc[df_BA['cluster'] < 3]
    #print(df_B.tail())
    return df_A, df_B, df_AB, df_BA
