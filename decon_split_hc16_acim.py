import sys, re
import os.path
import traceback
import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt


### split large df for faster processing ###
### process nanopolish data by read for deconvolution ###
def preprocess_df(df, seq, i, rn):
    # standardization of means from signal file for each read
    #### normalize for each read ####
    dfn = pd.DataFrame()
    reads = rn
    for read in reads:
        dft = df.loc[(df['read_index']==read) & (df['contig']==seq)]
        #### standardize means #####
        mod_mean = dft['event_level_mean'].mean()
        mod_std = dft['event_level_mean'].std()
        dft['event_level_mean'] = (dft['event_level_mean'] - mod_mean) / mod_std

        dwell_mean = dft['event_length'].mean()
        dwell_std = dft['event_length'].std()
        dft['event_length'] = (dft['event_length'] - dwell_mean) / dwell_std

        mod_mean = dft['event_stdv'].mean()
        mod_std = dft['event_stdv'].std()
        dft['event_stdv'] = (dft['event_stdv'] - mod_mean) / mod_std

        dft = dft.sort_values(by=['contig', 'read_index', 'position'])
        dfn = pd.concat([dfn, dft], ignore_index=True)

    dfn.to_csv("/home/jwbear/projects/def-jeromew/jwbear/decon_ACIM/" +seq+ "_"+
               str(i) + "_decon_signal_preprocess.csv")


    return


if __name__ == "__main__":
    i = int(sys.argv[1])  # get the value of the $SLURM_ARRAY_TASK_ID
    #sequences = ['RNAse_P', "cen_3'utr", 'cen_FL','T_thermophila', 'ik2_FL', 'HCV', "ik2_3'utr"]
    #sequences = ['RNAse_P']
    df = pd.read_csv("/home/jwbear/projects/def-jeromew/jwbear/temp/ACIM_hc16.txt", sep='\t')
    df = df.loc[df['contig'] == 'hc16_ligase']
    reads = df.loc[df['contig']=='hc16_ligase','read_index'].unique()
    read_split = np.array_split(reads, 20)
    ####### Collapse Multiple Events on Same Position and Sequence ###########
    ##### remove bad kmer reads ##########
    df = df.dropna()

    preprocess_df(df=df, seq='hc16_ligase', i=i, rn=read_split[i])

sys.exit(0)

