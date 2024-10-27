import sys, re
import platform
import os.path
import traceback
import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/dendrogram/Out/"
    save_path = "/home/jwbear/projects/def-jeromew/jwbear/dendrogram/heatmap/"
else:
    data_path = "/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/Out/"
    save_path = "/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/Figures/"

###### Heatmap of Reads Against Summed Reactivity Across different Metrics ######
def heatmap(seq="HCV"):
    df = pd.read_csv(data_path + seq + "_reactivity_full.csv")
    df = df[['position', 'contig', 'read_index', 'Reactivity']]
    df = df.groupby(by=['position', 'contig', 'read_index']).mean().reset_index()
    result = df.pivot(index="position", columns="read_index", values="Reactivity")
    fig = plt.gcf()
    plt.figure(figsize=(60, 60))
    plt.title("Deconvolution " + seq + " Reactivity Across Reads")
    seq_len = len(df['position'].unique())
    ax = sns.heatmap(result, cmap='crest',
                      yticklabels=range(0,seq_len))
    plt.show()
    plt.savefig(save_path + seq + '_heatmap.png', dpi=300)


if __name__ == "__main__":
    i = int(sys.argv[1])  # get the value of the $SLURM_ARRAY_TASK_ID
    sequences = ['RNAse_P', "cen_3'utr", "cen_3'utr_complex", 'cen_FL', 'cen_FL_complex',
                 "ik2_3'utr_complex", 'ik2_FL_complex', 'T_thermophila', 'ik2_FL', 'HCV',
                 "ik2_3'utr"]
    heatmap(seq=sequences[i])

sys.exit(0)
