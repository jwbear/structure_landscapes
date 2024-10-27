import sys, re
import os.path
import platform
import traceback
import numpy as np
import pandas as pd
import math
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


if platform.system() == 'Linux':
    ##### server #####
    dash_path = "/home/jwbear/projects/def-jeromew/jwbear/Deconvolution/"
else:
    dash_path = "/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/"
# reverse positions to 5'-3' from nanopolish
def reverse_positions(df):
    #print(self.reference_sequences)
    reference_sequences = parse_fasta()
    for contig in df['contig'].unique():
        seqlen = len(reference_sequences[contig])
        df.loc[df['contig'] == contig, 'position'] = seqlen - df['position']
    return df

#add complex tag to uncomplexed dmso data for comparison by name in analysis
def add_complex_dmso():
    complexes = ["cen_3'utr", "cen_FL", "ik2_3'utr", "ik2_FL"]
    df = pd.read_csv(dash_path + "Preprocess/DMSO_signal_preprocess_aligned.csv")
    df = df.drop(columns=['Unnamed: 0'])
    #print(df.columns)

    #signal data
    for c in complexes:
        dft = df[df['contig']==c]
        dft['contig'] =  c + "_complex"
        df = pd.concat([df,dft])

    #df.to_csv(dash_path + "Preprocess/DMSO_signal_preprocess_aligned.csv")

    #basecall data
    df = pd.read_csv(dash_path + "Basecall/DMSO_mod_rates_aligned.csv")
    df = df.drop(columns=['Unnamed: 0'])
    #print(df.columns)

    # signal data
    for c in complexes:
        dft = df[df['contig'] == c]
        dft['contig'] = c + "_complex"
        df = pd.concat([df, dft])

    print(df.head())
    df.to_csv(dash_path + "Basecall/DMSO_mod_rates_aligned.csv")


#check event level meean in single read
def ave_signal_by_read():
    #df = pd.read_csv(dash_path + "Deconvolution/Preprocess/ACIM_signal.txt", sep='\t')
    df = pd.read_csv(dash_path + 'Deconvolution/HCV_aligned.csv')
    df = df[(df['read_index']==12918) & (df['contig']=='HCV')]

    # mean of means from parse file
    mod_mean = df['event_level_mean'].mean()
    mod_std = df['event_level_mean'].std()
    df['event_level_mean'] = (df['event_level_mean'] - mod_mean) / mod_std

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df[['position','contig', 'reference_kmer', 'event_level_mean']]
    df.sort_values(by=['reference_kmer'], inplace=True)
    print(df)



# align by position and read for deconvolution
#align positiions from nanopore model to reference sequence
#this misalignment causes the 1-2 base shift in signal detection
#TODO: add remaining files
def align_positions_by_read():
    df = pd.read_csv(dash_path + "Deconvolution/HCV.csv")
    cols = ['position', 'Sequence', 'contig', 'reference_kmer','read_index',
            'event_level_mean','event_length','event_stdv','sequence']
    df = df[cols]
    # df['reference_kmer'].fillna(value=df['Sequence'], inplace=True)
    # df['sequence'] = df['reference_kmer'].astype(str).str[0]
    #correct gaps in sequence
    #save shifts for basecall alignment
    shift = 0
    contig = 'HCV'
    df_final = pd.DataFrame()
    df_shift = pd.DataFrame(columns=['contig', 'read_index', 'shift'])
    reads = df['read_index'].unique()
    for read in reads:
        dfr = df[df['read_index']==read]
        dfr.reset_index(drop=True, inplace=True)
        #fill in blanks with np.nan
        dfr['sequence'].fillna(value=np.nan)
        i = len(dfr)
        dtemp = dfr
        #rotate reference kmer sequence until aligned with reference sequence
        while not (dtemp['sequence'].equals(dtemp['Sequence'])) and (i>=0):
            dt = dfr[['read_index', 'reference_kmer', 'event_stdv', 'event_level_mean', 'event_length', 'sequence']]
            dt = dt.shift(periods=1, fill_value=np.nan)
            dfr.loc[:,['read_index', 'reference_kmer', 'event_stdv',
           'event_level_mean', 'event_length', 'sequence']] = dt
            # fill in blanks with np.nan, so not counted in alignment
            dtemp = dfr
            dtemp['sequence'].fillna(value=np.nan, inplace=True)
            dtemp = dtemp.dropna()
            i = i - 1   # sequence length
            #record shift position
            shift = dtemp['position'].iloc[0]
        df_final = pd.concat([df_final, dfr], ignore_index=True)
        #df.loc[len(df.index)] = ['Amy', 89, 93]
        df_shift.loc[len(df_shift.index)]=[contig,read,shift]
    df_final.to_csv(dash_path + "Deconvolution/" + contig + "_aligned.csv", index=False)
    df_shift.to_csv(dash_path + "Deconvolution/" + contig + "_shifts_0.csv", index=False)
    return df


#align positiions from nanopore model to reference sequence
#this misalignment causes the 1-2 base shift in signal detection
def align_positions(df):
    # avoid nans in sequence list
    sequences = df['contig'].unique()
    df['sequence'] = df['reference_kmer'].str[2]
    dfs = get_structure_ext()
    dfs = dfs[['Position','Sequence', 'Sequence_Name']]
    dfs = dfs.rename(columns={'Position': 'position', 'Sequence_Name': 'contig'})
    #retrieve representation of putative sequences not in structure file
    dfs = pd.concat([dfs, structuresforputativeseqs()], ignore_index=True)
    df = dfs.merge(df, on=['position','contig'], how="left")
    df = df[['position', 'Sequence', 'contig', 'reference_kmer', 'event_stdv',
             'event_level_mean', 'event_length', 'sequence']]
    df_final = pd.DataFrame()
    shift = {}
    for contig in sequences:
        print(contig)
        dft = df[df['contig'] == contig]
        dft['contig'].fillna(contig, inplace=True)
        #dft.set_index(['position', 'Sequence'])
        dft.reset_index(drop=True, inplace=True)
        i = len(dft)
        #rotate reference kmer sequence until aligned with reference sequence
        while not (dft['sequence'].equals(dft['Sequence'])) and (i>=0):
            #print(dft['sequence'].head())
            #print(dft['Sequence'].head())
            dt = dft[['reference_kmer', 'event_stdv', 'event_level_mean', 'event_length', 'sequence']]
            dt = dt.shift(periods=1, fill_value=np.nan)
            dft.loc[:,['reference_kmer', 'event_stdv',
           'event_level_mean', 'event_length', 'sequence']] = dt
            #print(dt.head())
            #print(dft.head())
            dft = dft.dropna()
            i = i - 1   # sequence length
            #record shift position
            shift[contig] = dft['position'].iloc[0]

        print(dft.head())
        df_final = pd.concat([df_final, dft], ignore_index=True)
    df = df_final
    print(shift)
    return df, shift


# accepts shifts calculated from signal data
def align_positions_bc(df, shift):
    # avoid nans in sequence list
    sequences = df['contig'].unique()
    df_final = pd.DataFrame()
    for contig in sequences:
        print(contig)
        dft = df[df['contig'] == contig]
        # dft.set_index(['position', 'Sequence'])
        dft.reset_index(drop=True, inplace=True)
        # rotate data by reference kmer shift in signal file
        dt = dft[['Basecall_Reactivity', 'Quality', 'Mismatch',
       'Deletion', 'Insertion', 'Aligned_Reads']]
        print(dt.columns)
        dt = dt.shift(periods=shift[contig], fill_value=np.nan)
        vals = {'Basecall_Reactivity':-1, 'Quality':0, 'Mismatch':-1,
       'Deletion':-1, 'Insertion':-1, 'Aligned_Reads':0}
        dt.fillna(vals, inplace=True)
        dft.loc[:, ['Basecall_Reactivity', 'Quality', 'Mismatch',
       'Deletion', 'Insertion', 'Aligned_Reads']] = dt
        print(dt.head())
        print(dft.head())
        print(dft.head())
        df_final = pd.concat([df_final, dft], ignore_index=True)
    df = df_final
    print(shift)
    return df

def unit_vector_norm(x):
    # unit vector normalization
    x = np.where(x != 0, ((x- np.min(x))/ (np.max(x)- np.min(x))), 0)
    #x = stats.zscore(x)
    return x

def structuresforputativeseqs():
    cen3 = "ACTTGTTTAGAGAATGTAAATAAGCAATTAAACAGTGCATTCTAGCCATAGGGCATTCTACCATTTTTAAATTGTGTGTGCCATGCAGTCTAGTCCGCTTTTTCATGTATAGACAGTTAAATAACAATAACTAAATAACTATAATCGGAAATTTAATTTTATTTCAGCATGATAAAATAAATAATTTAATGACCTACAG"
    ik23 = "ACGGGCATATCATGAAAGTGCAAGAATATTTTATTTGCCTTTACTTTGTAAGTTAACAATAAATGTTTACTTTTTTATATCTGAATTTGTAAAGCAACTACATATATTCCTATTGAAACTTGGCTGAATATCGTGAAAGAGTAAGATTTCTGTAGGTCATTAAATTATTTATTTTATCATGCTGAAATAAAATTAAATTTCCGATTAT"
    cenfl = "TTATAAATTTATTTCCGGTTCTGCTAGGAAGTTCATTTCGGTTCTTTGTCTGCAGGTCGATTGTGTATAATAAAAAGCATTAATTAACTATTCGTACATGCAGTTGCGCCCTAGTACGTAGGGCTTTTTCTGCACTCGCTGCAATACCATACTTTTTGTTTTTGGGGATGAAACAGTAAAAAAGTTACTGTTGTAGCAGTTTTCCACCATTGAGAGGACATTTGCCAAAGTTCGCATCAAGCCCAAGCTTAGAAAGGTCCAAGTCCAAGATGGAGGAATCCAATCACGGTTCGGCTGGCTGTGAAAACGTATCGCAGTTCATGCTCGATGACCTACAATTGGCAGCAGAGCTGGGAAAAACGCTCCTGGAACGCAATAAGGAGCTGGAAACCTTCATCAAGGAGTACAAGATTAAGGGGGATGAGCAGGAACTGGAGATCTTGCATCTGCGCAAGCACATTAATGCGATGACCGAGGTGAATGATTCACGGCTTAAGGTCTACGAGCAGCTGGAAGTGGGCATTCAGGATCTGGAGCGTGCCAATCAGCGGCTAAATCTGGAGAAGAATCGGGACAAGAAGCAGATTAAGACGCTGACCACCAACACGGAAGTCCTGGAAGCCCGCTGCGAGGAACTGAGTCAACTTCTGAGCGATGCTCGGCAATCCCTGAGCACCGAGCGGCGGAAGGTGGATCAGTACCAGCAGGAACGCTATAGGATGCAGCATTCGACCGAGGGTTCTGTGTCCAGTCACAGCATTCAATCGCTGTGCAAGGAGCAGAGTGTTGAGTTCTCCAAACTAGATGTTATGGCCATAGCGAACTCGACAGGATTGGAGGATATCTCCTTCAGTAATGCCACCATGTGCGAGAGGACCGCCGTTAAGGGTGAAGACAACGAGGAATTGGTAAAGCTGCTCAGCGAAATGGAGGTGTTAAAACGTGACTTCTTGGCCGAACAGCAGCGGTGCACCGAGCTGGAGGAGCAGTTGGTGACCATAATCCAAGACAACCAAGGTCTGCAGACACGTCTGCTGGAGAACAGTGCCAATGAGGGAACAATGTCGATGCACGAGGAGTTCAGCCTCCTGGACGACGTGAGACAAGGCCAAATGTGTAGTCGCTGCCTGAGGGATATTAACGAAAGCAACACCAATATGGATGATCAATCCTCCATTGCCCCAACCGAAGAAATCTACGAAGATGACGATCGCAGCATACTAAGCGAAAGCACCTCGAAATGCGATAATAGTGGTGCGGATTATAAGGAACGCTTCCGGATTCCCGAGGACCTCAATCCCAACTCTAGCGACAAACCAAATCCTTATCGGGACCTAGTCGAGAAATATGAAGCTCTGGTGGAGGTGAAGAGAACCTCGAATGCGGTCAAGAGCAACTTCACCAGCAATCCAGATGGGAAGACCATGACCGAGTCCAGTCAAGGCAAAAAATCAGAAACAATTGTTAACAGCTCAAAGGAATCTGACCTTATGTTGGATTCAACACGTAAGCGCACTCCAACCGAGTTTTCGGAATCAGAAACTACATCGTCGGGATTCTCGGATGAAACTAGCAACAAGTCTACGCAAACTGACGAGCGACCGAGTTACTTCCTTTGCTCAATAAGTAATGGAAACGATTGTAAATTCAGTATTTATGACGACGTCAGCCCCATTGAATCTCATTTCCGTAATCGCCCGGAGTACAGGGAACTCTTCAAGGAGATATTTGGAGTGCTCAAGAAGGCAGCTGATAATAACGAAGAAGACAAGCTTCCGTCCCTGCATGACGACGCACAGATTACAGAAAAATTACCCTTGGTGGCGGCCAAAGTACCCCCGGTAACACCCGAGAGGGAGGAATCTCCAGATGACTTCATTGACGACACACAGAGCGTCGTATCCTCAGTTATATCCAACCAGTCCATTGCAATGTCCGAGTGCGTTACCAAACTGGAGCGGAAGACAGCCAAAAAACACATCTTTGATGTCCGAAACCAACAAAACCAATCGTCCCTAACGCAATCCACTTTGTTAAACAGCTCCTCGAAGGAAACAACTGTCGGCGAACCGAATTATAAACCCATAAGGGAGAATGGCCAGATACTTACTCCACTCAAGCGCGAACCACTCGAATATCTAACCGTAGGTGTCGGAATCAAGAAGAAGAACCGACGAAAGCACCGAAACCAAAGCTCATCTGGAGATCGCATCGAATTGTTTAACTCCCGGGAATTTACTCCCAGGAACAGTCCTCTGGCCATGAACAACCGTGGTGGTGGACAAGGGAGCTCAAAGATGTGTACGGATACATTGAATGCTGAATTCGGACGCAGTAACCGCAGACGGACAACTCCATCCTCAAGCAACTGGAATGGCTCTCCCATGGTGATCTATAACAAGAATATGAATACTCCCCAAACTTCTCGGGGCCGAGTAATAGAGCTCAATGGCGTCGAGTTTTATCATAACACGGTATCGCAAGACTTACATAAACTTAAAAAGTTAGACTTATCCTACGCCGAAGTTTTGCGCCGGGCCGATGCCGGCGAACATGGACCCACAAGATCACATTCCCAGCGACAGCAACACAACGGAGCCAACATTCGCAAGAGTCATCATCATCAGTTTCGTCAAAAGTAAACTTGTTTAGAGAATGTAAATAAGCAATTAAACAGTGCATTCTAGCCATAGGGCATTCTACCATTTTTAAATTGTGTGTGCCATGCAGTCTAGTCCGCTTTTTCATGTATAGACAGTTAAATAACAATAACTAAATAACTATAATCGGAAATTTAATTTTATTTCAGCATGATAAAATAAATAATTTAATGACCTACAG"
    ik2fl = "CCTTCCTTACCAGTGTTTTTGTTTGTCACAATCGAGAAGGCGCTTGGAGTTATATCAAAAAATAAAAAATAGGAGGAAAACAGCCGCGAAATGTCCTTCCTGCGCGGTTCCGTGAGCTATGTGTGGTGCACCACCAGCGTCCTGGGAAAGGGAGCCACCGGTTCCGTGTTCCAGGGGGTCAACAAGATCACCGGCGAATCGGTGGCGGTGAAGACCTTTAATCCCTACAGTCACATGCGACCGGCTGATGTGCAGATGCGGGAGTTCGAGGCCCTGAAAAAGGTCAACCACGAGAATATAGTAAAGCTGTTGGCGATCGAGGAGGATCAAGAGGGGCGTGGTAAGGTGATCGTGATGGAGCTCTGCACAGGCGGAAGTCTCTTTAACATCCTGGACGATCCTGAGAACTCGTACGGTCTGCCGGAACACGAGTTCCTGCTGGTCTTGGAACACTTGTGCGCCGGAATGAAGCACTTGCGGGATAACAAGCTGGTGCATCGCGATCTGAAGCCCGGAAACATAATGAAGTTCATCTCGGAGGACGGGCAAACCATATACAAGCTTACTGATTTCGGTGCTGCTAGAGAACTGGAGGATAATCAGCCGTTTGCCTCTCTATACGGCACAGAAGAGTATCTTCATCCCGATCTCTACGAGCGCGCTGTGCTGAGGAAGTCAATCCAGCGATCGTTCACCGCCAATGTGGATTTGTGGTCGATTGGGGTCACGCTCTACCATGTGGCCACCGGAAATCTGCCTTTTAGACCTTTTGGTGGAAGGAAAAACCGAGAGACCATGCACCAAATCACTACCAAAAAGGCTTCTGGGGTGATTTCCGGCACTCAGCTGTCCGAGAATGGACCAATTGAATGGTCCACCACGTTGCCACCGCACGCCCATCTCTCGCAGGGACTGAAAACCCTGGTGACGCCTCTTCTAGCTGGACTTCTCGAGGAGAATAGGGAAAAGACCTGGTCTTTCGATCGTTTCTTCCACGAGGTGACGCTCATCCTCCGCAAGCGTGTCATTCATGTGTTCTTCACCAATCGGACTAGTTCGGTGGAAGTGTTCCTAGAGCCAGACGAGCAAATTGACAACTTCCGAGAGCGTATTTTCCTGCAAACAGAGGTGCCGCTGGAGAAGCAGATCCTTTTGTTCAACAACGAGCATCTGGAGAAGAAGGTTACTCCACGAACGATAGCTAAAGCATTCCCTGCCACCACAACAGATCAGCCAATATTCCTCTACAGCAACGACGACAACAATGTTCAGTTGCCCCAGCAATTAGATCTCCCCAAGTTCCCAGTGTTCCCTCCCAATGTTTCAGTGGAGAACGACGCCAGTCTGGCAAAGTCCGCTTGCAGCGTTGGCCATGAATGCAAAAGACGCGTGGACATCTTTACTTCCATGGATATCCTTATTAAGAAGGGAGTGGAGCACTTCATAGAGATGCTAGTGACCACAATAACTCTACTATTGAAAAAAACTGAGAGTTTTGACAACTTGCTTTCCACTGTGATTGATTATGCAGATGTGGTGCACAGCATGGCCAGAGTGACTAAAGGAGATCAGGAGATAAAGACCCTGCTGACTGCCTTGGAAAATGTCAAAAGCGACTTTGATGGAGCCGCGGATGTGATCTCGCAGATGCACAAACATTTTGTGATAGATGACGAACTTAACGATCAATGGACCTCCTCCATGCACGGAAAGAAGTGTCCTTGCAAGACCAGAGCCAGTGCCCAGGCCAAGTATCTGGTAGAAAGGCTGCGGGACTCCTGGCAGCACTTGCTCCGGGATCGTGCAACGCGCACACTGACCTACAACGACGAACAGTTTCATGCCCTGGAGAAGATTAAAGTGGATCACAATGGCAAACGGATAAAGGCCTTGCTTTTGGATAACGTAAATCCGACAGTGGCACAAATCGCAGAGTGCCTGGCGGACTGGTATAAGTTGGCTCAGACCGTCTACCTTAAAACTCAAATTTTGGAGAAGGACGTGCGCGATTGCGAAAGAAAGTTGAACGGAATACGCGATGAGTTGTATCACGTTAAGTCGGAGCTGAAGCTGGATGTGGACACAAAGACCATAAACAACAATAATCAGCTGGCCAAGATAGAGGAGCGGAATCGGCTAAGGGTCATGCAGCAGCAGCAACAGGAAGTTATGGCTGTCATGAGAACGAACAGCGATATAATAAGTTTGCTTAGCAAACTGGGCATTACCAACGGAAGTCTGGAAAGTAGTTAGACGGGCATATCATGAAAGTGCAAGAATATTTTATTTGCCTTTACTTTGTAAGTTAACAATAAATGTTTACTTTTTTATATCTGAATTTGTAAAGCAACTACATATATTCCTATTGAAACTTGGCTGAATATCGTGAAAGAGTAAGATTTCTGTAGGTCATTAAATTATTTATTTTATCATGCTGAAATAAAATTAAATTTCCGATTAT"
    HSP70_HSPA1A = "ACCAGACGCTGACAGCTACTCAGAACCAAATCTGGTTCCATCCAGAGACAAGCGAAGACAAGAGAAGCAGAGCGAGCGGCGCGTTCCCGATCCTCGGCCAGGACCAGCCTTCCCCAGAGCATCCCTGCCGCGGAGCGCAACCTTCCCAGGAGCATCCCTGCCGCGGAGCGCAACTTTCCCCGGAGCATCCACGCCGCGGAGCGCAGCCTTCCAGAAGCAGAGCGCGGCGCCATGGCCAAGAACACGGCGATCGGCATCGACCTGGGCACCACCTACTCGTGCGTGGGCGTGTTCCAGCACGGCAAGGTGGAGATCATCGCCAACGACCAGGGCAACCGCACGACCCCCAGCTACGTGGCCTTCACCGACACCGAGCGCCTCATCGGAGACGCCGCCAAGAACCAGGTGGCGCTGAACCCGCAGAACACCGTGTTCGACGCGAAGCGGCTGATCGGCCGCAAGTTCGGCGATGCGGTGGTGCAGTCCGACATGAAGCACTGGCCCTTCCAGGTGGTGAACGACGGCGACAAGCCCAAGGTGCAGGTGAACTACAAGGGCGAGAGCCGGTCGTTCTTCCCGGAGGAGATCTCGTCCATGGTGCTGACGAAGATGAAGGAGATCGCTGAGGCGTACCTGGGCCACCCGGTGACCAACGCGGTGATCACGGTGCCCGCCTACTTCAACGACTCTCAGCGGCAGGCCACCAAGGACGCGGGCGTGATCGCCGGTCTAAACGTGCTGCGGATCATCAACGAGCCCACGGCGGCCGCCATCGCCTACGGGCTGGACCGGACCGGCAAGGGCGAGCGCAACGTGCTCATCTTCGACCTGGGGGGCGGCACGTTCGACGTGTCCATCCTGACGATCGACGACGGCATCTTCGAGGTGAAGGCCACGGCGGGCGACACGCACCTGGGAGGGGAGGACTTCGACAACCGGCTGGTGAGCCACTTCGTGGAGGAGTTCAAGAGGAAGCACAAGAAGGACATCAGCCAGAACAAGCGCGCGGTGCGGCGGCTGCGCACTGCGTGTGAGAGGGCCAAGAGGACGCTGTCGTCCAGCACCCAGGCCAGCCTGGAGATCGACTCTCTGTTCGAGGGCATCGACTTCTACACATCCATCACGCGGGCGCGGTTCGAAGAGCTGTGCTCAGACCTGTTCCGCGGCACGCTGGAGCCCGTGGAGAAGGCCCTGCGCGACGCCAAGATGGACAAGGCGCAGATCCACGACCTGGTGCTGGTGGGCGGCTCGACGCGCATCCCCAAGGTGCAGAAGCTGCTGCAGGACTTCTTCAACGGGCGCGACCTGAACAAGAGCATCAACCCGGACGAGGCGGTGGCCTACGGGGCGGCGGTGCAGGCGGCCATCCTGATGGGGGACAAGTCGGAGAACGTGCAGGACCTGCTGCTGCTGGACGTGGCGCCGCTGTCGCTGGGCCTGGAGACTGCGGGCGGCGTGATGACGGCGCTCATCAAGCGCAACTCCACCATCCCCACCAAGCAGACGCAGACCTTCACCACCTACTCGGACAACCAGCCCGGGGTGCTGATCCAGGTGTACGAGGGCGAGAGGGCCATGACGCGCGACAACAACCTGCTGGGGCGCTTCGAACTGAGCGGCATCCCGCCGGCGCCCAGGGGCGTGCCACAGATCGAGGTGACCTTCGACATCGACGCCAACGGCATCCTGAACGTCACGGCCACCGACAAGAGCACCGGCAAGGCCAACAAGATCACCATCACCAACGACAAGGGCCGCCTGAGCAAGGAGGAGATCGAGCGCATGGTGCAGGAGGCCGAGCGCTACAAGGCCGAGGACGAGGTGCAGCGCGACAGGGTGGCCGCCAAGAACGCGCTCGAATCCTATGCCTTCAACATGAAGAGCGCCGTGGAGGACGAGGGTCTCAAGGGCAAGCTCAGCGAGGCTGACAAGAAGAAGGTGCTGGACAAGTGCCAGGAGGTCATCTCCTGGCTGGACTCCAACACGCTGGCCGACAAGGAGGAGTTCGTGCACAAGCGGGAGGAGCTGGAGCGGGTGTGCAGCCCCATCATCAGTGGGCTGTACCAGGGTGCGGGTGCTCCTGGGGCTGGGGGCTTCGGGGCCCAGGCGCCCAAGGGAGCCTCTGGCTCAGGACCCACCATCGAGGAGGTGGATTAGAGGCCTCTGCTGGCTCTCCCGGTGCTGGCTAGGAGACAGATATGTGGCCTTGAGGACTGTCATTATTTCAAGTTTAGTACTTCACTCCTTAGTTTGTCCTGCAATCAAGTCCTAGACTTAGGGAAACTAAACTGTCTTTCAGTTACTTTGTGTATTGCACGTGGGCTTTATCTTCCCTGTTAATTAACACTGCAAGTGTGTCTTTGTAAATATAAATAAATAAGTATATATATTCTTCAATTCAGCACTGCCCCGCTGATGTGATTTGTTTTGCAGGACAGCCAAAGCTATGTAGAGAGATATTCTGTATCAGAATACACAAAGAGACAGAGATATGTTATGAAAACATCAGGAGACTGTTGAGTTCTTTGTGTTTGGACTCTCCCCTGGGCCACATTGTTGATACATGCTTGTGTCGGGTCCTTCAGAGGCCAGGGCTGGATTACTGACAGCGGAGACTCTGCTGCTTCTCCTTGCGTTTATAATCTTGCATGGTGGTTGCACTGTAGGACTTGTTTCCAGGTTGGTGAACTTGGAGGTGAAGTGACAGCACCAGCATGTGTTCAGTTTTTACACAACCATCCTGAACTCGGGTCAATTTTTACCGGTCATTTGAAAATAAACTTCAAAATCACTTGCCA"

    data = [["cen_3'utr", np.array(list(cen3)), np.arange(0, len(cen3))],
            ["cen_3'utr_complex", np.array(list(cen3)), np.arange(0, len(cen3))],
            ["ik2_3'utr", np.array(list(ik23)), np.arange(0, len(ik23))],
            ["ik2_3'utr_complex", np.array(list(ik23)), np.arange(0, len(ik23))],
            ["cen_FL", np.array(list(cenfl)), np.arange(0, len(cenfl))],
            ["cen_FL_complex", np.array(list(cenfl)), np.arange(0, len(cenfl))],
            ["ik2_FL", np.array(list(ik2fl)), np.arange(0, len(ik2fl))],
            ["ik2_FL_complex", np.array(list(ik2fl)), np.arange(0, len(ik2fl))],
            ["HSP70_HSPA1A", np.array(list(HSP70_HSPA1A)), np.arange(0, len(HSP70_HSPA1A))]]
    #print(data)
    df = pd.DataFrame(columns=['contig', 'Sequence', 'position'])
    for d in data:
        dt = pd.DataFrame()
        dt['Sequence'] = d[1]
        dt['position'] = d[2]
        dt['contig'] = d[0]
        df = pd.concat([df,dt], ignore_index=True)
    return df


def get_shape():
    # shape files
    reactivities = {"T_thermophila": dash_path + "Shape/tetra_reactivities.txt",
                    "E_coli_tmRNA": dash_path + "Shape/ecoli_reactivities.txt",
                    "hc16_ligase": dash_path + "Shape/hc16_reactivities.txt",
                    "RNAse_P": dash_path + "Shape/RNAseP_reactivities.txt",
                    "HCV": dash_path + "Shape/hcv_reactivities.txt",
                    "cen_3'utr": dash_path + "Shape/cen-utr_reactivities.txt",
                    "ik2_3'utr": dash_path + "Shape/ik2-utr_reactivities.txt",
                    "cen_3'utr_complex": dash_path + "Shape/cen-ik2-utr_reactivities.txt",
                    "ik2_3'utr_complex": dash_path + "Shape/ik2-cen-utr_reactivities.txt",
                    "cen_FL_complex": dash_path + "Shape/cen-ik2-fl_reactivities.txt",
                    "ik2_FL_complex": dash_path + "Shape/ik2-cen-fl_reactivities.txt",
                    "cen_FL": dash_path + "Shape/cen-fl_reactivities.txt",
                    "ik2_FL": dash_path + "Shape/ik2-fl_reactivities.txt",
                    "FMN_Adaptor": dash_path + "Shape/FMN_reactivities.txt",
                    "Lysine_Adaptor": dash_path + "Shape/Lysine_reactivities.txt"
                    }

    df_shape = pd.DataFrame()
    for s, f in reactivities.items():
        df1 = pd.read_csv(f, names=['Position', 'Sequence', 'Reactivity_shape'])
        df1['Position'] = df1['Position'] - 1
        if "complex" in f:
            df1['Sequence_Name'] = s + '_complex'
        else:
            df1['Sequence_Name'] = s
        df_shape = pd.concat([df_shape, df1], ignore_index=True)

    df_shape["Reactivity_shape"] = np.where(df_shape["Reactivity_shape"] == -1, 0, df_shape["Reactivity_shape"])
    #df_shape["Reactivity_shape"] = unit_vector_norm(df_shape, "Reactivity_shape")
    #based on QuSHAPE paper defining range of reactivities: .4 < med, .85 < high
    shape_outlier = .4
    df_shape["Predicted_Shape"] = df_shape["Reactivity_shape"]
    df_shape["Predicted_Shape"] = np.where(df_shape["Predicted_Shape"] >= shape_outlier, -1,
                                            df_shape["Predicted_Shape"])
    df_shape["Predicted_Shape"] = np.where((df_shape["Predicted_Shape"] > 0) &
                                            (df_shape["Predicted_Shape"] < shape_outlier),
                                           1, df_shape["Predicted_Shape"])
    #df_shape["Predicted_Shape"] = np.where(df_shape["Reactivity_shape"] >= shape_outlier, -1, 1)
    #df_shape["Predicted_Shape"] = np.where(df_shape["Reactivity_shape"] == 0, 0, df_shape["Predicted_Shape"])
    #df_shape.to_csv("shape_data.csv")
    return df_shape

def get_shapemap():
    # shape files
    reactivities =  { "T_thermophila": dash_path + "ShapeMap/SepRep_ttRz_profile.txt",
     "E_coli_tmRNA" : dash_path + "ShapeMap/SepRep_tmRNA_profile.txt",
     "HCV" : dash_path + "ShapeMap/SepRep_HCV_IRES_profile.txt",
     "RNAse_P": dash_path + "ShapeMap/SepRep_RNAseP_profile.txt",
     "hc16_ligase": dash_path + "ShapeMap/SepRep_hc16_profile.txt"
    }

    # dft = pd.read_csv("ShapeMap/SepRep_ttRz_profile.txt",sep="\t")
    # print(dft.columns)
    # #remove ligands lowercase sequence has no values???
    # #dft = dft.dropna()
    # #dft['Sequence_Case'] = list(map(lambda x: x.islower(), dft['Sequence']))
    # #dft = dft[dft['Sequence_Case']==True]
    # dft['Sequence'] = dft['Sequence'].str.upper().str.replace("U", "T")
    # print(dft.head(300))
    # print(dft['Reactivity_profile'])
    df_shape = pd.DataFrame()
    for s, f in reactivities.items():
        df1 = pd.read_csv(f, usecols=['Nucleotide', 'Sequence', 'Reactivity_profile'], sep="\t")
        df1['Sequence'] = df1['Sequence'].str.upper().str.replace("U", "T")
        df1 = df1.rename(columns={'Nucleotide': 'Position'})
        df1['Position'] = df1['Position'].astype(int) - 1
        if "complex" in f:
            df1['Sequence_Name'] = s + '_complex'
        else:
            df1['Sequence_Name'] = s
        df_shape = pd.concat([df_shape, df1], ignore_index=True)

    df_shape['Reactivity_profile'] = df_shape['Reactivity_profile'].fillna(0)
    #df_shape["Reactivity_profile"] = np.where(df_shape["Reactivity_profile"] == "NaN", 0, df_shape["Reactivity_profile"])
    #df_shape["Reactivity_shape"] = unit_vector_norm(df_shape, "Reactivity_shape")
    #based on QuSHAPE paper defining range of reactivities: .4 < med, .85 < high
    shape_outlier = .4
    df_shape["Predicted_Shape_Map"] = df_shape["Reactivity_profile"]
    df_shape["Predicted_Shape_Map"] = np.where((df_shape["Predicted_Shape_Map"] != 0) &
                                               (df_shape["Predicted_Shape_Map"] < shape_outlier),
                                               1, df_shape["Predicted_Shape_Map"])
    df_shape["Predicted_Shape_Map"] = np.where((df_shape["Predicted_Shape_Map"] != 1) &
                                               (df_shape["Predicted_Shape_Map"] >= shape_outlier), -1,
                                           df_shape["Predicted_Shape_Map"])

    return df_shape



def get_shape_supervised(df_signal):
    df = df_signal

    # shape files
    reactivities = {"T_thermophila": "Shape/tetra_reactivities.txt",
                    "E_coli_tmRNA": "Shape/ecoli_reactivities.txt",
                    "HCV": "Shape/hcv_reactivities.txt",
                    "cen_3'utr": "Shape/cen-utr_reactivities.txt",
                    "ik2_3'utr": "Shape/ik2-utr_reactivities.txt",
                    "cen_3'utr_complex": "Shape/cen-ik2-utr_reactivities.txt",
                    "ik2_3'utr_complex": "Shape/ik2-cen-utr_reactivities.txt",
                    "cen_FL_complex": "Shape/cen-ik2-fl_reactivities.txt",
                    "ik2_FL_complex": "Shape/ik2-cen-fl_reactivities.txt",
                    "cen_FL": "Shape/cen-fl_reactivities.txt",
                    "ik2_FL": "Shape/ik2-fl_reactivities.txt"}

    df_shape = pd.DataFrame()
    for s, f in reactivities.items():
        df1 = pd.read_csv(f, names=['Position', 'Sequence', 'Reactivity_shape'])
        df1['Position'] = df1['Position'] - 1
        df1['Sequence_Name'] = s
        df_shape = pd.concat([df_shape, df1], ignore_index=True)

    df_shape["Reactivity_shape"] = np.where(df_shape["Reactivity_shape"] == -1, 0, df_shape["Reactivity_shape"])
    #df_shape["Reactivity_shape"] = unit_vector_norm(df_shape, "Reactivity_shape")
    #based on QuSHAPE paper defining range of reactivities: .4 < med, .85 < high
    shape_outlier = .4
    df_shape["Predicted_Shape"] = np.where(df_shape["Reactivity_shape"] >= shape_outlier, -1, 1)
    df_shape.rename(columns={'Position': 'position', 'Sequence_Name': 'contig', 'Predicted_Shape': 'Y'}, inplace=True)
    df = pd.merge(df, df_shape, on=['position', 'contig'], how='left')
    df = df.drop(columns=['Reactivity_shape', 'Sequence'])
    #df_shape.to_csv("shape_data.csv")
    return df

# returns raw shape values for distribution comparison to bc and predicted reactivities
def get_shape_continuous():
    # shape files
    reactivities = {"T_thermophila": dash_path + "Shape/tetra_reactivities.txt",
                    "E_coli_tmRNA": dash_path + "Shape/ecoli_reactivities.txt",
                    "HCV": dash_path + "Shape/hcv_reactivities.txt",
                    "cen_3'utr": dash_path + "Shape/cen-utr_reactivities.txt",
                    "ik2_3'utr": dash_path + "Shape/ik2-utr_reactivities.txt",
                    "cen_3'utr_complex": dash_path + "Shape/cen-ik2-utr_reactivities.txt",
                    "ik2_3'utr_complex": dash_path + "Shape/ik2-cen-utr_reactivities.txt",
                    "cen_FL_complex": dash_path + "Shape/cen-ik2-fl_reactivities.txt",
                    "ik2_FL_complex": dash_path + "Shape/ik2-cen-fl_reactivities.txt",
                    "cen_FL": dash_path + "Shape/cen-fl_reactivities.txt",
                    "ik2_FL": dash_path + "Shape/ik2-fl_reactivities.txt",
                    "FMN_Adaptor": dash_path + "Shape/FMN_reactivities.txt",
                    "Lysine_Adaptor": dash_path + "Shape/Lysine_reactivities.txt"
                    }

    df_shape = pd.DataFrame()
    for s, f in reactivities.items():
        df1 = pd.read_csv(f, names=['Position', 'Sequence', 'Reactivity_shape'])
        #print(f)
        df1['Position'] = df1['Position'].astype(int) - 1
        df1['Sequence_Name'] = s
        if "complex" in f:
            df1['Sequence_Name'] = s + '_complex'
        else:
            df1['Sequence_Name'] = s
        df_shape = pd.concat([df_shape, df1], ignore_index=True)

    #df_shape["Reactivity_shape"] = np.where(df_shape["Reactivity_shape"] == -1, 0, df_shape["Reactivity_shape"])
    # df_shape["Reactivity_shape"] = unit_vector_norm(df_shape, "Reactivity_shape")
    # based on QuSHAPE paper defining range of reactivities: .4 < med, .85 < high
    # remove missing data entries
    df_shape["Predicted_Shape"] = np.where(df_shape["Reactivity_shape"] == -1, 0, df_shape["Reactivity_shape"])
    # df_shape.to_csv("shape_data.csv")
    return df_shape



def get_bc_reactivity():
    # basecall error files
    #TODO: create single file, wo header in bash
    bc_files = {   "dmso_bc" :dash_path + "Basecall/DMSO_mod_rates_aligned.csv",
            "acim_bc" : dash_path + "Basecall/ACIM_mod_rates_aligned.csv"
            ,"acim_complex": dash_path + "Basecall/ACIM_mod_rates_complex_aligned.csv"
        }

    #,position,contig,Basecall_Reactivity,Quality,Mismatch,Deletion,Insertion,Aligned_Reads,Sequence
    cols = ['position','contig','Basecall_Reactivity', 'Quality',
                'Mismatch', 'Deletion', 'Insertion', 'Aligned_Reads', 'Sequence', 'Modification']
    csv_cols = ['position','contig','Basecall_Reactivity', 'Quality',
                'Mismatch', 'Deletion', 'Insertion', 'Aligned_Reads', 'Sequence']
    grp_cols = ['position','contig', 'Modification']
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
                          header=1)
        dfn['Modification'] = mod
        # if "complex" in f:
        #     dfn['contig'] = dfn['contig'].astype(str) + '_complex'

        df = pd.DataFrame(np.concatenate([df.values, dfn.values]), columns=cols)

    #average duplicate entries
    df = df.groupby(grp_cols, as_index=False)[mean_cols].mean()
    return df

def shape_statistics():
    df_shape = get_shape_continuous()
    print(df_shape.columns)
    df = df_shape.groupby(['Sequence_Name', 'Sequence'], as_index=False)\
        ['Reactivity_shape'].apply(lambda x: (x>.4).sum())
    print(df.columns)
    df = df.drop(columns=['Sequence_Name'])
    df = df.groupby(['Sequence'])['Reactivity_shape'].sum()
    print(df)
    sys.exit(0)

def bias_transitions():
    transition = np.array((4,4))
    mismatch = {'T_thermophila': 'Mismatch/T_thermophila_mismatch_bias_rates.csv',
                'HCV': 'Mismatch/HCV_mismatch_bias_rates.csv',
                'E_coli_tmRNA': 'Mismatch/E_coli_tmRNA_mismatch_bias_rates.csv',
                "cen_3'utr": "Mismatch/cen_3'utr_mismatch_bias_rates.csv",
                "ik2_3'utr": "Mismatch/ik2_3'utr_mismatch_bias_rates.csv",
                "ik2_FL": "Mismatch/ik2_FL_mismatch_bias_rates.csv",
                "cen_FL": "Mismatch/cen_FL_mismatch_bias_rates.csv",
                "FMN_Adaptor": "Mismatch/FMN_Adaptor_mismatch_bias_rates.csv",
                "Lysine_Adaptor": "Mismatch/Lysine_Adaptor_mismatch_bias_rates.csv",
                }

    df = pd.DataFrame()
    for s, f in mismatch.items():
        dft = pd.read_csv(f, names=['D', 'Position', 'Number of Mismatches', 'Modified Nt', 'Reference Nt'], header=0)
        dft['Sequence_Name'] = s
        df = pd.concat([df, dft], ignore_index=True)

    df = df.drop(columns=['D'])

    print(df.head())
    df_shape = get_shape_continuous()
    df_shape['Binary_Shape'] = np.where(df_shape['Reactivity_shape'] > .4, 1, 0)
    df = pd.merge(df, df_shape, on=['Position', 'Sequence_Name'])
    df = df[['Reference Nt', 'Modified Nt', 'Binary_Shape']]
    df = df[df['Binary_Shape']==0]
    df = df.groupby(['Reference Nt', 'Modified Nt']).count()
    print(df.columns)
    print(df.head())
    df.to_csv("mismatch_transistions.csv")
    sys.exit(0)

def kmer_transitions():
    #df_dmso = pd.read_csv
    df_dmso = pd.read_csv("DMSO_signal_preprocessed_aligned.csv")
    df_acim = pd.read_csv("ACIM_signal_preprocessed_aligned.csv")
    df_shape = get_shape_continuous()
    df_shape['Binary_Shape'] = np.where(df_shape['Reactivity_shape'] > .4, 1, 0)
    df = pd.merge(df_dmso, df_acim, on=['position', 'contig'],how="outer",
                  suffixes=('_dmso', '_acim'))
    df_shape = df_shape.rename(columns={'Position':'position', 'Sequence_Name':'contig'})
    df = pd.merge(df, df_shape, on=['position', 'contig'])
    print(df.columns)
    df = df[df['model_kmer_dmso'] != df['model_kmer_acim']]
    df.to_csv("kmer_transitions.csv")
    print(df.head())
    sys.exit(0)



######### Parse Reference Sequence from FASTA ###############
def parse_fasta(filepath="library.fasta"):
    sequence = {}
    sequence_name = ""
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('>'):
                s = re.split('[\s>]', line)
                sequence_name = s[1]
                # print(s[1])
            elif re.search("(?i)^[ACGTU]", line):
                sequence[sequence_name] = line[:-1].upper()
                # print(line)
    return sequence

### preprocess nanopolish data by read and sequence ###
def preprocess_df_byread(df):
    ####### Collapse Multiple Events on Same Position and Sequence ###########
    dft = df
    print(len(df))

    grp_cols = ['contig','position','reference_kmer','read_index']

    sequences = dft['contig'].unique()
    for seq in sequences:
        reads = dft['read_index'].unique()
        for read in reads:
            dft['position_count'] = dft['position']
            grp_cols = ['contig', 'position', 'reference_kmer', 'read_index']  # , 'model_kmer'
            mean_cols = ['event_stdv', 'event_level_mean']
            dft = dft.groupby(by=grp_cols, as_index=False).agg({'event_level_mean': 'mean',
                                                                'event_length': 'sum', 'event_stdv': 'mean'})
            dft = dft.sort_values(by=['contig', 'read_index', 'position'])
    sys.exit(0)

    # mean
    mod_mean = df['event_level_mean'].mean()
    mod_std = df['event_level_mean'].std()
    df['event_level_mean'] = (df['event_level_mean'] - mod_mean) / mod_std

    dwell_mean = df['event_length'].mean()
    dwell_std = df['event_length'].std()
    df['event_length'] = (df['event_length'] - dwell_mean) / dwell_std

    mod_mean = df['event_stdv'].mean()
    mod_std = df['event_stdv'].std()
    df['event_stdv'] = (df['event_stdv'] - mod_mean) / mod_std

    df = df.sort_values(['read_index', 'contig','position'])





    print(df.columns)
    print(len(df))

    #dt = reverse_positions(dt)
    # TODO Align individual signal files, with gaps
    df.to_csv(dash_path + "Deconvolution/Preprocess/ACIM_signal_preprocess_by_read.txt")

    ##### align kmer ######
    # df, shift = align_positions(df)

    return df

### preprocess nanopolish data ###
def preprocess_df(df):
    ####### Collapse Multiple Events on Same Position and Sequence ###########
    ##### remove bad kmer reads ##########
    df = df[df['standardized_level'].astype(str).str.contains('inf') == False]
    #### remove current means above 200 and below 0 ###
    #df = df[(df['event_level_mean'] > 200) & (df['event_level_mean'] < 0)]
    #### normalize event mean, current per strand to that of the expected model current mean ####
    #mod_mean = df['model_mean'].mean()
    #mod_std = df['model_mean'].std()

    #mean of means from parse file
    mod_mean = df['event_level_mean'].mean()
    mod_std = df['event_level_mean'].std()
    df['event_level_mean'] = (df['event_level_mean'] - mod_mean) / mod_std

    dwell_mean = df['event_length'].mean()
    dwell_std = df['event_length'].std()
    df['event_length'] = (df['event_length'] - dwell_mean) / dwell_std

    mod_mean = df['event_stdv'].mean()
    mod_std = df['event_stdv'].std()
    df['event_stdv'] = (df['event_stdv'] - mod_mean) / mod_std


    #### sum of event lengths ##########
    #### average signal/ translocation by position
    ### where read_index references reads, nanopolish --print - read - names to display reads
    grp_cols = ['contig', 'reference_kmer', 'position'] #, 'model_kmer'
    # df2 = df.groupby(grp_cols, as_index=False)[
    #     'event_length'].sum()

    #TODO: add count of position and contig groups, to see how many
    #may be important to remove dmso outliers
    #### weighted average of event mean, event std ########
    # TODO: what are the weights in the average?
    mean_cols = ['event_stdv', 'event_level_mean', 'event_length']
    df = df.groupby(grp_cols, as_index=False)[mean_cols].mean()
    # df2 = df.groupby(grp_cols, as_index=False)[mean_cols].std()
    # df2 = df2.rename(columns={'event_stdv': 'event_stdv_std', 'event_level_mean': 'event_level_mean_std',
    #                           'event_length': 'event_length_std'})
    # df = pd.merge(df1, df2, on=['position', 'contig','reference_kmer'])
    #df = pd.merge(df, df1, on=['position', 'contig', 'reference_kmer'])

    print(df.head())
    print(df.columns)

    #### join results ######
    #df = df.merge(df2)
    #reverse positions to 5'-3', from 3'-5' used in nanopolish
    df = reverse_positions(df)


    ##### position shifting ######
    # if position_shift > 0:
    #     df['position'] = df['position'] + self.position_shift

    return df

def get_structure():
    # structure files
    tetra_structure = dash_path + "Structure/T_PDB_7EZ0_1_N_annotated.csv"
    tetra_paper = dash_path + "Structure/tetra_paper.txt"
    ecoli_structure = dash_path + "Structure/E_PDB_3IZ4_1_A_annotated.csv"
    ecoli_paper = dash_path + "Structure/ecoli_paper.txt"
    hcv_structure = dash_path + "Structure/HCV_manual_annotated.csv"
    hc16_structure = dash_path + "Structure/hc16_ligase_manual_annotation.csv"
    rnase_structure = dash_path + "Structure/RNAse_P_PDB_2A64_1_A_annotated.csv"
    rnase_paper = dash_path + "Structure/rnase_paper.txt"


    df1 = pd.read_csv(tetra_structure, usecols=['Position', 'Sequence', 'BaseType'], header=0)
    df1['Sequence_Name'] = "T_thermophila"
    dft = pd.read_csv(tetra_paper, sep="\s", names=['Position', 'Sequence', 'BaseType'], header=0)
    dft['BaseType'] = np.where(dft['BaseType'] == 0, 'S', 'B')
    df1['BaseType'] = dft['BaseType']
    df2 = pd.read_csv(ecoli_structure, usecols=['Position', 'Sequence', 'BaseType'], header=0)
    df2['Sequence_Name'] = "E_coli_tmRNA"
    dft = pd.read_csv(ecoli_paper, sep="\s", names=['Position', 'Sequence', 'BaseType'], header=0)
    dft['BaseType'] = np.where(dft['BaseType'] == 0, 'S', 'B')
    df2['BaseType'] = dft['BaseType']
    df3 = pd.read_csv(hcv_structure, usecols=['Position', 'Sequence', 'BaseType'], header=0)
    df3['Sequence_Name'] = "HCV"
    df4 = pd.read_csv(hc16_structure, usecols=['Position', 'Sequence', 'BaseType'], header=0)
    df4['Sequence_Name'] = "hc16_ligase"
    df5 = pd.read_csv(rnase_structure, usecols=['Position', 'Sequence', 'BaseType'], header=0)
    df5['Sequence_Name'] = "RNAse_P"
    dft = pd.read_csv(rnase_paper, sep="\s", names=['Position', 'Sequence', 'BaseType'], header=0)
    dft['BaseType'] = np.where(dft['BaseType'] == 0, 'S', 'B')
    df5['BaseType'] = dft['BaseType']

    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    df = df.dropna()
    return df

def get_structure_ext():
    # structure files
    tetra_structure = dash_path + "Structure/T_PDB_7EZ0_1_N_annotated.csv"
    tetra_paper = dash_path + "Structure/tetra_paper.txt"
    ecoli_structure = dash_path + "Structure/E_PDB_3IZ4_1_A_annotated.csv"
    ecoli_paper = dash_path + "Structure/ecoli_paper.txt"
    hcv_structure = dash_path + "Structure/HCV_manual_annotated.csv"
    hc16_structure = dash_path + "Structure/hc16_ligase_manual_annotation.csv"
    rnase_structure = dash_path + "Structure/RNAse_P_PDB_2A64_1_A_annotated.csv"
    rnase_paper = dash_path + "Structure/rnase_paper.txt"

    df1 = pd.read_csv(tetra_structure, usecols=['Position', 'Sequence', 'BaseType', 'StructureType'], header=0)
    df1['Sequence_Name'] = "T_thermophila"
    dft = pd.read_csv(tetra_paper, sep="\s", names=['Position', 'Sequence', 'BaseType'], header=0)
    dft['BaseType'] = np.where(dft['BaseType'] == 0, 'S', 'B')
    df1['BaseType'] = dft['BaseType']
    df2 = pd.read_csv(ecoli_structure, usecols=['Position', 'Sequence', 'BaseType', 'StructureType'], header=0)
    df2['Sequence_Name'] = "E_coli_tmRNA"
    dft = pd.read_csv(ecoli_paper, sep="\s", names=['Position', 'Sequence', 'BaseType'], header=0)
    dft['BaseType'] = np.where(dft['BaseType'] == 0, 'S', 'B')
    df2['BaseType'] = dft['BaseType']
    df3 = pd.read_csv(hcv_structure, usecols=['Position', 'Sequence', 'BaseType', 'StructureType'], header=0)
    df3['Sequence_Name'] = "HCV"
    df4 = pd.read_csv(hc16_structure, usecols=['Position', 'Sequence', 'BaseType', 'StructureType'], header=0)
    df4['Sequence_Name'] = "hc16_ligase"
    df5 = pd.read_csv(rnase_structure, usecols=['Position', 'Sequence', 'BaseType', 'StructureType'], header=0)
    df5['Sequence_Name'] = "RNAse_P"
    dft = pd.read_csv(rnase_paper, sep="\s", names=['Position', 'Sequence', 'BaseType'], header=0)
    dft['BaseType'] = np.where(dft['BaseType'] == 0, 'S', 'B')
    df5['BaseType'] = dft['BaseType']


    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    df = df.dropna()
    return df

def parse_ACIM():
    signal_acim = { #'lysine': dash_path + 'Signal_ACIM/ACIM_lys.txt',
    # 'fmn': dash_path + 'Signal_ACIM/ACIM_fmn.txt',
    #'hc16_rnasep': dash_path + 'Signal_ACIM/ACIM_hc16_rnasep.txt',
    # 'fmn_lys': dash_path + 'Signal_ACIM/ACIM_fmn_lys.txt',
    # 'ciutr_separate': dash_path + 'Signal_ACIM/ACIM_ciutr_separate.txt',
    # 'ciutr_complex': dash_path + 'Signal_ACIM/ACIM_ciutr_complex.txt',
    # 'cifl_separate': dash_path + 'Signal_ACIM/ACIM_cifl_separate.txt',
    # 'cifl_complex': dash_path + 'Signal_ACIM/ACIM_cifl_complex.txt',
    # 'pool1' : dash_path + 'Signal_ACIM/ACIM_cen3utr_ik23utr_ecoli_tetra_hcv.txt',
    'pool2' : dash_path + 'Signal_ACIM/ACIM_cen3utr_cenFL_ik23utr_ik2FL_tetra_hcv_fmn_lys.txt'
    }

    df = pd.DataFrame()
    for key,fpath in signal_acim.items():
        dft = pd.read_csv(fpath, sep="\t")
        ##### remove bad kmer reads ##########
        dft = dft[dft['standardized_level'].astype(str).str.contains('inf') == False]
        #### normalize for each read ####
        dft['position_count'] = dft['position']
        grp_cols = ['contig', 'position', 'reference_kmer', 'read_index']  # , 'model_kmer'
        mean_cols = ['event_stdv', 'event_level_mean']
        dft = dft.groupby(by=grp_cols, as_index=False).agg({'event_level_mean':'mean',
            'event_length': 'sum', 'event_stdv': 'mean'})
        dft = dft.sort_values(by=['contig', 'read_index', 'position'])

        #### detect sequences in complex
        if "complex" in key:
            dft['contig'] = dft['contig'].astype(str) + '_complex'
        df = pd.concat([df, dft], ignore_index=True)

    #holdout 'cen_3'utr', 'ik2_3'utr', 'cen_FL', 'ik2_FL'
    #sequences = ['E_coli_tmRNA', 'HCV', 'T_thermophila', 'hc16_ligase', 'RNAse_P']
    #df = df[df['contig'].isin(sequences)]
    print(df.head())
    df.to_csv("ACIM_signal.txt", sep="\t", index=False)
    return df

def parse_DMSO(fa="/Users/timshel/Documents/NanoporeResearch/LecuyerII/DMSO_signal.txt"):
    df = pd.read_csv(fa, sep="\t")
    #holdout 'cen_3'utr', 'ik2_3'utr'
    sequences = ['E_coli_tmRNA', 'HCV', 'T_thermophila', 'hc16_ligase', 'RNAse_P']
    df = df[df['contig'].isin(sequences)]
    df.to_csv("DMSO_signal.txt", sep="\t", index=False)

def parse_ACIM_ssbp(fa="ACIM_signal.txt"):
    df = pd.read_csv(fa, sep="\t")
    #holdout 'cen_3'utr', 'ik2_3'utr'
    sequences = ['E_coli_tmRNA', 'HCV', 'T_thermophila']
    df = df[df['contig'].isin(sequences)]
    df2 = get_structure()
    df2 = df2.rename(columns={'Position':'position', 'Sequence_Name':'contig'})
    df = df.merge(df2, on=['position', 'contig'], how='left')
    dfss = df[df['BaseType']=='S']
    dfss.drop(['BaseType', 'Sequence'], axis=1, inplace=True)
    dfss.to_csv("ACIM_ss_signal.txt", sep="\t", index=False)
    dfbp = df[df['BaseType'] == 'B']
    dfbp.drop(['BaseType', 'Sequence'], axis=1, inplace=True)
    dfbp.to_csv("ACIM_bp_signal.txt", sep="\t", index=False)

def parse_DMSO_ssbp(fa="DMSO_signal.txt"):
    df = pd.read_csv(fa, sep="\t")
    #holdout 'cen_3'utr', 'ik2_3'utr'
    sequences = ['E_coli_tmRNA', 'HCV', 'T_thermophila']
    df = df[df['contig'].isin(sequences)]
    df2 = get_structure()
    df2 = df2.rename(columns={'Position':'position', 'Sequence_Name':'contig'})
    df = df.merge(df2, on=['position', 'contig'], how='left')
    dfss = df[df['BaseType']=='S']
    dfss.drop(['BaseType', 'Sequence'], axis=1, inplace=True)
    dfss.to_csv("DMSO_ss_signal.txt", sep="\t", index=False)
    dfbp = df[df['BaseType'] == 'B']
    dfbp.drop(['BaseType', 'Sequence'], axis=1, inplace=True)
    dfbp.to_csv("DMSO_bp_signal.txt", sep="\t", index=False)

#count tp within a specified range, offset, look ahead or behind offset number of nts
def count_agreement_with_offset(predicted, shape, offset=0):
    tp = []
    fp = []
    fn = []
    tn = []
    for i,p in enumerate(predicted):
        if i > len(shape)-1:
            break
        if shape[i] != -1:
            if p==1:
                if offset > 0:
                    if i + offset < len(predicted):
                        if any(shape[i:i+offset]):
                            tp.append(i)
                        else:
                            fp.append(i)
                    else:
                        if any(shape[i:]):
                            tp.append(i)
                        else:
                            fp.append(i)
                elif offset < 0:
                    if i - np.abs(offset) > 0:
                        if any(shape[i - np.abs(offset):i + 1]):
                            tp.append(i)
                        else:
                            fp.append(i)
                    else:
                        if any(shape[0:i + 1]):
                            tp.append(i)
                        else:
                            fp.append(i)
                elif offset==0:
                        if shape[i] == 1:
                             tp.append(i)
                        else:
                            fp.append(i)
            #fn
            elif p == 0:
                if offset > 0:
                    if i + offset < len(predicted):
                        if any(shape[i:i + offset]):
                            fn.append(i)
                        else:
                            tn.append(i)
                    else:
                        if any(shape[i:]):
                            fn.append(i)
                        else:
                            tn.append(i)
                elif offset < 0:
                    if i - np.abs(offset) > 0:
                        if any(shape[i - np.abs(offset):i + 1]):
                            fn.append(i)
                        else:
                            tn.append(i)
                    else:
                        if any(shape[0:i + 1]):
                            fn.append(i)
                        else:
                            tn.append(i)
                elif offset == 0:
                    if shape[i] == 1:
                        fn.append(i)
                    else:
                        tn.append(i)
#print(f"tp: {count}, fp: {fp}, fn: {fn}, tn: {tn}")
    return len(np.unique(np.array(tp))), len(np.unique(np.array(fp))), \
        len(np.unique(np.array(fn))), len(np.unique(np.array(tn)))

def plot_bc_shape_peaks(sequence="cen_3'utr"):
    print("Sequence: " , sequence)
    print()
    df = get_bc_reactivity()
    df1 = get_shape_continuous()
    bc_acim = df[(df['contig'].astype(str) == sequence) & (df['Modification'].astype(str) == "ACIM")]
    bc_dmso = df[(df['contig'].astype(str) == sequence) & (df['Modification'].astype(str) == "DMSO")]

    #define shapes
    df1 = df1[df1['Sequence_Name'] == sequence]
    # df1['Reactivity_shape'] = np.where((df1['Reactivity_shape'].to_numpy() >= .4), 1, 0)
    # bc_dmso.to_csv("dmso.csv")
    # bc_acim.to_csv("acim.csv")
    # df1['Reactivity_shape'].to_csv("y.csv")
    # sys.exit(0)


    #basecall columns used in delta analysis
    ocols = ['Mismatch', 'Basecall_Reactivity', 'Insertion', 'Deletion', 'Quality']

    #plot basecall columns histograms
    # visualize initialize distributions
    def bc():
        shapes = df1['Reactivity_shape'].to_numpy()
        bc = pd.DataFrame()
        a = bc_acim['Basecall_Reactivity'].astype("float32").to_numpy()
        b = bc_dmso['Basecall_Reactivity'].astype("float32").to_numpy()
        bc['Delta_Basecall_Reactivity'] = np.where((a > 0) & (b > 0), np.subtract(a, b), 0)
        plt.hist(b, bins=100)
        plt.title("BC ACIM")
        plt.show()
        plt.hist(bc['Delta_Basecall_Reactivity'], bins=100)
        plt.title("Delta Basecall Reactivity")
        plt.show()
        # print(df['Basecall_Reactivity'].to_numpy())
        # SHAPE threshold in unit
        threshold = (.4 - np.min(shapes)) / (np.max(shapes) - np.min(shapes))
        x = bc['Delta_Basecall_Reactivity'].to_numpy()
        basecalls = np.where(x > 0, ((x - np.min(shapes)) / (np.max(shapes) - np.min(shapes))), 0)
        shapes = unit_vector_norm(shapes)
        x = bc['Delta_Basecall_Reactivity'].to_numpy()
        basecalls = unit_vector_norm(basecalls)
        print("Shape reactivity unit threshold: ", threshold)
        y_true = np.where(shapes > threshold, 1, 0)
        max = 0
        shift = 0
        # for i in range(0, 1):
        #     basecalls = np.roll(basecalls, i)
        #     rx_sites = np.where((shapes >= threshold) & (basecalls >=threshold), True, False).sum()
        #     if rx_sites > max:
        #         max = rx_sites
        #         shift = i
        print("basecalls mean, ", np.mean(basecalls))
        print("basecalls median, ", np.median(basecalls))
        print("basecalls std, ", np.std(basecalls))
        # print("max agreement ", max)
        # print("shift at max ", shift)
        # print("Total Reactive Sites: ", np.where((shapes >= threshold), True, False).sum())
        # bcthreshold = np.median(basecalls) - .02

    #find optimal threshold that maximizes tp when comparing shape to bc
    def getbcthreshold():
        shapes = df1['Reactivity_shape'].to_numpy()
        # SHAPE threshold in unit
        threshold = (.4 - np.min(shapes)) / (np.max(shapes) - np.min(shapes))
        x = bc['Delta_Basecall_Reactivity'].to_numpy()
        basecalls = np.where(x > 0, ((x - np.min(shapes)) / (np.max(shapes) - np.min(shapes))), 0)
        shapes = unit_vector_norm(shapes)
        x = bc['Delta_Basecall_Reactivity'].to_numpy()
        basecalls = unit_vector_norm(basecalls)
        print("Shape reactivity unit threshold: ", threshold)
        y_true = np.where(shapes > threshold, 1, 0)
        bcthreshold = np.median(basecalls)
        max = 0
        for i in np.arange(bcthreshold, 0, step=-.001):
            y_pred = np.where(basecalls > bcthreshold, 1, 0)
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            #print(auc)
            if auc > max:
                max = auc
                bcthreshold = i
        print("maximum agreement ", max)
        print(bcthreshold)
        return bcthreshold

    def dbscan():
        print("Dbscan outliers......")
        print()
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes = np.where(shapes > .4, 1, 0)
        shape_outliers = np.where((shapes >= 1), True, False).sum()
        scols = ocols
        for c in scols:
            print("Column: ", c)
            a = unit_vector_norm(bc_acim[scols].astype("float32").to_numpy())
            b = unit_vector_norm(bc_dmso[scols].astype("float32").to_numpy())
            delta = np.subtract(a,b).reshape(-1,1)
            #delta = np.delete(delta, np.where(delta == 0), axis=0)

            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=2)
            nbrs = neigh.fit(delta)
            distances, indices = nbrs.kneighbors(delta)
            # Plotting K-distance Graph
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]
            #print(np.std(distances))
            md = np.median(distances) + .85*np.median(distances)
            if md == 0:
                md = np.mean(distances) * 2
            #print(f"max distances: {md}")
            # plt.figure(figsize=(8, 5))
            # plt.plot(distances)
            # plt.title('K-distance Graph', fontsize=20)
            # plt.xlabel('Data Points sorted by distance', fontsize=14)
            # plt.ylabel('Epsilon', fontsize=14)
            # plt.show()



            model = DBSCAN(eps=md, min_samples=2).fit(delta)
            colors = model.labels_
            plt.figure(figsize=(10, 7))
            plt.scatter(delta, model.labels_, c=colors)
            plt.title('Outliers Detection using DBSCAN', fontsize=20)
            plt.show()
            #print(model.labels_)
            unique, counts = np.unique(model.labels_, return_counts=True)
            #print(f"clusters labels: {unique}: {counts}")
            delta_outliers = np.where((model.labels_ == -1), True, False)
            pred_outliers = np.where((model.labels_ == -1), True, False).sum()
            print("Shape Outliers: ", shape_outliers)
            print("Predicted Outliers: ", pred_outliers)
            tp = count_agreement_with_offset(delta_outliers, shapes, 2)
            accuracy = tp / pred_outliers
            max_acc = 0
            max_tp = 0
            best_k = 0
            if accuracy > max_acc:
                max_tp = tp
                max_acc = accuracy
            print(f"Total agreed outlier sites : {max_tp}")
            print(f"Accuracy: {max_acc}")
            #sys.exit(0)


    #dbscan()

    def delta_bc():
        print("running delta bc...")
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes_cont = df1['Reactivity_shape'].to_numpy()
        shapes_cont = unit_vector_norm(shapes_cont)
        # SHAPE threshold in unit
        threshold = (.4 - np.min(shapes)) / (np.max(shapes) - np.min(shapes))
        #x = bc['Delta_Basecall_Reactivity'].to_numpy()
        #basecalls = np.where(x > 0, ((x - np.min(shapes)) / (np.max(shapes) - np.min(shapes))), 0)
        shapes = unit_vector_norm(shapes)
        #x = bc['Delta_Basecall_Reactivity'].to_numpy()
        #basecalls = unit_vector_norm(basecalls)
        print("Shape reactivity unit threshold: ", threshold)
        y_true = np.where(shapes > threshold, 1, 0)
        #bcthreshold = np.median(basecalls)
        #'Delta_Mismatch', 'Delta_Insertion', 'Delta_Deletion', 'Delta_Quality',
        cols = {'delta_reactivity'}
        ocols = ['Basecall_Reactivity']
        delta = pd.DataFrame()
        for c in ocols:
            a = bc_acim[c].astype("float32").to_numpy() #unit_vector_norm(bc_acim[c].astype("float32").to_numpy())
            b = bc_dmso[c].astype("float32").to_numpy() #unit_vector_norm(bc_dmso[c].astype("float32").to_numpy())
            ncol = 'Delta_' + c
            delta[ncol] = np.subtract(a, b)

            def plot_hist():
                plt.hist(delta[ncol], bins=100)
                plt.title(ncol)
                plt.show()
                plt.hist(b, bins=100)
                plt.title(c + ' ACIM')
                plt.show()
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes = np.where(shapes > .4, 1, 0)
        delta['shapes'] = shapes
        dist = []
        x = delta[ncol].to_numpy()
        b = np.where(x > 0, ((x - np.min(shapes)) / (np.max(shapes) - np.min(shapes))), 0)
        b = unit_vector_norm(b).reshape(-1)
        a = shapes.reshape(-1)
        for i in range(0, len(shapes)):
            d = 0
            if (float(a[i]) > 0) and (float(b[i]) > 0):
                d = np.linalg.norm(float(a[i]) - float(b[i]))
            if math.isnan(d):
                dist.append(0)
            else:
                dist.append(d)
        delta['distance'] = dist
        delta['reactive'] = np.where(shapes > threshold, 1, 0)
        delta['delta_reactivity'] = b
        delta['corr'] = delta['delta_reactivity'].astype("float32").corr(delta['shapes'].astype("float32"),
                                                                  method="pearson")

        delta = delta[delta['delta_reactivity'] > 0]
        rnorm = delta['delta_reactivity'].astype("float32")
            #unit_vector_norm(bc['Delta_Basecall_Reactivity'].astype("float32").to_numpy())
        delta['zscore'] = stats.zscore(rnorm)
        mean = delta['delta_reactivity'].median() #delta['zscore'].median()
        std = delta['delta_reactivity'].std() #delta['zscore'].std()
        upper = mean + (1 * std)
        lower = mean - (1 * std)
        #delta['outlier'] = np.where((delta['zscore']>upper) | (delta['zscore']<lower), 1, 0)
        delta['outlier'] = np.where((delta['delta_reactivity']>upper) , 1, 0)

        tot_reactive = np.where((shapes >= threshold), True, False).sum()
        print("Total Reactive Sites: ", tot_reactive)
        print("Total sites", len(shapes))
        print("Total Outlier Sites:")
        print("BC Outlier Sites: ", np.where((delta['outlier'] >= 1), True, False).sum())
        print(b)
        print(shapes_cont)
        print(f"MannWhitney U test:{stats.mannwhitneyu(b,shapes_cont, alternative='two-sided')}")
        print(f"KS test:{stats.ks_2samp(b,shapes_cont)}")
        #delta.to_csv("delta_bc_" + sequence + ".csv")

    #delta_bc()

    def ks():
        print(ocols)
        scols = ['Basecall_Reactivity']
        peaks = []
        peak_range = []
        peaks_value = {}
        shape_errors = df1[df1["Reactivity_shape"]== -1]
        shape_errors = np.array(shape_errors["Position"])
        shape_outliers = np.where((df1['Reactivity_shape'].to_numpy() >= .4), 1, 0)
        shape_pred = np.where((shape_outliers >= 1), True, False).sum()
        for c in scols:
            print("Column: ", c)
            k = {}
            ksv = []
            kss = []
            y = bc_acim[c].astype("float32").to_numpy()
            x = bc_dmso[c].astype("float32").to_numpy()
            a = unit_vector_norm(bc_acim[c].astype("float32").to_numpy())
            b = unit_vector_norm(bc_dmso[c].astype("float32").to_numpy())
            rate = np.where( (a != 0) & (b != 0), np.divide(a, b), a)
            rate = np.nan_to_num(rate, posinf=0, neginf=0, nan=0)
            diff = np.subtract(a, b)
            end = len(x)
            w = 2
            for i in range(1, end):
                if i >= (w+1) and i<= end -w :
                    d = stats.ks_2samp(x[i-w:i+w], y[i-w:i+w])
                elif i < (w+1):
                    d = stats.ks_2samp(x[1:i + w], y[1:i + w])
                elif i > end - w:
                    d = stats.ks_2samp(x[i-w:end], y[i-w:end])
                k[i-1] = d.statistic
                ksv.append(d.statistic_location)
                kss.append(d.statistic_sign)
                # print(d.statistic)
                # print(d.pvalue)
                # print(d.statistic_location)
                # print(d.statistic_sign)
            a = pd.DataFrame.from_dict(k, orient='index', columns=['statistic'])

            #print(peaks_d)
            a['location'] = ksv
            a['sign'] = kss

            if c == 'Quality':
                peaks_value[c] = unit_vector_norm(diff)
            else:
                peaks_value[c] = unit_vector_norm(np.multiply(np.array(list(k.values())), np.array(kss)))


            #m = np.sum(unit_vector_norm(a['statistic'].to_numpy()))
            #print(f"{c} : {np.divide(y['c'], x['c'])}")
            if c == "Quality":
                p = find_peaks(a['statistic'].to_numpy(), plateau_size=[0,6])
            elif c == "Insertion" or c == "Deletion" or c == "Mismatch":
                 p = [0]
            else:
                p = find_peaks(a['statistic'].to_numpy(), plateau_size=[0, 2])
            # s1 = np.zeros(len(a['statistic']))


            s1 = np.zeros(len(diff))

            #print(p[1]['left_edges'])
            #print(p[1]['right_edges'])
            for i in zip(p[1]['left_edges'], p[1]['right_edges']):
                if i[1] > i[0]:
                    for j in range(i[0], i[1] + 1):
                        peak_range.append(j)
                else:
                    peak_range.append(i[0])

                if i[0] == i[1]:
                    s1[i[0]] = 1
                else:
                    s1[i[0]:i[1]+1] = 1
            #set peak indexes
            #s1[p[0]] = 1
            #s1 = np.roll(s1,2)
            #store peaks for each column
            peaks = np.unique(np.concatenate((peaks, peak_range), axis=None))
            #peaks = np.unique(np.concatenate((peaks, p[0], p[1]['left_edges'], p[1]['right_edges']), axis=None))
            #s = np.where(a['statistic'] > m, 1, 0)
            pred_outliers = np.where((s1 >= 1), True, False).sum()
            print("Total Predicted Outlier Sites: ", pred_outliers)
            print("Total Shape Outlier Sites: ", shape_pred)
            for offset in range(-2, 3):
                tp = count_agreement_with_offset(s1, shape_outliers, offset)
                accuracy = tp / pred_outliers
                print(f"Total agreed outlier sites (offset: {offset}): {tp}")
                print(f"Accuracy: {accuracy}")
            #print("Total agreed outlier sites (shifted): ", count_agreement_with_offset(np.roll(s1,-2), shape_outliers, 0))
            #a.to_csv("hcv_ks.csv")
        seqlen = len(a)
        p_weight = np.zeros(seqlen).flatten()
        #print(peaks_value)
        # for i in range(0, seqlen-1):
        #     if (i in peaks) and (i not in shape_errors):
        #         weight = 0
        #         for c, ps in peaks_value.items():
        #             if c == "Basecall_Reactivity":
        #                 weight = weight + .6*(ps[i])
        #             else:
        #                 weight = weight + .4*(ps[i])
        print(peaks)
        for i in peaks:
            p_weight[int(i)] = 1


        print(p_weight)
        #p_weight = np.roll(p_weight, 2)

        # for c, ps in peaks.items():
        #     #print(len(peaks_d[c]))
        #     plt.plot(np.arange(0,len(peaks_d[c])), peaks_d[c].values(), label=c)
        # plt.legend()
        # plt.show()

        print()
        print("peak weight...")
        pw = np.mean(p_weight) + np.std(p_weight)
        large = np.mean(p_weight) + 1.5 * np.std(p_weight)
        pw = large
        print(pw)

        pred_outliers = np.where((p_weight >= pw), True, False).sum()
        p_weight_binary = np.where((p_weight >= pw), True, False)
        print("Total Predicted Outlier Sites: ", pred_outliers)
        print("Total Shape Outlier Sites: ", shape_pred)
        for offset in range(-2, 3):
            tp = count_agreement_with_offset(p_weight_binary, shape_outliers, offset)
            accuracy = tp / pred_outliers
            print(f"Total agreed outlier sites (offset: {offset}): {tp}")
            print(f"Accuracy: {accuracy}")

        #print(p_weight)
    #ks()

    def chebyshev():
        scols = ocols
        for c in scols:
            print("Column: ", c)
            y = bc_acim[c].to_numpy().flatten()
            x = bc_dmso[c].to_numpy().flatten()
            end = len(x)
            ##### manual cdf to measure ks #####
            # CDF
            # CDF(x) = "number of samples <= x"/"number of samples"
            def cdf(x, y):
                x1 = np.sort(x)
                y1 = np.sort(y)
                def ecdf(x,v):
                    res = np.searchsorted(x, v, side='right') / x.size
                    return res
                kp = []
                for v in x:
                    kp.append(ecdf(x1,v))
                cdfx = np.array(kp)
                kp = []
                for v in y:
                    kp.append(ecdf(y1, v))
                cdfy = np.array(kp)
                return cdfx, cdfy

            cdfx, cdfy = cdf(x,y)
            delta_ecdf = np.subtract(cdfy, cdfx)
            # print(delta_ecdf)
            plt.plot(delta_ecdf)
            plt.show()
            p = find_peaks(delta_ecdf, plateau_size=[0, 10])
            s1 = np.zeros(len(delta_ecdf))
            s1[p[0]] = 1

            pred_outliers = np.where((s1 >= 1), True, False).sum()
            print("Total Predicted Outlier Sites: ", pred_outliers)
            shape_outliers = np.where((df1['Reactivity_shape'].to_numpy() >= .4), 1, 0)
            shape_pred = np.where((shape_outliers >= 1), True, False).sum()
            print("Total Shape Outlier Sites: ", shape_pred)
            for offset in range(-2, 3):
                tp = count_agreement_with_offset(s1, shape_outliers, offset)
                accuracy = tp/ pred_outliers
                print(f"Total agreed outlier sites (offset: {offset}): {tp}")
                print(f"Accuracy: {accuracy}")

            # print("Total agreed outlier sites (shifted): ", count_agreement_with_offset(np.roll(s1,-2), shape_outliers, 0))
            # a.to_csv("hcv_ks.csv")

    def lr():

        print("Linear Regression outliers......")
        print()
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes = np.where(shapes > .4, 1, 0)
        shape_outliers = np.where((shapes >= 1), True, False).sum()
        # linear regression, get residuals
        # significant residuals/ bonferroni
        residuals = pd.DataFrame()
        #for c in ocols:
        print("\n" + " LR Outliers:")
        scols = ocols
        y = bc_acim[scols]
        for c in scols:
            y[c] = np.abs(unit_vector_norm(y[c].astype("float32").to_numpy()))

        #y = unit_vector_norm(bc_acim[c].astype("float32").to_numpy())
        #y = y.reshape(-1, 1)
        X = bc_dmso[scols]
        for c in scols:
            X[c] = np.abs(unit_vector_norm(X[c].astype("float32").to_numpy()))
        #X = unit_vector_norm(bc_dmso[c].astype("float32").to_numpy())
        #X = X.reshape(-1, 1)
        #r_sum = r.sum(axis=1)
        x_sum = X.to_numpy().sum(axis=1).reshape(-1,1)
        y_sum = y.to_numpy().sum(axis=1).reshape(-1,1)
        reg = LinearRegression().fit(x_sum, y_sum)
        r_sum = reg.predict(y_sum)
        #x_residuals = stats.zscore(x_residuals)
        residuals = np.subtract(y_sum, r_sum)
        #residuals = residuals.sum(axis=1)
        #residuals = stats.zscore(residuals)
        median = np.median(residuals)
        diff = np.abs(residuals-median)
        mad = np.median(diff)
        r = (.6745 * (residuals - median))/ (mad)
        #r = stats.zscore(residuals)
        plt.hist(r, bins=100)
        plt.show()

        # for i, r in enumerate(r):
        #     print(str(i) + ' ' + str(r))
                 # if r > 2 or r < -2:
                 #     print(str(i) +' ' + str(r))
        #r = stats.zscore(r)
        #y = stats.zscore(y)
        #predicted line

        plt.plot(y_sum, r_sum, color="black")
        plt.title("linear regression")
        #true values
        plt.scatter(x_sum, y_sum, color="blue", s=10)
        #plt.scatter(X, y, color="green")
        #plt.plot(X, r, color="blue", linewidth=3)
        plt.show()


        print("Shape Outliers: ", shape_outliers)

        absr = np.abs(r) #z score

        #print(f"absolute value: {absr}")
        preds = np.where(absr > 2, 1, 0)
        statshape = shapes.flatten()
        statpreds = preds.flatten()
        #print(shapes.shape)
        pred_outliers = np.where((preds >= 1), True, False).sum()
        print("Predicted Outliers: ", pred_outliers)
        print(f"MannWhitney U test:{stats.mannwhitneyu(statpreds, statshape, alternative='two-sided')}")
        print(f"KS test:{stats.ks_2samp(statpreds, statshape)}")

        max_acc = 0
        max_tp = 0
        max_offset = 0
        true_acc = 0
        for offset in range(-2, 3):
            tp, fp, fn, tn = count_agreement_with_offset(preds, shapes, offset)
            accuracy = tp / pred_outliers
            true_accuracy = (tp + tn) / (tp + tn + fp + fn)
            if true_accuracy > true_acc:
                true_acc = true_accuracy
            if accuracy > max_acc:
                max_tp = tp
                max_acc = accuracy
                max_offset = offset

        print(f"Total agreed outlier sites (offset: {max_offset}: {max_tp}")
        print(f"True Accuracy: {true_acc}")
        print(f"Accuracy: {max_acc}")

    lr()
    sys.exit(0)

    def delta_outliers():
        print("Delta outliers......")
        print()
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes = np.where(shapes > .4, 1, 0)
        shape_outliers = np.where((shapes >= 1), True, False).sum()
        df = pd.DataFrame()

        for c in ocols:
            print(c)
            a = unit_vector_norm(bc_acim[c].astype("float32").to_numpy())
            b = unit_vector_norm(bc_dmso[c].astype("float32").to_numpy())
            delta = np.subtract(a, b)
            df[c] = delta
            mean = np.median(delta)  # delta['zscore'].median()
            std = np.std(delta)  # delta['zscore'].std()
            max_acc = 0
            max_tp = 0
            max_offset = 0
            best_k = 0
            true_acc = 0
            for k in range(1,4):
                upper = mean + (k * std)
                lower = mean - (k * std)
                delta1 = np.where((delta > upper) | (delta < lower), 1, 0)
                pred_outliers = np.where((delta1 >= 1), True, False).sum()
                print("Shape Outliers: ", shape_outliers)
                print("Predicted Outliers: ", pred_outliers)


                for offset in range(-2, 3):
                    tp, fp, fn, tn = count_agreement_with_offset(delta1, shapes, offset)
                    accuracy = tp / pred_outliers
                    true_accuracy = (tp + tn) /(tp + tn + fp + fn)
                    if true_accuracy > true_acc:
                        true_acc = true_accuracy
                    if accuracy > max_acc:
                        max_tp = tp
                        max_acc = accuracy
                        max_offset = offset
                        best_k = k
            print(f"Total agreed outlier sites (offset: {max_offset}, k: {best_k}): {max_tp}")
            print(f"True Accuracy: {true_acc}")
            print(f"Accuracy: {max_acc}")
            df['Y'] = shapes
            #df.to_csv("probnn.csv")

    delta_outliers()
    sys.exit(0)


    def iqr_outliers():
        print("IQR outliers......")
        print()
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes = np.where(shapes > .4, 1, 0)
        shape_outliers = np.where((shapes >= 1), True, False).sum()

        for c in ocols:
            print(c)
            a = unit_vector_norm(bc_acim[c].astype("float32").to_numpy())
            b = unit_vector_norm(bc_dmso[c].astype("float32").to_numpy())
            delta = np.divide(a, b)
            delta = np.nan_to_num(delta, posinf=0, neginf=0, nan=0)
            q3, q1 = np.percentile(delta, [75, 25])
            iqr = q3 - q1
            iqr_lower = q1 - (1.5 * iqr)
            iqr_upper = q3 + (1.5 * iqr)
            mean = np.median(delta)  # delta['zscore'].median()
            std = np.std(delta)  # delta['zscore'].std()
            max_acc = 0
            max_tp = 0
            max_offset = 0
            best_k = 0
            for k in range(1, 4):
                upper = mean + (k * std)
                lower = mean - (k * std)
                delta1 = np.where((delta > upper) | (delta < lower), 1, 0)
                pred_outliers = np.where((delta1 >= 1), True, False).sum()
                print("Shape Outliers: ", shape_outliers)
                print("Predicted Outliers: ", pred_outliers)

                for offset in range(-2, 3):
                    tp = count_agreement_with_offset(delta1, shapes, offset)
                    accuracy = tp / shape_outliers
                    if accuracy > max_acc:
                        max_tp = tp
                        max_acc = accuracy
                        max_offset = offset
                        best_k = k
            print(f"Total agreed outlier sites (offset: {max_offset}, k: {best_k}): {max_tp}")
            print(f"Accuracy: {max_acc}")

    def rate_outliers():
        print("Modification rate outliers......")
        print()
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes = np.where(shapes > .4, 1, 0)
        shape_outliers = np.where((shapes >= 1), True, False).sum()

        for c in ocols:
            print(c)
            a = unit_vector_norm(bc_acim[c].astype("float32").to_numpy())
            b = unit_vector_norm(bc_dmso[c].astype("float32").to_numpy())
            delta = np.divide(a, b)
            delta = np.nan_to_num(delta, posinf=0, neginf=0, nan=0)
            print(f"{c} : {np.mean(delta)}")
            n = delta.size
            hist, binedges = np.histogram(delta, bins=n)
            binedges = binedges[:-1] + (binedges[1] - binedges[0]) / 2  # convert bin edges to centers
            f = UnivariateSpline(binedges, hist, s=n)
            probs = hist/n
            ### bound by chebyshevs or brute force threshold ????
            plt.plot(binedges, f(binedges))
            plt.show()
            q3, q1 = np.percentile(delta, [75, 25])
            iqr = q3 - q1
            iqr_lower = q1 - (1.5 * iqr)
            iqr_upper = q3 + (1.5 * iqr)
            mean = np.median(delta)  # delta['zscore'].median()
            std = np.std(delta)  # delta['zscore'].std()
            max_acc = 0
            max_tp = 0
            max_offset = 0
            best_k = 0
            for k in range(1, 4):
                upper = mean + (k * std)  # iqr_upper #mean + (k * std)
                lower = mean - (k * std)  # iqr_lower #mean - (k * std)
                delta1 = np.where((delta > upper) | (delta < lower), 1, 0)
                pred_outliers = np.where((delta1 >= 1), True, False).sum()
                print("Shape Outliers: ", shape_outliers)
                print("Predicted Outliers: ", pred_outliers)

                for offset in range(-2, 3):
                    tp = count_agreement_with_offset(delta1, shapes, offset)
                    accuracy = tp / pred_outliers
                    if accuracy > max_acc:
                        max_tp = tp
                        max_acc = accuracy
                        max_offset = offset
                        best_k = k
            print(f"Total agreed outlier sites (offset: {max_offset}, k: {best_k}): {max_tp}")
            print(f"Accuracy: {max_acc}")
    #rate_outliers()
    #sys.exit(0)
    def original_plotting():
        shapes = df1['Reactivity_shape'].to_numpy()
        shapes = np.where(shapes > .4, 1, 0)
        shape_outliers = np.where((shapes >= 1), True, False).sum()
        #delta = delta[(delta['basecalls']>= threshold) & (delta['shapes']>= threshold)]
        #delta.to_csv("delta_" + str(threshold) + "_" + sequence + ".csv")
        #print(delta.head())
        #print("Agreed reactive sites: ", np.where((shapes >= threshold) & (basecalls >= bcthreshold), True, False).sum())
        #ave_dist = sum(dist)/ (sum(i > 0 for i in dist))
        #print("Average distance: ", ave_dist)
        #print("Agreed sites: ", np.where(np.abs(shapes- basecalls) < np.abs(ave_dist), True, False).sum())
        y_true = np.where(shapes > threshold, 1, 0)
        y_pred = np.where(basecalls > threshold, 1, 0)
        d1 = RocCurveDisplay.from_predictions(y_true, y_pred)
        pt = "AUC of Basecalling vs Shape-CE \n Reactivities for " + sequence
        plt.title(pt)
        fig = plt.gcf()
        dir_name = 'Basecall_Shape_Plots'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        figname = os.path.join(dir_name + '/' + pt + '.png')
        fig.savefig(figname, dpi=300)
        plt.show()
        print("Shapes", np.median(shapes))
        print(np.std(shapes))
        print("Basecall", np.median(basecalls))
        print(np.std(basecalls))
        position1 = len(basecalls)
        position2 = len(shapes)
        basecall_rx = pd.DataFrame({'Position': position1, 'Basecall_Reactivity': basecalls,
                                    'Shape_Reactivity': shapes})
        basecall_rx.to_csv("basecall_shape_" + sequence + ".csv")

        # sum = 1000000
        # for i in range(0, len(basecalls)):
        #     diff = np.abs(np.roll(basecalls, i) - shapes)
        #     tmpsum= np.sum(diff)
        #     if tmpsum < sum:
        #         sum = tmpsum
        #         print(i)
        #         print(sum)



        xmin = 0
        xmax = position1
        ymin = shapes.min()
        ymax = shapes.max()
        x = np.arange(xmin, xmax)
        y_b = np.roll(basecalls[xmin:xmax], 0)
        y_s = shapes[xmin:xmax]

        #### set figure length #####
        figlen = len(x) * .3
        if figlen < 6.4:
            figlen = 6.4
        #### plot modifications #####
        fig = plt.figure(figsize=(figlen, 4))  # 6.4, 4.8 default size
        ax = fig.add_axes([.1, .2, .8, .7])
        ax.margins(.005, .05)
        ax.set_xticks(range(xmin, xmax))
        ax.set_yticks(np.arange(0,1, step=.1))
        ax.set_xticklabels(range(xmin, xmax))
        ax.tick_params(axis='x', direction='out', length=2, rotation=90)
        ax.set_xlabel('Reference Nucleotide Position', fontsize="xx-large")
        ax.set_ylabel('Reactivity', fontsize="xx-large")
        ax.set_title(sequence + " Basecall Reactivity vs SHAPE-CE Peaks", fontsize="xx-large")
        ax.grid(axis='y')
        # proper ticks
        #X = np.arange(len(positions))
        #ax.plot(X, dels, color='r', label="Deletions")
        # ax.bar_label(pps, label_type='edge')
        ax.bar(x,y_b, color='r', label="Basecall Reactivity",
               align="edge", edgecolor="black", width=.2)
        ax.bar(x, y_s, color='g', label="Shape Reactivity",
               align="edge", edgecolor="black", width=-.2)
        #ax.plot(x,y_b, color='r', label="Basecall Reactivity")
        #ax.plot(x, y_s, color='g', label="Shape Reactivity")
        ax.hlines(y=threshold, xmin=xmin, xmax=xmax, color='g', linestyle='dashed')
        ax.legend()
        fig = plt.gcf()
        dir_name = 'Basecall_Shape_Plots'
        if not os.path.exists(dir_name):
             os.makedirs(dir_name)
        figname = os.path.join(dir_name + '/' + sequence +  'Basecall Reactivity vs SHAPE-CE Peaks' + '.png')
        fig.savefig(figname, dpi=300)
        plt.show()





# df = get_shape_continuous()
# print(df.head())
# print(df['Predicted_Shape'].max())
# print(df['Predicted_Shape'].min())
# min = df['Predicted_Shape'].min()
# max = df['Predicted_Shape'].max()
# df['Predicted_Shape'] = np.where(df["Predicted_Shape"] > 0, ((df["Predicted_Shape"] - min)/ (max - min)), 0)
# print(df['Predicted_Shape'].max())
# print(df['Predicted_Shape'].min())
# for sequence in ['E_coli_tmRNA']: #['E_coli_tmRNA', 'HCV', 'T_thermophila', 'FMN_Adaptor', 'Lysine_Adaptor']:
#      plot_bc_shape_peaks(sequence)
#parse_ACIM()
#parse_ACIM_ssbp()
#parse_DMSO()
#parse_DMSO_ssbp()
#ksm()

### ACM Plots
#plot_average_mod_rate(mod="ACIM")
#plot_average_mod_by_pos_rate()
# plot_bc_kde()
#plot_bc_shape_peaks(sequence='Lysine_Adaptor')
#get_bc_reactivity_peaks()
#bias_transitions()
#plot_accuracies()
#plot_coverage()
#shape_statistics()
#kmer_transitions()

#parse_ACIM()

#preprocess_df()

def align_data(mod):
    df = pd.read_csv(dash_path + 'Preprocess/DMSO_signal_preprocess.csv')
    dft, shift = align_positions(df)
    dft.to_csv(dash_path + 'Preprocess/DMSO_signal_preprocess_aligned.csv')

    #Basecall alignment requires shift value from kmer alignment
    f = dash_path + "Basecall/DMSO_mod_rates.csv"
    df = pd.read_csv(f)
    if 'complex' in f:
        print("complex")
        df['contig'].loc[df['contig']=="cen_3'utr"] = "cen_3'utr_complex"
        df['contig'].loc[df['contig']=="ik2_3'utr"] = "ik2_3'utr_complex"
        df['contig'].loc[df['contig']=="cen_FL"] = "cen_FL_complex"
        df['contig'].loc[df['contig']=="ik2_FL"] = "ik2_FL_complex"
    df = align_positions_bc(df, shift)
    df.to_csv(dash_path + 'Basecall/DMSO_mod_rates_aligned.csv')


#add_complex_dmso()
# df = pd.read_csv(dash_path + "Deconvolution/Preprocess/ACIM_signal.txt",
#                  sep='\t')
# preprocess_df_byread(df)
# df = get_shapemap()
# df = df.loc[df['Sequence_Name']=='T_thermophila', ['Predicted_Shape_Map']]
# df['Predicted_Shape_Map'].loc[df['Predicted_Shape_Map'] == 1] = 0
# df['Predicted_Shape_Map'].loc[df['Predicted_Shape_Map'] < 0] = 1
# df.to_csv(dash_path + 'ShapeMap/t_thermophila.csv', index=False)

#preprocess acim file
#df = pd.read_csv(dash_path + "Deconvolution/Preprocess/ACIM_signal.txt", sep='\t')
#preprocess_df_byread(df)

#align dmso decon file
# df = pd.read_csv(dash_path + "Deconvolution/Preprocess/DMSO_decon_signal_preprocess.csv")
# df, shift = align_positions(df)
# df.to_csv(dash_path + "Deconvolution/Preprocess/DMSO_decon_signal_preprocess_aligned.csv",
#           index=False)

#align acim decon file
# df = pd.read_csv(dash_path + "Deconvolution/Preprocess/ACIM_signal_preprocess_by_read.txt")
# df, shift = align_positions_by_read(df)
# df.to_csv(dash_path + "Deconvolution/Preprocess/ACIM_signal_preprocess_by_read_aligned.txt",
#           index=False)

#align_positions_by_read()
#ave_signal_by_read()
#sys.exit(0)
