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
from matplotlib.patches import Ellipse

#### Paper GMM positional induced clusters reflect predictions, nice
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = "/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/Out/"
save_path = "/Users/timshel/structure_landscapes/DashML/Deconvolution/Decon/"

# df = pd.read_csv("/Users/timshel/structure_landscapes/DashML/Shape/tetra_reactivities.txt",
#                 names=['position', 'base', 'reactivity'], sep=',',)
# df['reactivity'] = np.where(df['reactivity']>=.45, 1, 0)
# print(df.head())
# df.to_csv(save_path+"tetra_ce.csv", index=False, header=False)

# df = pd.read_csv("/Users/timshel/structure_landscapes/DashML/Deconvolution/ShapeMap/SepRep_ttRz_profile.txt", sep='\t')
# df = df[['Nucleotide','Reactivity_profile']]
# df['Reactivity_profile'] = np.where(df['Reactivity_profile']>=.45, 1, 0)
# print(df.head())
# df.to_csv(save_path+"tetra_map.csv", index=False, header=False)

# df = pd.read_csv("/Users/timshel/structure_landscapes/DashML/Shape/hcv_reactivities.txt",
#                 names=['position', 'base', 'reactivity'], sep=',',)
# df['reactivity'] = np.where(df['reactivity']>=.45, 1, 0)
# print(df.head())
# df.to_csv(save_path+"hcv_ce.csv", index=False, header=False)

# df = pd.read_csv("/Users/timshel/structure_landscapes/DashML/Shape/ecoli_reactivities.txt",
#                 names=['position', 'base', 'reactivity'], sep=',',)
# df['reactivity'] = np.where(df['reactivity']>=.45, 1, 0)
# print(df.head())
# df.to_csv(save_path+"ecoli_ce.csv", index=False, header=False)

# df = pd.read_csv("/Users/timshel/structure_landscapes/DashML/Deconvolution/ShapeMap/SepRep_tmRNA_profile.txt", sep='\t')
# df = df[['Nucleotide','Reactivity_profile']]
# df['Reactivity_profile'] = np.where(df['Reactivity_profile']>=.45, 1, 0)
# print(df.head())
# df.to_csv(save_path+"ecoli_map.csv", index=False, header=False)

df = pd.read_csv("/Users/timshel/structure_landscapes/DashML/Shape/hc16_reactivities.txt",
                names=['position', 'base', 'reactivity'], sep=',',)
df['reactivity'] = np.where(df['reactivity']>=.45, 1, 0)
print(df.head())
df.to_csv(save_path+"hc16_ce.csv", index=False, header=False)

sys.exit(0)
