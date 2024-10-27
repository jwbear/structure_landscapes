#### get conserved regions for structural landscape #####
#### use 30 centroids (kmeans/hamming) and count across landscape #####
#### set threshold percentage of landscape after calculation ####
#### conserved regions are more interesting for conformational or interaction changes ####

import sys, os
import re
import platform
import sys
import pandas as pd
import numpy as np
import varnaapi
import library as lib

varnaapi.set_VARNA(sys.path[1] + '/DashML/VARNA/VARNAv3-93.jar')



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Deconvolution/Out/"
    save_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Dendrogram/Clusters/"
else:
    data_path = sys.path[1] + "/DashML/Deconvolution/Dendrogram/Clusters/"
    save_path = sys.path[1] + "/DashML/Deconvolution/Dendrogram/Figures/Conserved_Regions/"



# get % conserved for each position
# where conserved is greater than eq to size of dominant cluster || some percentage
# Hamming 10%, Kmeans 6%
# highly conserved > 50%

#TODO add skip function in highly reactive regions to improve coverage eg if one base in 3 is
# unreactive still consider it a region

def getConservedRegions(df, cons_thresh=.1, metric='hamming'):
    clust_num = 30
    region_size = 3
    #convert nonbinary df (kmeans)
    if metric != 'hamming':
        df['centroid'] = np.where(df['centroid'] > .8, -1, 1)
    df = (df.groupby(['position', 'centroid'], observed=False).size().
                       unstack(fill_value=0).reset_index())
    df['percent_mod'] = df[-1]/clust_num
    df['percent_unmod'] = df[1]/clust_num


    #print(df)
    cbp = []
    css = []
    bpregion = []
    rxregion = []
    i = 0

    def is_gap(i, gap=1):
        if df['percent_mod'][i + gap] >= cons_thresh:
            return True
        else: return False

    while i < len(df):
        if df['percent_mod'][i] >= cons_thresh:
            rxregion.append(i+1)
            i = i + 1
            while i < len(df):
                if df['percent_mod'][i] >= cons_thresh:
                    rxregion.append(i+1)
                    i  = i + 1
                #gap for highly reactive regions
                elif df['percent_mod'][i] <= cons_thresh and is_gap(i,0):
                    rxregion.append(i + 1)
                    rxregion.append(i + 2)
                    i = i + 2
                else:
                    if len(rxregion)>=region_size:
                        css.append(rxregion)
                    rxregion=[]
                    break

        elif df['percent_unmod'][i] >= (1 -cons_thresh):
            bpregion.append(i+1)
            i = i + 1
            while i < len(df):
                if df['percent_unmod'][i] >= (1- cons_thresh):
                    bpregion.append(i+1)
                    i  = i + 1
                else:
                    if len(bpregion)>=region_size:
                        cbp.append(bpregion)
                    bpregion = []
                    break

    return css, cbp

#get secondary structure
 # TODO try save yml
def save_bpseq(seq, cluster, structure):
    sequence = structure.sequence
    structure = structure.structure
    seqlen = lib.get_seqlen(seq)

    base_pairs = []
    left = []
    for i in range(len(structure)):
        if re.search('\(', structure[i]):
            left.append(i + 1)
        if re.search('\)', structure[i]):
            base_pairs.append([left.pop(), i + 1])
    base_pairs.sort(key=lambda tup: (tup[0], tup[1]))
    print(base_pairs)

    f = open(save_path + seq + "_" + str(cluster) + ".bpseq", "w")
    f.write('# ' + seq + ' ' + str(cluster) + '\n')
    n = 0
    for i in range(1,seqlen+1):
        #print(i)
        if (n < len(base_pairs)) and (base_pairs[n][0] == i):
            f.write(str(i) + ' ' + sequence[i-1] + ' ' + str(base_pairs[n][1]) + '\n')
            n = n + 1
        else:
            bp2 = [x[1] for x in base_pairs]
            try:
                m = bp2.index(i)
                f.write(str(i) + ' ' + sequence[i-1] + ' ' + str(base_pairs[m][0]) + '\n')
            except ValueError:
                f.write(str(i) + ' ' + sequence[i-1] + ' ' + str(0) + '\n')
    f.write('\n')
    f.close()
    return

def get_vplot(seq, cons_ss, cons_bp, metric='hamming'):
    # get sequences and lengths
    seq_name = seq
    seq_len = lib.get_seqlen(seq)
    sequence = lib.get_sequence(seq)
    #get secondary structure
    ss = lib.get_free_structure(seq)

    v = varnaapi.Structure(structure=ss, sequence=sequence)
    v.update(resolution=10, zoom=1, algorithm='radiate', flat=True)
    #save_bpseq(seq, cluster, v)
    out_fig = save_path + seq + "_" + metric + ".png"
    v.dump_param(save_path + seq + "_" + metric + ".yml")
    #annotating high reactivity regions.
    for r in cons_ss:
        v.add_highlight_region(r[0],r[-1], fill='#f16849', outline='#f16849')
    #conserved inaccessible regions
    for r in cons_bp:
        v.add_highlight_region(r[0],r[-1], fill='#c5def2', outline='#c5def2')
    #v.add_colormap(values=np.arange(1, 10), vmin=30, vmax=40, style='bw')
    #values is an array where each position indicates color 0-n
    #overall style is applied
    # annotating interactions
    cmap = np.ones(seq_len)
    v.add_colormap(values=[3], style='energy')
    #v.add_aux_BP(1, 10, color='red')
    v.savefig(out_fig)
    v.show()


    #v = varnaapi.FileDraw('/Users/timshel/NanoporeAnalysis/DashML/VARNA/RNAseP_as-in-original-paper.bpseq')
    #df = pd.read_csv("/Users/timshel/NanoporeAnalysis/DashML/ShapeMap/varna_RNAse_P.csv")
    #df['Position'] = df['Position'] + 1
    #varnaapi.load_config('/Users/timshel/NanoporeAnalysis/DashML/VARNA/RNAseP_as-in-original-paper.bpseq')
    # sequence = 'GUUAAUCAUGCUCGGGUAAUCGCUGCGGCCGGUUUCGGCCGUAGAGGAAAGUCCAUGCUCGCACGGUGCUGAGAUGCCCGUAGUGUUCGUGCCUAGCGAAUCCAUAAGCUAGGGCAGCCUGGCUUCGGCUGGGCUGACGGCGGGGAAAGAACCUACGUCCGGCUGGGAUAUGGUUCGAUUACCCUGAAAGUGCCACAGUGACGGAGCUCUAAGGGAAACCUUAGAGGUGGAACGCGGUAAACCCCACGAGCGAGAAACCCAAAUGAUGGUAGGGGCACCUUCCCGAAGGAAAUGAACGGAGGGAAGGACAGGCGGCGCAUGCAGCCUGUAGAUAGAUGAUUACCGCCGGAGUACGAGGCGCAAAGCCGCUUGCAGUACGAAGGUACAGAACAUGGCUUAUAGAGCAUGAUUAACGUC'
    # ss = "(((((((((((((((((((((.(((((((((....)))))))))...[.[[.[[[[[(((((((((.(((...).)))))).....((((((((((((........)))))))((((((((((....))))))))).)((.(((((((((((((..((((.....))))).))))))).))))).....(((((............((((((((....)))))))).........)))..))))))))))))))...((((.........))))...((((((((((((.(...)))))))))))))(((((((........)))))))........)))))))((((((((((..(((.....))).......))))))))))......]]]]]]]].).)))))))))))))..."
    # v = varnaapi.Structure(structure=ss, sequence=sequence)
    #v = varnaapi.FileDraw('/Users/timshel/NanoporeAnalysis/DashML/VARNA/RNAseP_as-in-original-paper.bpseq')
    #v.savefig("RNAse_Pp.png")







#data sets
#cluster,position,centroid
seq = 'T_thermophila'
df = pd.read_csv(data_path + seq + "_mode_centroids.csv")
cons_ss, cons_bp = getConservedRegions(df, .1, metric='hamming')
# plot native structure with conserved regions in varna
get_vplot(seq, cons_ss, cons_bp, metric='hamming')
df = pd.read_csv(data_path + seq + "_centroids.csv")
cons_ss, cons_bp = getConservedRegions(df, .4, metric='kmeans')
# plot native structure with conserved regions in varna
get_vplot(seq, cons_ss, cons_bp, metric='kmeans')
sys.exit(0)
