import sys, os, re
import platform
import subprocess
import sys
import pandas as pd
import library as lib
import numpy as np


#RNA basepairing with reactivity is probably better here
# RNAfold -p -d2 --noLP --MEA --shape=HCV_rnafold2.dat < hcv.fa > hcv_bp.out
# RNAcofold -a -d2 --noLP < sequences.fa > cofold.out
# todo bp percentages where predict is true but over 95% can be unmodified
# ignore non-predicted or missing values
#RNAfold -p -d2 --noLP < test_sequenc.fa > test_sequenc.out
# RNAcofold -a -d2 --noLP < sequences.fa > cofold.out
# $ RNAfold --shape=reactivities.dat < sequence.fa
# where the file reactivities.dat is a two column text file with sequence positions (1-based)
# normalized reactivity values (usually between 0 and 2. Missing values may be left out, or assigned a negative score:


#### TODO rerun with correct reactivities (kmeans) from centroid files
#### was still running the first time we wrote this

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/dendrogram/Out/"
    save_path = "/home/jwbear/projects/def-jeromew/jwbear/dendrogram/Dendrogram/Dendrogram_Out/"
else:
    data_path = sys.path[1] + "/DashML/Deconvolution/Dendrogram/Clusters/"
    save_path = sys.path[1] + "/DashML/Deconvolution/Dendrogram/Putative_Structures/"

#out files
def get_putative_mfe(seq, path):
    putative_mfes = path + seq + "_putative_mfes.txt"
    return putative_mfes


def extract_mfes(seq, cluster, clust_type,out_file):
    # create sequence folder
    save_path_dir = save_path + seq
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    #extract putative mfes from file
    mfe = []
    df = pd.DataFrame(columns=['cluster', 'mfe'])
    f = open(out_file, "r")
    lines = f.readlines()
    for i, line in enumerate(lines):
        if (i > 1) and (i <= 6):
            l = re.split('\s|\n', line)
            n = re.sub('[\{\}\[\]\(\)]', '', l[1])
            if re.search('^-|[0-9]', n):
                mfe.append(n)
    f.close()
    #write putative mfes to file
    f = open(get_putative_mfe(seq, save_path_dir + '/'), "a")
    for m in mfe:
        f.write(seq +','+ clust_type +','+ str(cluster) + ',' + m + '\n')
    f.close()
    return

##### get putative structures for cluster centroids #####
# RNAfold -p -a -d2 --noLP --MEA  < sequence.fa > sequence_cofold.out
def getRNAfold(seqname, sequence, clust_num, clust_rx, clust_type):
    try:
        # create sequence save directory
        save_path_dir = save_path + seqname
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)


        #create input file
        f = open(save_path_dir + "/sequence.fa", "w")
        f.write(">" + seqname + "-" + str(clust_num) + "\n" + sequence + "\n")
        f.close()

        ### dat file, print reactivities to properly formatted dat file
        out = save_path_dir + '/' + seqname + "-" + str(clust_num)
        f = open(out + ".dat", "w")
        seqlen = len(sequence)
        for i in range(0, seqlen):
            f.write(str(i + 1) + "\t" + str(clust_rx[i]) + "\n")
        f.close()
        dat_file = "--shape=" + os.path.abspath(out + ".dat")
        print("dat file: ", dat_file)

        #send to rnacofold
        in_path = save_path_dir + "/sequence.fa"
        print(in_path)
        out_path = save_path_dir + '/' + seqname + "_" + str(clust_num) + ".out"
        p1 = subprocess.Popen(["RNAfold", "-p", "-d2", "--noLP", "--MEA", dat_file, in_path],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=save_path_dir + '/')
        #reval = sequence + "\n" + ss + "\n@"
        #p1.communicate(input=reval.encode())
        output = str(p1.communicate(timeout=300)).split("\\n")
        # create output file
        f = open(out_path, "w")
        f.write('\n'.join(output))
        f.close()
        p1.stdout.close()
        p1.stdin.close()
        extract_mfes(seqname, clust_num, clust_type,out_path)
    except subprocess.CalledProcessError as e:
        print(e)
    except Exception as e:
        print(e)
    finally:
        f.close()
        p1.stdout.close()
        p1.stdin.close()

# for each of k reactivities
# get rnafold data with varying reactivities from clusters
def get_putative_structure(seq):
    # get sequences and lengths
    seq_name = seq
    sequence = lib.get_sequence(seq)

    # dataframes of centroids
    df = pd.read_csv(data_path + seq + "_centroids.csv")
    clusters = df['cluster'].unique()
    for cluster in clusters:
        reactivities = df.loc[df['cluster']==cluster, 'centroid'].to_numpy()
        getRNAfold(seq_name, sequence, cluster, reactivities, 'kmeans')

    # dataframes of hamming centroids
    df = pd.read_csv(data_path + seq + "_mode_centroids.csv")
    # scale reactivities
    df['centroid'].loc[df['centroid']==1] = 0
    df['centroid'].loc[df['centroid']==-1] = 2
    clusters = df['cluster'].unique()
    for cluster in clusters:
        reactivities = df.loc[df['cluster'] == cluster, 'centroid'].to_numpy()
        getRNAfold(seq_name, sequence, cluster, reactivities, 'hamming')


get_putative_structure('T_thermophila')
sys.exit(0)
