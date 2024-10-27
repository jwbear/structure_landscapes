import sys, os, re
import platform
import subprocess
import sys
import pandas as pd
import library as lib
import numpy as np


#RNA basepairing with reactivity is probably better here
# RNAeval -v -d2 < input.txt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/dendrogram/Out/"
    save_path = "/home/jwbear/projects/def-jeromew/jwbear/dendrogram/Dendrogram/Dendrogram_Out/"
else:
    data_path = "/Users/timshel/structure_landscapes/DashML/Deconvolution/Dendrogram/Clusters/"
    save_path = "/Users/timshel/structure_landscapes/DashML/Deconvolution/Dendrogram/Putative_Structures/"


##### Get basepair DMSO probabilities for sequences #####
# RNAeval -v -d2 < input.txt
def get_MFE(seq):
    try:
        sequence = lib.get_sequence(seq)
        structure = lib.get_free_structure(seq)
        # create input file
        f = open(save_path + seq + "_native_structure.txt", "w")
        f.write(sequence + "\n" + structure + "\n")
        f.close()
        in_file = save_path + seq + "_native_structure.txt"

        #send to rnaeval
        p1 = subprocess.Popen(["RNAeval", "-v", "-d2", "-i", in_file],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=save_path)
        #reval = sequence + "\n" + ss + "\n@"
        #p1.communicate(input=reval.encode())
        output = str(p1.communicate(timeout=300)).split("\\n")
        # create output file
        out_path = save_path + seq + "_" + "native_MFE.out"
        f = open(out_path, "w")
        f.write('\n'.join(output))
        f.close()
        p1.stdout.close()
        p1.stdin.close()
        #save mfes to file
        extract_mfe(seq,structure,out_path)
    except subprocess.CalledProcessError as e:
        print(e)
    except Exception as e:
        print(e)
    finally:
        f.close()
        p1.stdout.close()
        p1.stdin.close()

def extract_mfe(seq, ss, f):
    # get sequences and lengths
    f = open(f, "r")
    lines = f.readlines()
    for line in lines:
        if ss in line:
            l = line.strip().split(' ')
            mfe = re.sub('\(|\)', '', l[1])
            break
    f.close()
    f = open(save_path + 'native_mfes.txt', 'a')
    f.write(seq + ', ' + mfe + '\n')
    f.close()
    return


for s in ['HCV', 'RNAse_P', 'T_thermophila']:
    get_MFE(s)
sys.exit(0)
