import sys, re
import os.path
import platform
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DashML.Deconvolution.BpProbabilities.library as lib
import DashML.Deconvolution.BpProbabilities.datafx_bp as dfx



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

if platform.system() == 'Linux':
    ##### server #####
    data_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/DashML/Deconvolution/BpProbabilities/DMSO/"
    f_path = "/home/jwbear/projects/def-jeromew/jwbear/StructureLandscapes/Deconvolution/BpProbabilities/Interaction_Rx/"
else:
    bp_path = sys.path[1] + "/DashML/Deconvolution/BpProbabilities/DMSO/"
    rx_path = sys.path[1] + "/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/BpProbabilities/Interaction_Rx/"

# TODO add intramolecular probabilities from DMSO
# TODO add path for interaction probabilities

### get probabilities for a single sequence for dmso, eg no cluster reactivities
### could use dmso reactivities ???, need to calculate dmso reactivities
def get_probabilities(seq, seq2=None, c1=None, c2=None, strand=None):
    #dmso reactivities
    if (c1==None) | (c2==None):
        with (open(bp_path + "A" + seq +","+ seq + "_dp5.ps", "r+") as f):
            lines = f.readlines()
    else:
        with (open(rx_path + strand + seq + "_" + str(c1) + "," + seq2 + "_" + str(c2) + "_dp5.ps", "r+") as f):
            lines = f.readlines()

    i = 0
    while not lines[i].strip().__contains__("%start of base pair probability data"):
        i = i + 1
        #print(i)

    probs = {}
    j = i+2
    ## end of probabilities
    endl = ("showpage|end|lbox")
    while j < len(lines):
        #print(j)
        bases = lines[j].strip().split()
        if (re.search(endl, lines[j].strip())) is None:
            key = int(bases[0].strip())
            #print(key)
            child_bases = {}
            child_bases[int(bases[1].strip())] = float(bases[2].strip())
            while (j + 1 < len(lines)):
                j = j + 1
                b = lines[j].strip()
                if (re.search(endl, b)) is not None:
                    break
                b= b.split()
                if int(b[0].strip())==key:
                    child_bases[int(b[1].strip())]=float(b[2].strip())
                else:
                    break
        else: break
        #print(child_bases)
        probs[key] = child_bases
    return probs

### get interaction probabilities from file for dmso by sequence
def get_intrx_probabilities_dmso(seq, seq2=None, c1=None, c2=None, strand=None):
    seqlen = lib.get_seqlen(seq)
    #update bases > seqlen to corrolated single sequence base
    def get_basel(n):
        if n > seqlen:
            return n - seqlen
        else:
            return n
    # dmso reactivities
    if (c1 == None) | (c2 == None):
        with (open(bp_path + "AA" + seq + "," + seq + "_dp5.ps", "r+") as f):
            lines = f.readlines()
    else:
        with (open(rx_path + strand + seq + "_" + str(c1) + "," + seq2 + "_" + str(c2) + "_dp5.ps", "r+") as f):
            lines = f.readlines()



    i = 0
    while not lines[i].strip().__contains__("%start of base pair probability data"):
        i = i + 1
        #print(i)

    probs = {}
    j = i+2 ##duplicate line in interaction file
    endl = ("showpage|end|lbox")
    while j < len(lines):
        #print(j)
        bases = lines[j].strip().split()
        if (re.search(endl, lines[j].strip())) is None:
            #print(bases)
            a = get_basel(int(bases[0].strip()))
            b = get_basel(int(bases[1].strip()))
            probi = float(bases[2].strip())
            #print(key)
            child_bases = {}
            child_bases[b] = probi
            while (j + 1 < len(lines)):
                j = j + 1
                b = lines[j].strip().split()
                if (re.search(endl, lines[j].strip())) is not None:
                    break
                #print(b)
                nexta = get_basel(int(b[0].strip()))
                if nexta==a:
                    nextb = get_basel(int(b[1].strip()))
                    nextprob = float(b[2].strip())
                    # update duplicates child keys with max probability
                    val = child_bases.get(nextb, None)
                    if val==None:
                         child_bases[nextb]=nextprob
                    else:
                        child_bases[nextb] = max(val, nextprob)
                else:
                    break
        else: break
        #print(child_bases)
        #update duplicates main keys by merging children
        childdict = probs.get(a, None)
        if childdict==None:
            probs[a] = child_bases
        else:
            #merge childbases
            for k,v in childdict.items():
                val = child_bases.get(k, None)
                if val==None:
                    child_bases[k] = v
                else:
                    #get max prob if present
                    child_bases[k] = max(v, val)
            probs[a] = child_bases

    # for key, val in probs.items():
    #     print(key)
    #     for b, p in val.items():
    #         print(b, p)
    # sys.exit(0)
    return probs

### get interaction probabilities for dmso or single molecule
def get_intrx_probabilities_dmso(seq, seq2=None, c1=None, c2=None, strand=None):
    seqlen = lib.get_seqlen(seq)
    #update bases > seqlen to corrolated single sequence base
    def get_basel(n):
        if n > seqlen:
            return n - seqlen
        else:
            return n
    # dmso reactivities
    if (c1 == None) | (c2 == None):
        with (open(bp_path + "AA" + seq + "," + seq + "_dp5.ps", "r+") as f):
            lines = f.readlines()
    else:
        with (open(rx_path + strand + seq + "_" + str(c1) + "," + seq2 + "_" + str(c2) + "_dp5.ps", "r+") as f):
            lines = f.readlines()


    i = 0
    while not lines[i].strip().__contains__("%start of base pair probability data"):
        i = i + 1
        #print(i)

    probs = {}
    j = i+2 ##duplicate line in interaction file
    endl = ("showpage|end|lbox")
    while j < len(lines):
        #print(j)
        bases = lines[j].strip().split()
        if (re.search(endl, lines[j].strip())) is None:
            #print(bases)
            a = get_basel(int(bases[0].strip()))
            b = get_basel(int(bases[1].strip()))
            probi = float(bases[2].strip())
            #print(key)
            child_bases = {}
            child_bases[b] = probi
            while (j + 1 < len(lines)):
                j = j + 1
                b = lines[j].strip().split()
                if (re.search(endl, lines[j].strip())) is not None:
                    break
                #print(b)
                nexta = get_basel(int(b[0].strip()))
                if nexta==a:
                    nextb = get_basel(int(b[1].strip()))
                    nextprob = float(b[2].strip())
                    # update duplicates child keys with max probability
                    val = child_bases.get(nextb, None)
                    if val==None:
                         child_bases[nextb]=nextprob
                    else:
                        child_bases[nextb] = max(val, nextprob)
                else:
                    break
        else: break
        #print(child_bases)
        #update duplicates main keys by merging children
        childdict = probs.get(a, None)
        if childdict==None:
            probs[a] = child_bases
        else:
            #merge childbases
            for k,v in childdict.items():
                val = child_bases.get(k, None)
                if val==None:
                    child_bases[k] = v
                else:
                    #get max prob if present
                    child_bases[k] = max(v, val)
            probs[a] = child_bases

    # for key, val in probs.items():
    #     print(key)
    #     for b, p in val.items():
    #         print(b, p)
    # sys.exit(0)
    return probs

### get interaction probabilities for dmso or single molecule
def get_intrx_probabilities_mod(seq, seq2=None, c1=None, c2=None, strand=None):
    seqlen = lib.get_seqlen(seq)
    #update bases > seqlen to corrolated single sequence base
    def get_basel(n):
        return n

    # dmso reactivities
    if (c1 == None) | (c2 == None):
        with (open(bp_path + "AA" + seq + "," + seq + "_dp5.ps", "r+") as f):
            lines = f.readlines()
    else:
        with (open(rx_path + strand + seq + "_" + str(c1) + "," + seq2 + "_" + str(c2) + "_dp5.ps", "r+") as f):
            lines = f.readlines()


    i = 0
    while not lines[i].strip().__contains__("%start of base pair probability data"):
        i = i + 1
        #print(i)

    probs = {}
    j = i+2 ##duplicate line in interaction file
    endl = ("showpage|end|lbox")
    while j < len(lines):
        #print(j)
        bases = lines[j].strip().split()
        if (re.search(endl, lines[j].strip())) is None:
            #print(bases)
            a = get_basel(int(bases[0].strip()))
            b = get_basel(int(bases[1].strip()))
            probi = float(bases[2].strip())
            #print(key)
            child_bases = {}
            child_bases[b] = probi
            while (j + 1 < len(lines)):
                j = j + 1
                b = lines[j].strip().split()
                if (re.search(endl, lines[j].strip())) is not None:
                    break
                #print(b)
                nexta = get_basel(int(b[0].strip()))
                if nexta==a:
                    nextb = get_basel(int(b[1].strip()))
                    nextprob = float(b[2].strip())
                    # update duplicates child keys with max probability
                    val = child_bases.get(nextb, None)
                    if val==None:
                         child_bases[nextb]=nextprob
                    else:
                        child_bases[nextb] = max(val, nextprob)
                else:
                    break
        else: break
        #print(child_bases)
        #update duplicates main keys by merging children
        childdict = probs.get(a, None)
        if childdict==None:
            probs[a] = child_bases
        else:
            #merge childbases
            for k,v in childdict.items():
                val = child_bases.get(k, None)
                if val==None:
                    child_bases[k] = v
                else:
                    #get max prob if present
                    child_bases[k] = max(v, val)
            probs[a] = child_bases

    # for key, val in probs.items():
    #     print(key)
    #     for b, p in val.items():
    #         print(b, p)
    # sys.exit(0)
    return probs

### get interaction probabilities from file
def get_intrx_probabilities(f):
    print("old intrx probabilities, go correct it")
    sys.exit(0)
    with (open(f, "r+") as f):
        lines = f.readlines()
    i = 0
    while not lines[i].strip().__contains__("%start of base pair probability data"):
        i = i + 1
        #print(i)

    probs = {}
    j = i+2 ##duplicate line in interaction file
    endl = ("showpage|end|lbox")
    while j < len(lines):
        #print(j)
        bases = lines[j].strip().split()
        #print(bases)
        if (re.search(endl, lines[j].strip())) is None:
            key = int(bases[0].strip())
            #print(key)
            child_bases = {}
            child_bases[int(bases[1].strip())] = float(bases[2].strip())
            while (j + 1 < len(lines)):
                j = j + 1
                b = lines[j].strip()
                if (re.search(endl, lines[j].strip())) is not None:
                    break
                b= b.split()
                if int(b[0].strip())==key:
                    child_bases[int(b[1].strip())]=float(b[2].strip())
                else:
                    break
        else: break
        #print(child_bases)
        probs[key] = child_bases
    return probs



#todo retrieve probs[b1][b2]
#build and store in massive dict of sequences
#todo get remaining bp probs
#probsrx = bp_probabilities("HCV_rx")

# if probsrx == probs:
#     print("same")

#
####### get max probability for each base
###### used by predict

def adjust_probabilities(df, probsA ,probsAA):
    positions = df['position'].to_numpy()
    #get max bp probability from A intra, AA inter
    def get_base_prob(base):
        pbs = probsA.get(base)
        pbss = probsAA.get(base)
        maxA = 0
        maxAA = 0
        #print(pbs)
        # if key not found must search secondary key
        if pbs is None:
            p = 0
            for v in probsA.values():
                t = v.get(base)
                if t is not None and t > p:
                    p = t
            maxA = p
        else:
            p = 0
            for key, val in pbs.items():
                if val > p:
                    p = val
            maxA = p
        if pbss is None:
            p = 0
            for v in probsAA.values():
                t = v.get(base)
                if t is not None and t > p:
                    p = t
            maxAA = p
        else:
            p = 0
            for key, val in pbss.items():
                if val > p:
                    p = val
            maxAA = p

        #print(maxA, maxAA)
        return max(maxA, maxAA)

    ps = np.zeros(len(positions))
    for i, base in enumerate(positions):
        ps[i] = get_base_prob(base)
        #print(i, ps[i])

    df['base_pair_prob'] = ps
    df['Predict'] = np.where(((df['Predict']==-1) & (df['base_pair_prob'] > .90)), 1, df['Predict'])
    return df

##### get max probabilities between A vs AA
#for same molecule, equal lengths
# positions
def get_max_probabilities_sep(seq, probs1):
    slen = lib.get_seqlen(seq)
    positions = np.arange(1, slen+1)
    #get max bp probability from A intra, AA inter
    def get_base_prob(base):
        pbs = probs1.get(base)
        maxA = 0
        maxAA = 0
        #print(pbs)
        # if key not found must search secondary key
        if pbs is None:
            p = 0
            for v in probs1.values():
                t = v.get(base)
                if t is not None and t > p:
                    p = t
            maxA = p
        else:
            p = 0
            for key, val in pbs.items():
                if val > p:
                    p = val
            maxA = p

        #print(maxA, maxAA)
        return maxA

    ps = np.zeros(len(positions))
    for i, base in enumerate(positions):
        ps[i] = get_base_prob(base)
        #print(i, ps[i])

    return ps

def get_max_probabilities_sep_bases(seq, probs1):
    slen = lib.get_seqlen(seq)
    positions = np.arange(1, slen+1)
    #get max bp probability from A intra, AA inter
    def get_base_prob(base):
        pbs = probs1.get(base)
        maxA = 0
        b2 = 0
        maxAA = 0
        #print(pbs)
        # if key not found must search secondary key
        if pbs is None:
            p = 0
            b = 0
            for v in probs1.values():
                t = v.get(base)
                if t is not None:
                    #get key by value from child dict
                    lkey = list(v.keys())
                    lval = list(v.values())
                    position = lval.index(t)
                    b = lkey[position]
                if t is not None and t > p:
                    p = t
                    lkey = list(v.keys())
                    lval = list(v.values())
                    position = lval.index(t)
                    b = lkey[position]
            maxA = p
            b2 = b
        else:
            p = 0
            b = 0
            for key, val in pbs.items():
                if val > p:
                    p = val
                    b = key
            maxA = p
            b2 = b

        #print(maxA, maxAA)
        return maxA, b2

    ps = np.zeros(len(positions))
    pos = np.zeros(len(positions))
    for i, base in enumerate(positions):
        ps[i], b2 = get_base_prob(base)
        pos[i] = b2
        #print(i, ps[i])

    return ps, pos

### Combines probabilities extracted from files and get max bpp for each base
### Used in Predict Function, based on dmso reactivities only
def get_adjusted_probabilities(df, seq="cen_3'utr"):
    probsA = get_probabilities(seq)
    probsAA = get_intrx_probabilities_dmso(seq)
    # get max probabilities for single molecule folding
    # or intra eg A vs AA
    df = adjust_probabilities(df,probsA, probsAA)
    return df

### Get probabilities using cluster reactivities
# probsB = get_probabilities(seq="cen_3'utr", seq2="ik2_3'utr",c1=0,c2=0, strand="B")
# probsBB = get_intrx_probabilities_dmso(seq="cen_3'utr", seq2="ik2_3'utr",c1=0,c2=0, strand="BB")
# probsAB = get_intrx_probabilities_dmso(seq="cen_3'utr", seq2="ik2_3'utr",c1=0,c2=0, strand="AB")
# #df = get_max_probabilities_sep(seq2="ik2_3'utr",probs1=probsB, probs11=probsBB)
# print(probsB)
# print(probsBB)
# print(probsAB)
#sys.exit(0)

####### get max probability for each base

# def get_max_probabilities(f):
#     probs = get_intrx_probabilities(f)
#     def get_base_prob(base):
#         # if key not found must search secondary key
#         pbs = probs.get(base)
#         #print(pbs)
#         if pbs is None:
#             p = 0
#             for v in probs.values():
#                 t = v.get(base)
#                 if t is not None and t > p:
#                     p = t
#             return base, p
#         else:
#             p = 0
#             k = 0
#             for key, val in pbs.items():
#                 if val > p:
#                     p = val
#                     k = key
#             return k, p
#
#     #store base and max probability pairing
#     df = pd.DataFrame(columns=['base1', 'base2', 'max_prob'])
#     #ps = np.zeros(shape=(len(probs), 2))
#     for i, base in enumerate(probs.keys()):
#         print(base)
#         b, p = get_base_prob(base)
#         df.loc[i] = [base,b,p]
#     df.sort_values(by='base1', inplace=True, ignore_index=True)
#     df.reset_index(inplace=True)
#     return df
#probs = bp_probabilities("HCV_norx")
#sys.exit(0)
