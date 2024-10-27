# Calculate accuracy metrics on a method using the prediction output in the dataframe,
# df, must contain position, contig

import sys, re
import os.path
import traceback
import numpy as np
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
import data_fx as dfx

def get_metrics():
    # get metrics
    df_ce = dfx.get_shape()
    # print(df_ce.columns)
    df_map = dfx.get_shapemap()
    # print(df_map.columns)
    df_shape = df_map.merge(df_ce, on=['Position', 'Sequence', 'Sequence_Name'], how="left")
    df_shape = df_shape[['Position', 'Sequence_Name', 'Sequence', 'Reactivity_shape', 'Reactivity_profile',
                         'Predicted_Shape', 'Predicted_Shape_Map']]
    dfs = dfx.get_structure_ext()
    df = dfs.merge(df_shape, on=['Position', 'Sequence', 'Sequence_Name'])
    #print(df.columns)
    # get measurement metrics; ignoring base-paired bases except when indicated by both shapes that they are
    # acessiblel to Acim
    # an unranked base
    df['Metric'] = 5
     # single stranded bases
    df.loc[(df['BaseType'] == 'S') & (df['Metric'] == 5) , ['Metric']] = 4
    # base paired  that is part of a loop structure and predicted by SHAPE
    df.loc[((df['StructureType'] == 'M') | (df['StructureType'] == 'I') | (df['StructureType'] == 'E')) &
           (df['Metric'] >= 3), ['Metric']] = 3
    # bases detected either by SHape CE or Shape Map
    df.loc[
        ((df['Predicted_Shape'] == -1) | (df['Predicted_Shape_Map'] == -1)) & (df['Metric'] >= 3), ['Metric']] = 2
    # part of a loop structure and predicted by SHAPE
    df.loc[((df['StructureType'] == 'M') | (df['StructureType'] == 'I') | (df['StructureType'] == 'E')) &
           ((df['Predicted_Shape'] == -1) | (df['Predicted_Shape_Map'] == -1)) & (df['Metric'] != 1), ['Metric']] = 0
    # bases detected by both Shape CE and Shape MAp
    df.loc[(df['Predicted_Shape'] == -1) & (df['Predicted_Shape_Map'] == -1), ['Metric']] = 1


    df = df[['Position', 'Sequence_Name', 'Sequence', 'Predicted_Shape', 'Predicted_Shape_Map', 'Metric']]
    return df

def get_Metric(df, seq=""):
    dfm = get_metrics()
    dfm.rename(columns={'Position':'position', 'Sequence_Name':'contig'}, inplace=True)
    dfm.drop(columns=['Sequence'], inplace=True)
    df = df.merge(dfm, on=['position', 'contig'], how="left")
    #remove sequences with no shape data
    df = df.dropna()

    if len(df) > 1 :
        # metric counts
        zero = np.where((df['Predict']==-1) & (df['Metric']==0),1,0).sum()
        ztot = np.where((df['Metric']==0),1,0).sum()
        one = np.where((df['Predict']==-1) & (df['Metric']==1),1,0).sum()
        otot = np.where((df['Metric']==1),1,0).sum()
        two = np.where((df['Predict']==-1) & (df['Metric']==2),1,0).sum()
        ttot = np.where((df['Metric']==2),1,0).sum()
        three = np.where((df['Predict']==-1) & (df['Metric']==3),1,0).sum()
        thtot = np.where((df['Metric']==3),1,0).sum()
        four = np.where((df['Predict'] == -1) & (df['Metric'] == 4), 1, 0).sum()
        ftot = np.where((df['Metric'] == 4), 1, 0).sum()
        smap = np.where((df['Predict'] == -1) & (df['Predicted_Shape_Map'] == -1), 1, 0).sum()
        smaptot = np.where((df['Predicted_Shape_Map'] == -1), 1, 0).sum()
        sce =  np.where((df['Predict'] == -1) & (df['Predicted_Shape'] == -1), 1, 0).sum()
        scetot =  np.where((df['Predicted_Shape'] == -1), 1, 0).sum()



        mline = "ShapeMap Agreed Detection: Predicted: " + str(smap) + " Total Shape Map: " + str(smaptot) + ", (%" + \
                 str((smap / smaptot) * 100) + ")" + "\n" + \
            "ShapeCE Agreed Detection: Predicted: " + str(sce) + " Total ShapeCE: " + str(scetot) + ", (%" + \
                 str((sce / scetot) * 100) + ")" + "\n" + \
            "ShapeCE and ShapeMap Agreed Detection: \n" + \
            "\t Predicted: " + str(one) + "\n" + \
            "\t Shape Agreed: " + str(otot) + ", (%" + str((one / otot) * 100)+ ")" + "\n" + \
            "Annotated Shape XOR Detected Loops: \n" + \
            "\n\tPredicted: " + str(zero) + "\n\tShape Predicted: " + str(ztot) + ", (%" + str((zero/ztot) * 100) +  ")\n" + \
            "Single Stranded: \n\tPredicted: " +  str(four) + \
            "\n\tShape Predicted: " + str(ftot) + ", (%" + str((four / ftot) * 100) + ")\n" + \
            "Predicted Shape XOR Detection: \n\tPredicted:" + str(two) + \
            "\n\tShape Predicted: " + str(ttot) + ", (%" +  str((two/ttot) * 100) + ")\n" + \
            "Base Paired Loop Positions: \n\tPredicted: " + str(three) + \
            "\n\tShape Predicted: " + str(thtot) + ", (%" + str((three/thtot) * 100) + ")\n"

        fp = np.where((df['Predict'] == -1) & (df['Metric']==5), 1, 0).sum()
        fn = np.where((df['Predict'] == 1) & (df['Metric']==5), 1, 0).sum()
        tp = np.where((df['Predict'] == -1) & (df['Metric']<5),1,0).sum()
        tn = np.where((df['Predict'] == 1) & (df['Metric']>=4), 1, 0).sum()
        pos =np.where((df['Metric']<4),1,0).sum()
        neg =np.where((df['Metric']>=4),1,0).sum()

        pline = "Negatives: " + str(neg) + "\n" + \
            "Positives: " + str(pos)  + "\n" + \
            "True Positives: " + str(tp) + " Total: " +  str(pos) + ", %" + str((tp/(pos)) *100)  + "\n" + \
            "False Positives: " + str(fp) + " Total: " + str(neg) + ", %" + str((fp /neg)*100)  + "\n" + \
            "True Negatives: " + str(tn) + " Total: " + str(neg) + ", %" + str((tn / (neg))*100)  + "\n" + \
            "False Negatives: " + str(fn) + " Total: " +  str(neg) + ", %" + str((fn / neg)*100)  + "\n"

        print(str(mline))
        print(str(pline))

        if seq=="":
            s = df['contig'].unique()
            if len(s) == 1:
                seq = str(s)
            else:
                seq = str(random.randint(10, 10000 )) + str(s[0])

        f = open("/Users/timshel/NanoporeAnalysis/DashML/Deconvolution/Out/Metrics/"+ str(seq) +"_"+ "metric.txt", "w")
        f.write(str(mline))
        f.write(str(pline))
        f.close()

        # count per sequence, metrics

    return df
