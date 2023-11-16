## ---------------------------------------------------------------------------------------------------------------
## Script name: Performance.py
## Purpose of script: Evaluating the performance of D-HAT using different thresholds
## Author: Loujain Liekah
## Date Created: 15-05-2021
## Copyright (c) Loujain Liekah, 2021
## ----------------------------------------------------------------------------------------------------------------
##

import pandas as pd
import numpy as np
import ProcessBlock
from pathlib import Path
from ast import literal_eval
import os
import sys

# featname is block attribute, fm is the feature matrix
def findminsim(source,target,featmname, fm,truthname,clean):
    try:
        fm['pair'] = list(map(literal_eval, fm['pair']))
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    rv = []
    em = []
    try:
        for col in fm:
            if 'embed' in str(col) or 'pair' in str(col):
                em.append(col)
            elif '_x' in str(col) or '_y' in str(col):
                rv.append(col)
        fm.drop(rv,axis=1,inplace=True)
    except Exception as e:
        print(e)
    fm[['subje1', 'subje2']] = pd.DataFrame(fm['pair'].tolist(), index=fm.index) #, index=featurematrix.index
    truthf = "./Truth/" + truthname + 'truth.json'
    import json
    with open(truthf) as f:
        truth = json.loads(f.read())
    truth = [tuple(l) for l in truth]
    truth = set(truth)
    duplicates = len(truth)

    try:
        included_src = fm['subje1'].nunique()
        included_trg = fm['subje2'].nunique()
        print("included from source", included_src)
        print("included from target", included_trg)
    except Exception as e:
        included_src = None
        included_trg = None
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)

    allthresh = []
    for threshold in np.arange(0.5,1.01,0.1):
        featurematrix =  fm.copy()
        try:
            # Matching with different thresholds [0.5,1]
            threshold = round(threshold,1)
            # threshold = 0.7
            matchscores = ProcessBlock.matchscore(featurematrix, threshold)
            # exclude pairs with matching score < min_thr [0,1] for dirty EM
            for min_thr in np.arange(0, 1.1, 0.1):
                    thresRes = {}
                    thresRes['Truth_duplicates'] = duplicates
                    thresRes['included_src'] = included_src
                    thresRes['included_trg'] = included_trg
                    thresRes['Threshold'] = round(threshold,1)
                    thresRes['minimum_similarity'] = min_thr
                    answer = ProcessBlock.MutMax(matchscores, min_thr, clean)
                    # process feature matrix : get binary vectors, create matching scores
                    # and cluster, return predictions
                    # precision, recall, f1 = evaluate(answer,truthname)
                    # Evaluate Performance, avoid case sensitive ids, lower all
                    # truth = {(i.lower(), j) if isinstance(i,str) else (i,j)  for i, j  in truth }
                    # answer= {(i.lower(), j)  if isinstance(i,str) else(i,j) for i, j  in answer}
                    tp = len(list(set(answer).intersection(set(truth))))
                    thresRes['TP'] = tp
                    fp = len(answer) - tp
                    thresRes['FP'] = fp
                    fn = len(truth) - tp
                    thresRes['FN'] = fn
                    precision = (tp) / ((tp) + (fp) + 0.0001)
                    thresRes['Precision'] = precision
                    recall = (tp) / ((tp) + (fn) + 0.0001)
                    thresRes['Recall'] = recall
                    f1 = 2 * precision * recall / (precision + recall + 0.0001)
                    thresRes['F1'] = f1
                    thresRes['Source'] = source
                    thresRes['Target'] = target
                    thresRes['BlockingKey'] = featmname
                    allthresh.append(thresRes)
                    print("threshold", "%.1f" %threshold, "min_match", min_thr, "precision", "%.3f" % precision, "recall",
                          "%.3f" % recall, "f1", "%.3f" % f1)
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    varpath = str(source) + str(featmname)
    varpath = varpath.replace('/', '').replace('.', ' ')
    resName = Path("./ResEM/" + varpath + str(threshold) + ".csv")
    df_res = pd.DataFrame(allthresh)
    df_res.to_csv(resName, index=False)

    # Choose the threshold with the best F1
    bestperf = df_res.groupby('Threshold', group_keys=False).apply(lambda x: x.loc[x.F1.idxmax()])
    print(bestperf.loc[:, bestperf.columns.isin(['Threshold', 'minimum_similarity', 'F1'])])
    return bestperf

# calculates p,r,f1 for a given answer and truth
def evaluate(answer, truthname):
    truthf = "./Truth/" + truthname + 'truth.json'
    import json
    with open(truthf) as f:
        truth = json.loads(f.read())

    truth = [tuple(l) for l in truth]
    truth = set(truth)
    tp = len(list(set(answer).intersection(set(truth))))
    fp = len(answer) - tp
    fn = len(truth) - tp
    precision = (tp) / ((tp) + (fp) + 0.0001)
    recall = (tp) / ((tp) + (fn) + 0.0001)
    f1 = 2 * precision * recall / (precision + recall + 0.0001)
    return precision, recall, f1
