## ---------------------------------------------------------------------------------------------------------------
## Script name: features.py
## Purpose of script: Deploying the various similarity functions on the attribute values of candidate pairs. 
## Author: Loujain Liekah
## Date Created: 15-05-2021
## Copyright (c) Loujain Liekah, 2021
## ----------------------------------------------------------------------------------------------------------------
##

import nltk
import Levenshtein as l
import numpy as np
from scipy import stats,spatial
from fastDamerauLevenshtein import damerauLevenshtein
import pandas as pd
import collections
import sys, os

def embed_eucscipy(x,y):
    try:
        x = np.asarray(x, dtype='float64')
        y = np.asarray(y, dtype='float64')
        sim = 1/(1+spatial.distance.euclidean(x,y))
        return sim
    except:
        try:
            if x is None or y is None:
                return np.nan
            if pd.isnull(x) or pd.isnull(y):
                return np.nan
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)

def embed_cosinscipy(x,y):
    try:
        x = np.asarray(x, dtype='float64')
        y = np.asarray(y, dtype='float64')
        sim =  1 - spatial.distance.cosine(x,y)
        return sim
    except:
        try:
            if x is None or y is None:
                return np.nan
            if pd.isnull(x) or pd.isnull(y):
                return np.nan

        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)

def wordmover(x,y):
    try:
        x = np.asarray(x, dtype='float64')
        y = np.asarray(y, dtype='float64')
        sim = 1/(1+stats.wasserstein_distance(x,y))
        return sim
    except:
        try:
            if x is None or y is None:
                return np.nan
            if pd.isnull(x) or pd.isnull(y):
                return np.nan
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)

# numeric similarity measures
# for boolean and numbers!
def is_equal(x, y):
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(x) in nuls or str(y) in nuls:
        return np.nan
    if x is None or y is None:
        return np.nan
    if pd.isnull(x) or pd.isnull(y):
        return np.nan
    try:
        if float(x) == float(y):
            return 1.0
        return 0.0
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

# p = 1 manhatan, p = 2 euclidean.. etc
def minkowski_dis(num1,num2,p):
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(num1) in nuls or str(num2) in nuls:
        return np.nan
    if num1 is None or num2 is None:
        return np.nan
    if pd.isnull(num1) or pd.isnull(num2):
        return np.nan
    try:
        num1 = float(num1)
        num2 = float(num2)
        dist = (abs(num1 - num2)**float(p))**(1/float(p))
        return round(1.0 - dist, 5)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

def euclid(num1,num2):
    if num1 is None or num2 is None:
        return np.nan
    if pd.isnull(num1) or pd.isnull(num2):
        return np.nan
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(num1) in nuls or str(num2) in nuls:
        return np.nan
    try:
        return minkowski_dis(num1,num2,2)
    except :
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

# compute normalized manhattan similarity
def abs_norm(x, y):
    if x is None or y is None:
        return np.nan
    if pd.isnull(x) or pd.isnull(y):
        return np.nan
    try:
        x = float(x) # value error
        y = float(y)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        return np.nan
    if x == 0.0 and y == 0.0:
        return 0
    else:
        try:
            d = (abs(x - y) / max(abs(x), abs(y)))
            if d <= 10e-5:
                d = 0
            return 1.0 - d
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)

# Code from py_entitymatching
def rel_sim(d1, d2):
    if d1 is None or d2 is None:
        return np.nan
    if pd.isnull(d1) or pd.isnull(d2):
        return np.nan
    try:
        d1 = float(d1)
        d2 = float(d2)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
    if d1 == 0.0 and d2 == 0.0:
        return 0
    else:
        # Compute the relative difference between two numbers
        try:
            x = 1- (abs(d1 - d2)) / max(d1 , d2)  # TypeError
            return x
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)

def func(typefunc):
    sim_names = [ "euclid", "equal", "cosine", "embed_euc", 'wordmover',#'energydistance',
                 "leven","edit","containment", "tokenjac","bag","overlapjac","jac" ]
    sim_func= [euclid,is_equal,embed_cosinscipy,embed_eucscipy, wordmover,#energydistance,
               leven,editns,jaccard_token_sim,bag_sim,overlap_jac_coef,jaccard_similarity]
    dictfunc = dict(zip(sim_names,sim_func))

    dictfunc = {k: dictfunc.get(v, v) for k, v in typefunc.items()}
    return dictfunc

# takes columntypes and return functions on eachtype
def type_simfun(types,entityid="subject_id"):
    typefunc = {}
    for key,value in types.items():
        if str(entityid) in str(key) or "id" in str(key):
            pass
        elif "Categorical" in str(value):
            typefunc[str(key)] = [is_equal]
        elif pd.api.types.is_numeric_dtype(value):
            typefunc[str(key)] = [rel_sim, abs_norm,euclid,is_equal]
        elif "Short" in str(value) and "combined" not in str(key):
            typefunc[str(key)] = [leven,editns,containment_sim,jaccard_token_sim,bag_sim,overlap_jac_coef,jaccard_similarity]
        elif "Long" in str(value) and "combined" not in str(key):
            typefunc[str(key)] = [containment_sim,jaccard_token_sim,bag_sim,overlap_jac_coef,jaccard_similarity]
            # add euclidean and WMD
        elif "vector" in str(value):
            typefunc[str(key)] = [embed_cosinscipy, embed_eucscipy,wordmover]

        # rest is either categorical or boolean
        elif pd.api.types.is_bool_dtype(value):
            typefunc[str(key)] =[is_equal]
    return typefunc

# short strings
def str_sim(text1, text2, method):
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(text1) in nuls or str(text2) in nuls:
        return np.nan
    if text1 is None or text2 is None:
        return np.nan
    if pd.isnull(text1) or pd.isnull(text2):
        return np.nan
    try:
        text1 = str(text1).lower()
        text2 = str(text2).lower()
        max_length = max(len(text1), len(text2))
        if method == 'edit':
            # similarity is complement of normalized distance, keep 5 numbers after comma
            return round(1.0 - damerauLevenshtein(text1, text2,similarity = False)/max_length+0.0001, 5)
        elif method == 'leven':
            return round(1.0 - l.distance(text1, text2)/(max_length+0.0001), 5)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

def leven(text1,text2):
    try:
        return str_sim(text1, text2, 'leven')
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

def editns(text1,text2):
    try:
        return str_sim(text1, text2, 'edit')
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

#define a flexible method of aggregation
def aggregate_scores(scores, method):
    if method == 'mean':
        sim_scores = np.nanmean(scores, axis=1)
    return sim_scores

def jaccard_similarity(text1, text2):
    # print("Jaccard for", text1, "AND", text2)
    text1 = str(text1)
    text2 = str(text2)
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(text1) in nuls or str(text2) in nuls:
        return np.nan
    if text1 is None or text2 is None:
        return np.nan
    if pd.isnull(text1) or pd.isnull(text2):
        return np.nan
    try:
            s1 = set(str(text1).split())
            s2 = set(str(text2).split())
            if s1 in nuls or s2 in nuls:
                return np.nan
            else: return round(float(len(s1.intersection(s2)) / (len(s1.union(s2))+0.0001)),5)
    except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)

# intersection normalized by smallest set
def overlap_jac_coef(text1, text2):
    text1 = str(text1)
    text2 = str(text2)
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(text1) in nuls or str(text2) in nuls:
        return np.nan
    if text1 is None or text2 is None:
        return np.nan
    if pd.isnull(text1) or pd.isnull(text2):
        return np.nan
    try:
        s1 = set(str(text1).split())
        s2 = set(str(text2).split())
        if s1 in nuls or s2 in nuls:
            return np.nan
        else:
            return round(float(len(s1.intersection(s2)) / (min(len(s1),len(s2)) + 0.0001)), 5)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        # return np.nan

def get_relaxed_jaccard_sim(str1, str2, n_grams=1):
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(str1) in nuls or str(str2) in nuls:
        return np.nan
    if str1 is None or str2 is None:
        return np.nan
    if pd.isnull(str1) or pd.isnull(str2):
        return np.nan
    try:
            a = set(str1.split())
            b = set(str2.split())
            if not a or not b: return np.nan
            c = []
            for a_ in a:
                for b_ in b:
                    if str_sim(a_, b_,'leven') > 0.7:
                        c.append(a_)
            intersection = len(c)
            min_length = min(len(a), len(b))
            if intersection > min_length:
                intersection = min_length
            return float(intersection)/ float(len(a) + len(b) - intersection+0.0001)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

# Short and Long
def containment_sim(str1, str2):
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    if str(str1) in nuls or str(str2) in nuls:
        return np.nan
    if str1 is None or str2 is None:
        return np.nan
    if pd.isnull(str1) or pd.isnull(str2):
        return np.nan
    try:
        if (len(set(str1.split())) > 1 and len(set(str2.split())) > 1):
            a = set(str1.split())
            b = set(str2.split())
        else:  # consider the characters if one word
            a = set(str1)
            b = set(str2)
        c = a.intersection(b)
        if str(str1) in nuls or str(str2) in nuls:
            return -1.0
        elif len(a) == 0 or len(b) == 0:
            return np.nan
        else:
            return float(len(c)) / float(min(len(a), len(b))+0.0001)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

# long and short jaccard
def jaccard_token_sim(str1, str2):
    nuls = ['nan', 'NA', 'NaN', '<NA>', '']
    if str(str1) in nuls or str(str2) in nuls:
        return np.nan
    if str1 is None or str2 is None:
        return np.nan
    if pd.isnull(str1) or pd.isnull(str2):
        return np.nan
    str1 = str(str1)
    str2 = str(str2)
    try:
        return 1 - nltk.jaccard_distance(set(str1), set(str2))
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)

#short and long string
def bag_sim(str1,str2):
    nuls = ['nan', 'NA', 'NaN', '<NA>', '']
    if str(str1) in nuls or str(str2) in nuls:
        return np.nan
    if str1 is None or str2 is None:
        return np.nan
    if pd.isnull(str1) or pd.isnull(str2):
        return np.nan
    str1 = str(str1)
    str2 = str(str2)
    try:
        len_str1 = len(str1)
        len_str2 = len(str2)
        if len_str1 == 0:
            return len_str2
        if len_str2 == 0:
            return len_str1
        bag1 = collections.Counter(str1)
        bag2 = collections.Counter(str2)
        size1 = sum((bag1 - bag2).values())
        size2 = sum((bag2 - bag1).values())
        # returning the max of difference of sets
        maxleng= max(size1, size2)
        string1_len = len(str1)
        string2_len = len(str2)
        if string1_len == 0 and string2_len == 0:
            return 1.0
        return 1 - (maxleng / max(string1_len, string2_len))
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
