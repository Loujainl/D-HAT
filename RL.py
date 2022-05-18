import numpy as np
import itertools
from transformers import logging
import sys
import os
import py_entitymatching as em
import pandas as pd
from ast import literal_eval
import pint

ureg = pint.UnitRegistry()
pd.options.mode.chained_assignment = None  # default='warn'
logging.set_verbosity_error()

def convertstrToNum(col):
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    try:
        colcon = col.apply(lambda x: ureg(x).magnitude if str(x) not in nuls else np.nan)
        colcon = colcon.convert_dtypes(convert_string=True, convert_boolean=False, convert_integer=False, convert_floating=False)
        print(col,"string column successfully transformed to float")
        return colcon
    except:
        return col

def inBlock(truth,featureMatrix):

    pairss = featureMatrix['pair'].apply(lambda x: literal_eval(str(x)))

    res = truth.intersection(set(pairss))
    return len(res)

# Assessment of Blocking
def intheBlock(truth,blocks,idfeat):
    res = {}
    sid = "ltable_" + str(idfeat)
    tid = "rtable_" + str(idfeat)
    pairss = set(zip(blocks[sid],blocks[tid]))
    truth = {item.lower() if isinstance(item, str) else item for item in truth}
    pairss = {item.lower() if isinstance(item, str) else item for item in pairss}
    tp = len(truth.intersection(set(pairss)))
    fp = blocks.shape[0] - tp
    fn = len(truth) - tp
    precision = (tp) / ((tp) + (fp) + 0.0001)
    recall = (tp) / ((tp) + (fn) + 0.0001)
    f1 = 2 * precision * recall / (precision + recall + 0.0001)
    res['block_precision'] = precision
    res['block_recall'] = recall
    res['block_f1'] = f1
    return res, f1, precision

# Not Used
def magellan_attrequivblock(source, target,key,attribute):
    ab = em.AttrEquivalenceBlocker()
    l = list(source.columns)
    l.remove(key)
    # print(l)
    r = list(target.columns)
    r.remove(key)
    C = ab.block_tables(source, target, l_block_attr=attribute, r_block_attr=attribute, l_output_attrs=l,
        r_output_attrs=r, allow_missing=False)
    return C

def block_candidate(blocks,attribute)   :
    ob = em.OverlapBlocker()
    C2 = ob.block_candset(blocks, attribute, attribute, word_level=True, overlap_size=2, show_progress=True)
    return C2


def magellan_overlapblock(source,target,keysource,keytarget, blockattribute):
    print(source.columns,target.columns,keysource,keytarget, blockattribute)
    em.set_key(source, str(keysource))
    em.set_key(target, str(keytarget))
    ob = em.OverlapBlocker()
    l = list(source.columns)
    l.remove(keysource)
    r = list(target.columns)
    r.remove(keytarget)
    C = ob.block_tables(source, target, str(blockattribute), str(blockattribute),
                        l_output_attrs=l,
                        r_output_attrs=r)
    return C


# Keep only attributes with same column name
def schematcher(df1,df2):
    try:
            aligned = set(df1.columns).intersection(set(df2.columns))

            for col in df1.columns:
                if col not in aligned:
                    df1.drop(col, axis=1,inplace = True)
                else:
                    pass
                    # merge.append(col)
            for col2 in df2.columns:
                if col2 not in aligned:
                     df2.drop(col2, axis=1,inplace = True)
    except Exception as e:
            print(e)
    return df1, df2

# long or short string
def checkString(column):
    column_values = column.dropna()
    length = 0
    for value in column:
        value = str(value)
        length = length + len(value.split())

    avg_length = length / (len(column_values)+0.0001)

    if (avg_length > 4):
        typestring ='Long String'
    else:
        typestring = 'Short String'
    return typestring

def columnsTypes(data):
    types = dict()
    try:
        for col in data:
            column_values = data[col].dropna()
            types1= str(column_values.dtypes)
            if "pair" in str(col):
                    pass
            if "embed" in str(col):
                    types[col] = "vector"
            elif pd.api.types.is_string_dtype(types1):
                numberunique = data[col].nunique()
                if numberunique <= 6:
                    types[col] = "Categorical"
                else:
                    types[col] = checkString(data[col])
            else:
                    types[col] = types1
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return types


# Not used
def fill_null(data):
    print("Filling Nulls")
    for col in data:
        dt = data[col].dtype
        if dt == int or dt == float:
            data[col].fillna(data[col].mean(), inplace=True)
            data[col] = data[col].convert_dtypes()
        else:
            data[col].fillna("", inplace=True)
            data[col] = data[col].convert_dtypes()
    return data

def convert_types(data):
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    countcateg = 0
    countconst = 0
    # Count distinct values per column, use nunique:
    for col in data:
        numberunique = data[col].nunique()
        if 'subject_id' in str(col):
            pass
        elif numberunique == 0 or numberunique == 1:
                data.drop(col,axis=1, inplace=True)
                print("dropped constant column", str(col))
                countconst +=1
        elif numberunique < 6 and data.shape[0] > 6:
            countcateg += 1
            print("this column either bool or categorical with unique values", numberunique)
            try:
                d = {'No': False, 'NO':False, 'no': False, 'non': False,'yes': True, 'Yes':True, 'YES':True, 'oui': True,pd.NaT: np.nan, pd.NA: np.nan ,'nan': np.nan,'<NA>': np.nan ,'NA': np.nan, 'False': False, 'True': True, 'M': False, 'F': True}
                data[col] = data[col].map(d)

                # for categorical:
                mapper = {k: int(i) for i, k in enumerate(data[col].unique()) if str(k) not in nuls}
                data[col] = data[col].map(mapper)
                data[col] = data[col].convert_dtypes(convert_string=True, convert_boolean=False, convert_integer=False,
                                                     convert_floating=False)
            except Exception as e:
                print(e)
        else:
            column_values = data[col].dropna()
            types1 = str(column_values.dtypes)
            # convert columns with units:
            if pd.api.types.is_string_dtype(types1):
                try:
                    data[col]= convertstrToNum(data[col])

                except:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
    print("total number of categorical (and boolean)", countcateg)
    print("total number of dropped constant", countconst)
    return data

# Not used
def all_types_string(data):
    for col in data:
        data[col] = data[col].astype(str)
        data[col] = data[col].convert_dtypes()
    return data

# normalize numeric columns in a dataframe
def normalize(df):
    normalized = pd.DataFrame()

    for i in df:
        column_values = df[i].dropna()
        typec = column_values.dtypes
        if pd.api.types.is_numeric_dtype(typec) and not pd.api.types.is_bool_dtype(typec):
            normalized[i] =(df.loc[:,i]-df.loc[:,i].min())/(df.loc[:,i].max()-df.loc[:,i].min()+0.0001)
        else: normalized[i] = df.loc[:,i]
    return normalized

# Not Used
def create_pairs(datafr):
# create pairs in a list:
    pairs = list()
    from itertools import combinations
    for index in list(combinations(datafr.index, 2)):
        pairs.append(datafr.loc[index,:])
    return pairs

def creat_pairs(df1,df2):
    c = list(itertools.product(df1.index, df2.index))
    return c

# truth is pairs of matches, predict is pairs predicted matches
def measurements(truth,predict):
    tp = set(truth)&set(predict)
    fp = set(predict) - set(truth)
    fn = set(truth) - set(predict)
    precision = len(tp)/(len(tp)+len(fp)+0.00001)
    recall = len(tp)/(len(tp)+len(fn))
    f1 = 2*precision*recall/(precision+recall+0.00001)
    print("fp:",len(fp))
    print("fn",len(fn))
    print("tp:",len(tp))
    return precision, recall, f1


def findItemMax(data, key):
    # data is a dictionary
    key = str(key)
    dict = max(data, key=key)
    return dict

# Not Used
def transitive_closure(a):
    closure = set(a)
    while True:
        new_relations = set((x,w) for x,y in closure for q,w in closure if str(q) == str(y))
        closure_until_now = closure | new_relations

        if closure_until_now == closure:
            break
        closure = closure_until_now
    return closure

