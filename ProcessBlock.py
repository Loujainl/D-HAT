import RL
import features
import numpy as np
import pandas as pd
from ast import literal_eval
import sys, os
from multiprocessing import  Pool
from datetime import datetime


def parallelize_dataframe(df, func, entityid, withembedding, n_cores=12):
    # n_cores = multiprocessing.cpu_count()
    result_list = []
    try:
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        for i in range(len(df_split)):
           df_split[i].reset_index(drop=True, inplace=True)
           r = pool.apply_async(func, args=(df_split[i],entityid, withembedding))
           result_list.append(r.get())
        df = pd.concat(result_list)
        pool.close()
        pool.join()
        print("Result after parallel")
        print(df.shape)
    except UnboundLocalError:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return df
# Create Feature Matrix for block

def process(blocks,withembedding,entityid):
    if not blocks.empty:
        try:
            # returns name and type of each columns
            typedicc = RL.columnsTypes(blocks)
            sim_fun = features.type_simfun(typedicc,withembedding)
            # find pair of corresponding attributes between source and target
            corresponds = []
            for key, value in sim_fun.items():
                if key.startswith('rtable'):
                    corresponds.extend([col, key, value] for col in blocks if col[6:] == key[6:] and col != key)

            start = datetime.now()
            featss = []
            for att in corresponds:
                funcss = att[2]
                for tuple in blocks[[att[0], att[1], 'pair']].itertuples(index=False):
                    ltable_val = tuple[0]
                    rtable_val = tuple[1]
                    tuplefeats = {}
                    tuplefeats['pair'] = tuple[2]
                    for func in funcss:
                        func_name = str(att[0])[7:] + str(func).split()[1]
                        tuplefeats[func_name] = func(ltable_val, rtable_val)
                    featss.append(tuplefeats)

            mergedfeat = pd.DataFrame(featss)
            mergedfeat = mergedfeat.groupby(['pair'], as_index=False).first()
        except:
            print("could not create features, attribute in correspondance:", print(att))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    else:
        answer = set()
        return answer

    # mergedfeat = pd.DataFrame(featdata)
    # featurematrix.to_csv(featmname, index=False)

    return  mergedfeat


# Calculate matching score from Feature Matrix and a Threshold
def matchscore(mergedfeat, tn):
    tn = float(tn)
    # print("Matching using features:", mergedfeat.columns)
    try:
        mergedfeat.loc[:, ~mergedfeat.columns.isin(['subje1', 'subje2', 'pair'])] = \
            np.where(mergedfeat.loc[:, ~mergedfeat.columns.isin(['subje1', 'subje2', 'pair'])] >= tn, 1, np.where(mergedfeat.loc[:, ~mergedfeat.columns.isin(['subje1', 'subje2', 'pair'])] < tn, 0, np.nan))
        # Calculate Matching Scores, nan mean
        mergedfeat['scores'] = np.nanmean(mergedfeat.loc[:, ~mergedfeat.columns.isin(['subje1', 'subje2', 'pair'])], axis=1)

        mergedfeat['pair'] = mergedfeat['pair'].apply(lambda x: literal_eval(str(x)))

        # Return match scores, source id, target id and pair
        intermediate = mergedfeat.filter(['pair','subje1','subje2','scores'], axis=1)

    except:
        print("can not calculate matching")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return intermediate

# Mutually Maximum Greedy Clustering
def MutMax(mergedfeat, min_sim, clean=True):
    # merged has 'pair','subje1','subje2','scores'
    min_sim = float(min_sim)
    mergedfeat[['subje1', 'subje2']] = pd.DataFrame(mergedfeat['pair'].tolist(), index=mergedfeat.index)

    try:
        if clean == 1:
            print("performing clean clean ER")
            # ids of max mean (per source)
            # sort scores in ascending order and keep last
            result_source = mergedfeat.sort_values('scores').drop_duplicates(['subje1'], keep='last')
            result_source = result_source[result_source['scores'] > min_sim]

            # ids of max mean per target
            result_target = mergedfeat.sort_values('scores').drop_duplicates(['subje2'], keep='last')
            result_target = result_target[result_target['scores'] > min_sim]
            # MutMax takes intersection by pairid, keep only one copy
            s1 = result_source.merge(result_target, on=['pair'], how='inner')
            s1 = s1.drop_duplicates(subset='pair', keep="last")
            answer = s1['pair'].apply(lambda x: literal_eval(str(x)))
            answer = set(answer)

            return answer
        else:
            # if it is dirty ER, result union of target and source
            print("Performing Dirty ER")
            # result_source = mergedfeat.sort_values('scores', ascending=False).groupby('subje1').head(2)
            result_source = mergedfeat.sort_values('scores').drop_duplicates(['subje1'], keep='last')
            # remove tuples less than a minimum matching score
            result_source = result_source[result_source['scores'] > min_sim]
            result_target = mergedfeat.sort_values('scores').drop_duplicates(['subje2'], keep='last')
            result_target = result_target[result_target['scores'] > min_sim]
            s1 = result_source.merge(result_target, on=['pair'], how='outer')
            s1 = s1.drop_duplicates(subset='pair', keep="last")
            answer = s1['pair'].apply(lambda x: literal_eval(str(x)))
    except:
            answer = set()
    return answer