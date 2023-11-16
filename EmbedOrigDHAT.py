## ---------------------------------------------------------------------------------------------------------------
## Script name: EmbedOrigDHAT.py
## Purpose of script: Entity Matching or Deduplicating datasets with Heterogeneous Attribute Types
## Author: Loujain Liekah
## Date Created: 15-05-2021
## Copyright (c) Loujain Liekah, 2021
## ----------------------------------------------------------------------------------------------------------------
##

import RL
import pandas as pd
import numpy as np
from datetime import datetime
import ProcessBlock
from pathlib import Path
import EmbeddingTools
import sys, os
import Performance
import warnings
warnings.filterwarnings("ignore")



def runexp(source,target,idfeat='subject_id'):
    local = input("Type 1 for Running Local, Cluster otherwise\n")
    local = int(local)
    if(int(local)==1):
        print("local mode")
        sourcedf = pd.read_csv('./OriginalDatasets/' + source + '.csv', encoding = "ISO-8859-1",on_bad_lines = 'skip')#, sep = '|')
        targetdf = pd.read_csv('./OriginalDatasets/' + target + '.csv', encoding = "ISO-8859-1",on_bad_lines = 'skip')#, sep = '\|\|')
    else:
        print("Cluster Mode")
        sourcedf = pd.read_csv('./OriginalDatasets/' + source + '.csv', encoding = "ISO-8859-1",on_bad_lines = 'skip')
        targetdf = pd.read_csv('./OriginalDatasets/' + target + '.csv', encoding = "ISO-8859-1",on_bad_lines = 'skip')

    sourcedf = sourcedf.rename(columns={'pid': 'subject_id'})
    targetdf = targetdf.rename(columns={'pid': 'subject_id'})
    sourcedf = sourcedf.loc[:, ~sourcedf.columns.str.startswith('Unnamed')]
    targetdf = targetdf.loc[:, ~targetdf.columns.str.startswith('Unnamed')]

    truthname = input(("Truthfilename: "))
    truthf = "./Truth/"+truthname + 'truth.json'
    import json
    with open(truthf) as f:
        truth = json.loads(f.read())
    truth = [tuple(l) for l in truth]
    truth = set(truth)

    print("source shape", sourcedf.shape)
    print("target shape", targetdf.shape)

    sourcedf, targetdf = RL.schematcher(sourcedf, targetdf)
    print("1. Schema Matching, OK ")

    sourcedf = RL.convert_types(sourcedf)
    targetdf = RL.convert_types(targetdf)
    print("2. Converting Types, OK")

    # 2. normalize both dataset, skip the entityid column
    sourcedf.loc[:, sourcedf.columns != str(idfeat)] = RL.normalize(sourcedf.loc[:, sourcedf.columns != str(idfeat)])
    targetdf.loc[:, targetdf.columns != str(idfeat)] = RL.normalize(targetdf.loc[:, targetdf.columns != str(idfeat)])
    print("3. Normalizing Numerics, OK")

    try:
        # remove from blocks column with coverage <0.1 (keep columns with percrntage nulls < 90)
        nuls = ['nan', 'NA', 'NaN', '<NA>', '', np.nan, pd.NA]
        columnsbefore = sourcedf.columns
        sourcedf = sourcedf.loc[:, sourcedf.isin(nuls).mean() < .9]
        targetdf = targetdf.loc[:, sourcedf.isin(nuls).mean() < .9]
        columnsafter = sourcedf.columns
        dropped = set(columnsbefore).difference(set(columnsafter))
        print("Number of columns with coverage < 0.1", len(dropped))
    except Exception as e:
        print(e)

    print("4. Selecting Attributes, OK")

    # 3. Blocking using magellan overlapblocker
    bt = datetime.now()
    try:
        blocking_F1 = {}
        for col in sourcedf.columns:
            if str(col) == str(idfeat):
                pass
            else:
                sourcecolumn_values = sourcedf[col].dropna()
                targetcolumn_values = targetdf[col].dropna()
                types1 = str(sourcecolumn_values.dtypes)
                types2 = str(targetcolumn_values.dtypes)
                # try blocking only with string columns:
                if pd.api.types.is_string_dtype(types1) and pd.api.types.is_string_dtype(types2):
                    blocktemps = RL.magellan_overlapblock(sourcedf, targetdf, idfeat, idfeat, col)
                    _,f1, precision = RL.intheBlock(truth,blocktemps,idfeat)
                    blocking_F1[str(col)] = f1
        blockattribute = max(blocking_F1, key=blocking_F1.get)

        blocks = RL.magellan_overlapblock(sourcedf, targetdf, idfeat, idfeat, blockattribute)
        print(blocks.columns)
        print("Blocking using best key:", blockattribute)
        print("blocking time =", datetime.now() - bt)
        print("5. Blocking successful")
    except:
        print("Can not block this dataset, Cartesian Product instead:")
        sourcedf.columns = ['ltable_'+ str(col)  for col in sourcedf.columns]
        targetdf.columns = [  'rtable_' + str(col) for col in targetdf.columns]
        blocks =  sourcedf.merge(targetdf, how='cross')
        print("cross product columns", blocks.columns)
        blockattribute = ''

    print("Block shape: ", blocks.shape)

    # Does the block contain all pairs in the truth:
    cont = RL.intheBlock(truth, blocks, idfeat)
    print("Blocking Performance", cont)

    # # remove from blocks column with coverage <0.1 (keep columns with percrntage nulls < 90)
    # nuls = ['nan', 'NA', 'NaN', '<NA>', '', np.nan, pd.NA, None]
    # blocks = blocks.loc[:, blocks.isin(nuls).mean() < .9]
    print("currentshape of blocks ", blocks.shape)

    withembedding = input(" Type: 0 for Classic similarity, 1 for Semantic, 2 for Hybrid ")
    withembedding = int(withembedding)
    clean = input("Please write 1 for clean-clean EM, 0 for dirty EM: ")
    clean = int(clean)


    startTime = datetime.now()
    print("Starting at:", startTime)


    # Check if the features are already created, pass to the evaluation
    if withembedding==0:
        settingstr='classic'
    elif withembedding==1:
        settingstr = 'Embedding'
    else:
        settingstr = 'Hybrid'

    featmname = str(settingstr)+str(blockattribute)+str(source)+".csv"
    featmname = featmname.replace("/","")
    featmname = "./FeatureMatrices/"+featmname
    my_file = Path(featmname)

    if my_file.is_file():
        #  Feature Matrix Exists, read it
        print("loading feature matrix")
        featurematrix = pd.read_csv(my_file)

    else:
        try:
            # 4. convert types in case 2 unique to boolean.
            idss = ['ltable_subject_id', 'rtable_subject_id']
            blocks = RL.convert_types(blocks)
            # Normalize here if there are new Numerical Attributes
            blocks.loc[:, ~blocks.columns.isin(idss)] = RL.normalize(blocks[blocks.columns[~blocks.columns.isin(idss)]])
            blocks['pair'] = list(zip(blocks['ltable_subject_id'], blocks['rtable_subject_id']))
            print("4. Converted Types")
            print(RL.columnsTypes(blocks))

            # embed Blocks : keep numerical, keep string , add embedded string
            embedmname = str(blockattribute) + str(source) + ".csv"
            embedmname = embedmname.replace("/", "")
            embedmname = "./Embeddings/blockey_" + embedmname
            my_emd_file = Path(embedmname)

            # for classic similarity
            if int(withembedding) == 0:
                print("not embedding")
                embed_blocks = blocks
                    # try to load existing embedding
            elif my_emd_file.is_file() and withembedding !=0: # and False
                try:
                    print("loading embedding file")
                    embed_blocks = pd.read_csv(my_emd_file, index_col=False, error_bad_lines=False) # , index_col=[0]
                except:
                    print("Can not load embedding")
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                # Only Embedding
            elif withembedding == 1:
                ft_em = blocks.copy()
                w2v_em = blocks.copy()
                if int(local) != 1:
                    print("Embedding on Parallel")
                    embed_ft = EmbeddingTools.parallelize_embedding(ft_em,idfeat ,EmbeddingTools.embedmydf,'FT')
                    embed_w2v = EmbeddingTools.parallelize_embedding(w2v_em, idfeat ,EmbeddingTools.embedmydf, 'w2v')
                    embed_glove = EmbeddingTools.parallelize_embedding(blocks, idfeat ,EmbeddingTools.embedmydf, 'glove')

                    feats1 = pd.merge(embed_w2v, embed_glove, on='pair')
                    embed_blocks = pd.merge(embed_ft, feats1, on='pair')
                    middles = datetime.now()

                else:
                 try:
                    embed_ft = EmbeddingTools.embedmydf(ft_em,idfeat,'FT')
                    # word2vec
                    embed_w2v = EmbeddingTools.embedmydf(w2v_em,idfeat,'w2v')
                    embed_glove = EmbeddingTools.embedmydf(blocks,idfeat,'glove')
                    print("embedding succeeded with GloVe")
                    # embed_glove.to_csv("./Embeddings/embed_glove.csv",index=False)

                    # combine Embeddings into one
                    feats1 = pd.merge(embed_w2v, embed_glove, on='pair')
                    embed_blocks = pd.merge(embed_ft, feats1, on='pair')
                    midTime = datetime.now() - startTime
                    print("Embedding ok, Time", midTime, "shape", embed_blocks.shape)
                    middles = datetime.now()
                 except:
                    print("Error embedding")

            # Hybrid: Classic and Embedding each column alone!
            else:
                ft_em = blocks.copy()
                w2v_em = blocks.copy()
                if int(local) != 1:
                    embed_ft = EmbeddingTools.parallelize_embedding(ft_em,idfeat ,EmbeddingTools.embedmydfhybrid,'FT')
                    collis = []
                    # drop the extra column before merge
                    for cols in embed_ft.columns:
                        if 'embed' in str(cols) or 'pair' in str(cols):
                            pass
                        else:
                            collis.append(cols)
                    embed_w2v = EmbeddingTools.parallelize_embedding(w2v_em, idfeat ,EmbeddingTools.embedmydfhybrid, 'w2v')
                    collis2 = []
                    for cols in embed_w2v.columns:
                        if 'embed' in str(cols) or 'pair' in str(cols):
                            pass
                        else:
                            collis2.append(cols)
                    embed_glove = EmbeddingTools.parallelize_embedding(blocks, idfeat ,EmbeddingTools.embedmydfhybrid, 'glove')
                    try:
                        embed_ft.drop(collis, axis=1, inplace=True)
                        embed_w2v.drop(collis2, axis=1, inplace=True)
                    except Exception as e:
                        print(e)
                    feats1 = pd.merge(embed_w2v, embed_glove, on='pair')
                    embed_blocks = pd.merge(embed_ft, feats1, on='pair')
                    middles = datetime.now()
                else:
                    ft_em = blocks.copy()
                    w2v_em = blocks.copy()
                    colstrright = ['ltable_subject_id', 'rtable_subject_id']
                    embed_w2v = EmbeddingTools.embedmydfhybrid(w2v_em,idfeat,'w2v')
                    embed_glove = EmbeddingTools.embedmydfhybrid(blocks,idfeat,'glove')
                    embed_ft = EmbeddingTools.embedmydfhybrid(ft_em,idfeat,'FT')
                    try:
                        embed_ft.drop(colstrright, axis=1, inplace=True)
                        embed_w2v.drop(colstrright, axis=1, inplace=True)
                        embed_glove.drop(colstrright, axis=1, inplace=True)
                    except:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print("could not drop combined dataframes")

                    feats1 = pd.merge(embed_w2v, embed_glove, on='pair')
                    embed_blocks = pd.merge(embed_ft,feats1)
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


        try:
            # Create Feature Matrix For Block
            print("Creating Feature Matrix")
            if int(local) != 1 :
                featurematrix = ProcessBlock.parallelize_dataframe(embed_blocks,ProcessBlock.process, idfeat, withembedding)
                res = RL.inBlock(truth,featurematrix)
                # Save Feature Matrix to Avoid recreating
                featurematrix.to_csv(featmname, index=False)
                currentTly = datetime.now() - middles
                print("Creating features", currentTly)
            else:
                featurematrix = ProcessBlock.process(embed_blocks,idfeat,withembedding)
                featurematrix.to_csv(featmname, index=False)
                currentTly = datetime.now() - middles
                print("Creating features", currentTly)
        except Exception as e:
            print(e)

    # Perform Matching and MutMax , Try different similarity and matching thresholds and return best combination, Save Results
    bestperf = Performance.findminsim(source,target,featmname, featurematrix,truthname,clean)
    print("Recalling Blocking Performance", cont)
    print("using blocking key", blockattribute)
    Runtime_total = datetime.now() - startTime
    print("Total Processing Runtime", Runtime_total)
    print("End Time:", datetime.now())

if __name__ == '__main__':
    runexp(*sys.argv[1:])


