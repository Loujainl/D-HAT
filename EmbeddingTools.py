## ---------------------------------------------------------------------------------------------------------------
## Script name: EmbeddingTools.py
## Purpose of script: Creating Pre-trained Embeddings for Textual Data of the entities
## Author: Loujain Liekah
## Date Created: 15-05-2021
## Copyright (c) Loujain Liekah, 2021
## ----------------------------------------------------------------------------------------------------------------
##

import numpy as np
import pandas as pd
from nltk import word_tokenize
import os, sys
import gensim.downloader as api
from multiprocessing import Pool
from datetime import datetime
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
modelfasttext = fasttext.load_model('cc.en.300.bin')
modelglove = api.load("glove-twitter-25")
modelw2v = api.load('word2vec-google-news-300')
# modelfasttext = 'testmode'
# modelglove = 'testmode'
# modelw2v = 'test'
import multiprocessing

def preprocessing(text):
    text = str(text)
    words = word_tokenize(text)
    tokens = [w.lower() for w in words]
    # remove words less than 3 letters
    tokens = [word for word in tokens if len(word)>3]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def embedmytext(stext,model,poolingmethod='mean'):
    stext = preprocessing(stext)
    embed_sent = np.zeros(300,dtype='float32')
    if model == 'w2v':
            embedlist = []
            for word in stext.split():
                try:
                    emword = modelw2v[word]
                    embedlist.append(emword)
                except KeyError:
                    pass
            # POOLING
            try:
                embedarr = np.array(embedlist)
                embed_sent = np.nanmean(embedarr, axis=0)
                # if str(poolingmethod) =='max':
                #     # embed_sent = np.array(embedlist).nanmax(axis=0)
                # elif str(poolingmethod) == 'sum':
                #     # embed_sent = np.sum(embedlist,axis=0)
                # elif str(poolingmethod) == 'min':
                #     # embed_sent = np.nanmin(embedlist,axis=0)
                # else:
                #     # embed_sent = np.nanmean(embedarr,axis=0)

            except:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

    elif model =='glove':
            embedlist = []
            for word in stext.split():
                try:
                    emword = modelglove[word]
                    embedlist.append(emword)
                except KeyError:
                    pass
            # POOLING
            try:
                    embedarr = np.array(embedlist)
                    embed_sent = np.nanmean(embedarr, axis=0)
            except Exception as e:
                print(e)


    elif model == 'FT':
            try:
                embed_sent = modelfasttext.get_sentence_vector(stext)
            except Exception as e:
                print(e)
                embed_sent = np.zeros(300,dtype='float32')

    elif model =='bert':
        pass
        # embed_sent = bertmodel.encode(stext)
    return embed_sent

# To use the record as a textual document
def combine_attributes(df,entityidcol):
    colstrleft = []
    colstrright = []
    colnum = []
    try:
        for i in df.columns:
            column_values = df[i].dropna()
            type = column_values.dtypes
            if pd.api.types.is_string_dtype(type) and str(entityidcol) not in str(i) and 'ltable' in str(i) and 'pair' not in str(i) :
                colstrleft.append(i)
            elif pd.api.types.is_string_dtype(type) and str(entityidcol) not in str(i) and 'rtable' in str(i) and 'pair' not in str(i):
                colstrright.append(i)
            else:
                colnum.append(i)
        df['ltable_combined'] = df[colstrleft].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        df['ltable_combined'] = df['ltable_combined'].astype(str)
        df['rtable_combined'] = df[colstrright].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        df['rtable_combined'] = df['rtable_combined'].astype(str)
        colstrright.extend(colstrleft)
        df.drop(colstrright, axis=1, inplace=True)
        print("Combination of attribute is done successfully")
        print("columns of current dataframe", df.columns)
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("could not combine")
    return df


def parallelize_embedding(df,entityidcol, func, model, n_cores=12):
    # n_cores = multiprocessing.cpu_count()
    try:
        result_list = []
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        for i in range(len(df_split)):
           start1 = datetime.now()
           df_split[i].reset_index(drop=True, inplace=True)
           r = pool.apply_async(func, args=(df_split[i],entityidcol,model))
           result_list.append(r.get())
        df = pd.concat(result_list)
        pool.close()
        pool.join()

    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return df


# If values of textual attributes were concatenated, embed using this method
def embedAllAsOne(dfcombined, model):
    try:
        if model =='FT':
            dfcombined['ltable_comb_embed_FT'] = dfcombined['ltable_combined'].apply(
                lambda x: embedmytext(x, model,'mean'))
            dfcombined['rtable_comb_embed_FT'] = dfcombined['rtable_combined'].apply(
                lambda x: embedmytext(x, model,'mean'))

        elif model =='bert':
            pass

        else:
            dfcombined['ltable_comb_embed_mean'] = dfcombined['ltable_combined'].apply(
                lambda x: embedmytext(x, model,'mean'))
            dfcombined['rtable_comb_embed_mean'] = dfcombined['rtable_combined'].apply(
                lambda x: embedmytext(x, model,'mean'))  # if str(x) not in nuls else np.nan
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return dfcombined


# Hybrid embed each textual column alone, and keep the original column to apply classical similarity
def embedmydfhybrid(df,entityidcol, model):
    embed_df = pd.DataFrame()
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    try:
        for i in df.columns:
            column_values = df[i].dropna()
            type = column_values.dtypes
            if pd.api.types.is_string_dtype(type) and str(entityidcol) not in str(i):
                    embedding = df[i].apply(lambda x: embedmytext(x,model) if str(x) not in nuls else np.nan)

                    namembed = str(i) + "_embedded"
                    embed_df[namembed] = pd.Series(data = embedding.astype('object'), index = df.index)
                    embed_df[i] = df[i] # Also keep original valuse
            else:
                    embed_df[i] = df[i]
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return embed_df


def embedmydf(df,entityidcol, model):
    embed_df = pd.DataFrame()
    nuls = ['nan', 'NA', 'NaN','<NA>', '']
    try:
        for i in df.columns:
            if str(entityidcol)  in str(i) or 'pair'  in str(i):
                embed_df[i] = df[i]
            else:
                column_values = df[i].dropna()
                type = column_values.dtypes
                if pd.api.types.is_string_dtype(type):
                        embedding = df[i].apply(lambda x: embedmytext(x,model) if str(x) not in nuls else np.nan)
                        namembed = str(i) + "_embedded"
                        embed_df[namembed] = pd.Series(data = embedding.astype('object'), index = df.index)
                else:
                        embed_df[i] = df[i]
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Could not perform embedding")

    return embed_df
