#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime as dt
import pool_algos
import multiprocessing
import os
import itertools
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pickle
import scipy.sparse as sparse
import Knn
from SpectralCf import SpectralCF
import random as rd 
import tensorflow as tf
import statistics

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

os.environ['PYSPARK_PYTHON']='/usr/bin/python3'

print('Number of recommendations : ')
nm = int(input())

print('You want to split again ? ')
splt = int(input())

train_model = pd.DataFrame(columns=['ARM_PRECISION','K50_PRECISION','ALS_PRECISION','BPR_PRECISION',\
'VAES_PRECISION','SPEC_PRECISION','CUST_ID','ARTICLE_ID','AGE','NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST'])
   
directory = 'Tafeng'

def function(x,i):
    x = str(x)
    x = x.split(',')
    if i :
        return int(x[i][1:-1])
    else:
        return int(x[i][1:])

if splt==1:
    head = ['CUST_ID','ARTICLE_ID','AGE','RATING']

    data = pd.read_csv('data_set/Fa_teng.csv')
    data = data.rename(columns={'AMOUNT':'RATING'})

    nb_i = 50
    nb_u = 75

    data = data[data.groupby('ARTICLE_ID')['ARTICLE_ID'].transform('count').ge(nb_i)]
    data = data[data.groupby('CUST_ID')['CUST_ID'].transform('count').ge(nb_u)]

    us = data.CUST_ID.unique()

    data_train = pd.DataFrame(columns=head)
    data_train_meta = pd.DataFrame(columns=head)
    data_test_meta = pd.DataFrame(columns=head)

    for u in us :
        data_local = data[data['CUST_ID']==u]
        data_local = data_local.sample(frac=1)
        deb,inter = round(0.5*len(data_local)),round(0.9*len(data_local))
        
        data_train = data_train.append(data_local[:deb])
        data_train_meta = data_train_meta.append(data_local[deb:inter])
        data_test_meta = data_test_meta.append(data_local[inter:])

    data.to_csv( directory+'/data.csv' )

    data_train.to_csv( directory+'/data_train.csv', index=False )
    data_train_meta.to_csv( directory+'/data_train_meta.csv', index=False )
    data_test_meta.to_csv( directory+'/data_test_meta.csv', index=False )

elif splt==0:
    data_train = pd.read_csv( directory+'/data_train.csv' )
    data_train_meta = pd.read_csv( directory+'/data_train_meta.csv' )
    data_test_meta = pd.read_csv( directory+'/data_test_meta.csv' )


print('Split Done')

data_train_2 = data_train[['CUST_ID','AGE']]
data_train_2 = data_train_2.drop_duplicates()
data_train_2 = data_train_2[~data_train_2['AGE'].isnull()]

data_train = data_train[data_train['CUST_ID'].isin(data_train_2.CUST_ID.unique())]

k = data_train.groupby(['CUST_ID','ARTICLE_ID']).size().reset_index(name='FREQUENCY')
k = k.groupby('CUST_ID').size().reset_index(name='NUMBER')
k = k[k['CUST_ID']>10].CUST_ID.unique()

data_train = data_train[data_train['CUST_ID'].isin(k)]
data_train_meta = data_train_meta[data_train_meta['CUST_ID'].isin(k)]
data_test_meta = data_test_meta[data_test_meta['CUST_ID'].isin(k)]

data_train.to_csv( directory+'/data_train.csv', index=False )
data_train_meta.to_csv( directory+'/data_train_meta.csv', index=False )
data_test_meta.to_csv( directory+'/data_test_meta.csv', index=False )

data_train = data_train.groupby(['CUST_ID','ARTICLE_ID']).size().reset_index(name='FREQUENCY')

# Frequency of items and users
nb_purch_train = data_train.groupby('CUST_ID')['FREQUENCY'].agg('sum').reset_index(name='NB_PURCH_TRAIN')
nb_article_purch_train = data_train.groupby('ARTICLE_ID')['FREQUENCY'].agg('sum').reset_index(name='NB_ARTICLE_PURCH_TRAIN')

nb_purch_test = data_train_meta.groupby(['CUST_ID','ARTICLE_ID']).size().reset_index(name='FREQUENCY')
nb_article_purch_test = nb_purch_test.groupby('ARTICLE_ID')['FREQUENCY'].agg('sum').reset_index(name='NB_ARTICLE_PURCH_TEST')
nb_purch_test = nb_purch_test.groupby('CUST_ID')['FREQUENCY'].agg('sum').reset_index(name='NB_PURCH_TEST')

# Data for Factorization Machines
data_train_FM = pd.merge(data_train,nb_purch_test, on='CUST_ID')
data_train_FM = pd.merge(data_train_FM,nb_article_purch_test, on='ARTICLE_ID')
data_train_FM = pd.merge(data_train_FM,data_train_2, on='CUST_ID')

data_train_FM.CUST_ID = data_train_FM.CUST_ID.astype('object')
data_train_FM.ARTICLE_ID = data_train_FM.ARTICLE_ID.astype('object')

object_col = ['CUST_ID','AGE']
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) #One Hot Encoder
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(data_train_FM[object_col]))

user_FM = data_train_FM[object_col]
OH_cols_train.index = data_train_FM.index
user_FM = pd.concat([user_FM, OH_cols_train], axis=1)

user_FM = user_FM.drop(columns=['AGE'])

object_col = ['ARTICLE_ID']
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) #One Hot Encoder
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(data_train_FM[object_col]))

item_FM = data_train_FM[object_col]
OH_cols_train.index = data_train_FM.index
item_FM = pd.concat([item_FM, OH_cols_train], axis=1)

data_train_FM.to_csv(directory+'/data_train_FM.csv',index=False)

item_FM = item_FM.drop_duplicates()
user_FM = user_FM.drop_duplicates()

item_FM.to_csv(directory+'/item_FM.csv',index=False)
user_FM.to_csv(directory+'/user_FM.csv',index=False)

### Train Pool of recommendation algorithms
if len(data_train)>1:
    reco_algo = pool_algos.RecommendationAlgos(None,None,data_train,num=nm)

    print('====== USERS : ',len(reco_algo.users) )
    if len(reco_algo.users) > 1 :
        
        #TRAIN 
        
        reco_algo.get_association_matrix() #ARM
        
        reco_algo.fit_als() #ALS
        reco_algo.cosine_similarity() #CF
        
        sim = reco_algo.Similarity_matrix
        sim = sim.tocsr()
        data = reco_algo.data.tocsr()
        
        knn50 = Knn.KNN(data)
        knn50.fit(sim, selectTopK = True, topK=50)
        print(' #------------------ KNN 50 ... Done !-------------------#')
        
        reco_algo.fit_bpr() #BPR
        reco_algo.fit_vae(500,150,0.2,1) #VAES
        
        test_dict = reco_algo.test_dict(data_train_meta,data_train) #Test users
        
        with open(directory +'/test_dict', 'w') as f:
            print(test_dict, file=f) 
        
        data = data.astype(np.float32)

        d = data_train.groupby('CUST_ID')['ARTICLE_ID'].agg(list).reset_index()

        train_items = dict()

        for u in reco_algo.users:
            l = d[d['CUST_ID']==u]['ARTICLE_ID'].tolist()[0]
            l = [reco_algo.items_d[i] for i in l]
            train_items[reco_algo.users_d[u]] = l

        del d

        def sample():
            if BATCH_SIZE <= len(reco_algo.users):
                users_ = rd.sample(range(len(reco_algo.users)), BATCH_SIZE)
            else:
                users_ = [ rd.choice( range(len(reco_algo.users)) ) for _ in range(BATCH_SIZE) ]

            def sample_pos_items_for_u(u, num):
                pos_items = train_items[ u ] #TRAIN ITEMS A FAIRE
                if len(pos_items) >= num:
                    return rd.sample(pos_items, num)
                else:
                    return [rd.choice(pos_items) for _ in range(num)]

            def sample_neg_items_for_u(u, num):
                neg_items = list(set(range(len(reco_algo.items))) - set(train_items[ u ]))
                return rd.sample(neg_items, num)

            pos_items, neg_items = [], []
            for u in users_:
                pos_items += sample_pos_items_for_u(u, 1)
                neg_items += sample_neg_items_for_u(u, 1)
            
            print('END SAMPLE')

            return users_, pos_items, neg_items

        EMB_DIM = 128
        BATCH_SIZE = 512
        DECAY = 0.001
        K = 3
        N_EPOCH = 400
        LR = 0.01

        print('N_EPOCH : {0} - BATCH : {1} - EMB : {2} - DECAY : {3} - LR : {4}'\
        .format(N_EPOCH,BATCH_SIZE,EMB_DIM,DECAY,LR))

        tf.compat.v1.reset_default_graph()

        model = SpectralCF(K=K, graph=data, n_users=len(reco_algo.users), n_items=len(reco_algo.items),\
        emb_dim=EMB_DIM, lr=LR, decay=DECAY, batch_size=BATCH_SIZE)

        print("Instantiating model... done!")

        config = tf.compat.v1.ConfigProto()
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())

        print("Training model... ")

        res_spec = dict()
        
        for epoch in range(N_EPOCH):
            print('========= EPOCH : {0} ========='.format(epoch))
            users_, pos_items, neg_items = sample()
            _, loss = sess.run([model.updates, model.loss],\
            feed_dict={model.users: users_, model.pos_items: pos_items,model.neg_items: neg_items})

            if epoch == N_EPOCH-1:
                test_users = list(test_dict.keys())
                index = 0
                print(len(test_users)/BATCH_SIZE)
                while True:
                    print(index/BATCH_SIZE)
                    if index >= len(test_users):
                        break
                    
                    user_batch = test_users[index:index + BATCH_SIZE]
                    index += BATCH_SIZE

                    user_inter = []

                    for u in user_batch:
                        user_inter.append(reco_algo.users_d[u])
                    
                    user_batch = user_inter

                    if len(user_batch) < BATCH_SIZE:
                        user_batch += [user_batch[-1]] * (BATCH_SIZE - len(user_batch))

                    user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})

                    user_batch_rating_uid = zip(user_batch_rating, user_batch)

                    for x in user_batch_rating_uid:
                        rating = x[0]
                        u = x[1]
                        u = reco_algo.users[u]
                        item_score = []
                        for i in reco_algo.items:
                            item_score.append((i, rating[ reco_algo.items_d[i] ]))
                        item_score = sorted(item_score, key=lambda x: x[1],reverse=True)
                        #item_score.reverse()
                        res_spec[u]=item_score

        with open( directory+'/res_spec', 'w' ) as f:
            print(res_spec, file=f)            
        
        reco_algo.res_vae(1)
        
        test = data_train_meta
        test = test.groupby(['CUST_ID','ARTICLE_ID']).size().reset_index(name='RATING')
        test = test[['CUST_ID','ARTICLE_ID','RATING']]

        test2 = pd.DataFrame(columns=['CUST_ID','ARTICLE_ID','RATING','NUM'])

        for u,v in test_dict.items():
            inter = test[test['CUST_ID']==u]
            inter = inter[inter['ARTICLE_ID'].isin(v)]
            inter = inter.sample(frac=1)
            inter = inter.sort_values(by='RATING',ascending=False)
            inter['NUM'] = list(range(0,len(inter)))
            test2 = test2.append(inter, ignore_index = True)

        test = test2

        test.to_csv( directory+'/test_test.csv', index=False )
        del test2
        

        numbers = []
        for u,v in test_dict.items():
            numbers.append(len(v))

        med = statistics.median(numbers)
        mean = statistics.mean(numbers)
        
        def top_N_KNN50(user, n, val_values):
            if user in reco_algo.users_d:
                user_ind = reco_algo.users_d[user]
                #user_ind_v = cores_custs_v[user]
                user_ind_v = 0
                
                pred,score = knn50.recommend(user_ind,return_scores=True)
                
                items_purchased = reco_algo.get_purchased_2(user_ind, user_ind_v, val_values) ##
                ensemble_produits = set(items_purchased)

                rec_list = []
                score_list = []
                i=0
                for k,ind in enumerate(pred):
                    if i >= n:
                        break
                    else:
                        item_id = reco_algo.items[ind]
                        if item_id in ensemble_produits:
                            continue
                        else:
                            rec_list.append(item_id)
                            score_list.append(score[ind])
                            i=i+1
                return rec_list, score_list
            else:
                return None
        
        def top_N_SPECTRAL(user, n, val_values):
            if user in reco_algo.users_d:
                user_ind = reco_algo.users_d[user]
                #user_ind_v = cores_custs_v[user]
                user_ind_v = 0
                
                pred = res_spec[user]
                #pred = nmf_pure.recommend(user_ind,return_scores=False)

                items_purchased = reco_algo.get_purchased_2(user_ind, user_ind_v, val_values) ##
                ensemble_produits = set(items_purchased)

                rec_list= []
                i=0
                for k,ind in enumerate(pred):
                    if i >= n:
                        break
                    else:
                        item_id = ind[0]
                        if item_id in ensemble_produits:
                            continue
                        else:
                            rec_list.append(item_id)
                            i=i+1
                score = list(np.full(len(rec_list),1))
                return rec_list, score
            else:
                return None
        
        print('Test of Algos and Train for meta model , num of users :',len(test_dict.keys()))
        
        def worker(i,key, value, nb_list):

            val = set()
            classifier = dict()
            classifier2 = dict()

            nb = len(reco_algo.items)
            
            liste_ARM, score_ARM = reco_algo.top_N_ARM(key,nb,val)
            liste_BPR, score_BPR = reco_algo.top_N_BPR(key,nb,val)
            liste_VAES, score_VAES = reco_algo.top_N_VAES(key,nb,val)
            liste_ALS, score_ALS = reco_algo.top_N_ALS(key,nb,val)
            liste_knn_50, score_knn_50 = top_N_KNN50(key,nb,val)
            liste_SPEC, score_SPEC = top_N_SPECTRAL(key,nb,val)
            
            #d = test[test['CUST_ID']==key]

            liste = set(liste_ARM).intersection(set(liste_BPR))
            liste = liste.intersection(set(liste_ALS))
            liste = liste.intersection(set(liste_SPEC))
            liste = liste.intersection(set(liste_VAES))
            liste = liste.intersection(set(liste_knn_50))

            liste2 = set(liste_ARM[:500]).intersection(set(liste_BPR[:500]))
            liste2 = liste2.intersection(set(liste_ALS[:500]))
            liste2 = liste2.intersection(set(liste_VAES[:500]))
            liste2 = liste2.intersection(set(liste_knn_50[:500]))
            liste2 = liste2 - value

            k = 0

            for item in value :
                if item in liste:
                    k = k+1
                    clas = []
                    clas2 = []

                    #j = int(d[d['ARTICLE_ID']==item]['NUM'])

                    clas.append( int(item in liste_ARM[:nb_list]) )
                    clas.append( int(item in liste_knn_50[:nb_list]) )
                    clas.append( int(item in liste_ALS[:nb_list]) )
                    clas.append( int(item in liste_BPR[:nb_list]) )
                    clas.append( int(item in liste_VAES[:nb_list]) )
                    clas.append( int(item in liste_SPEC[:nb_list]) )

                    '''
                    clas.append( abs(j-liste_ARM.index(item)) )
                    clas.append( abs(j-liste_knn_50.index(item)) )
                    clas.append( abs(j-liste_ALS.index(item)) )
                    clas.append( abs(j-liste_BPR.index(item)) )
                    clas.append( abs(j-liste_VAES.index(item)) )
                    clas.append( abs(j-liste_SPEC.index(item)) )
                    '''

                    clas2.append( score_ARM[ liste_ARM.index(item) ] )
                    clas2.append( score_knn_50[ liste_knn_50.index(item) ] )
                    clas2.append( score_ALS[ liste_ALS.index(item) ] )
                    clas2.append( score_BPR[ liste_BPR.index(item) ] )
                    clas2.append( score_VAES[ liste_VAES.index(item) ] ) 
                    clas2.append( score_SPEC[ liste_SPEC.index(item) ] )
                    clas2.append( 1 )

                    classifier[key,item] = clas
                    classifier2[key,item] = clas2
            
            for item in liste2:
                clas2 = []

                clas2.append( score_ARM[ liste_ARM.index(item) ] )
                clas2.append( score_knn_50[ liste_knn_50.index(item) ] )
                clas2.append( score_ALS[ liste_ALS.index(item) ] )
                clas2.append( score_BPR[ liste_BPR.index(item) ] )
                clas2.append( score_VAES[ liste_VAES.index(item) ] ) 
                clas2.append( score_SPEC[ liste_SPEC.index(item) ] )
                clas2.append( 0 )

                classifier2[key,item] = clas2

            
            print(i,' NUUUM : ',k)

            return classifier, classifier2
      
        manager = multiprocessing.Manager()
        pred = dict()
        pred_score = dict()
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result = []

        for i,(key,value) in enumerate(test_dict.items()):
            result = pool.apply_async(worker,args=(i,key,value,int(mean)))
            r = result.get()
            r_score = r[1]
            r = r[0]
            pred.update(r)
            pred_score.update(r_score)
        pool.close()
        pool.join()


        col = {0:'ARM_PRECISION',1:'K50_PRECISION',2:'ALS_PRECISION',3:'BPR_PRECISION',\
        4:'VAES_PRECISION',5:'SPEC_PRECISION'}

        col_score = {0:'ARM_PRECISION',1:'K50_PRECISION',2:'ALS_PRECISION',3:'BPR_PRECISION',\
        4:'VAES_PRECISION',5:'SPEC_PRECISION',6:'SCORE'}

        print('Data for Training meta model is okey')

        data_prec = pd.DataFrame.from_dict(pred,orient='index')
        data_prec.rename(columns=col, inplace=True)

        data_prec['CUST_ID'] = pred.keys()

        custs = data_prec['CUST_ID'].apply(function,i=0)
        items = data_prec['CUST_ID'].apply(function,i=1)

        data_prec['CUST_ID']=custs
        data_prec['ARTICLE_ID']=items

        data_prec = pd.merge(data_prec,data_train_2,on='CUST_ID') #Add Sex and Age for each customer
        data_prec.to_csv( directory+'/data_prec.csv', index=False)
        '''
        data_prec = pd.read_csv( directory+'/data_prec.csv')
        #'''
        
        data_prec_score = pd.DataFrame.from_dict(pred_score,orient='index')
        data_prec_score.rename(columns=col_score, inplace=True)

        data_prec_score['CUST_ID'] = pred_score.keys()

        custs = data_prec_score['CUST_ID'].apply(function,i=0)
        items = data_prec_score['CUST_ID'].apply(function,i=1)

        data_prec_score['CUST_ID']=custs
        data_prec_score['ARTICLE_ID']=items

        data_prec_score.to_csv( directory+'/data_prec_score.csv', index=False)
        '''

        data_prec_score = pd.read_csv( directory+'/data_prec_score.csv')
        #'''

        ### Test Set of Meta Model and doing tests on recommendationa algorithms
        test_dict_2 = reco_algo.test_dict_2(data_test_meta, data_prec,data_train)
        
        print('Test of algos and Test for meta model , num of users :',len(test_dict_2.keys()))
        
        with open( directory+'/test_dict_2', 'w') as fil:
            print(test_dict_2, file=fil)
        
        def worker2(i,key, value, nb_list):
            val = test_dict[key]
            nb = len(reco_algo.items)

            classifier = dict()

            liste_ARM, score_ARM = reco_algo.top_N_ARM(key,nb,val)
            liste_BPR, score_BPR = reco_algo.top_N_BPR(key,nb,val)
            liste_VAES, score_VAES = reco_algo.top_N_VAES(key,nb,val)
            liste_ALS, score_ALS = reco_algo.top_N_ALS(key,nb,val)
            liste_knn_50, score_knn_50 = top_N_KNN50(key,nb,val)
            liste_SPEC, score_SPEC = top_N_SPECTRAL(key,nb,val)
            
            k = 0

            liste = set(liste_ARM).intersection(set(liste_BPR))
            liste = liste.intersection(set(liste_ALS))
            liste = liste.intersection(set(liste_SPEC))
            liste = liste.intersection(set(liste_VAES))
            liste = liste.intersection(set(liste_knn_50))

            for item in reco_algo.items :
                if item in liste:
                    k = k+1
                    clas = []

                    clas.append( liste_ARM.index(item) )
                    clas.append( liste_knn_50.index(item) )
                    clas.append( liste_ALS.index(item) )
                    clas.append( liste_BPR.index(item) )
                    clas.append( liste_VAES.index(item) )
                    clas.append( liste_SPEC.index(item) )

                    classifier[key,item] = clas
            
            print(i,' NUUUM : ',k)

            return classifier
        
        results=[]
        manager = multiprocessing.Manager()
        #precisions = manager.dict()
        pred = dict()
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        for i,(key,value) in enumerate(test_dict_2.items()):
            result = pool.apply_async(worker2,args=(i,key,value,nm))
            r = result.get()
            pred.update(r)
        pool.close()
        pool.join()
        
        col = {0:'ARM_PRECISION',1:'K50_PRECISION',2:'ALS_PRECISION',3:'BPR_PRECISION',\
        4:'VAES_PRECISION',5:'SPEC_PRECISION'}

        print('Data for Testing meta model is okey')
        data_test_meta = pd.DataFrame.from_dict(pred,orient='index')
        data_test_meta.rename(columns=col, inplace=True)

        data_test_meta['CUST_ID'] = pred.keys()

        custs = data_test_meta['CUST_ID'].apply(function,i=0)
        items = data_test_meta['CUST_ID'].apply(function,i=1)

        data_test_meta['CUST_ID']=custs
        data_test_meta['ARTICLE_ID']=items

        data_test_meta = pd.merge(data_test_meta,data_train_2,on='CUST_ID') #Add Sex and Age for each customer
        data_test_meta.to_csv( directory+'/data_reco_baselines.csv', index=False)
        '''
        data_test_meta = pd.read_csv( directory+'/data_reco_baselines.csv')
        #'''

        def worker3(i,key, value, nb_list):
            val = test_dict[key]
            nb = len(reco_algo.items)

            classifier2 = dict()

            liste_ARM, score_ARM = reco_algo.top_N_ARM(key,nb,val)
            liste_BPR, score_BPR = reco_algo.top_N_BPR(key,nb,val)
            liste_VAES, score_VAES = reco_algo.top_N_VAES(key,nb,val)
            liste_ALS, score_ALS = reco_algo.top_N_ALS(key,nb,val)
            liste_knn_50, score_knn_50 = top_N_KNN50(key,nb,val)
            liste_SPEC, score_SPEC = top_N_SPECTRAL(key,nb,val)
            
            k = 0

            liste = set(liste_ARM).intersection(set(liste_BPR))
            liste = liste.intersection(set(liste_ALS))
            liste = liste.intersection(set(liste_SPEC))
            liste = liste.intersection(set(liste_VAES))
            liste = liste.intersection(set(liste_knn_50))

            for item in reco_algo.items :
                if item in liste:
                    k = k+1

                    clas2 = []

                    clas2.append( score_ARM[ liste_ARM.index(item) ] )
                    clas2.append( score_knn_50[ liste_knn_50.index(item) ] )
                    clas2.append( score_ALS[ liste_ALS.index(item) ] )
                    clas2.append( score_BPR[ liste_BPR.index(item) ] )
                    clas2.append( score_VAES[ liste_VAES.index(item) ] ) 
                    clas2.append( score_SPEC[ liste_SPEC.index(item) ] )

                    classifier2[key,item] = clas2
            
            print(i,' NUUUM : ',k)

            return classifier2
            
        results=[]
        manager = multiprocessing.Manager()
        pred_score = dict()
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        for i,(key,value) in enumerate(test_dict_2.items()):
            result = pool.apply_async(worker3,args=(i,key,value,nm))
            r = result.get()
            pred_score.update(r)
        pool.close()
        pool.join()

        data_test_meta_score = pd.DataFrame.from_dict(pred_score,orient='index')
        data_test_meta_score.rename(columns=col, inplace=True)

        data_test_meta_score['CUST_ID'] = pred_score.keys()

        custs = data_test_meta_score['CUST_ID'].apply(function,i=0)
        items = data_test_meta_score['CUST_ID'].apply(function,i=1)

        data_test_meta_score['CUST_ID']=custs
        data_test_meta_score['ARTICLE_ID']=items
        
        data_test_meta_score.to_csv( directory+'/data_reco_baselines_score.csv', index=False)
        '''
        data_test_meta_score = pd.read_csv( directory+'/data_reco_baselines_score.csv')
        #'''

        
        us = list(test_dict_2.keys())
        n = np.full(len(us),1,dtype='uint8')

        us = pd.DataFrame({'CUST_ID':us,'KEY':n})
        #us = pd.merge(us,nb_purch_train,on='CUST_ID')
        us = pd.merge(us,nb_purch_test,on='CUST_ID')

        n = np.full(len(reco_algo.items),1,dtype='uint8')

        it = pd.DataFrame({'ARTICLE_ID':reco_algo.items,'KEY':n})
        #it = pd.merge(it,nb_article_purch_train,on='ARTICLE_ID')
        it = pd.merge(it,nb_article_purch_test,on='ARTICLE_ID')

        us = pd.merge(us,it,on='KEY')
        us.to_csv( directory+'/test_protocol.csv', index=False )

        del us
        
        purch_items = list()
        purch_users = list()

        us = list(test_dict_2.keys())
        for u in us:
            user_ind = reco_algo.users_d[u]
            user_ind_v = 0
            purchased = reco_algo.get_purchased_2(user_ind, user_ind_v, test_dict[u])

            purch_items.extend(purchased)
            purch_users.extend( np.full(len(purchased),u,dtype='uint8') )
        
        purch = pd.DataFrame()
        purch['CUST_ID']= purch_users
        purch['ARTICLE_ID']= purch_items

        purch.to_csv( directory+'/purch.csv', index=False )

        print('Meta Model Data')
        
        #Meta Model Training set

        data_prec = pd.merge(data_prec,nb_purch_test, on='CUST_ID')
        data_prec_score = pd.merge(data_prec_score,nb_purch_test, on='CUST_ID')

        data_prec = pd.merge(data_prec,nb_article_purch_test, on='ARTICLE_ID')
        data_prec_score = pd.merge(data_prec_score,nb_article_purch_test, on='ARTICLE_ID')

        data_test_meta = pd.merge(data_test_meta,nb_purch_test, on='CUST_ID')
        data_test_meta_score = pd.merge(data_test_meta_score,nb_purch_test, on='CUST_ID')

        data_test_meta = pd.merge(data_test_meta,nb_article_purch_test, on='ARTICLE_ID')
        data_test_meta_score = pd.merge(data_test_meta_score,nb_article_purch_test, on='ARTICLE_ID')

        data_test_meta.to_csv( directory+'/data_reco_baselines.csv', index=False)
        data_test_meta_score.to_csv( directory+'/data_reco_baselines_score.csv', index=False)

        nb_purch_train.to_csv( directory+'/nb_purch_train.csv', index=False )
        nb_purch_test.to_csv( directory+'/nb_purch_test.csv', index=False )
        nb_article_purch_train.to_csv( directory+'/nb_article_purch_train.csv', index=False )
        nb_article_purch_test.to_csv( directory+'/nb_article_purch_test.csv', index=False )
        
        train_model = train_model.append(data_prec)
    del reco_algo

train_model.to_csv( directory+'/train_model.csv', index=False )
data_prec_score.to_csv( directory+'/train_model_score.csv', index=False )

train_model = pd.read_csv( directory+'/train_model.csv')
data_prec_score = pd.read_csv( directory+'/train_model_score.csv')

#Train Meta Models

X_train = train_model[['AGE','NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST']]

y_train_ARM = train_model[['ARM_PRECISION']]
y_train_KNN = train_model[['K50_PRECISION']]
y_train_ALS = train_model[['ALS_PRECISION']]
y_train_BPR = train_model[['BPR_PRECISION']]
y_train_VAES = train_model[['VAES_PRECISION']]
y_train_SPEC = train_model[['SPEC_PRECISION']]

print('Normalize')

x = X_train[['NB_PURCH_TEST']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_train['NB_PURCH_TEST'] = x_scaled

x = X_train[['NB_ARTICLE_PURCH_TEST']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_train['NB_ARTICLE_PURCH_TEST'] = x_scaled

print('Categorical Columns')
#Categorical columns
object_col = train_model.dtypes == 'object'
object_col = list( object_col[object_col].index )

print(object_col)

if 'CUST_ID' in object_col:
    object_col.remove('CUST_ID')

if 'ARTICLE_ID' in object_col:
    object_col.remove('ARTICLE_ID')

f = X_train[object_col]

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) #One Hot Encoder

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_col]))

with open( directory+'/encoder', 'wb') as fil: 
    pickle.dump(OH_encoder, fil)

OH_cols_train.index = X_train.index #One Hot Remoes indexes, they are replaced
X_train = X_train.drop(object_col, axis=1) #Drop categorical columns
X_train = pd.concat([X_train, OH_cols_train], axis=1) #Add hot one encoding columns

X_train = sparse.csr_matrix(X_train.values)

y_train_ARM = y_train_ARM.to_numpy().ravel()
y_train_KNN = y_train_KNN.to_numpy().ravel()
y_train_BPR = y_train_BPR.to_numpy().ravel()
y_train_ALS = y_train_ALS.to_numpy().ravel()
y_train_VAES = y_train_VAES.to_numpy().ravel()
y_train_SPEC = y_train_SPEC.to_numpy().ravel()

d = directory

print('Logistic Regression Model Now')

param_log_reg = {'penalty':['l2'],\
'C': [0.1,0.5,1],\
'solver' :['lbfgs', 'sag', 'saga'],\
'max_iter':[100, 200, 500]}

model = LogisticRegression()
model_ARM = RandomizedSearchCV(model, param_log_reg, n_iter=11, cv=2, n_jobs=-1)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM,open( d+'/models/ARM_log_reg','wb'))

model = LogisticRegression()
model_KNN = RandomizedSearchCV(model, param_log_reg, n_iter=11, cv=2, n_jobs=-1)
model_KNN.fit(X_train,y_train_KNN)
pickle.dump(model_KNN,open( d+'/models/KNN_log_reg','wb'))

model = LogisticRegression()
model_ALS = RandomizedSearchCV(model, param_log_reg, n_iter=11, cv=2, n_jobs=-1)
model_ALS.fit(X_train,y_train_ALS)
pickle.dump(model_ALS,open( d+'/models/ALS_log_reg','wb'))

model = LogisticRegression()
model_BPR = RandomizedSearchCV(model, param_log_reg, n_iter=11, cv=2, n_jobs=-1)
model_BPR.fit(X_train,y_train_BPR)
pickle.dump(model_BPR,open( d+'/models/BPR_log_reg','wb'))

model = LogisticRegression()
model_VAES = RandomizedSearchCV(model, param_log_reg, n_iter=11, cv=2, n_jobs=-1)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES,open( d+'/models/VAES_log_reg','wb'))

model = LogisticRegression()
model_SPEC = RandomizedSearchCV(model, param_log_reg, n_iter=11, cv=2, n_jobs=-1)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC,open( d+'/models/SPEC_log_reg','wb'))

print('Decision Tree Classifier Model Now')

param_dec = {'criterion':['gini', 'entropy'],\
'max_depth':[3,5,10,20],\
'splitter': ['best', 'random'],\
'max_features' : ['auto', 'sqrt', 'log2']}

model = DecisionTreeClassifier()
model_ARM = RandomizedSearchCV(model, param_dec, n_iter=11, cv=2, n_jobs=-1)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM,open( d+'/models/ARM_dec_tree','wb'))

model = DecisionTreeClassifier()
model_KNN = RandomizedSearchCV(model, param_dec, n_iter=11, cv=2, n_jobs=-1)
model_KNN.fit(X_train,y_train_KNN)
pickle.dump(model_KNN,open( d+'/models/KNN_dec_tree','wb'))

model = DecisionTreeClassifier()
model_ALS = RandomizedSearchCV(model, param_dec, n_iter=11, cv=2, n_jobs=-1)
model_ALS.fit(X_train,y_train_ALS)
pickle.dump(model_ALS,open( d+'/models/ALS_dec_tree','wb'))

model = DecisionTreeClassifier()
model_BPR = RandomizedSearchCV(model, param_dec, n_iter=11, cv=2, n_jobs=-1)
model_BPR.fit(X_train,y_train_BPR)
pickle.dump(model_BPR,open( d+'/models/BPR_dec_tree','wb'))

model = DecisionTreeClassifier()
model_VAES = RandomizedSearchCV(model, param_dec, n_iter=11, cv=2, n_jobs=-1)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES,open( d+'/models/VAES_dec_tree','wb'))

model = DecisionTreeClassifier()
model_SPEC = RandomizedSearchCV(model, param_dec, n_iter=11, cv=2, n_jobs=-1)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC,open( d+'/models/SPEC_dec_tree','wb'))

print('SGD Classifier Model Now')

param_sgd = {'penalty':['l2'],\
'alpha': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'max_iter':[100, 200, 500, 1000]}

model = SGDClassifier()
model_ARM = RandomizedSearchCV(model, param_sgd, n_iter=11, cv=2, n_jobs=-1)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM,open( d+'/models/ARM_sgd','wb'))

model = SGDClassifier()
model_KNN = RandomizedSearchCV(model, param_sgd, n_iter=11, cv=2, n_jobs=-1)
model_KNN.fit(X_train,y_train_KNN)
pickle.dump(model_KNN,open( d+'/models/KNN_sgd','wb'))

model = SGDClassifier()
model_ALS = RandomizedSearchCV(model, param_sgd, n_iter=11, cv=2, n_jobs=-1)
model_ALS.fit(X_train,y_train_ALS)
pickle.dump(model_ALS,open( d+'/models/ALS_sgd','wb'))

model = SGDClassifier()
model_BPR = RandomizedSearchCV(model, param_sgd, n_iter=11, cv=2, n_jobs=-1)
model_BPR.fit(X_train,y_train_BPR)
pickle.dump(model_BPR,open( d+'/models/BPR_sgd','wb'))

model = SGDClassifier()
model_VAES = RandomizedSearchCV(model, param_sgd, n_iter=11, cv=2, n_jobs=-1)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES,open( d+'/models/VAES_sgd','wb'))

model = SGDClassifier()
model_SPEC = RandomizedSearchCV(model, param_sgd, n_iter=11, cv=2, n_jobs=-1)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC,open( d+'/models/SPEC_sgd','wb'))

print('Decision XGBoost Model Now')

param_xgb = {'n_estimators':[500, 1000, 2000],\
'learning_rate': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'gamma': [0.5, 1, 2, 5],\
'reg_lambda':[0.001, 0.005, 0.01, 0.05],\
'max_depth':[3,5,10]}

model = XGBClassifier()
model_ARM = RandomizedSearchCV(model, param_xgb, n_iter=11, cv=2, n_jobs=-1)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM,open( d+'/models/ARM_xgb','wb'))

model = XGBClassifier()
model_KNN = RandomizedSearchCV(model, param_xgb, n_iter=11, cv=2, n_jobs=-1)
model_KNN.fit(X_train,y_train_KNN)
pickle.dump(model_KNN,open( d+'/models/KNN_xgb','wb'))

model = XGBClassifier()
model_ALS = RandomizedSearchCV(model, param_xgb, n_iter=11, cv=2, n_jobs=-1)
model_ALS.fit(X_train,y_train_ALS)
pickle.dump(model_ALS,open( d+'/models/ALS_xgb','wb'))

model = XGBClassifier()
model_BPR = RandomizedSearchCV(model, param_xgb, n_iter=11, cv=2, n_jobs=-1)
model_BPR.fit(X_train,y_train_BPR)
pickle.dump(model_BPR,open( d+'/models/BPR_xgb','wb'))

model = XGBClassifier()
model_VAES = RandomizedSearchCV(model, param_xgb, n_iter=11, cv=2, n_jobs=-1)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES,open( d+'/models/VAES_xgb','wb'))

model = XGBClassifier()
model_SPEC = RandomizedSearchCV(model, param_xgb, n_iter=11, cv=2, n_jobs=-1)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC,open( d+'/models/SPEC_xgb','wb'))

#Train Stacking models

X_train = data_prec_score[['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','K50_PRECISION','ALS_PRECISION','BPR_PRECISION','VAES_PRECISION','SPEC_PRECISION']]

y_train = data_prec_score[['SCORE']].to_numpy().ravel()

print('Normalize')

for c in ['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','K50_PRECISION','ALS_PRECISION','BPR_PRECISION','VAES_PRECISION','SPEC_PRECISION']:
    x = X_train[[c]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_train[c] = x_scaled

d = directory

print('Logistic Regression Model Now')

param_log_reg = {'penalty':['l2'],\
'C': [0.1,0.5,1],\
'solver' :['lbfgs', 'sag', 'saga'],\
'max_iter':[100, 200, 500]}

model = LogisticRegression()
model_Log = RandomizedSearchCV(model, param_log_reg, n_iter=11, cv=2, n_jobs=-1)
model_Log.fit(X_train,y_train)
pickle.dump(model_Log,open( d+'/models/STACKING_log_reg','wb'))

print('Decision Tree Classifier Model Now')

param_dec = {'criterion':['gini', 'entropy'],\
'max_depth':[3,5,10,20],\
'splitter': ['best', 'random'],\
'max_features' : ['auto', 'sqrt', 'log2']}

model = DecisionTreeClassifier()
model_Ded = RandomizedSearchCV(model, param_dec, n_iter=11, cv=2, n_jobs=-1)
model_Ded.fit(X_train,y_train)
pickle.dump(model_Ded,open( d+'/models/STACKING_dec_tree','wb'))

print('SGD Classifier Model Now')

param_sgd = {'penalty':['l2'],\
'alpha': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'max_iter':[100, 200, 500, 1000]}

model = SGDClassifier()
model_Sgd = RandomizedSearchCV(model, param_sgd, n_iter=11, cv=2, n_jobs=-1)
model_Sgd.fit(X_train,y_train)
pickle.dump(model_Sgd,open( d+'/models/STACKING_sgd','wb'))

print('Decision XGBoost Model Now')

param_xgb = {'n_estimators':[500, 1000, 2000],\
'learning_rate': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'gamma': [0.5, 1, 2, 5],\
'reg_lambda':[0.001, 0.005, 0.01, 0.05],\
'max_depth':[3,5,10]}

model = XGBClassifier()
model_Xgb = RandomizedSearchCV(model, param_xgb, n_iter=11, cv=2, n_jobs=-1)
model_Xgb.fit(X_train,y_train)
pickle.dump(model_Xgb,open( d+'/models/STACKING_xgb','wb'))