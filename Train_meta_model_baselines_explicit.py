import pandas as pd
import numpy as np
import datetime as dt
import multiprocessing
import os
import itertools
from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing as pp
import implicit
import scipy.sparse as sparse
import pickle
import time
import Knn
import svd
import nmf
import sys
import pool_algos
from SpectralCf import SpectralCF
import tensorflow as tf
import random as rd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR 
from xgboost import XGBRegressor

print('mo : Movies \nmu : Music  ? ')
d = str(input())

print('Number of recommendations : ')
nm = int(input())

print('You want to split again ? ')
splt = int(input())

train_model = pd.DataFrame(columns=['ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION',\
'K100_PRECISION','VAES_PRECISION','BPR_PRECISION','CUST_ID','ARTICLE_ID',\
'NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST'])

def f(x,i):
    x = str(x)
    x = x.split(',')
    if i : #Item
        return x[i][2:-2]
    else: #Cust
        return x[i][2:-1]

if d == 'mu':
    d = 'Music_split'
elif d == 'mo' :
    d = 'Movies_split'

if splt==1:
    head = ['CUST_ID','ARTICLE_ID','RATING','TIME']

    if d == 'Music_split':
        data = pd.read_csv('data_set/Digital_Music.csv',names=head)
        time = data.drop_duplicates(subset=['CUST_ID','ARTICLE_ID'])[['CUST_ID','ARTICLE_ID','TIME']]
        data = data.groupby(['CUST_ID','ARTICLE_ID'])['RATING'].mean().reset_index(name='RATING')
        data = pd.merge(data,time,on=['CUST_ID','ARTICLE_ID'])
        nb_i = 10
        nb_u = 10
    elif d == 'Movies_split':
        data = pd.read_csv('data_set/Movies_and_TV.csv',names=head)
        time = data.drop_duplicates(subset=['CUST_ID','ARTICLE_ID'])[['CUST_ID','ARTICLE_ID','TIME']]
        data = data.groupby(['CUST_ID','ARTICLE_ID'])['RATING'].mean().reset_index(name='RATING')
        data = pd.merge(data,time,on=['CUST_ID','ARTICLE_ID'])
        nb_i = 20
        nb_u = 100


    data = data[data.groupby('ARTICLE_ID')['ARTICLE_ID'].transform('count').ge(nb_i)]
    data = data[data.groupby('CUST_ID')['CUST_ID'].transform('count').ge(nb_u)]

    us = data.CUST_ID.unique()

    data_train = pd.DataFrame(columns=head)
    data_train_meta = pd.DataFrame(columns=head)
    data_test_meta = pd.DataFrame(columns=head)

    for u in us :
        data_local = data[data['CUST_ID']==u]
        data_local = data_local.sort_values(by='TIME')
        data_local = data_local[['CUST_ID','ARTICLE_ID','RATING']]
        deb,inter = round(0.5*len(data_local)),round(0.9*len(data_local))
        
        data_train = data_train.append(data_local[:deb])
        data_train_meta = data_train_meta.append(data_local[deb:inter])
        data_test_meta = data_test_meta.append(data_local[inter:])

    data.to_csv( d+'/data.csv', index=False)

    data_train.to_csv( d+'/data_train.csv', index=False)
    data_train_meta.to_csv( d+'/data_train_meta.csv', index=False)
    data_test_meta.to_csv( d+'/data_test_meta.csv', index=False)

elif splt==0:
    data_train = pd.read_csv( d+'/data_train.csv' )
    data_train_meta = pd.read_csv( d+'/data_train_meta.csv' )
    data_test_meta = pd.read_csv( d+'/data_test_meta.csv' )

print('Split Done')

data_train_2 = data_train.groupby(['CUST_ID','ARTICLE_ID'])['RATING'].mean().reset_index(name='FREQUENCY')
data_train_2['FREQUENCY'] = data_train_2['FREQUENCY'].map(lambda x : int(x>3))
reco_algo = pool_algos.RecommendationAlgos(None,None,data_train_2,num=nm)
reco_algo.get_association_matrix(amazon=True) #ARM

reco_algo.fit_vae(500,150,0.2,2) #VAES
reco_algo.res_vae(2)

vaes_res = reco_algo.res

data_train = data_train.groupby(['CUST_ID','ARTICLE_ID'])['RATING'].mean().reset_index(name='FREQUENCY')

#NEEEEEW

nb_purch_train = data_train.groupby('CUST_ID')['FREQUENCY'].size().reset_index(name='NB_PURCH_TRAIN')
nb_article_purch_train = data_train.groupby('ARTICLE_ID')['FREQUENCY'].size().reset_index(name='NB_ARTICLE_PURCH_TRAIN')

nb_purch_test = data_train_meta.groupby(['CUST_ID','ARTICLE_ID']).size().reset_index(name='FREQUENCY')
nb_article_purch_test = nb_purch_test.groupby('ARTICLE_ID')['FREQUENCY'].size().reset_index(name='NB_ARTICLE_PURCH_TEST')
nb_purch_test = nb_purch_test.groupby('CUST_ID')['FREQUENCY'].size().reset_index(name='NB_PURCH_TEST')

nb_purch_train.to_csv(d+'/nb_purch_train.csv', index=False)
nb_purch_test.to_csv(d+'/nb_purch_test.csv', index=False)
nb_article_purch_train.to_csv(d+'/nb_article_purch_train.csv', index=False)
nb_article_purch_test.to_csv(d+'/nb_article_purch_test.csv', index=False)

### Train Pool of recommendation algorithms

########################################################################
num = nm

items = list(np.sort(data_train.ARTICLE_ID.unique()))  # all unique items
users = list(np.sort(data_train.CUST_ID.unique()))    # all unique users
rating = list(np.sort(data_train.FREQUENCY))
########################################################################
rows = data_train.CUST_ID.astype(pd.api.types.CategoricalDtype(categories = users)).cat.codes    # Get the associated row indices
cols = data_train.ARTICLE_ID.astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes    # Get the associated row indices

index_list = list(cols.values)
items_2 = list(data_train.ARTICLE_ID)

items_d = dict()
items_dd = dict()
users_d = dict()
users_dd = dict()

j=0
for i in items_2:
    if i not in items_d:
        items_d[i] = index_list[j]
        items_dd[ index_list[j] ] = i
    j=j+1

index_list = list(rows.values)
items_2 = list(data_train.CUST_ID)

j=0
for i in items_2:
    if i not in users_d:
        users_d[i] = index_list[j]
        users_dd[ index_list[j] ] = i
    j=j+1

data = sparse.csr_matrix((rating, (rows, cols)), shape = (len(users), len(items)  )  )

rating2 = list()
for i in rating:
    if i<4:
        rating2.append(0)
    else:
        rating2.append(1)

rating_matrix = sparse.csr_matrix((rating2, (rows, cols)), shape = (len(users), len(items)  )  )
Association_matrix = rating_matrix.T * rating_matrix # P.j(t) * P.i
c = rating_matrix.T.sum(axis=1).A.ravel()

c2 = list()
for i in range(c.shape[0]):
    if c[i]==0:
        c2.append(0.0001)
    else:
        c2.append(c[i])
c2 = np.array(c2)

c = sparse.diags(1/c2) # ||P.j||

Association_matrix= c.dot(Association_matrix)

print(Association_matrix.shape)
print(' #------------------ ARM ... Done !-------------------#')

nmf_pure = nmf.NMFRecommender(data)
nmf_pure.fit(num_factors=200)
print(' #------------------ NMF ... Done !-------------------#')

svd_pure = svd.PureSVDRecommender(data)
svd_pure.fit()
print(' #------------------ SVD Pure ... Done !-------------------#')


Similarity_matrix = pp.normalize(data.tocsc(), axis=0)
Similarity_matrix = Similarity_matrix.T * Similarity_matrix
print('SIMILARITY DONE')

print(Similarity_matrix.shape)

knn100 = Knn.KNN(data)
knn100.fit(Similarity_matrix, selectTopK = True, topK=100)
print(' #------------------ KNN 100 ... Done !-------------------#')

data = data.astype(np.float32)

inter = data_train.groupby('CUST_ID')['ARTICLE_ID'].agg(list).reset_index()
train_items = dict()

for u in users:
    l = inter[inter['CUST_ID']==u]['ARTICLE_ID'].tolist()[0]
    l = [items_d[i] for i in l]
    train_items[users_d[u]] = l

del inter

def sample():
    if BATCH_SIZE <= len(users):
        users_ = rd.sample(range(len(users)), BATCH_SIZE)
    else:
        users_ = [ rd.choice( range(len(users)) ) for _ in range(BATCH_SIZE) ]

    def sample_pos_items_for_u(u, num):
        pos_items = train_items[ u ] #TRAIN ITEMS A FAIRE
        if len(pos_items) >= num:
            return rd.sample(pos_items, num)
        else:
            return [rd.choice(pos_items) for _ in range(num)]

    def sample_neg_items_for_u(u, num):
        neg_items = list(set(range(len(items))) - set(train_items[ u ]))
        return rd.sample(neg_items, num)

    pos_items, neg_items = [], []
    for u in users_:
        pos_items += sample_pos_items_for_u(u, 1)
        neg_items += sample_neg_items_for_u(u, 1)

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

model = SpectralCF(K=K, graph=data, n_users=len(users), n_items=len(items), emb_dim=EMB_DIM,\
lr=LR, decay=DECAY, batch_size=BATCH_SIZE)

print("Instantiating model... done!")

config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session(config=config)
sess.run(tf.compat.v1.global_variables_initializer())

print("Training model... ")

res_spec = dict()

########################################################################

test_set = data_train_meta[['CUST_ID', 'ARTICLE_ID']]
test = data_train_meta

keys = list(data_train[['CUST_ID', 'ARTICLE_ID']].columns.values)

i1 = test_set.set_index(keys).index
i2 = data_train[['CUST_ID', 'ARTICLE_ID']].set_index(keys).index

test_set = test_set[~i1.isin(i2)]
test_set = test_set[test_set['CUST_ID'].isin(set(users))]  # Keep only users appearing in the training set

test = pd.merge(test_set,test,on=['CUST_ID','ARTICLE_ID'])
test = test[['CUST_ID','ARTICLE_ID','RATING']]

test2 = pd.DataFrame(columns=['CUST_ID','ARTICLE_ID','RATING','NUM'])

for u in test.CUST_ID.unique():
    inter = test[test['CUST_ID']==u]
    inter = inter.sample(frac=1)
    inter = inter.sort_values(by='RATING',ascending=False)
    inter['NUM'] = list(range(0,len(inter)))
    test2 = test2.append(inter, ignore_index = True)

test = test2
test.to_csv(d+'/test_test.csv', index=False)

del test2

# Create a dictionnary where keys are customers and values are set of purchased products in the test set
test_dict = test_set.groupby('CUST_ID').ARTICLE_ID.agg(set).to_dict()
del test_set


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
                user_inter.append(users_d[u])
            
            user_batch = user_inter

            if len(user_batch) < BATCH_SIZE:
                user_batch += [user_batch[-1]] * (BATCH_SIZE - len(user_batch))

            user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})

            user_batch_rating_uid = zip(user_batch_rating, user_batch)

            for x in user_batch_rating_uid:
                rating = x[0]
                u = x[1]
                u = users_dd[u]
                item_score = []
                for i in items:
                    item_score.append((i, rating[ items_d[i] ]))
                item_score = sorted(item_score, key=lambda x: x[1],reverse=True)
                #item_score.reverse()
                res_spec[u]=item_score

with open(d+'/res_spec', 'w') as fil:
    print(res_spec, file=fil)
########################################################################

print('Train Done')
print('Meta training , num of users :',len(test_dict.keys()))


def get_purchased(cust_ind, user_ind_v, validation_values):
    purchased_ind = data[cust_ind,:].nonzero()[1]
    purchased_liste = []
    for i in purchased_ind:
        purchased_liste.append(items[i])
    
    for i in validation_values:
        purchased_liste.append(i)

    return purchased_liste

def top_N_SVD(user, n, validation_values):
    if user in users_d:
        user_ind = users_d[user]
        user_ind_v = 0
        
        pred, score = svd_pure.recommend(user_ind,return_scores=True)

        items_purchased = get_purchased(user_ind, user_ind_v, validation_values) ##
        ensemble_produits = set(items_purchased)

        rec_list= []
        score_list = []
        i=0
        for k,ind in enumerate(pred):
            if i >= n:
                break
            else:
                item_id = items[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    rec_list.append(item_id)
                    score_list.append(score[ind])
                    i=i+1
        
        return rec_list, score_list
    else:
        return None

def top_N_KNN10(user, n, validation_values):
    if user in users_d:
        user_ind = users_d[user]
        user_ind_v = 0
        
        pred, score = knn10.recommend(user_ind,return_scores=True)

        items_purchased = get_purchased(user_ind, user_ind_v, validation_values) ##
        ensemble_produits = set(items_purchased)

        rec_list= []
        score_list = []
        i=0
        for k,ind in enumerate(pred):
            if i >= n:
                break
            else:
                item_id = items[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    rec_list.append(item_id)
                    score_list.append(score[ind])
                    i=i+1
        
        return rec_list, score_list
    else:
        return None

def top_N_KNN100(user, n, validation_values):
    if user in users_d:
        user_ind = users_d[user]
        user_ind_v = 0
        
        pred, score = knn100.recommend(user_ind,return_scores=True)

        items_purchased = get_purchased(user_ind, user_ind_v, validation_values) ##
        ensemble_produits = set(items_purchased)

        rec_list= []
        score_list = []
        i=0
        for k,ind in enumerate(pred):
            if i >= n:
                break
            else:
                item_id = items[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    rec_list.append(item_id)
                    score_list.append(score[ind])
                    i=i+1
        
        return rec_list, score_list
    else:
        return None

def top_N_KNN50(user, n, validation_values):
    if user in users_d:
        user_ind = users_d[user]
        user_ind_v = 0
        
        pred, score = knn50.recommend(user_ind,return_scores=True)

        items_purchased = get_purchased(user_ind, user_ind_v, validation_values) ##
        ensemble_produits = set(items_purchased)

        rec_list= []
        score_list = []
        i=0
        for k,ind in enumerate(pred):
            if i >= n:
                break
            else:
                item_id = items[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    rec_list.append(item_id)
                    score_list.append(score[ind])
                    i=i+1
        
        return rec_list, score_list
    else:
        return None

def top_N_ARM(user, n, validation_values):
    if user in users_d:
        user_ind = users_d[user]
        user_ind_v = 0
        
        user_pref = rating_matrix[user_ind,:].toarray() * Association_matrix
        user_pref = user_pref.reshape(-1)

        items_purchased = get_purchased(user_ind, user_ind_v, validation_values)
        ensemble_produits = set(items_purchased)

        items_ind = np.argsort(user_pref)[::-1]      # Sort the indices of the items into order of best recommendations
        rec_list= []
        score_list = []
        i=0

        for k,ind in enumerate(items_ind):
            if i >= n:
                break
            else:
                item_id = items[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    rec_list.append(item_id)
                    score_list.append(user_pref[ind])
                    i=i+1
        return rec_list,score_list
    else:
        return None

def top_N_NMF(user, n, validation_values):
    if user in users_d:
        user_ind = users_d[user]
        user_ind_v = 0
        
        pred, score = nmf_pure.recommend(user_ind,return_scores=True)

        items_purchased = get_purchased(user_ind, user_ind_v, validation_values) ##
        ensemble_produits = set(items_purchased)

        rec_list= []
        score_list = []
        i=0
        for k,ind in enumerate(pred):
            if i >= n:
                break
            else:
                item_id = items[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    rec_list.append(item_id)
                    score_list.append(score[ind])
                    i=i+1
        
        return rec_list, score_list
    else:
        return None

def top_N_SPECTRAL(user, n, validation_values):
    if user in users_d:
        user_ind = users_d[user]
        user_ind_v = 0
        
        pred = res_spec[user]

        items_purchased = get_purchased(user_ind, user_ind_v, validation_values) ##
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

def top_N_VAE(user, n, validation_values):
    if user in users_d :
        user_ind = users_d[user]
        user_ind_v = 0

        score = vaes_res[user_ind,:]
        sort = np.argsort(-score)

        #liste = reco_algo.top_N_VAES(user,n, validation_values)
        items_purchased = get_purchased(user_ind, user_ind_v, validation_values) ##
        ensemble_produits = set(items_purchased)
        
        top = []
        scr = []
        i = 0
        for ind in sort:
            if i >= n:
                break
            else:
                item_id = items[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    top.append(item_id)
                    scr.append(score[i])
                    i = i+1
        return top,scr

def worker_rank(i,key, value, classifier, classifier2):
    num = len(items)
    
    val = set()

    liste_ARM, score_ARM = top_N_ARM(key,num, val)
    liste_SVD_pure, score_SVD_pure = top_N_SVD(key,num, val)
    liste_nmf, score_nmf = top_N_NMF(key,num, val)
    liste_knn_100, score_knn_100 = top_N_KNN100(key,num, val)
    liste_VAES, score_VAES = top_N_VAE(key,num, val)
    liste_SPEC, score_SPEC = top_N_SPECTRAL(key,num, val)

    d = test[test['CUST_ID']==key]

    liste = set(liste_ARM).intersection(set(liste_SVD_pure))
    liste = liste.intersection(set(liste_nmf))
    liste = liste.intersection(set(liste_SPEC))
    liste = liste.intersection(set(liste_VAES))
    liste = liste.intersection(set(liste_knn_100))

    k=0
    for item in value :
        if item in liste:

            j = int(d[d['ARTICLE_ID']==item]['NUM'])
            h = int(d[d['ARTICLE_ID']==item]['RATING'])
            clas = []
            clas2 = []

            clas.append( abs(j-liste_ARM.index(item)) )
            clas.append( abs(j-liste_SVD_pure.index(item)) )
            clas.append( abs(j-liste_nmf.index(item)) )
            clas.append( abs(j-liste_knn_100.index(item)) )
            clas.append( abs(j-liste_VAES.index(item)) )
            clas.append( abs(j-liste_SPEC.index(item)) )

            clas2.append( score_ARM[ liste_ARM.index(item) ] )
            clas2.append( score_SVD_pure[ liste_SVD_pure.index(item) ] )
            clas2.append( score_nmf[ liste_nmf.index(item) ] )
            clas2.append( score_knn_100[ liste_knn_100.index(item) ] )
            clas2.append( score_VAES[ liste_VAES.index(item) ] ) 
            clas2.append( score_SPEC[ liste_SPEC.index(item) ] )
            clas2.append( h )

            k=k+1

            classifier[key,item] = clas
            classifier2[key,item] = clas2


results=[]
manager = multiprocessing.Manager()
precisions = manager.dict()
precisions_score = manager.dict()
pool = multiprocessing.Pool(multiprocessing.cpu_count())

print('Num of Process : ',pool._processes)

for i,(key,value) in enumerate(test_dict.items()):
    result = pool.apply_async(worker_rank,args=(i,key,value,precisions,precisions_score))
    #result.get()
pool.close()
pool.join()

col = {0:'ARM_PRECISION',1:'SVD_PURE_PRECISION',2:'NMF_PRECISION',3:'K100_PRECISION',\
4:'VAES_PRECISION',5:'SPEC_PRECISION'}

col_score = {0:'ARM_PRECISION',1:'SVD_PURE_PRECISION',2:'NMF_PRECISION',3:'K100_PRECISION',\
4:'VAES_PRECISION',5:'SPEC_PRECISION',6:'SCORE'}

data_prec = pd.DataFrame.from_dict(precisions,orient='index')
data_prec.rename(columns=col,inplace=True)

data_prec['CUST_ID'] = precisions.keys()

custs = data_prec['CUST_ID'].apply(f,i=0)
it_it = data_prec['CUST_ID'].apply(f,i=1)

data_prec['CUST_ID']=custs
data_prec['ARTICLE_ID']=it_it

data_prec = data_prec[['ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION',\
'K100_PRECISION','VAES_PRECISION','SPEC_PRECISION','CUST_ID','ARTICLE_ID']]

data_prec.to_csv(d+'/data_prec.csv', index=False)

'''
data_prec = pd.read_csv(d+'/data_prec.csv')
#'''


data_prec_score = pd.DataFrame.from_dict(precisions_score,orient='index')
data_prec_score.rename(columns=col_score,inplace=True)

data_prec_score['CUST_ID'] = precisions_score.keys()

custs = data_prec_score['CUST_ID'].apply(f,i=0)
it_it = data_prec_score['CUST_ID'].apply(f,i=1)

data_prec_score['CUST_ID']=custs
data_prec_score['ARTICLE_ID']=it_it

data_prec_score = data_prec_score[['ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION',\
'K100_PRECISION','VAES_PRECISION','SPEC_PRECISION','CUST_ID','ARTICLE_ID','SCORE']]

data_prec_score.to_csv(d+'/data_prec_score.csv', index=False)

'''
data_prec_score = pd.read_csv(d+'/data_prec_score.csv')
#'''

print('Data for Training meta model Done')

test_set = data_test_meta[['CUST_ID', 'ARTICLE_ID']]
test = data_test_meta

keys = list(data_train[['CUST_ID', 'ARTICLE_ID']].columns.values)
i1 = test_set.set_index(keys).index
i2 = data_train[['CUST_ID', 'ARTICLE_ID']].set_index(keys).index
test_set = test_set[~i1.isin(i2)]

keys = list(data_prec[['CUST_ID', 'ARTICLE_ID']].columns.values)
i1 = test_set.set_index(keys).index
i2 = data_prec[['CUST_ID', 'ARTICLE_ID']].set_index(keys).index
test_set = test_set[~i1.isin(i2)]

test_set = test_set[test_set['CUST_ID'].isin( set(users) )]
test_set = test_set[test_set['CUST_ID'].isin( set(data_prec.CUST_ID.unique()) )]

test = pd.merge(test_set,test,on=['CUST_ID','ARTICLE_ID'])
test = test[['CUST_ID','ARTICLE_ID','RATING']]

test2 = pd.DataFrame(columns=['CUST_ID','ARTICLE_ID','RATING','NUM'])

for u in test.CUST_ID.unique():
    inter = test[test['CUST_ID']==u]
    inter = inter.sample(frac=1)
    inter = inter.sort_values(by='RATING',ascending=False)
    inter['NUM'] = list(range(0,len(inter)))
    test2 = test2.append(inter, ignore_index = True)

test = test2
#test.to_csv(d+'/test_test_2.csv', index=False)
del test2

# Create a dictionnary where keys are customers and values are set of purchased products in the test set
test_dict_2 = test_set.groupby('CUST_ID').ARTICLE_ID.agg(set).to_dict()
del test_set

print('Test , num of users :',len(test_dict_2.keys()))

def worker_rank2(i,key, value):
    num = len(items)
    maxi = len(items)
    val = test_dict[key]

    classifier = dict()
    classifier2 = dict()
    
    liste_ARM, score_ARM = top_N_ARM(key,num, val)
    liste_SVD_pure, score_SVD_pure = top_N_SVD(key,num, val)
    liste_nmf, score_nmf = top_N_NMF(key,num, val)
    liste_knn_100,score_knn_100 = top_N_KNN100(key,num, val)
    liste_VAES, score_VAES = top_N_VAE(key,num, val)
    liste_SPEC, score_SPEC = top_N_SPECTRAL(key,num, val)

    liste = set(liste_ARM).intersection(set(liste_SVD_pure))
    liste = liste.intersection(set(liste_nmf))
    liste = liste.intersection(set(liste_SPEC))
    liste = liste.intersection(set(liste_VAES))
    liste = liste.intersection(set(liste_knn_100))

    #liste = list(liste)

    n = len(liste)
    k=0
    for item in liste :
        clas = []
        clas2 = []

        idx_ARM = liste_ARM.index(item)
        idx_SVD = liste_SVD_pure.index(item)
        idx_NMF = liste_nmf.index(item)
        idx_KNN = liste_knn_100.index(item)
        idx_VAES = liste_VAES.index(item)
        idx_SPEC = liste_SPEC.index(item)

        clas.append( idx_ARM )
        clas.append( idx_SVD ) 
        clas.append( idx_NMF )
        clas.append( idx_KNN )
        clas.append( idx_VAES )
        clas.append( idx_SPEC )

        clas2.append( '{:.7f}'.format(score_ARM[ idx_ARM ]) )
        clas2.append( '{:.7f}'.format(score_SVD_pure[ idx_SVD ]) )
        clas2.append( '{:.7f}'.format(score_nmf[ idx_NMF ]) )
        clas2.append( '{:.7f}'.format(score_knn_100[ idx_KNN ]) )
        clas2.append( '{:.7f}'.format(score_VAES[ idx_VAES ]) ) 
        clas2.append( '{:.7f}'.format(score_SPEC[ idx_SPEC ]) )

        classifier[key,item] = clas
        classifier2[key,item] = clas2

        k=k+1
    return classifier, classifier2


manager = multiprocessing.Manager()
precisions = dict()
precisions_score = dict()
pool = multiprocessing.Pool(multiprocessing.cpu_count())
result = []

for i,(key,value) in enumerate(test_dict_2.items()):
    result.append(pool.apply_async(worker_rank2,args=(i,key,value)))   
pool.close()
pool.join()

for re in result:
    r = re.get()
    r_score = r[1]
    r = r[0]
    precisions.update(r)
    precisions_score.update(r_score)

col = {0:'ARM_PRECISION',1:'SVD_PURE_PRECISION',2:'NMF_PRECISION',3:'K100_PRECISION',\
4:'VAES_PRECISION',5:'SPEC_PRECISION'}

print('Data for Testing meta model is okey')

data_test_meta = pd.DataFrame.from_dict(precisions,orient='index')
data_test_meta.rename(columns=col,inplace=True)

data_test_meta['CUST_ID'] = precisions.keys()

custs = data_test_meta['CUST_ID'].apply(f,i=0)
it_it = data_test_meta['CUST_ID'].apply(f,i=1)

data_test_meta['CUST_ID']=custs
data_test_meta['ARTICLE_ID']=it_it

data_test_meta = data_test_meta[['ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION',\
'K100_PRECISION','VAES_PRECISION','SPEC_PRECISION','CUST_ID','ARTICLE_ID']]

data_test_meta.to_csv(d+'/data_reco_baselines.csv', index=False)

'''
data_test_meta = pd.read_csv(d+'/data_reco_baselines.csv')
#''' 

data_test_meta_score = pd.DataFrame.from_dict(precisions_score,orient='index')
data_test_meta_score.rename(columns=col,inplace=True)

data_test_meta_score['CUST_ID'] = precisions_score.keys()

custs = data_test_meta_score['CUST_ID'].apply(f,i=0)
it_it = data_test_meta_score['CUST_ID'].apply(f,i=1)

data_test_meta_score['CUST_ID']=custs
data_test_meta_score['ARTICLE_ID']=it_it

data_test_meta_score = data_test_meta_score[['ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION',\
'K100_PRECISION','VAES_PRECISION','SPEC_PRECISION','CUST_ID','ARTICLE_ID']]

data_test_meta_score.to_csv(d+'/data_reco_baselines_score.csv', index=False)

'''
data_test_meta_score = pd.read_csv(d+'/data_reco_baselines_score.csv')
#'''

print('Data for Test Done')


data_test_meta_score[['ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION','VAES_PRECISION','SPEC_PRECISION']].\
apply(pd.to_numeric, errors='coerce').fillna(0)

with open(d+'/test_dict_2', 'w') as fil:
    print(test_dict_2, file=fil)

purch_items = list()
purch_users = list()

us = list(test_dict_2.keys())

for u in us:
    user_ind = users_d[u]
    user_ind_v = 0

    purchased = get_purchased(user_ind, user_ind_v, test_dict[u])

    purch_items.extend(purchased)
    purch_users.extend( np.full(len(purchased),u) )

purch = pd.DataFrame()
purch['CUST_ID']= purch_users
purch['ARTICLE_ID']= purch_items

purch.to_csv(d+'/purch.csv',index=False)

del us

#Number of purchases of users and items in the test set

us = list(test_dict_2.keys())
n = np.full(len(us),1,dtype='uint8')

us = pd.DataFrame({'CUST_ID':us,'KEY':n})
us = pd.merge(us,nb_purch_test,on='CUST_ID')

n = np.full(len(items),1,dtype='uint8')

it = pd.DataFrame({'ARTICLE_ID':items,'KEY':n})
it = pd.merge(it,nb_article_purch_test,on='ARTICLE_ID')

us = pd.merge(us,it,on='KEY')
us.to_csv(d+'/test_protocol.csv',index=False)

del us


#Adding meta features
data_prec = pd.merge(data_prec,nb_purch_test, on='CUST_ID')
data_prec_score = pd.merge(data_prec_score,nb_purch_test, on='CUST_ID')

data_prec = pd.merge(data_prec,nb_article_purch_test, on='ARTICLE_ID')
data_prec_score = pd.merge(data_prec_score,nb_article_purch_test, on='ARTICLE_ID')

data_test_meta = pd.merge(data_test_meta,nb_purch_test, on='CUST_ID')
data_test_meta_score = pd.merge(data_test_meta_score,nb_purch_test, on='CUST_ID')

data_test_meta = pd.merge(data_test_meta,nb_article_purch_test, on='ARTICLE_ID')
data_test_meta_score = pd.merge(data_test_meta_score,nb_article_purch_test, on='ARTICLE_ID')

data_test_meta.to_csv(d+'/data_reco_baselines.csv', index=False)
data_test_meta_score.to_csv(d+'/data_reco_baselines_score.csv', index=False)

train_model = train_model.append(data_prec)
train_model.to_csv(d+'/train_model.csv', index=False)

#train_model = pd.read_csv(d+'/train_model.csv')


data_prec_score.to_csv(d+'/train_model_score.csv', index=False)
#data_prec_score = pd.read_csv(d+'/train_model_score.csv')

# Meta training
X_train = train_model[['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST']]

y_train_ARM = train_model[['ARM_PRECISION']]
y_train_SVD = train_model[['SVD_PURE_PRECISION']]
y_train_NMF = train_model[['NMF_PRECISION']]
y_train_KNN100 = train_model[['K100_PRECISION']]
y_train_VAES = train_model[['VAES_PRECISION']]
y_train_SPEC = train_model[['SPEC_PRECISION']]

print('Normalize')

x = X_train[['NB_PURCH_TEST']].values.astype(float)
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_train['NB_PURCH_TEST'] = x_scaled

x = X_train[['NB_ARTICLE_PURCH_TEST']].values.astype(float)
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_train['NB_ARTICLE_PURCH_TEST'] = x_scaled

print('Categorical Columns')
#Categorical columns
object_col = train_model.dtypes == 'object'
print(object_col)

object_col = list( object_col[object_col].index )
object_col.remove('CUST_ID')
object_col.remove('ARTICLE_ID')

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) #One Hot Encoder
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_col]))

with open(d+'/encoder', 'wb') as fil: 
    pickle.dump(OH_encoder, fil)

OH_cols_train.index = X_train.index #One Hot Remoes indexes, they are replaced
X_train = X_train.drop(object_col, axis=1) #Drop categorical columns
X_train = pd.concat([X_train, OH_cols_train], axis=1) #Add hot one encoding columns

param_dec = {'criterion':['mse', 'friedman_mse', 'mae'],\
'max_depth':[3,5,10,20],\
'splitter': ['best', 'random'],\
'max_features' : ['auto', 'sqrt', 'log2']}

param_xgb = {'n_estimators':[100, 500, 1000],\
'learning_rate': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'gamma': [0.5, 1, 2, 5], 'subsample': [0.5, 1.0],\
'reg_lambda':[0.001, 0.005, 0.01, 0.05],\
'max_depth':[3,5,10]}

param_sgd = {'penalty':['l2'],\
'alpha': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'max_iter':[100, 200, 500, 1000]}

X_train = sparse.csr_matrix(X_train.values)

y_train_ARM = y_train_ARM.to_numpy().ravel()
y_train_SVD = y_train_SVD.to_numpy().ravel()
y_train_NMF = y_train_NMF.to_numpy().ravel()
y_train_KNN100 = y_train_KNN100.to_numpy().ravel()
y_train_VAES = y_train_VAES.to_numpy().ravel()
y_train_SPEC = y_train_SPEC.to_numpy().ravel()


print('Linear Regression Model Now')
#----------------------------------------------------------#

model_ARM = LinearRegression(normalize=True)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM, open(d+'/models/lin_reg_arm', 'wb'))

model_SVD = LinearRegression(normalize=True)
model_SVD.fit(X_train,y_train_SVD)
pickle.dump(model_SVD, open(d+'/models/lin_reg_svd_pure', 'wb'))

model_NMF = LinearRegression(normalize=True)
model_NMF.fit(X_train,y_train_NMF)
pickle.dump(model_NMF, open(d+'/models/lin_reg_nmf', 'wb'))

model_KNN100 = LinearRegression(normalize=True)
model_KNN100.fit(X_train,y_train_KNN100)
pickle.dump(model_KNN100, open(d+'/models/lin_reg_knn100', 'wb'))

model_VAES = LinearRegression(normalize=True)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES, open(d+'/models/lin_reg_vaes', 'wb'))

model_SPEC = LinearRegression(normalize=True)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC, open(d+'/models/lin_reg_spec', 'wb'))

#----------------------------------------------------------#

print('Decision Tree Regressor Model Now')
model = DecisionTreeRegressor()
model_ARM = RandomizedSearchCV(model, param_dec,n_jobs=-1)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM, open(d+'/models/des_tree_arm', 'wb'))

model = DecisionTreeRegressor()
model_SVD = RandomizedSearchCV(model, param_dec,n_jobs=-1)
model_SVD.fit(X_train,y_train_SVD)
pickle.dump(model_SVD, open(d+'/models/des_tree_svd_pure', 'wb'))

model = DecisionTreeRegressor()
model_NMF = RandomizedSearchCV(model, param_dec,n_jobs=-1)
model_NMF.fit(X_train,y_train_NMF)
pickle.dump(model_NMF, open(d+'/models/des_tree_nmf', 'wb'))

model = DecisionTreeRegressor()
model_KNN100 = RandomizedSearchCV(model, param_dec,n_jobs=-1)
model_KNN100.fit(X_train,y_train_KNN100)
pickle.dump(model_KNN100, open(d+'/models/des_tree_knn100', 'wb'))

model = DecisionTreeRegressor()
model_VAES = RandomizedSearchCV(model, param_dec,n_jobs=-1)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES, open(d+'/models/des_tree_vaes', 'wb'))

model = DecisionTreeRegressor()
model_SPEC = RandomizedSearchCV(model, param_dec,n_jobs=-1)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC, open(d+'/models/des_tree_spec', 'wb'))

#----------------------------------------------------------#

print('SGD Classifier Model Now')

model = SGDRegressor()
model_ARM = RandomizedSearchCV(model, param_sgd,n_jobs=-1)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM, open(d+'/models/sgd_arm', 'wb'))

model = SGDRegressor()
model_SVD = RandomizedSearchCV(model, param_sgd,n_jobs=-1)
model_SVD.fit(X_train,y_train_SVD)
pickle.dump(model_SVD, open(d+'/models/sgd_svd_pure', 'wb'))

model = SGDRegressor()
model_NMF = RandomizedSearchCV(model, param_sgd,n_jobs=-1)
model_NMF.fit(X_train,y_train_NMF)
pickle.dump(model_NMF, open(d+'/models/sgd_nmf', 'wb'))

model = SGDRegressor()
model_KNN100 = RandomizedSearchCV(model, param_sgd,n_jobs=-1)
model_KNN100.fit(X_train,y_train_KNN100)
pickle.dump(model_KNN100, open(d+'/models/sgd_knn100', 'wb'))

model = SGDRegressor()
model_VAES = RandomizedSearchCV(model, param_sgd,n_jobs=-1)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES, open(d+'/models/sgd_vaes', 'wb'))

model = SGDRegressor()
model_SPEC = RandomizedSearchCV(model, param_sgd,n_jobs=-1)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC, open(d+'/models/sgd_spec', 'wb'))

#----------------------------------------------------------#

print('Decision XGBoost Model Now')

model = XGBRegressor()
model_ARM = RandomizedSearchCV(model, param_xgb,n_jobs=-1)
model_ARM.fit(X_train,y_train_ARM)
pickle.dump(model_ARM, open(d+'/models/xgb_arm', 'wb'))

model = XGBRegressor()
model_SVD = RandomizedSearchCV(model, param_xgb,n_jobs=-1)
model_SVD.fit(X_train,y_train_SVD)
pickle.dump(model_SVD, open(d+'/models/xgb_svd_pure', 'wb'))

model = XGBRegressor()
model_NMF = RandomizedSearchCV(model, param_xgb,n_jobs=-1)
model_NMF.fit(X_train,y_train_NMF)
pickle.dump(model_NMF, open(d+'/models/xgb_nmf', 'wb'))

model = XGBRegressor()
model_KNN100 = RandomizedSearchCV(model, param_xgb,n_jobs=-1)
model_KNN100.fit(X_train,y_train_KNN100)
pickle.dump(model_KNN100, open(d+'/models/xgb_knn100', 'wb'))

model = XGBRegressor()
model_VAES = RandomizedSearchCV(model, param_xgb,n_jobs=-1)
model_VAES.fit(X_train,y_train_VAES)
pickle.dump(model_VAES, open(d+'/models/xgb_vaes', 'wb'))

model = XGBRegressor()
model_SPEC = RandomizedSearchCV(model, param_xgb,n_jobs=-1)
model_SPEC.fit(X_train,y_train_SPEC)
pickle.dump(model_SPEC, open(d+'/models/xgb_spec', 'wb'))
#'''

# Stacking Model
X_train = data_prec_score[['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION','VAES_PRECISION','SPEC_PRECISION']]

y_train = data_prec_score[['SCORE']].to_numpy().ravel()

print('Normalize')

for c in ['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION','VAES_PRECISION','SPEC_PRECISION']:
    x = X_train[[c]].values.astype(float)
    min_max_scaler = pp.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_train[c] = x_scaled

param_dec = {'criterion':['mse', 'friedman_mse', 'mae'],\
'max_depth':[3,5,10,20],\
'splitter': ['best', 'random'],\
'max_features' : ['auto', 'sqrt', 'log2']}

param_xgb = {'n_estimators':[100, 500, 1000],\
'learning_rate': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'gamma': [0.5, 1, 2, 5], 'subsample': [0.5, 1.0],\
'reg_lambda':[0.001, 0.005, 0.01, 0.05],\
'max_depth':[3,5,10]}

param_sgd = {'penalty':['l2'],\
'alpha': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06],\
'max_iter':[100, 200, 500, 1000]}

X_train = sparse.csr_matrix(X_train.values)

print('Linear Regression Model Now')

model = LinearRegression(normalize=True)
model.fit(X_train,y_train)
pickle.dump(model, open(d+'/models/lin_reg_stacking', 'wb'))

print('Decision Tree Regressor Model Now')

model = DecisionTreeRegressor()
model_STACKING = RandomizedSearchCV(model, param_dec,n_jobs=-1)
model_STACKING.fit(X_train,y_train)
pickle.dump(model_STACKING, open(d+'/models/des_tree_stacking', 'wb'))

print('SGD Classifier Model Now')

model = SGDRegressor()
model_STACKING = RandomizedSearchCV(model, param_sgd,n_jobs=-1)
model_STACKING.fit(X_train,y_train)
pickle.dump(model_STACKING, open(d+'/models/sgd_stacking', 'wb'))

print('Decision XGBoost Model Now')

model = XGBRegressor()
model_STACKING = RandomizedSearchCV(model, param_xgb,n_jobs=-1)
model_STACKING.fit(X_train,y_train)
pickle.dump(model_STACKING, open(d+'/models/xgb_stacking', 'wb'))