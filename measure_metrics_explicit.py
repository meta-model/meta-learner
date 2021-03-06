import pandas as pd
import numpy as np
import random
import time
import math
import pickle
import sklearn
import multiprocessing
import scipy.sparse as sparse
import sklearn.preprocessing as pp
import sys
import os

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR 
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

from tffm import TFFMRegressor
import tensorflow as tf

import pywFM

reg_algos = ['lin_reg','des_tree','sgd','xgb']
col = ['ARM','SVD_PURE','NMF','KNN100','VAES','SPEC']+reg_algos+['STACKING_lin_reg','STACKING_dec_tree','STACKING_sgd','STACKING_xgb']+['FM_PRECISION','FM_PRECISION_tf']

number = int(input('TOP N - Enter Value Of N : '))

d = str(input('Amazon Dataset : \nmo : Movies\nmu : Music\n'))

if d == 'mo':
    d = 'Movies_split'
elif d == 'mu':
    d = 'Music_split'

purch = pd.read_csv(d+'/purch.csv')

with open(d+'/test_dict_2') as f:
    users = f.read()
    users = eval(users)

data_test_meta = pd.read_csv( d+'/data_test_meta.csv' )

test = dict()

# Verify that None of items in test set exists in Training and meta training
for i,(key,value) in enumerate(users.items()):
    prch = set(purch[purch['CUST_ID']==key].ARTICLE_ID)
    clas = []
    for j in value:
        if j not in prch:
            m2 = data_test_meta[(data_test_meta['CUST_ID']==key)&(data_test_meta['ARTICLE_ID']==j)]
            if int(m2.RATING) > 3: #Items with ratings inferior to 3 are considered as irrelevant
                clas.append(j)
    test[key]=clas

users = test

# Factorization Machines
item_FM = pd.read_csv(d+'/item_FM.csv')
user_FM = pd.read_csv(d+'/user_FM.csv')
data_train_FM = pd.read_csv(d+'/data_train_FM.csv')

data_train_FM = pd.merge(data_train_FM,user_FM,on='CUST_ID')
data_train_FM = pd.merge(data_train_FM,item_FM,on='ARTICLE_ID')

y = data_train_FM[['FREQUENCY']].to_numpy()
y = np.reshape( y, (y.shape[0],) )

X = sparse.csr_matrix(data_train_FM.drop(columns=['FREQUENCY','CUST_ID','ARTICLE_ID']).to_numpy())

del data_train_FM

rank = 20
l_r = 0.05
reg = 0.001
epoch =  200

model_tf = TFFMRegressor(
    order=2,
    rank=rank,
    optimizer=tf.train.AdamOptimizer(learning_rate=l_r),
    reg=reg,
    n_epochs=epoch,
    init_std=0.001,
    input_type='sparse'
)


data_reco_baselines_score = pd.read_csv(d+'/data_reco_baselines_score.csv')

data_reco_baselines_score = data_reco_baselines_score[['NB_PURCH_TEST', 'NB_ARTICLE_PURCH_TEST' ,'ARM_PRECISION', 'SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION',\
 'VAES_PRECISION','SPEC_PRECISION','CUST_ID','ARTICLE_ID']]

protocol = pd.read_csv(d+'/test_protocol.csv')
protocol = protocol.drop_duplicates()

data_reco_baselines = pd.read_csv(d+'/data_reco_baselines.csv')

data_reco_baselines = data_reco_baselines.drop_duplicates()\
[['ARM_PRECISION', 'SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION', 'VAES_PRECISION','SPEC_PRECISION', 'CUST_ID','ARTICLE_ID']]

protocol = pd.merge(protocol,data_reco_baselines,on=['CUST_ID','ARTICLE_ID'])


model_tf.fit(X, y, show_progress=False)
data_test_FM = protocol[['CUST_ID','ARTICLE_ID','NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST']]


for enu, us in enumerate(data_test_FM.CUST_ID.unique()):
    local = data_test_FM[data_test_FM.CUST_ID==us]
    local = pd.merge(local,user_FM,on='CUST_ID')
    local = pd.merge(local,item_FM,on='ARTICLE_ID')

    X_test = sparse.csr_matrix(local.drop(columns=['CUST_ID','ARTICLE_ID']).to_numpy())

    if len(local) > 0 :
        local['FM_PRECISION_tf'] =  model_tf.predict( X_test )
        local = local[['ARTICLE_ID','FM_PRECISION_tf']]
        local.to_csv( d+'/FM/local_test_{}.csv'.format(us),index=False )

def work(enu,us):
    local = data_test_FM[data_test_FM.CUST_ID==us]
    local = pd.merge(local,user_FM,on='CUST_ID')
    local = pd.merge(local,item_FM,on='ARTICLE_ID')
    if len(local) > 0 :
        X_test = sparse.csr_matrix(local.drop(columns=['CUST_ID','ARTICLE_ID']).to_numpy())

        os.environ['LIBFM_PATH']='/home/slide/bouaroun/libfm/bin/'
        fm = pywFM.FM(task='regression', num_iter=150, learning_method='als', learn_rate=0.05, r2_regularization=0.001) 
        model = fm.run(X, y, X_test , np.array([random.randint(1,5) for i in range(len(local))]) )

        local = pd.read_csv( d+'/FM/local_test_{}.csv'.format(us))
        local['FM_PRECISION'] =  model.predictions
        local = local[['ARTICLE_ID','FM_PRECISION','FM_PRECISION_tf']]
        local.to_csv( d+'/FM/local_test_{}.csv'.format(us),index=False )
    print(enu)


manager = multiprocessing.Manager()
pool = multiprocessing.Pool(multiprocessing.cpu_count())

print('Num of Process : ',pool._processes)

for enu, us in enumerate(data_test_FM.CUST_ID.unique()):
    result = pool.apply_async(work,args=(enu,us))
    #result.get()
pool.close()
pool.join()


#Predictions of each stacking model
us_it = protocol[['CUST_ID','ARTICLE_ID']]
data_reco_baselines_score = pd.merge(data_reco_baselines_score, us_it, on=['CUST_ID','ARTICLE_ID'])

for c in ['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION','VAES_PRECISION','SPEC_PRECISION']:
    x = data_reco_baselines_score[[c]].values.astype(float)
    min_max_scaler = pp.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_reco_baselines_score[c] = x_scaled

X_test = data_reco_baselines_score[['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION',\
'VAES_PRECISION','SPEC_PRECISION']]

X_test = sparse.csr_matrix(X_test.values)

stacking_models = pd.DataFrame()
stacking_models['CUST_ID'] = data_reco_baselines_score['CUST_ID']
stacking_models['ARTICLE_ID'] = data_reco_baselines_score['ARTICLE_ID']

model_log_reg = pickle.load(open(d+'/models/lin_reg_stacking', 'rb'))
model_dec_tree = pickle.load(open(d+'/models/des_tree_stacking', 'rb'))
model_sgd = pickle.load(open(d+'/models/sgd_stacking', 'rb'))
model_xgb = pickle.load(open(d+'/models/xgb_stacking', 'rb'))

stacking_models['STACKING_lin_reg'] = model_log_reg.predict(X_test)
stacking_models['STACKING_dec_tree'] = model_dec_tree.predict(X_test)
stacking_models['STACKING_sgd'] = model_sgd.predict(X_test)
stacking_models['STACKING_xgb'] = model_xgb.predict(X_test)

#END of predictions of stacking models

X_test = protocol[['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST']]

print('Normalize')
x = X_test[['NB_PURCH_TEST']].values.astype(float)
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_test['NB_PURCH_TEST'] = x_scaled

x = X_test[['NB_ARTICLE_PURCH_TEST']].values.astype(float)
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_test['NB_ARTICLE_PURCH_TEST'] = x_scaled

object_col = X_test.dtypes == 'object'
object_col = list( object_col[object_col].index )

OH_encoder = pickle.load(open(d+'/encoder', 'rb'))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_col])) 
OH_cols_test.index = X_test.index
X_test = X_test.drop(object_col, axis=1)
X_test = pd.concat([X_test, OH_cols_test], axis=1)

X_test = sparse.csr_matrix(X_test.values)

all_results = None
best_regression = None

print('Data done')

#Predictions of each meta-learner (Regression models)
for reg_algo in reg_algos:

    results = protocol[['CUST_ID','ARTICLE_ID','ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION',\
    'K100_PRECISION','VAES_PRECISION','SPEC_PRECISION']]

    model_ARM = pickle.load(open(d+'/models/'+reg_algo+'_arm', 'rb'))
    model_SVD = pickle.load(open(d+'/models/'+reg_algo+'_svd_pure', 'rb'))
    model_NMF = pickle.load(open(d+'/models/'+reg_algo+'_nmf', 'rb'))
    model_KNN100 = pickle.load(open(d+'/models/'+reg_algo+'_knn100', 'rb'))
    model_VAES = pickle.load(open(d+'/models/'+reg_algo+'_vaes', 'rb'))
    model_SPEC = pickle.load(open(d+'/models/'+reg_algo+'_spec', 'rb'))

    results['ARM_PRECISION_'+reg_algo] = model_ARM.predict(X_test)
    results['SVD_PURE_PRECISION_'+reg_algo] = model_SVD.predict(X_test)
    results['NMF_PRECISION_'+reg_algo] = model_NMF.predict(X_test)
    results['K100_PRECISION_'+reg_algo] = model_KNN100.predict(X_test)
    results['VAES_PRECISION_'+reg_algo] = model_VAES.predict(X_test)
    results['SPEC_PRECISION_'+reg_algo] = model_SPEC.predict(X_test)

    if reg_algo == 'lin_reg':
        best_regression = results
    else:
        best_regression = pd.merge(best_regression,\
        results[['CUST_ID','ARTICLE_ID','ARM_PRECISION_'+reg_algo,'SVD_PURE_PRECISION_'+\
        reg_algo,'NMF_PRECISION_'+reg_algo,'K100_PRECISION_'+reg_algo,'VAES_PRECISION_'+reg_algo,\
        'SPEC_PRECISION_'+reg_algo]],\
        on=['CUST_ID','ARTICLE_ID'])

    predicted_algos = results[['ARM_PRECISION_'+reg_algo,'SVD_PURE_PRECISION_'+\
    reg_algo,'NMF_PRECISION_'+reg_algo,'K100_PRECISION_'+reg_algo,'VAES_PRECISION_'+reg_algo,\
    'SPEC_PRECISION_'+reg_algo]].min(axis=1)

    results = results[['CUST_ID','ARTICLE_ID','ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION',\
    'K100_PRECISION','VAES_PRECISION','SPEC_PRECISION']]

    results[reg_algo] = predicted_algos

    if reg_algo == 'lin_reg':
        results[reg_algo] = predicted_algos
        all_results = results
    else:
        all_results[reg_algo] = predicted_algos

all_results = pd.merge(all_results,stacking_models,on=['CUST_ID','ARTICLE_ID'])
all_results.to_csv(d+'/all_test_results.csv')

#all_results = pd.read_csv(d+'/all_test_results.csv')

mth = ['ARM_PRECISION','SVD_PURE_PRECISION','NMF_PRECISION','K100_PRECISION','VAES_PRECISION',\
'SPEC_PRECISION']+reg_algos+['STACKING_lin_reg','STACKING_dec_tree','STACKING_sgd','STACKING_xgb']+['FM_PRECISION','FM_PRECISION_tf']

# Metrics
def worker(i,key, value, classifier, classifier2, classifier3, classifier4):
    if len(value) > 0 :
        if os.path.isfile(d+'/FM/local_test_{}.csv'.format(key)):
            value = set(value)
            m1 = all_results[all_results['CUST_ID']==key]
            rank = list(range(1,number+1))
            precision = []
            recall = []
            ap = []
            dcg = []
            prch = set(purch[purch['CUST_ID']==key].ARTICLE_ID)
            m1 = m1[~m1['ARTICLE_ID'].isin(prch)]
            for methode in mth:
                if methode in ['STACKING_lin_reg','STACKING_dec_tree','STACKING_sgd','STACKING_xgb']:
                    m_local = m1.sort_values(by=methode, ascending=False)
                elif methode in ['FM_PRECISION','FM_PRECISION_tf']:
                    m_local = pd.read_csv(d+'/FM/local_test_{}.csv'.format(key))
                    m_local = m_local[~m_local['ARTICLE_ID'].isin(prch)]
                    m_local = m_local.sort_values(by=methode, ascending=False )
                else:
                    m_local = m1.sort_values(by=methode, ascending=True)
                    
                m_local = m_local.head(number)
                lst = m_local.ARTICLE_ID

                m_local['RELEV'] = m_local['ARTICLE_ID'].map(lambda x: x in value)
                m_local['RANK'] = rank
                m_local['SOM'] = m_local['RELEV'].cumsum()
                m_local['MAP'] = (m_local['RELEV']*m_local['SOM'])/m_local['RANK']
                m_local['DCG'] = m_local['RANK'].map(lambda x : math.log2(x+1))
                m_local['DCG'] = m_local['RELEV']/m_local['DCG']

                precision.append(len(value.intersection(set(lst)))/number)
                recall.append(len(value.intersection(set(lst)))/len(value))
                ap.append(m_local['MAP'].sum()/number)
                dcg.append(m_local['DCG'].sum()/number)
            
            classifier.append( precision )
            classifier2.append( recall )
            classifier3.append( ap )
            classifier4.append( dcg )

            print(i)

results=[]
manager = multiprocessing.Manager()
precisions = manager.list()
recalls = manager.list()
aps = manager.list()
dcgs = manager.list()
pool = multiprocessing.Pool(multiprocessing.cpu_count())

print('Num of Process : ',pool._processes)

for i,(key,value) in enumerate(users.items()):
    result = pool.apply_async(worker,args=(i,key,value,precisions,recalls,aps,dcgs))
    #result.get()
pool.close()
pool.join()

precisions = pd.DataFrame(np.array(precisions), columns=col)
recalls = pd.DataFrame(np.array(recalls), columns=col)
aps = pd.DataFrame(np.array(aps), columns=col)
dcgs = pd.DataFrame(np.array(dcgs), columns=col)


print('PRECISIONS : ')
print(precisions.mean()*100)

print('RECALLS : ')
print(recalls.mean()*100)

print('MAPS : ')
print(aps.mean()*100)

print('DCGS : ')
print(dcgs.mean()*100)
