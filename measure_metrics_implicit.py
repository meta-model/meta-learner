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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder

from tffm import TFFMClassifier,TFFMRegressor
import tensorflow as tf

import os
import pywFM


reg_algos = ['log_reg','dec_tree','sgd','xgb']

col = ['ARM','K50','ALS','BPR','VAES','SPEC']+reg_algos+['STACKING_log_reg','STACKING_dec_tree','STACKING_sgd','STACKING_xgb']+['FM_PRECISION','FM_PRECISION_tf']

number = int(input('TOP N - Enter Value Of N : '))

d = 'Tafeng'

purch = pd.read_csv(d+'/purch.csv')

with open(d+'/test_dict_2') as f:
    users = f.read()
    users = eval(users)

test = dict()

# Verify that None of items in test set exists in Training and meta training

for i,(key,value) in enumerate(users.items()):
    prch = set(purch[purch['CUST_ID']==key].ARTICLE_ID)
    clas = []
    for j in value:
        if j not in prch:
            clas.append(j)
    test[key]=clas

users = test


# Factorization Machines
item_FM = pd.read_csv(d+'/item_FM.csv')
user_FM = pd.read_csv(d+'/user_FM.csv')
data_train_FM = pd.read_csv(d+'/data_train_FM.csv')

items = list(data_train_FM.ARTICLE_ID.unique())
data_train_FM.FREQUENCY = data_train_FM.FREQUENCY.apply(lambda x: 1)

nb_purch = pd.read_csv(d+'/nb_purch_test.csv')
nb_article_purch = pd.read_csv(d+'/nb_article_purch_test.csv')

for enu, us in enumerate(data_train_FM.CUST_ID.unique()):
    items_local = list(data_train_FM[data_train_FM.CUST_ID==us].ARTICLE_ID.unique())
    items_local = list(set(items)-set(items_local))
    local = pd.DataFrame()
    local['CUST_ID']= np.full(5,us)
    local['ARTICLE_ID'] = random.sample(items_local, 5)
    local['FREQUENCY']= np.full(5,0)
    local = pd.merge(local,nb_purch,on='CUST_ID')
    local = pd.merge(local,nb_article_purch,on='ARTICLE_ID')
    local['AGE']= np.full(5,'')
    data_train_FM = pd.concat([data_train_FM,local],axis=0)

data_train_FM = pd.merge(data_train_FM,user_FM,on='CUST_ID')
data_train_FM = pd.merge(data_train_FM,item_FM,on='ARTICLE_ID')

y = data_train_FM[['FREQUENCY']].to_numpy()
y = np.reshape( y, (y.shape[0],) )

X = data_train_FM.drop(columns=['FREQUENCY','CUST_ID','ARTICLE_ID','AGE']).to_numpy()
X = X.astype(np.float32)

del data_train_FM

rank = 20
l_r = 0.05
reg = 0.001
epoch =  200

model_tf = TFFMClassifier(
    order=2,
    rank=rank,
    optimizer=tf.train.AdamOptimizer(learning_rate=l_r),
    reg=reg,
    n_epochs=epoch,
    init_std=0.0001
)

protocol = pd.read_csv(d+'/test_protocol.csv')
protocol = protocol.drop_duplicates()

data_train = pd.read_csv(d+'/train_model.csv')[['CUST_ID','AGE']].drop_duplicates()

protocol = pd.merge(protocol,data_train,on='CUST_ID')

data_reco_baselines = pd.read_csv(d+'/data_reco_baselines.csv').drop_duplicates()\
[['ARM_PRECISION', 'K50_PRECISION','ALS_PRECISION','BPR_PRECISION', 'VAES_PRECISION',\
 'SPEC_PRECISION', 'CUST_ID','ARTICLE_ID']]


protocol = pd.merge(protocol,data_reco_baselines,on=['CUST_ID','ARTICLE_ID'])

user_FM = user_FM[user_FM.CUST_ID.isin( protocol.CUST_ID.unique() )]
item_FM = item_FM[item_FM.ARTICLE_ID.isin( protocol.ARTICLE_ID.unique() )]

data_test_FM = protocol[['CUST_ID','ARTICLE_ID','NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST']]

model_tf.fit(X, y, show_progress=True)

for enu, us in enumerate(data_test_FM.CUST_ID.unique()):
    local = data_test_FM[data_test_FM.CUST_ID==us]
    local = pd.merge(local,user_FM,on='CUST_ID')
    local = pd.merge(local,item_FM,on='ARTICLE_ID')
    if len(local) > 0 :
        X_test = local.drop(columns=['CUST_ID','ARTICLE_ID']).to_numpy()

        local['FM_PRECISION_tf'] =  model_tf.predict( X_test )

        local = local[['ARTICLE_ID','FM_PRECISION_tf']]
        local.to_csv( d+'/FM/local_test_{}.csv'.format(us),index=False )
    print(enu)

def work(enu,us):
    local = data_test_FM[data_test_FM.CUST_ID==us]
    local = pd.merge(local,user_FM,on='CUST_ID')
    local = pd.merge(local,item_FM,on='ARTICLE_ID')
    if len(local) > 0 :
        os.environ['LIBFM_PATH']='/home/slide/bouaroun/libfm/bin/'
        X_test = sparse.csr_matrix(local.drop(columns=['CUST_ID','ARTICLE_ID']).to_numpy())

        fm = pywFM.FM(task='classification', num_iter=150, learning_method='als', learn_rate=0.05, r2_regularization=0.001) 
        model = fm.run(X, y, X_test , np.array([random.randint(0,1) for i in range(len(local))]) )

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

data_reco_baselines_score = pd.read_csv(d+'/data_reco_baselines_score.csv')

data_reco_baselines_score = data_reco_baselines_score[['NB_PURCH_TEST', 'NB_ARTICLE_PURCH_TEST' ,'ARM_PRECISION', 'K50_PRECISION','ALS_PRECISION','BPR_PRECISION',\
'VAES_PRECISION','SPEC_PRECISION','CUST_ID','ARTICLE_ID']]

us_it = protocol[['CUST_ID','ARTICLE_ID']]
data_reco_baselines_score = pd.merge(data_reco_baselines_score, us_it, on=['CUST_ID','ARTICLE_ID'])

for c in ['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','K50_PRECISION','ALS_PRECISION','BPR_PRECISION','VAES_PRECISION','SPEC_PRECISION']:
    x = data_reco_baselines_score[[c]].values.astype(float)
    min_max_scaler = pp.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_reco_baselines_score[c] = x_scaled

X_test = data_reco_baselines_score[['NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST','ARM_PRECISION','K50_PRECISION','ALS_PRECISION','BPR_PRECISION','VAES_PRECISION','SPEC_PRECISION']]

stacking_models = pd.DataFrame()
stacking_models['CUST_ID'] = data_reco_baselines_score['CUST_ID']
stacking_models['ARTICLE_ID'] = data_reco_baselines_score['ARTICLE_ID']

model_log_reg = pickle.load(open(d+'/models/STACKING_log_reg', 'rb'))
model_dec_tree = pickle.load(open(d+'/models/STACKING_dec_tree', 'rb'))
model_sgd = pickle.load(open(d+'/models/STACKING_sgd', 'rb'))
model_xgb = pickle.load(open(d+'/models/STACKING_xgb', 'rb'))

stacking_models['STACKING_log_reg'] = model_log_reg.predict(X_test)
stacking_models['STACKING_dec_tree'] = model_dec_tree.predict(X_test)
stacking_models['STACKING_sgd'] = model_sgd.predict(X_test)
stacking_models['STACKING_xgb'] = model_xgb.predict(X_test)

X_test = protocol[['AGE','NB_PURCH_TEST','NB_ARTICLE_PURCH_TEST']]

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
best_classification = None

#Predictions of each meta-learner (Classification models)

for reg_algo in reg_algos:

    results = protocol[['CUST_ID','ARTICLE_ID','ARM_PRECISION','K50_PRECISION','ALS_PRECISION',\
    'BPR_PRECISION','VAES_PRECISION','SPEC_PRECISION']]

    model_ARM = pickle.load(open(d+'/models/ARM_'+reg_algo, 'rb'))
    model_KNN50 = pickle.load(open(d+'/models/KNN_'+reg_algo, 'rb'))
    model_ALS = pickle.load(open(d+'/models/ALS_'+reg_algo, 'rb'))
    model_BPR = pickle.load(open(d+'/models/BPR_'+reg_algo, 'rb'))
    model_VAES = pickle.load(open(d+'/models/VAES_'+reg_algo, 'rb'))
    model_SPEC = pickle.load(open(d+'/models/SPEC_'+reg_algo, 'rb'))

    results['ARM_PRECISION_'+reg_algo] = model_ARM.predict(X_test)
    results['K50_PRECISION_'+reg_algo] = model_KNN50.predict(X_test)
    results['ALS_PRECISION_'+reg_algo] = model_ALS.predict(X_test)
    results['BPR_PRECISION_'+reg_algo] = model_BPR.predict(X_test)
    results['VAES_PRECISION_'+reg_algo] = model_VAES.predict(X_test)
    results['SPEC_PRECISION_'+reg_algo] = model_SPEC.predict(X_test)

    if reg_algo == 'log_reg':
        best_classification = results
    else:
        best_classification = pd.merge(best_classification,\
        results[['CUST_ID','ARTICLE_ID','ARM_PRECISION_'+reg_algo,'K50_PRECISION_'+\
        reg_algo,'ALS_PRECISION_'+reg_algo,'BPR_PRECISION_'+reg_algo,'VAES_PRECISION_'+reg_algo,\
        'SPEC_PRECISION_'+reg_algo]],\
        on=['CUST_ID','ARTICLE_ID'])
    
    predicted_algos = results['ARM_PRECISION_'+reg_algo] + results['K50_PRECISION_'+reg_algo] +\
    results['ALS_PRECISION_'+reg_algo] + results['BPR_PRECISION_'+reg_algo] +\
    results['VAES_PRECISION_'+reg_algo] + results['SPEC_PRECISION_'+reg_algo]

    results = results[['CUST_ID','ARTICLE_ID','ARM_PRECISION','K50_PRECISION','ALS_PRECISION',\
    'BPR_PRECISION','VAES_PRECISION','SPEC_PRECISION']]

    results[reg_algo] = predicted_algos

    if reg_algo == 'log_reg':
        results[reg_algo] = predicted_algos
        all_results = results
    else:
        all_results[reg_algo] = predicted_algos

all_results = pd.merge(all_results,stacking_models,on=['CUST_ID','ARTICLE_ID'])
all_results.to_csv(d+'/all_test_results.csv')

all_results= pd.read_csv(d+'/all_test_results.csv')

mth = ['ARM_PRECISION','K50_PRECISION','ALS_PRECISION','BPR_PRECISION','VAES_PRECISION',\
'SPEC_PRECISION']+reg_algos+['STACKING_log_reg','STACKING_dec_tree','STACKING_sgd','STACKING_xgb']+['FM_PRECISION','FM_PRECISION_tf']
# Metrics

def worker(i,key, value, classifier, classifier2, classifier3, classifier4):
    if len(value)>0:
        if os.path.isfile(d+'/FM/local_test_{}.csv'.format(key)):
            m1 = all_results[all_results['CUST_ID']==key]
            rank = list(range(1,number+1))
            precision = []
            recall = []
            ap = []
            dcg = []
            value = set(value)
            prch = set(purch[purch['CUST_ID']==key].ARTICLE_ID)
            m1 = m1[~m1['ARTICLE_ID'].isin(prch)]
            for methode in mth:
                if methode in reg_algos+['STACKING_log_reg','STACKING_dec_tree','STACKING_sgd','STACKING_xgb']:
                    m_local = m1.sort_values(by=methode,ascending=False)
                elif methode in ['FM_PRECISION','FM_PRECISION_tf']:
                    m_local = pd.read_csv(d+'/FM/local_test_{}.csv'.format(key))
                    m_local = m_local[~m_local['ARTICLE_ID'].isin(prch)]
                    m_local = m_local.sort_values(by=methode, ascending=False )
                else:
                    m_local = m1.sort_values(by=methode)
                    
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

precisions = pd.DataFrame(np.array(precisions),columns=col)
recalls = pd.DataFrame(np.array(recalls),columns=col)
aps = pd.DataFrame(np.array(aps),columns=col)
dcgs = pd.DataFrame(np.array(dcgs),columns=col)

print('PRECISION : ')
print(precisions.mean()*100)

print('RECALL : ')
print(recalls.mean()*100)

print('MAP : ')
print(aps.mean()*100)

print('DCG : ')
print(dcgs.mean()*100)
