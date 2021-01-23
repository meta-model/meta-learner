import pandas as pd
import numpy as np
import scipy.sparse as sparse
import multiprocessing
import os
import sklearn.preprocessing as pp
import pickle
import json
import time
import itertools

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel
import implicit
import tensorflow as tf
import VAES

class RecommendationAlgos (object):
    def __init__(self, ct=None, cont=None, data=None, action='train', num=10):
        self.num = num
        self.ct = ct
        self.cont = cont
        
        if action == 'train':
            if data is None:
                self.data = pd.read_csv('data_set\\full_dataset2.csv')
                self.data = self.data[self.data.groupby('CUST_ID')['CUST_ID'].transform('count').ge(self.num)] #Users having more than 10 purshases
            else :
                self.data = data


            self.data.to_csv('data_set/train_set_als.csv', sep= ',', index= False) #For ALS
            self.items = list(np.sort(self.data.ARTICLE_ID.unique()))  # all unique items
            self.users = list(np.sort(self.data.CUST_ID.unique()))    # all unique users
            self.rating = list(np.sort(self.data.FREQUENCY))
        else:
            if action != 'load':
                assert action != 'load'
    
    def get_association_matrix(self, amazon=False):
        rows = self.data.CUST_ID.astype(pd.api.types.CategoricalDtype(categories = self.users)).cat.codes    # Get the associated row indices
        cols = self.data.ARTICLE_ID.astype(pd.api.types.CategoricalDtype(categories = self.items)).cat.codes    # Get the associated row indices

        index_list = list(cols.values)
        items = list(self.data.ARTICLE_ID)

        self.items_d = dict()
        self.items_dd = dict()
        self.users_d = dict()

        j=0
        for i in items:
            if i not in self.items_d:
                self.items_d[i] = index_list[j]
                self.items_dd[ index_list[j] ] = i
            j=j+1

        print('ITEMS IN CF : ',len(self.items_d.keys()))

        index_list = list(rows.values)
        items = list(self.data.CUST_ID)

        j=0
        for i in items:
            if i not in self.users_d:
                self.users_d[i] = index_list[j]
            j=j+1

        print('USERS IN CF : ',len(self.users_d.keys()))

        self.data = sparse.csc_matrix((self.rating, (rows, cols)), shape = (len(self.users), len(self.items)  )  )

        if amazon==False:
            #convert into a purchase matrix
            self.data[self.data >= 1] =1
            
            self.Association_matrix = self.data.T.dot(self.data) # P.j(t) * P.i
            c = sparse.diags(1/self.data.T.sum(axis=1).A.ravel()) # ||P.j||
            self.Association_matrix= c.dot(self.Association_matrix)
        else:
            self.data[self.data >= 1] =1

    def __set_context_als(self):
        conf = SparkConf().setAppName("als")
        sc = SparkContext.getOrCreate(conf = conf)
        sc.setCheckpointDir('checkpoint/')
        return sc

    def res_vae(self,nm):
        p_dims = [200, 600, len(self.items)]
        tf.compat.v1.reset_default_graph()
        vae = VAES.MultiVAE(p_dims, lam=0.0)

        saver, logits_var, _, _ = vae.build_graph()

        with tf.compat.v1.Session() as sess:
            if nm==1:
                saver.restore(sess, 'vaes_1/model.ckpt')
            elif nm==2:
                saver.restore(sess, 'vaes_2/model.ckpt')
            else:
                saver.restore(sess, 'vaes_3/model.ckpt')

            X = self.data

            if sparse.isspmatrix(X):
                X = X.toarray()
                X = X.astype('float32')
            
            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
            self.res = pred_val

    def fit_vae(self, batch_size, n_epochs, anneal_cap,nm):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.anneal_cap = anneal_cap # largest annealing parameter

        N = self.data.shape[0] #Number of custs
        idxlist = list(range(N))

        batches_per_epoch = int(np.ceil(float(N) / batch_size))

        # the total number of gradient updates for annealing
        total_anneal_steps = 200000
    
        p_dims = [200, 600, len(self.items)]

        tf.compat.v1.reset_default_graph()
        vae = VAES.MultiVAE(p_dims, lam=0.0, random_seed=568978)

        saver, logits_var, loss_var, train_op_var = vae.build_graph()

        with tf.compat.v1.Session() as sess:

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            update_count = 0.0

            for epoch in range(self.n_epochs):
                print('===========================\n\tepoch : ',epoch)
                np.random.shuffle(idxlist)
                # train for one epoch
                for bnum, st_idx in enumerate(range(0, N, self.batch_size)): #For all Batches
                    end_idx = min(st_idx + self.batch_size, N) #End of the batch
                    X = self.data[idxlist[st_idx:end_idx]]
                    
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')       
                    
                    if total_anneal_steps > 0:
                        anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                    else:
                        anneal = anneal_cap
                    
                    feed_dict = {vae.input_ph: X, 
                                vae.keep_prob_ph: 0.5, 
                                vae.anneal_ph: anneal,
                                vae.is_training_ph: 1}        
                    sess.run(train_op_var, feed_dict=feed_dict)
                    
                    update_count += 1
                
            if nm==1:
                saver.save(sess, 'vaes_1/model.ckpt')
            elif nm==2:
                saver.save(sess, 'vaes_2/model.ckpt')
            else:
                saver.save(sess, 'vaes_3/model.ckpt')
            
    def fit_als(self):
        sc = self.__set_context_als()
        
        data = sc.textFile('data_set/train_set_als.csv')
        header = data.first()
        data = data.filter(lambda row: row!=header)

        trans = open('data_set/train_set_als.csv', 'r')
        lines = trans.readlines()

        users_dict= dict() # {CUST_ID : Num}
        items_dict = dict()
        i=0
        for line in lines[1:]:
            parts = line.split(',')
            user= int(parts[0])
            if user not in users_dict:
                users_dict[user]=i
                i+=1
        j=0
        for line in lines[1:]:
            parts = line.split(',')
            item= int(parts[1])
            if item not in items_dict:
                items_dict[item]=j
                j+=1

        self.users_als = {v: k for k, v in users_dict.items()} #{Num : CUST_ID}
        self.items_als = {v: k for k, v in items_dict.items()}

        self.users_dict = users_dict
        self.items_dict = items_dict

        ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(users_dict[int(l[0])], items_dict[int(l[1])], float(l[2])))
        self.model_ALS = ALS.trainImplicit(ratings= ratings, rank= 40, iterations=30, lambda_=0.001, blocks=-1, alpha=10.0)

    def fit_bpr(self):
        self.model_BPR = implicit.bpr.BayesianPersonalizedRanking (20,learning_rate=0.001,regularization=0.05,verify_negative_samples=True, num_threads=0)
        self.model_BPR.fit(self.data.transpose().tocoo(), show_progress=True)

    def cosine_similarity(self):
        self.Similarity_matrix = pp.normalize(self.data.tocsc(), axis=0)
        self.Similarity_matrix = self.Similarity_matrix.T * self.Similarity_matrix

    def __item_similarities(self, item_id):
        item_ind = self.items_d[item_id]
        item_sim_vector = self.Similarity_matrix[item_ind,:]
        return item_sim_vector

    def __profile_similarities(self, purchased):
        liste_sim = []
        for p in purchased:
            liste_sim.append(self.__item_similarities(p))
        
        reco_vector = liste_sim[0]
        for l in liste_sim[1:]:
            reco_vector= reco_vector + l
        return reco_vector/len(liste_sim)

    def top_N_ARM(self, user, n, val_values):
        if user in self.users_d:
            user_ind = self.users_d[user]
            #user_ind_v = self.cores_custs_v[user]
            user_ind_v = 0
            
            user_pref = self.data[user_ind,:].toarray()*self.Association_matrix
            user_pref = user_pref.reshape(-1)
            #user_pref = (self.rating_v[user_ind_v,:].dot(self.Association_matrix)).toarray().reshape(-1)
            items_purchased = self.__get_purchased(user_ind, user_ind_v, val_values)
            ensemble_produits = set(items_purchased)


            items_ind = np.argsort(user_pref)[::-1]      # Sort the indices of the items into order of best recommendations
            rec_list= []
            score_list = []
            i=0
            for ind in items_ind:
                if i >= n:
                    break
                else:
                    item_id = self.items[ind]
                    if item_id in ensemble_produits:
                        continue
                    else:
                        #if item_id.split('_')[1] == typ: #TO REMOVE AFTER
                        rec_list.append(item_id)
                        score_list.append(user_pref[ind])
                        i=i+1
            return rec_list,score_list
        else:
            return None
    
    def top_N_CF(self, user, n):
        if user in self.users_d:
            user_ind = self.users_d[user]

            #user_ind_v = self.cores_custs_v[user]
            user_ind_v = 0

            purchased = self.__get_purchased(user_ind, user_ind_v)
            ensemble_produits = set(purchased)

            pref = self.__profile_similarities(purchased).toarray().reshape(-1)

            items_ind = np.argsort(pref)[::-1]
            rec_list= []
            i=0
            for ind in items_ind:
                if i >= n:
                    break
                else:
                    item_id = self.items[ind]
                    if item_id in ensemble_produits:
                        continue
                    else:
                        rec_list.append(item_id)
                        i=i+1

            return rec_list
        else:
            return None

    def top_N_BPR(self, user, n, val_values):
        if user in self.users_d:
            user_ind = self.users_d[user]

            #user_ind_v = self.cores_custs_v[user]
            user_ind_v = 0

            liste = self.model_BPR.recommend(user_ind , self.data.tocsr(), len(self.items) , filter_already_liked_items=True)

            purchased = self.__get_purchased(user_ind, user_ind_v, val_values)
            ensemble_produits = set(purchased)

            l = [ self.items[i] for i in [l[0] for l in liste] ]
            rec = [i for i in l if i not in ensemble_produits ]

            score = [l[1] for l in liste if self.items[l[0]] in rec]

            rec = rec[:n]
            score = score[:n]
            return rec,score
        else:
            return None

    def top_N_ALS(self, user, n, val_values): 
        if user in self.users_d:
            #user_ind_v = self.cores_custs_v[user]
            user_ind_v = 0
            user = self.users_dict[user] # get row of ALS from

            predictions = self.model_ALS.recommendProducts(user,len(self.items_dict))

            user_ind = self.users_d[ np.int64(self.users_als[user]) ]

            purchased = self.__get_purchased(user_ind, user_ind_v, val_values)
            purchased_set = set(purchased)
            liste_reco = []
            score_list = []
            i=0
            for pre in predictions:
                if i>=n:
                    break
                else:
                    produit = int(self.items_als[pre.product])
                    if produit in purchased_set:
                        continue
                    else:
                        #if item_id.split('_')[1] == typ: #TO REMOVE AFTER
                        liste_reco.append(produit)
                        score_list.append(pre.rating)
                        i=i+1
            return liste_reco, score_list
        else:
            return None

    def top_N_VAES(self, user, n, val_values):
        user_ind_train = self.users_d[user]
        #user_ind_v = self.cores_custs_v[user]
        user_ind_v = 0

        #sort = np.argsort(-self.res[user_ind_v,:])
        score = self.res[user_ind_train,:]
        sort = np.argsort(-score)

        purch = self.__get_purchased(user_ind_train, user_ind_v, val_values)

        top = []
        scr = []
        num = 0
        for i in sort:
            if num >= n:
                break
            else:
                item_id = self.items_dd[i]
                if item_id in purch:
                    continue
                else:
                    top.append(item_id)
                    scr.append(score[i])
                    num = num+1
        return top,scr