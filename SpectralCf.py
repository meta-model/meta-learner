import tensorflow as tf
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import multiprocessing

def worker(i,X):
    if (i<99) & (i>0):
        c2 = X[i*2735:(i+1)*2735,:]
    elif i == 0:
        c2 = X[:2735,:]
    else :
        c2 = X[i*2735:,:]
    print('DOT - NUM : ',i)
    c2 = c2.dot(X.T) # P.j(t) * P.i
    print(i,' ',c2.shape)
    c2.data = np.round(c2.data,5)
    sparse.save_npz('DOT_'+str(i)+'.npz', c2)

def worker2(i,X,Y):
    if (i<99) & (i>0):
        c2 = X[i*2735:(i+1)*2735,:]
    elif i == 0:
        c2 = X[:2735,:]
    else :
        c2 = X[i*2735:,:]
    print('DOT - NUM : ',i)
    c2 = c2.dot(Y) # P.j(t) * P.i
    print(i,' ',c2.shape)
    c2.data = np.round(c2.data,5)
    sparse.save_npz('A2_DOT_'+str(i)+'.npz', c2)

class SpectralCF(object):
    def __init__(self, K, graph, n_users, n_items, emb_dim, lr, batch_size, decay):
        self.model_name = 'GraphCF with eigen decomposition'
        self.graph = graph
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.K = K
        self.decay = decay

        print("SpectralCF: Computing adjacient_matrix...")
        self.A = self.adjacient_matrix(self_connection=True)

        print("SpectralCF: Computing degree_matrix...")
        self.D = self.degree_matrix()

        print("SpectralCF: Computing laplacian_matrix...")
        self.L = self.laplacian_matrix(normalized=True)

        print("SpectralCF: Computing eigenvalues...")
        self.lamda, self.U = linalg.eigs(self.L)
        self.lamda = sparse.diags(self.lamda).tocsc()
        self.U = sparse.csc_matrix(self.U)

        print("SpectralCF: Building Tensorflow graph...")

        # placeholder definition
        self.users = tf.compat.v1.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(self.batch_size, ))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(self.batch_size,))
        '''
        self.user_embeddings = tf.Variable(
            tf.random.normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        '''
        self.user_embeddings = tf.compat.v1.get_variable(name='user_embeddings',\
         shape=[self.n_users, self.emb_dim], initializer=\
         tf.random_normal_initializer(mean=0.001, stddev=0.02, seed=None, dtype=tf.dtypes.float32))
        '''
        self.item_embeddings = tf.Variable(
            tf.random.normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')
        '''
        self.item_embeddings = tf.compat.v1.get_variable(name='item_embeddings',\
         shape=[self.n_items, self.emb_dim], initializer=\
         tf.random_normal_initializer(mean=0.001, stddev=0.02, seed=None, dtype=tf.dtypes.float32))

        self.filters = []
        for k in range(self.K):
            self.filters.append(tf.compat.v1.get_variable(name=str(k),\
                shape=[self.emb_dim, self.emb_dim], initializer=\
                tf.random_normal_initializer(mean=0.001, stddev=0.02, seed=None, dtype=tf.dtypes.float32)))

        self.U = self.U.astype(np.float32)
        self.lamda = self.lamda.astype(np.float32)

        A_hat = self.U.dot(self.U.T)

        A_2 = self.U.dot(self.lamda)
        A_2 = A_2.dot(self.U.T)

        A_hat = A_hat + A_2

        #A_hat += np.dot(np.dot(self.U, self.lamda_2), self.U.T)
        A_hat = A_hat.todense()
        A_hat = A_hat.astype(np.float32)

        print('A HAT DONE')
        print(A_hat.shape)

        shap = int(A_hat.shape[0]/2)

        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [embeddings]

        print(embeddings.shape)

        for k in range(0, self.K):
            
            embeddings_inter_1 = tf.matmul(A_hat[:shap,:], embeddings)
            embeddings_inter_2 = tf.matmul(A_hat[shap:,:], embeddings)

            embeddings = tf.concat([embeddings_inter_1, embeddings_inter_2], axis=0)

            print(embeddings.shape)
            
            #embeddings = tf.matmul(A_hat, embeddings)

            #filters = self.filters[k]#tf.squeeze(tf.gather(self.filters, k))
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings, self.filters[k]))
            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        self.u_embeddings, self.i_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        self.u_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.neg_items)

        self.all_ratings = tf.matmul(self.u_embeddings, self.i_embeddings, transpose_a=False, transpose_b=True)


        self.loss = self.create_bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings)


        self.opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr)

        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings] + self.filters)

        print("SpectralCF: Building Tensorflow graph... done!")

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_mean(maxi)) + self.decay * regularizer
        return loss

    def adjacient_matrix(self, self_connection=False):
        A = sparse.csc_matrix((self.n_users, self.n_users), dtype=np.float32)
        A2 = sparse.csc_matrix((self.n_items, self.n_items), dtype=np.float32)

        A = sparse.hstack([A,self.graph])
        A2 = sparse.hstack([self.graph.T,A2])
        A = sparse.vstack([A,A2])

        del A2
        
        if self_connection == True:
            return sparse.identity(self.n_users+self.n_items,dtype=np.float32) + A
        return A

    def degree_matrix(self):
        degree = np.sum(self.A, axis=1)
        degree = degree.reshape((1,degree.shape[0]))
        degree = np.asarray(degree).reshape(-1)

        #degree = np.diag(degree)
        return degree

    def laplacian_matrix(self, normalized=False):
        if normalized == False:
            return self.D - self.A

        temp = np.power(self.D, -1)
        temp = sparse.diags(temp).tocsc()
        temp = temp.dot(self.A)
    
        #temp = np.dot(temp, np.power(self.D, -0.5))
        return sparse.identity(self.n_users+self.n_items,dtype=np.float32) - temp