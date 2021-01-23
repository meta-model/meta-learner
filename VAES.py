import pandas as pd
import numpy as np 
import tensorflow as tf 
from tensorflow.compat.v1.keras.regularizers import l2

class MultiVAE (object) :
    '''
        p_dims : Dimension of genrative part
        q_dims : Dimension of Inference part
        lam : Regularization rate
        lr : Learning Rate
    '''
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        tf.compat.v1.disable_eager_execution()
        self.p_dims = p_dims #Dimensions of generative part
        if q_dims is None: 
            self.q_dims = p_dims[::-1] #Dimension of inference part
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self.dims = self.q_dims + self.p_dims[1:]
        self.lam = lam 
        self.lr = lr 
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):
        self.input_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,self.dims[0]]) #dims[0] = Number of items - Our Input
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1.0, shape=None) #For Dropout
        self.is_training_ph = tf.compat.v1.placeholder_with_default(0.0, shape=None) #To remove std from the Latent Vector
        self.anneal_ph = tf.compat.v1.placeholder_with_default(1., shape=None) #Annealing parameter

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], [] #Weights of inference part

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out = d_out * 2  #Two sets of parameters : Mean and Variance
            
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)

            self.weights_q.append( tf.compat.v1.get_variable(name=weight_key, shape=[d_in,d_out],\
                initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)) ) #Create weights initialized
            
            self.biases_q.append( tf.compat.v1.get_variable(name=bias_key, shape=[d_out],\
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)) ) #Create bias initialized

        self.weights_p, self.biases_p = [], [] #Weights of generative part

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)

            self.weights_p.append( tf.compat.v1.get_variable(name=weight_key, shape=[d_in,d_out],\
                initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed)) ) #Create weights initialized
            
            self.biases_p.append( tf.compat.v1.get_variable(name=bias_key, shape=[d_out],\
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)) ) #Create bias initialized

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b #Linear function
            
            if i != len(self.weights_p) - 1: #All layers except the last one
                h = tf.nn.tanh(h)
        return h

    def q_graph(self):
        mu_q, std_q, KL = None, None, None

        h = tf.nn.l2_normalize(self.input_ph, 1) #Nornalization by rows (users)
        h = tf.nn.dropout(h, self.keep_prob_ph)  #Applying Dropout

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_q) - 1: #All Layers except the last
                h = tf.nn.tanh(h)
            else: #Last layer (Mean - Variance)
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum( 0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1 ), axis=1))
        return mu_q, std_q, KL

    def forward_pass(self):
        #Ineference part
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random.normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph * epsilon * std_q #The Latent vector

        #Generative part
        logits= self.p_graph(sampled_z) #Return the last layer

        return tf.compat.v1.train.Saver(), logits, KL

    def build_graph(self):
        self._construct_weights() #Construct p and q weights

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits) #Softmax the last layer

        neg_ll = -tf.reduce_mean(tf.reduce_sum( log_softmax_var * self.input_ph, axis=-1) )
        
        reg = l2(self.lam)
        reg_var = reg(self.weights_q + self.weights_p) # Apply regularization to weights
        # tensorflow l2 regularization multiply 0.5 to the l2 norm multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var
        
        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(neg_ELBO) #Train the model

        return saver, logits, neg_ELBO, train_op
    