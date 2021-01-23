from sklearn.decomposition import NMF
from sklearn.model_selection import RandomizedSearchCV
import scipy.sparse as sps
import numpy as np

class NMFRecommender():

    SOLVER_VALUES = {"coordinate_descent": "cd", "multiplicative_update": "mu"}

    INIT_VALUES = ["random", "nndsvda", "nndsvdar",'nndsvd']
    # random: non-negative random matrices, scaled with sqrt(X.mean() / n_components)
    # nndsvda: Nonnegative Double Singular Value Decomposition with zeros filled with the average of X

    BETA_LOSS_VALUES = ["frobenius", "kullback-leibler"]

    def __init__(self, URM_train, bias = False, verbose = True):
        # URM_train sparse CSR
        self.URM_train = URM_train
        self.URM_train.eliminate_zeros()

        self.verbose = verbose

        self.use_bias = bias

    def fit(self, num_factors=500, l1_ratio = 0.1, alpha = 0.05 , solver = "multiplicative_update",\
    init_type = "nndsvda", beta_loss = "frobenius", random_seed = None):

        assert l1_ratio>= 0 and l1_ratio<=1, "l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

        if solver not in self.SOLVER_VALUES:
           raise ValueError("Value for 'solver' not recognized. Acceptable values are {}, provided was '{}'".format(self.SOLVER_VALUES.keys(), solver))

        if init_type not in self.INIT_VALUES:
           raise ValueError("Value for 'init_type' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_VALUES, init_type))

        if beta_loss not in self.BETA_LOSS_VALUES:
           raise ValueError("Value for 'beta_loss' not recognized. Acceptable values are {}, provided was '{}'".format(self.BETA_LOSS_VALUES, beta_loss))


        print("Computing NMF decomposition...")

        param = {'n_components' : [50, 100, 200],\
        'init':['nndsvda','nndsvdar', 'nndsvd'],\
        'solver':['mu'],\
        'beta_loss': ['frobenius', 'kullback-leibler'],\
        'max_iter' : [200,500,1000],\
        'l1_ratio' : [0.05, 0.1 ,0.5],\
        'alpha' : [0.005,0.05,0.1,0.5],\
        'verbose' : [True]}

        #frobenius - mu - 200 - 0.1 - 0.05 - nndsvda

        
        nmf_solver = NMF(n_components = num_factors, init = init_type,solver = self.SOLVER_VALUES[solver],\
        beta_loss = beta_loss, random_state = random_seed, l1_ratio = l1_ratio, alpha = alpha,\
         shuffle = True, verbose = self.verbose, max_iter = 1000)

        '''
        model = NMF()
        nmf_solver = RandomizedSearchCV(model, param, scoring= 'max_error', n_jobs=-1)
        '''
        nmf_solver.fit(self.URM_train)

        self.ITEM_factors = nmf_solver.components_.copy().T
        self.USER_factors = nmf_solver.transform(self.URM_train)

        print("Computing NMF decomposition... Done!")

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array], self.ITEM_factors[items_to_compute,:].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)


        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores

    def recommend(self, user_id_array, cutoff = None, items_to_compute = None, return_scores = False):
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1]-1 
        
        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]
        
        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking[0]
            scores_batch = -scores_batch[0]
        else :
            ranking_list = ranking
        
        if return_scores:
            return ranking_list, scores_batch
        else:
            return ranking_list
