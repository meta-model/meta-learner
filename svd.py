from sklearn.utils.extmath import randomized_svd 
import scipy.sparse as sps
import numpy as np

class PureSVDRecommender():
    def __init__(self, URM_train, bias = False, verbose = True):
        # URM_train sparse CSR
        self.URM_train = URM_train
        self.URM_train.eliminate_zeros()

        self.verbose = verbose

        self.use_bias = bias

    def fit(self, num_factors=100, lambda_user=10, lambda_item=25, random_seed = None):
        print("Computing SVD decomposition...")
        if self.use_bias:
            self.URM_train = self.URM_train.tocsc()

            mu = self.URM_train.data.sum(dtype=np.float32) / self.URM_train.data.shape[0]
            self.GLOBAL_bias = mu

            col_nnz = np.diff(self.URM_train.indptr)

            URM_train_unbiased = self.URM_train.copy()
            URM_train_unbiased.data -= mu

            self.ITEM_bias = URM_train_unbiased.sum(axis=0) / (col_nnz + lambda_item)
            self.ITEM_bias = np.asarray(self.ITEM_bias).ravel()  # converts 2-d matrix to 1-d array without anycopy

            URM_train_unbiased.data -= np.repeat(self.ITEM_bias, col_nnz)

            # now convert the csc matrix to csr for efficient row-wise computation
            URM_train_unbiased_csr = URM_train_unbiased.tocsr()

            row_nnz = np.diff(URM_train_unbiased_csr.indptr)
            # finally, let's compute the bias

            self.USER_bias = URM_train_unbiased_csr.sum(axis=1).ravel() / (row_nnz + lambda_user)
            self.USER_bias = np.asarray(self.USER_bias).ravel()

            self.URM_train = self.URM_train.tocsr()

        print('Computin Biases ... Done!')

        U, Sigma, QT = randomized_svd(self.URM_train, n_components=num_factors, \
        random_state = random_seed)

        U_s = U * sps.diags(Sigma)

        self.USER_factors = U_s
        self.ITEM_factors = QT.T

        print("Computing SVD decomposition... Done!")

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
