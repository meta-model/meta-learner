import numpy as np
import scipy.sparse as sparse 
import time

class KNN(object):
    # URM_train sparse CSR
    def __init__(self, URM_train, verbose=True):
        self.URM_train = URM_train
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape
        self.verbose = verbose

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if (self.URM_train.getformat() == "csr") & (self.W_sparse.getformat() == "csr") :

            user_profile_array = self.URM_train[user_id_array]

            if items_to_compute is not None:
                item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
                item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
                item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
            else:
                item_scores = user_profile_array.dot(self.W_sparse).toarray()

            return item_scores
        
        else :
            print('URM TRAIN AND SPARSE HAVE NOT THE SAME FORMAT (CSR) ')
            return None

    def similarityMatrixTopK(self, item_weights, k=100, verbose = False):
        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        start_time = time.time()

        if verbose:
            print("Generating topK matrix")

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        if sparse_weights:
            if not isinstance(item_weights, sparse.csc_matrix):
                item_weights = item_weights.tocsc()
        else:
            column_row_index = np.arange(nitems, dtype=np.int32)

        for item_idx in range(nitems):

            cols_indptr.append(len(data))

            if sparse_weights:
                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx+1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

            else:
                column_data = item_weights[:,item_idx]

            non_zero_data = column_data!=0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])


        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse

    def fit(self, W_sparse, selectTopK = False, topK=100):

        assert W_sparse.shape[0] == W_sparse.shape[1],\
            "ItemKNNCustomSimilarityRecommender: W_sparse matrice is not square. Current shape is {}".format(W_sparse.shape)

        assert self.URM_train.shape[1] == W_sparse.shape[0],\
            "ItemKNNCustomSimilarityRecommender: URM_train and W_sparse matrices are not consistent. " \
            "The number of columns in URM_train must be equal to the rows in W_sparse. " \
            "Current shapes are: URM_train {}, W_sparse {}".format(self.URM_train.shape, W_sparse.shape)

        if selectTopK:
            W_sparse = self.similarityMatrixTopK(W_sparse, topK)

        if not isinstance(W_sparse, sparse.csr_matrix):
            self.W_sparse = W_sparse.tocsr()
    
    def recommend(self, user_id_array, cutoff = None, items_to_compute = None, return_scores = False):
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

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
