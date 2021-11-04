import numpy as np

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils import check_matrix

from scipy.sparse.linalg import eigs
from sklearn.utils.extmath import randomized_svd


class DCT(Recommender):

    """

    Decoupled Completion and Transduction
    Cold-Start Item and User Recommendation with Decoupled Completion and Transduction
    Iman Barjasteh et al.

    """

    RECOMMENDER_NAME = "DCT"

    def __init__(self, URM_train, item_similarity_matrix):

        super(DCT, self).__init__(URM_train)
        self.item_similarity_matrix = check_matrix(item_similarity_matrix)


    def compute_item_score(self, user_id_array, items_to_compute=None):

        assert self.W.shape[0] > user_id_array.max(),\
                "MatrixFactorization_Cython: Cold users not allowed. " \
                "Users in trained model are {}, requested prediction for users up to {}"\
                .format(self.W.shape[0], user_id_array.max())

        if items_to_compute is not None:
            item_scores = np.dot(self.W[user_id_array], self.H[:, items_to_compute])
        else:
            item_scores = np.dot(self.W[user_id_array], self.H)

        item_scores = np.dot(np.dot(item_scores, self.Ub_hat_complete), self.Ub.T)

        return item_scores


    def fit(self, num_factors=10, num_eigs=10):

        super(DCT, self).fit()

        self.num_factors = num_factors
        self.num_eigs = num_eigs

        self.items_to_keep = np.arange(self.n_items)[np.ediff1d(self.URM_train.tocsc().indptr) > 0]

        self._print("Calculating URM factorization")
        u, s, vt = randomized_svd(self.URM_train, num_factors)

        self.W = np.dot(u, np.diag(s))
        self.H = vt[:, self.items_to_keep]

        self._print("Calculating eigenvalues of B")
        _, self.Ub = eigs(self.item_similarity_matrix, k=num_eigs)

        self._print("Calculating Ub complete")
        Ub_hat = self.Ub[self.items_to_keep]
        self.Ub_hat_complete = np.dot(Ub_hat, np.linalg.pinv(np.dot(Ub_hat.T, Ub_hat)))


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict = {"W": self.W,
                     "H": self.H,
                     "Ub": self.Ub}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete")
