import numpy as np
import scipy.sparse as sps
import pandas as pd
import os, pickle

from .Utils import reconcile_mapper_with_removed_tokens, removeFeatures


class Dataset(object):


    def __init__(self, dataset_name, base_folder=None, postprocessings=None,
                 URM_dict=None, URM_mappers_dict=None,
                 ICM_dict=None, ICM_mappers_dict=None,
                 UCM_dict=None, UCM_mappers_dict=None):

        super(Dataset, self).__init__()

        self.dataset_name = dataset_name
        self.base_folder = base_folder if base_folder is not None else dataset_name + os.sep

        if postprocessings is None:
            postprocessings = []
        self.postprocessings = postprocessings

        self.n_items = None
        self.n_users = None

        self.URM_dict = {}
        self.URM_mappers_dict = {}
        if URM_dict is not None:
            assert len(URM_dict) == len(URM_mappers_dict), "Number of URMs ({}) different from number of mappers ({})"\
                                                           .format(len(URM_dict), len(URM_mappers_dict))
            for key in URM_dict.keys():
                assert key in URM_mappers_dict.keys(), "Missing mapper for {}".format(key)
                assert len(URM_mappers_dict[key]) == 2, "Expected 2 mappers for {}, given {}".format(key, len(URM_mappers_dict[key]))
                self.add_URM(key, URM_dict[key], URM_mappers_dict[key][0], URM_mappers_dict[key][1])

        self.ICM_dict = {}
        self.ICM_mappers_dict = {}
        if ICM_dict is not None:
            assert len(ICM_dict) == len(ICM_mappers_dict), "Number of ICMs ({}) different from number of mappers ({})"\
                                                           .format(len(ICM_dict), len(ICM_mappers_dict))
            for key in ICM_dict.keys():
                assert key in ICM_mappers_dict.keys(), "Missing mapper for {}".format(key)
                assert len(ICM_mappers_dict[key]) == 2, "Expected 2 mappers for {}, given {}".format(key, len(ICM_mappers_dict[key]))
                self.add_ICM(key, ICM_dict[key], ICM_mappers_dict[key][0], ICM_mappers_dict[key][1])

        self.UCM_dict = {}
        self.UCM_mappers_dict = {}
        if UCM_dict is not None:
            assert len(UCM_dict) == len(UCM_mappers_dict), "Number of UCMs ({}) different from number of mappers ({})"\
                                                           .format(len(UCM_dict), len(UCM_mappers_dict))
            for key in UCM_dict.keys():
                assert key in UCM_mappers_dict.keys(), "Missing mapper for {}".format(key)
                assert len(UCM_mappers_dict[key]) == 2, "Expected 2 mappers for {}, given {}".format(key, len(UCM_mappers_dict[key]))
                self.add_UCM(key, UCM_dict[key], UCM_mappers_dict[key][0], UCM_mappers_dict[key][1])
        
    
    def _check_UCM(self, name, UCM, row_mapper, column_mapper):

        assert UCM.shape[0] == self.n_users, "Wrong number of users for {}: {} - {}" \
            .format(name, UCM.shape[0], self.n_users)
        assert UCM.shape[0] == len(row_mapper), "Wrong user mapper size for {}: {} - {}" \
            .format(name, UCM.shape[0], len(row_mapper))
        assert UCM.shape[1] == len(column_mapper), "Wrong feature mapper size for {}: {} - {}" \
            .format(name, UCM.shape[1], len(column_mapper))

        nonzero = np.arange(UCM.shape[0], dtype=np.int)[np.ediff1d(UCM.tocsr().indptr) > 0]
        assert np.isin(nonzero, np.array(list(row_mapper.values()))).all(), \
            "There exist users with features that do not have a mapper entry in {}".format(name)

        nonzero = np.arange(UCM.shape[1], dtype=np.int)[np.ediff1d(UCM.tocsc().indptr) > 0]
        assert np.isin(nonzero, np.array(list(column_mapper.values()))).all(), \
            "There exist user features with interactions that do not have a mapper entry in {}".format(name)
        
    
    def _check_ICM(self, name, ICM, row_mapper, column_mapper):

        assert ICM.shape[0] == self.n_items, "Wrong number of items for {}: {} - {}" \
            .format(name, ICM.shape[0], self.n_items)
        assert ICM.shape[0] == len(row_mapper), "Wrong item mapper size for {}: {} - {}" \
            .format(name, ICM.shape[0], len(row_mapper))
        assert ICM.shape[1] == len(column_mapper), "Wrong feature mapper size for {}: {} - {}" \
            .format(name, ICM.shape[1], len(column_mapper))

        nonzero = np.arange(ICM.shape[0], dtype=np.int)[np.ediff1d(ICM.tocsr().indptr) > 0]
        assert np.isin(nonzero, np.array(list(row_mapper.values()))).all(), \
            "There exist items with features that do not have a mapper entry in {}".format(name)

        nonzero = np.arange(ICM.shape[1], dtype=np.int)[np.ediff1d(ICM.tocsc().indptr) > 0]
        assert np.isin(nonzero, np.array(list(column_mapper.values()))).all(), \
            "There exist item features with interactions that do not have a mapper entry in {}".format(name)
        
    
    def _check_URM(self, name, URM, row_mapper, column_mapper):

        assert URM.shape[0] == self.n_users, "Wrong number of users for {}: {} - {}" \
            .format(name, URM.shape[0], self.n_users)
        assert URM.shape[1] == self.n_items, "Wrong number of items for {}: {} - {}" \
            .format(name, URM.shape[1], self.n_items)
        assert URM.shape[0] == len(row_mapper), "Wrong user mapper size for {}: {} - {}" \
            .format(name, URM.shape[0], len(row_mapper))
        assert URM.shape[1] == len(column_mapper), "Wrong item mapper size for {}: {} - {}" \
            .format(name, URM.shape[1], len(column_mapper))

        nonzero = np.arange(URM.shape[0], dtype=np.int)[np.ediff1d(URM.tocsr().indptr) > 0]
        assert np.isin(nonzero, np.array(list(row_mapper.values()))).all(), \
            "There exist users with interactions that do not have a mapper entry in {}".format(name)

        nonzero = np.arange(URM.shape[1], dtype=np.int)[np.ediff1d(URM.tocsc().indptr) > 0]
        assert np.isin(nonzero, np.array(list(column_mapper.values()))).all(), \
            "There exist items with interactions that do not have a mapper entry in {}".format(name)


    def get_name(self):
        return self.dataset_name


    def set_base_folder(self, base_folder):
        self.base_folder = base_folder


    def get_base_folder(self):
        return self.base_folder


    def get_complete_folder(self):
        save_folder_path = self.base_folder
        if self.get_postprocessings():
            save_folder_path += os.path.join(*[p.get_name() for p in self.get_postprocessings()]) + os.sep
        return save_folder_path


    def get_UCM(self, UCM_name="UCM_all"):
        return self.UCM_dict[UCM_name].copy()


    def get_ICM(self, ICM_name="ICM_all"):
        return self.ICM_dict[ICM_name].copy()


    def get_URM(self, URM_name="URM_all"):
        return self.URM_dict[URM_name].copy()


    def get_UCM_mapper(self, UCM_name="UCM_all"):
        return (self.UCM_mappers_dict[UCM_name][0].copy(), self.UCM_mappers_dict[UCM_name][1].copy())


    def get_ICM_mapper(self, ICM_name="ICM_all"):
        return (self.ICM_mappers_dict[ICM_name][0].copy(), self.ICM_mappers_dict[ICM_name][1].copy())


    def get_URM_mapper(self, URM_name="URM_all"):
        return (self.URM_mappers_dict[URM_name][0].copy(), self.URM_mappers_dict[URM_name][1].copy())


    def get_UCM_names(self):
        return self.UCM_dict.keys()


    def get_ICM_names(self):
        return self.ICM_dict.keys()


    def get_URM_names(self):
        return self.URM_dict.keys()


    def get_UCM_dict(self):
        return self.UCM_dict


    def get_ICM_dict(self):
        return self.ICM_dict


    def get_URM_dict(self):
        return self.URM_dict


    def get_UCM_mappers_dict(self):
        return self.UCM_mappers_dict


    def get_ICM_mappers_dict(self):
        return self.ICM_mappers_dict


    def get_URM_mappers_dict(self):
        return self.URM_mappers_dict
    
    
    def add_UCM(self, UCM_name, UCM, row_mapper, column_mapper):
        if self.n_users is None:
            self.n_users = UCM.shape[0]
        self._check_UCM(UCM_name, UCM, row_mapper, column_mapper)
        self.UCM_dict[UCM_name] = UCM.copy()
        self.UCM_mappers_dict[UCM_name] = (row_mapper.copy(), column_mapper.copy())
    
    
    def add_ICM(self, ICM_name, ICM, row_mapper, column_mapper):
        if self.n_items is None:
            self.n_items = ICM.shape[0]
        self._check_ICM(ICM_name, ICM, row_mapper, column_mapper)
        self.ICM_dict[ICM_name] = ICM.copy()
        self.ICM_mappers_dict[ICM_name] = (row_mapper.copy(), column_mapper.copy())
    
    
    def add_URM(self, URM_name, URM, row_mapper, column_mapper):
        if self.n_users is None:
            self.n_users = URM.shape[0]
        if self.n_items is None:
            self.n_items = URM.shape[1]
        self._check_URM(URM_name, URM, row_mapper, column_mapper)
        self.URM_dict[URM_name] = URM.copy()
        self.URM_mappers_dict[URM_name] = (row_mapper.copy(), column_mapper.copy())


    def print_statistics(self):

        n_users, n_items = self.URM_dict["URM_all"].shape

        n_interactions = self.URM_dict["URM_all"].nnz

        URM_all = sps.csr_matrix(self.URM_dict["URM_all"])
        user_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_user = user_profile_length.max()
        avg_interactions_per_user = n_interactions/n_users
        min_interactions_per_user = user_profile_length.min()

        URM_all = sps.csc_matrix(self.URM_dict["URM_all"])
        item_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_item = item_profile_length.max()
        avg_interactions_per_item = n_interactions/n_items
        min_interactions_per_item = item_profile_length.min()

        print("Current dataset is: {}\n"
              "\tNumber of items: {}\n"
              "\tNumber of users: {}\n"
              "\tNumber of interactions in URM_all: {}\n"
              "\tInteraction density: {:.4f}%\n"
              "\tInteractions per user:\n"
              "\t\t Min: {}\n"
              "\t\t Avg: {:.2f}\n"    
              "\t\t Max: {}\n"     
              "\tInteractions per item:\n"    
              "\t\t Min: {}\n"
              "\t\t Avg: {:.2f}\n"    
              "\t\t Max: {}\n".format(
            self.get_name(),
            n_items,
            n_users,
            n_interactions,
            n_interactions / (n_items * n_users) * 100,
            min_interactions_per_user,
            avg_interactions_per_user,
            max_interactions_per_user,
            min_interactions_per_item,
            avg_interactions_per_item,
            max_interactions_per_item
        ))


    def add_postprocessing(self, postprocessing):
        if postprocessing.get_name() in [p.get_name() for p in self.postprocessings]:
            print("WARNING! Postprocessing {} already applied to the dataset.".format(postprocessing.get_name()))
        self.postprocessings.append(postprocessing)


    def get_postprocessings(self):
        return self.postprocessings.copy()


    def remove_items(self, items_to_remove, keep_original_shape=False):

        for key in self.URM_dict.keys():
            if keep_original_shape:
                # Gives memory error...
                # self.URM_dict[key][:, items_to_remove] = 0
                self.URM_dict[key] = self.URM_dict[key].tocsc()
                for i in items_to_remove.tolist():
                    self.URM_dict[key].data[self.URM_dict[key].indptr[i]:self.URM_dict[key].indptr[i+1]] = 0.0
            else:
                mask = np.ones(self.n_items, dtype=np.bool)
                mask[items_to_remove] = False
                self.URM_dict[key] = self.URM_dict[key][:, np.arange(self.n_items)[mask]]
                new_mapper = reconcile_mapper_with_removed_tokens(self.URM_mappers_dict[key][1], items_to_remove)
                self.URM_mappers_dict[key] = (self.URM_mappers_dict[key][0], new_mapper)
            self.URM_dict[key].eliminate_zeros()
            self.URM_dict[key] = self.URM_dict[key].tocsr()

        for key in self.ICM_dict.keys():
            if keep_original_shape:
                # Gives memory error...
                # self.ICM_dict[key][items_to_remove, :] = 0
                for i in items_to_remove.tolist():
                    self.ICM_dict[key].data[self.ICM_dict[key].indptr[i]:self.ICM_dict[key].indptr[i+1]] = 0.0
            else:
                mask = np.ones(self.n_items, dtype=np.bool)
                mask[items_to_remove] = False
                self.ICM_dict[key] = self.ICM_dict[key][np.arange(self.n_items)[mask], :]
                new_mapper = reconcile_mapper_with_removed_tokens(self.ICM_mappers_dict[key][0], items_to_remove)
                self.ICM_mappers_dict[key] = (new_mapper, self.ICM_mappers_dict[key][1])
            self.ICM_dict[key].eliminate_zeros()

        self.n_items -= len(items_to_remove)


    def remove_users(self, users_to_remove, keep_original_shape=False):

        for key in self.URM_dict.keys():
            if keep_original_shape:
                # Gives memory error...
                # self.URM_dict[key][users_to_remove, :] = 0
                for i in users_to_remove.tolist():
                    self.URM_dict[key].data[self.URM_dict[key].indptr[i]:self.URM_dict[key].indptr[i+1]] = 0.0
            else:
                mask = np.ones(self.n_users, dtype=np.bool)
                mask[users_to_remove] = False
                self.URM_dict[key] = self.URM_dict[key][np.arange(self.n_users)[mask], :]
                new_mapper = reconcile_mapper_with_removed_tokens(self.URM_mappers_dict[key][0], users_to_remove)
                self.URM_mappers_dict[key] = (new_mapper, self.URM_mappers_dict[key][1])
            self.URM_dict[key].eliminate_zeros()

        for key in self.UCM_dict.keys():
            if keep_original_shape:
                # Gives memory error...
                # self.UCM_dict[key][users_to_remove, :] = 0
                for i in users_to_remove.tolist():
                    self.UCM_dict[key].data[self.UCM_dict[key].indptr[i]:self.UCM_dict[key].indptr[i+1]] = 0.0
            else:
                mask = np.ones(self.n_users, dtype=np.bool)
                mask[users_to_remove] = False
                self.UCM_dict[key] = self.UCM_dict[key][np.arange(self.n_users)[mask], :]
                new_mapper = reconcile_mapper_with_removed_tokens(self.UCM_mappers_dict[key][0], users_to_remove)
                self.UCM_mappers_dict[key] = (new_mapper, self.UCM_mappers_dict[key][1])

            self.UCM_dict[key].eliminate_zeros()

        self.n_users -= len(users_to_remove)


    def remove_item_features(self, min_occurrence=5, max_perc_occurrence=0.3, keep_original_shape=False):

        for key in self.ICM_dict.keys():
            self.ICM_dict[key], _, new_mapper = removeFeatures(self.ICM_dict[key],
                                                               minOccurrence=min_occurrence,
                                                               maxPercOccurrence=max_perc_occurrence,
                                                               reshape=not keep_original_shape,
                                                               reconcile_mapper=self.ICM_mappers_dict[key][1])
            self.ICM_mappers_dict[key] = (self.ICM_mappers_dict[key][0], new_mapper)


    def remove_user_features(self, min_occurrence=5, max_perc_occurrence=0.3, keep_original_shape=False):

        for key in self.UCM_dict.keys():
            self.UCM_dict[key], _, new_mapper = removeFeatures(self.UCM_dict[key],
                                                               minOccurrence=min_occurrence,
                                                               maxPercOccurrence=max_perc_occurrence,
                                                               reshape=not keep_original_shape,
                                                               reconcile_mapper=self.UCM_mappers_dict[key][1])
            self.UCM_mappers_dict[key] = (self.UCM_mappers_dict[key][0], new_mapper)


    def copy(self):
        return Dataset(self.dataset_name, base_folder=self.base_folder, postprocessings=self.postprocessings.copy(),
                       URM_dict=self.URM_dict.copy(), URM_mappers_dict=self.URM_mappers_dict.copy(),
                       ICM_dict=self.ICM_dict.copy(), ICM_mappers_dict=self.ICM_mappers_dict.copy(),
                       UCM_dict=self.UCM_dict.copy(), UCM_mappers_dict=self.UCM_mappers_dict.copy())


    def save_data(self, save_folder_path=None, filename_suffix=""):

        if save_folder_path is None:
            save_folder_path = self.get_complete_folder()

        os.makedirs(save_folder_path, exist_ok=True)

        for URM_name in self.get_URM_names():
            print("Reader: Saving {}...".format(URM_name + filename_suffix))
            sps.save_npz(save_folder_path + "{}.npz".format(URM_name + filename_suffix), self.get_URM(URM_name))
            with open(save_folder_path + "{}_mapper".format(URM_name + filename_suffix), "wb") as file:
                pickle.dump(self.get_URM_mapper(URM_name), file, protocol=pickle.HIGHEST_PROTOCOL)

        for ICM_name in self.get_ICM_names():
            print("Reader: Saving {}...".format(ICM_name + filename_suffix))
            sps.save_npz(save_folder_path + "{}.npz".format(ICM_name + filename_suffix), self.get_ICM(ICM_name))
            with open(save_folder_path + "{}_mapper".format(ICM_name + filename_suffix), "wb") as file:
                pickle.dump(self.get_ICM_mapper(ICM_name), file, protocol=pickle.HIGHEST_PROTOCOL)

        for UCM_name in self.get_UCM_names():
            print("Reader: Saving {}...".format(UCM_name + filename_suffix))
            sps.save_npz(save_folder_path + "{}.npz".format(UCM_name + filename_suffix), self.get_UCM(UCM_name))
            with open(save_folder_path + "{}_mapper".format(UCM_name + filename_suffix), "wb") as file:
                pickle.dump(self.get_UCM_mapper(UCM_name), file, protocol=pickle.HIGHEST_PROTOCOL)

        print("Reader: Saving complete!")


    def save_data_as_csv(self, save_folder_path=None, filename_suffix="", sep=",", decimal="."):

        if save_folder_path is None:
            save_folder_path = self.get_complete_folder()

        os.makedirs(save_folder_path, exist_ok=True)

        for URM_name in self.get_URM_names():
            print("Reader: Saving {}.csv...".format(URM_name + filename_suffix))
            urm = self.get_URM(URM_name).tocoo()
            pd.DataFrame({'row': urm.row, 'col': urm.col, 'data': urm.data}).reset_index()\
              .to_csv(save_folder_path + "{}.csv".format(URM_name + filename_suffix),
                      sep=sep, index=False, decimal=decimal)

        for ICM_name in self.get_ICM_names():
            print("Reader: Saving {}.csv...".format(ICM_name + filename_suffix))
            urm = self.get_ICM(ICM_name).tocoo()
            pd.DataFrame({'row': urm.row, 'col': urm.col, 'data': urm.data})\
              .to_csv(save_folder_path + "{}.csv".format(ICM_name + filename_suffix),
                      sep=sep, index=False, decimal=decimal)

        for UCM_name in self.get_UCM_names():
            print("Reader: Saving {}.csv...".format(UCM_name + filename_suffix))
            urm = self.get_UCM(UCM_name).tocoo()
            pd.DataFrame({'row': urm.row, 'col': urm.col, 'data': urm.data})\
              .to_csv(save_folder_path + "{}.csv".format(UCM_name + filename_suffix),
                      sep=sep, index=False, decimal=decimal)

        print("Reader: Saving complete!")
