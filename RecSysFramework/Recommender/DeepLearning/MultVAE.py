#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31/10/18
@author: Maurizio Ferrari Dacrema, Cesare Bernardis
"""


import tensorflow as tf

import os
import shutil
import sys

import numpy as np
from scipy import sparse

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils import EarlyStoppingModel



def get_unique_temp_folder(input_temp_folder_path):
    """
    The function returns the path of a folder in result_experiments
    The function guarantees that the folder is not already existent and it creates it
    :return:
    """

    if input_temp_folder_path[-1] == "/":
        input_temp_folder_path = input_temp_folder_path[:-1]

    progressive_temp_folder_name = input_temp_folder_path
    counter_suffix = 0

    while os.path.isdir(progressive_temp_folder_name):
        counter_suffix += 1
        progressive_temp_folder_name = input_temp_folder_path + "_" + str(counter_suffix)

    progressive_temp_folder_name += "/"
    os.makedirs(progressive_temp_folder_name)

    return progressive_temp_folder_name


class BaseTempFolder(object):

    def __init__(self):
        super(BaseTempFolder, self).__init__()

        self.DEFAULT_TEMP_FILE_FOLDER = './result_experiments/__Temp_{}/'.format(self.RECOMMENDER_NAME)


    def _get_unique_temp_folder(self, input_temp_file_folder=None):

        if input_temp_file_folder is None:
            print("{}: Using default Temp folder '{}'".format(self.RECOMMENDER_NAME, self.DEFAULT_TEMP_FILE_FOLDER))
            self._use_default_temp_folder = True
            output_temp_file_folder = get_unique_temp_folder(self.DEFAULT_TEMP_FILE_FOLDER)
        else:
            print("{}: Using Temp folder '{}'".format(self.RECOMMENDER_NAME, input_temp_file_folder))
            self._use_default_temp_folder = False
            output_temp_file_folder = get_unique_temp_folder(input_temp_file_folder)

        if not os.path.isdir(output_temp_file_folder):
            os.makedirs(output_temp_file_folder)

        return output_temp_file_folder



    def _clean_temp_folder(self, temp_file_folder):
        """
        Clean temporary folder only if the default one
        :return:
        """

        if  self._use_default_temp_folder:
            print("{}: Cleaning temporary files from '{}'".format(self.RECOMMENDER_NAME, temp_file_folder))
            shutil.rmtree(temp_file_folder, ignore_errors=True)

        else:
            print("{}: Maintaining temporary files due to a custom temp folder being selected".format(self.RECOMMENDER_NAME))



class MultDAE_original(object):

    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None, debug=False):
        self.debug = debug
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
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
        self.input_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1.0, shape=None)

    def build_graph(self):

        self.construct_weights()

        saver, logits = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        # per-user average negative log-likelihood
        neg_ll = -tf.reduce_mean(input_tensor=tf.reduce_sum(
            input_tensor=log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights
        reg = tf.keras.regularizers.l2(self.lam)
        loss = neg_ll + sum(reg(w) for w in self.weights)

        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

        if self.debug:
            # add summary statistics
            tf.compat.v1.summary.scalar('negative_multi_ll', neg_ll)
            tf.compat.v1.summary.scalar('loss', loss)
            merged = tf.compat.v1.summary.merge_all()
        else:
            merged = None
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, 1 - (self.keep_prob_ph))

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.compat.v1.train.Saver(), h

    def construct_weights(self):

        self.weights = []
        self.biases = []

        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)

            self.weights.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed)))

            self.biases.append(tf.compat.v1.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            if self.debug:
                # add summary stats
                tf.compat.v1.summary.histogram(weight_key, self.weights[-1])
                tf.compat.v1.summary.histogram(bias_key, self.biases[-1])



class MultVAE_original(MultDAE_original):

    def construct_placeholders(self):
        super(MultVAE_original, self).construct_placeholders()
        # placeholders with default values when scoring
        self.is_training_ph = tf.compat.v1.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.compat.v1.placeholder_with_default(1., shape=None)

    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(input_tensor=tf.reduce_sum(
            input_tensor=log_softmax_var * self.input_ph,
            axis=-1))
        # apply regularization to weights
        reg = tf.keras.regularizers.l2(self.lam)
        neg_ELBO = neg_ll + self.anneal_ph * KL + sum(reg(w) for w in self.weights_q + self.weights_p)

        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        if self.debug:
            tf.compat.v1.summary.scalar('negative_multi_ll', neg_ll)
            tf.compat.v1.summary.scalar('KL', KL)
            tf.compat.v1.summary.scalar('neg_ELBO_train', neg_ELBO)
            merged = tf.compat.v1.summary.merge_all()
        else:
            merged = None

        return saver, logits, neg_ELBO, train_op, merged

    def q_graph(self):
        mu_q, std_q, KL = None, None, None

        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, rate = 1 - self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(input_tensor=tf.reduce_sum(
                        input_tensor=0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random.normal(tf.shape(input=std_q))

        sampled_z = mu_q + self.is_training_ph *\
            epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return tf.compat.v1.train.Saver(), logits, KL

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)

            self.weights_q.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed)))

            self.biases_q.append(tf.compat.v1.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            if self.debug:
                # add summary stats
                tf.compat.v1.summary.histogram(weight_key, self.weights_q[-1])
                tf.compat.v1.summary.histogram(bias_key, self.biases_q[-1])

        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform", seed=self.random_seed)))

            self.biases_p.append(tf.compat.v1.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            if self.debug:
                # add summary stats
                tf.compat.v1.summary.histogram(weight_key, self.weights_p[-1])
                tf.compat.v1.summary.histogram(bias_key, self.biases_p[-1])



class MultVAE(Recommender, EarlyStoppingModel, BaseTempFolder):

    RECOMMENDER_NAME = "MultVAE"

    def __init__(self, URM_train):
        super(MultVAE, self).__init__(URM_train)


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        URM_train_user_slice = self.URM_train[user_id_array]

        if sparse.isspmatrix(URM_train_user_slice):
            URM_train_user_slice = URM_train_user_slice.toarray()

        URM_train_user_slice = URM_train_user_slice.astype('float32')

        item_scores_to_compute = self.sess.run(self.logits_var, feed_dict={self.vae.input_ph: URM_train_user_slice})

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = item_scores_to_compute[:, items_to_compute]
        else:
            item_scores = item_scores_to_compute

        return item_scores


    def fit(self,
            epochs=100,
            learning_rate=1e-3,
            batch_size=500,
            dropout=0.5,
            total_anneal_steps=200000,
            anneal_cap=0.2,
            p_dims=None,
            debug=False,
            l2_reg=0.01,
            temp_file_folder=None,
            **earlystopping_kwargs):

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        self.n_users, self.n_items = self.URM_train.shape

        self.batch_size = batch_size
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.batches_per_epoch = int(np.ceil(float(self.n_users) / batch_size))
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.debug = debug
        self.learning_rate = learning_rate

        self.update_count = 0.0

        if p_dims is None:
            self.p_dims = [200, 600]
        else:
            self.p_dims = p_dims

        if self.p_dims[-1] != self.n_items:
            self.p_dims.append(self.n_items)

        self.q_dims = self.p_dims[::-1]

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        self.vae = MultVAE_original(self.p_dims, q_dims=self.q_dims, lr=self.learning_rate, lam=self.l2_reg, random_seed=98765, debug=self.debug)
        self.saver, self.logits_var, self.loss_var, self.train_op_var, self.merged_var = self.vae.build_graph()

        arch_str = "I-%s-I" % ('-'.join([str(d) for d in self.vae.dims[1:-1]]))

        self.log_dir = self.temp_file_folder + 'log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
            total_anneal_steps/1000, anneal_cap, arch_str)

        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        print("Mult_VAE_RecommenderWrapper: log directory: %s" % self.log_dir)
        
        if self.debug:
            self.summary_writer = tf.compat.v1.summary.FileWriter(self.log_dir, graph=tf.compat.v1.get_default_graph())

        self.chkpt_dir = self.temp_file_folder + 'chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
            total_anneal_steps/1000, anneal_cap, arch_str)

        if not os.path.isdir(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        print("Mult_VAE_RecommenderWrapper: checkpoint directory: %s" % self.chkpt_dir)


        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)


    def clear_session(self):
        tf.keras.backend.clear_session()
        if self.sess is not None:
            self.sess.close()
        self.sess = None
        print("------------ SESSION DELETED -----------------")


    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")


    def _run_epoch(self, num_epoch):

        user_index_list_train = list(range(self.n_users))

        np.random.shuffle(user_index_list_train)

        # train for one epoch
        nbatches = int(self.n_users / self.batch_size) + 1
        for bnum in range(nbatches):

            st_idx = self.batch_size * bnum
            end_idx = min(st_idx + self.batch_size, self.n_users)
            X = self.URM_train[user_index_list_train[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            feed_dict = {self.vae.input_ph: X,
                         self.vae.keep_prob_ph: self.dropout,
                         self.vae.anneal_ph: anneal,
                         self.vae.is_training_ph: 1}
            self.sess.run(self.train_op_var, feed_dict=feed_dict)

            if self.debug and bnum % 100 == 0:
                summary_train = self.sess.run(self.merged_var, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_train,
                                           global_step=num_epoch * self.batches_per_epoch + bnum)

            self.update_count += 1


    def save_model(self, folder_path, file_name=None):

        #https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, folder_path + file_name + "_session")

        data_dict_to_save = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
            "total_anneal_steps": self.total_anneal_steps,
            "anneal_cap": self.anneal_cap,
            "update_count": self.update_count,
            "p_dims": self.p_dims,
            "debug": self.debug,
            "batches_per_epoch": self.batches_per_epoch,
            "log_dir": self.log_dir,
            "chkpt_dir": self.chkpt_dir,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        tf.compat.v1.reset_default_graph()
        self.vae = MultVAE_original(self.p_dims, lam=self.l2_reg, debug=self.debug)
        self.saver, self.logits_var, self.loss_var, self.train_op_var, self.merged_var = self.vae.build_graph()

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver.restore(self.sess, folder_path + file_name + "_session")
        
        if self.debug:
            self.summary_writer = tf.compat.v1.summary.FileWriter(self.log_dir, graph=tf.compat.v1.get_default_graph())

        self._print("Loading complete")


class MultVAE_OptimizerMask(MultVAE):

    def fit(self, epochs=100, batch_size=500, total_anneal_steps=200000, learning_rate=1e-3, l2_reg=0.01,
            dropout=0.5, anneal_cap=0.2, temp_file_folder=None, **kwargs):

        p_dims = {}
        for key, value in kwargs.items():
            if "dl_layer_" in key:
                p_dims[key] = value

        super(MultVAE_OptimizerMask, self).fit(self, epochs=epochs, batch_size=batch_size, dropout=dropout, learning_rate=learning_rate,
                total_anneal_steps=total_anneal_steps, anneal_cap=anneal_cap, p_dims=p_dims, l2_reg=l2_reg,
                temp_file_folder=temp_file_folder, **kwargs)
