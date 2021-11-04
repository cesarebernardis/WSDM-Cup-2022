#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/10/21
@author: Cesare Bernardis

Adapted implementation from https://github.com/ilya-shenbin/RecVAE

"""

import numpy as np

import os
import shutil
import sys
import time
import torch
import scipy.sparse as sps

from copy import deepcopy

from torch import nn
from torch.nn import functional as F

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils import EarlyStoppingModel


torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


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
        self.DEFAULT_TEMP_FILE_FOLDER = './experiments_data/__Temp_{}/'.format(self.RECOMMENDER_NAME)


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


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3 / 20, 3 / 4, 1 / 10]):
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class VAE_original(nn.Module):

    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE_original, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)

        if calculate_loss:
            norm = user_ratings.sum(dim=-1)
            kl_weight = gamma * norm

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo

        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


class Batch:

    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx(), device=self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.as_tensor(self.get_ratings(is_out).toarray(), device=self._device)


def generate(batch_size, device, data_in, data_out=None, shuffle=False):

    total_samples = data_in.shape[0]
    samples_per_epoch = total_samples

    idxlist = np.arange(total_samples)
    if shuffle:
        np.random.shuffle(idxlist)

    for st_idx in range(0, len(idxlist), batch_size):
        end_idx = min(st_idx + batch_size, len(idxlist))
        idx = idxlist[st_idx:end_idx]
        yield Batch(device, idx, data_in, data_out)


class RecVAE(Recommender, EarlyStoppingModel, BaseTempFolder):

    RECOMMENDER_NAME = "RecVAE"

    def __init__(self, URM_train):
        super(RecVAE, self).__init__(URM_train)


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        self.model.eval()
        URM_train_user_slice = self.URM_train[user_id_array]
        URM_train_user_slice = URM_train_user_slice.astype('float32')
        item_scores_to_compute = np.empty((len(user_id_array), self.n_items), dtype=np.float32)

        for batch in generate(batch_size=64, device=self.device, data_in=URM_train_user_slice):
            ratings_dev = batch.get_ratings_to_dev()
            ratings_pred = self.model(ratings_dev, calculate_loss=False).cpu().detach().numpy()
            item_scores_to_compute[batch.get_idx(), :] = ratings_pred
            del ratings_dev

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = item_scores_to_compute[:, items_to_compute]
        else:
            item_scores = item_scores_to_compute

        return item_scores


    def fit(self,
            epochs=100,
            hidden_dim=512,
            latent_dim=256,
            gamma=0.005,
            learning_rate=5e-4,
            batch_size=64,
            dropout=0.5,
            n_enc_epochs=3,
            n_dec_epochs=1,
            temp_file_folder=None,
            **earlystopping_kwargs):

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        self.URM_train = sps.csr_matrix(self.URM_train, dtype=np.float32)

        self.batch_size = batch_size
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_enc_epochs = n_enc_epochs
        self.n_dec_epochs = n_dec_epochs

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu:0")

        self.log_dir = self.temp_file_folder + '{}-{}'.format(self.RECOMMENDER_NAME, time.time())
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_model_file = self.log_dir + "/best_model_sd.pt"

        model_kwargs = {
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'input_dim': self.n_items
        }

        self.model = VAE_original(**model_kwargs).to(self.device)

        decoder_params = set(self.model.decoder.parameters())
        encoder_params = set(self.model.encoder.parameters())

        self.optimizers = [torch.optim.Adam(encoder_params, lr=self.learning_rate),
                           torch.optim.Adam(decoder_params, lr=self.learning_rate)]

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.model.load_state_dict(torch.load(self.best_model_file))
        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)


    def clear_session(self):
        del self.model

    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        torch.save(self.model.state_dict(), self.best_model_file)

    def _run_epoch(self, num_epoch):

        self.model.train()

        def run(dropout, epochs=1):
            for e in range(epochs):
                for batch in generate(batch_size=self.batch_size, device=self.device,
                                      data_in=self.URM_train, shuffle=True):
                    ratings = batch.get_ratings_to_dev()

                    for optimizer in self.optimizers:
                        optimizer.zero_grad(set_to_none=True)

                    _, loss = self.model(ratings, gamma=self.gamma, dropout_rate=dropout)
                    loss.backward()

                    for optimizer in self.optimizers:
                        optimizer.step()

                    del ratings

        run(self.dropout, epochs=self.n_enc_epochs)
        self.model.update_prior()
        run(0.0, epochs=self.n_dec_epochs)


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        torch.save(self.model, folder_path + file_name + "_model.pt")

        data_dict_to_save = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "n_enc_epochs": self.n_enc_epochs,
            "n_dec_epochs": self.n_dec_epochs,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name=None):
        super(RecVAE, self).load_model(folder_path, file_name=file_name)
        self.model = torch.load(folder_path + file_name + "_model.pt")

