import torch
import abc
import torch.nn.functional as F
from utils.helper_math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)
import math
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

RECON_DIST = ["bernoulli", "laplace", "gaussian"]


class ResettableReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super(ResettableReduceLROnPlateau, self).__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose,
                                                          threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)

        self.optimizer = optimizer

    def reset(self, initial_lr):
        self.best = float('inf') if self.mode == 'min' else -float('inf')
        self.num_bad_epochs = 0
        self.last_epoch = -1
        self._reset2(initial_lr)

    def _reset2(self, initial_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr


def ordinal_regression(predictions, targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:int(target)] = 1

    return torch.nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1).mean()


def ordinal_to_binary_relevance(labels, num_classes):
    binary_labels = np.zeros((len(labels), num_classes-1))
    for i, label in enumerate(labels):
        for j in range(num_classes - 1):
            binary_labels[i, j] = 1 if label > j else 0
    return torch.tensor(binary_labels, dtype=torch.float32)


def focal_loss(inputs, targets, gamma_fl=5):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    F_loss = ((1 - pt) ** gamma_fl) * BCE_loss
    return F_loss.mean()


def focal_loss_alpha(inputs, targets, gamma_fl=5, alpha=0.15):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
    at = torch.clone(inputs)
    for i, input in enumerate(inputs):
        if input == 1:
            at[i] = alpha
        else:
            at[i] = 1
    F_loss = at*((1 - pt) ** gamma_fl) * BCE_loss
    return F_loss.mean()


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, z_mean, z_log_var, is_train):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        kwargs:
            Loss specific arguments
        """


class BtcvaeLoss(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, alpha=.1, beta=.6, gamma=.1, distribution='gaussian'):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.distribution = distribution

    def __call__(self, features, decoded, z_mean, z_log_var, encoded, n_data, is_mss, btc, train=False):

        rec_loss = _reconstruction_loss(
            features, decoded, distribution=self.distribution)

        if btc:

            log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(encoded,
                                                                                 z_mean,
                                                                                 z_log_var,
                                                                                 n_data, is_mss=is_mss)
            mi_loss = (log_q_zCx - log_qz).mean()
            tc_loss = (log_qz - log_prod_qzi).mean()
            dw_kl_loss = (log_prod_qzi - log_pz).mean()

            if sum([torch.isnan(mi_loss), torch.isnan(tc_loss), torch.isnan(dw_kl_loss)]):
                print("mi_loss, tc_loss, dw_kl_loss: ",
                      mi_loss, tc_loss, dw_kl_loss)
                print("log_pz, log_qz, log_prod_qzi, log_q_zCx: ", torch.isnan(log_pz.mean()), torch.isnan(
                    log_qz.mean()), torch.isnan(log_prod_qzi.mean()), torch.isnan(log_q_zCx.mean()))
                print("log_pz, log_qz, log_prod_qzi, log_q_zCx: ", torch.isnan(
                    log_pz), torch.isnan(log_qz), torch.isnan(log_prod_qzi), torch.isnan(log_q_zCx))

            anneal_reg = 1

            loss = rec_loss + self.alpha * mi_loss + self.beta * \
                tc_loss + anneal_reg * self.gamma * dw_kl_loss

            return loss, rec_loss, tc_loss
        elif self.beta < 0.5:
            loss = rec_loss
            return loss, loss, 0
        else:
            kl_loss = _kl_normal_loss(z_mean, z_log_var)
            loss = rec_loss + self.beta * kl_loss
            return loss, rec_loss, 0


def contrastive_loss_lvf_v3(y, encoded):
    batch_size, ld = encoded.shape
    y_c = y.squeeze()
    labels = (y_c == y_c[0]).float()
    distances = (encoded.unsqueeze(1) -
                 encoded.unsqueeze(0)).norm(dim=2).pow(2)
    margin = 30
    loss = 0.5 * labels * distances + 0.5 * \
        (1 - labels) * F.relu(margin - distances)
    loss = loss.sum() / batch_size
    return loss


def contrastive_loss_lvf_v2(y, encoded):
    batch_size, ld = encoded.shape
    y_c = y.squeeze()
    labels = (y_c == y_c[0]).float()
    distances = F.pairwise_distance(encoded.unsqueeze(1), encoded.unsqueeze(0))
    margin = 0.5
    target = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    target = 2 * target - 1  # convert from 0/1 to -1/1
    target *= margin  # set the target to the desired margin value
    loss = F.cosine_embedding_loss(distances, target, margin=1)
    print(loss.shape)
    return loss


def contrastive_loss_lvf(y, encoded):

    batch_size, ld = encoded.shape
    y_c = y.squeeze()
    L_total = 0
    for i in range(batch_size):
        y_c_i = (y_c == y_c[i]).int()
        encoded_i_repeated = encoded[i].reshape([1, ld]).repeat(batch_size, 1)
        mse = F.mse_loss(encoded_i_repeated, encoded,
                         reduction="none").sum(dim=1)
        m = 5
        L = 0.5 * y_c_i * mse + 0.5 * \
            (torch.ones_like(y_c_i) - y_c_i) * \
            torch.max(torch.zeros_like(mse), m - mse)
        L_total += L.mean()

    return L_total / batch_size


def contrastive_loss_ordinal(y, encoded, margin=5, ordinal_margin_scale=1, pred2=2):
    batch_size, ld = encoded.shape
    y_c = y.squeeze()

    y_c_matrix = y_c.view(-1, 1) == y_c.view(1, -1)
    y_c_matrix = y_c_matrix.float()

    encoded_expanded = encoded.unsqueeze(1)
    encoded_tiled = encoded.unsqueeze(0)

    mse_matrix = F.mse_loss(encoded_expanded.expand(-1, batch_size, -1),
                            encoded_tiled.expand(batch_size, -1, -1), reduction="none").sum(dim=2)/pred2

    # Ordinal term
    y_diff_matrix = torch.abs(y_c.view(-1, 1) - y_c.view(1, -1)).float()
    ordinal_margin = ordinal_margin_scale * y_diff_matrix

    # Positive and negative terms for contrastive loss
    pos_term = y_c_matrix * mse_matrix
    neg_term = (1 - y_c_matrix) * torch.clamp(margin +
                                              ordinal_margin - mse_matrix, min=0)

    L = 0.5 * (pos_term + neg_term)
    L_total = L.mean()

    return L_total


def contrastive_loss_missing(y, encoded, margin=5, ordinal_margin_scale=1, pred2=2):
    batch_size, ld = encoded.shape
    y_c = y.squeeze()

    # Create mask to identify missing labels (np.nan)
    missing_mask = torch.isnan(y_c.view(-1, 1)) | torch.isnan(y_c.view(1, -1))

    y_c_matrix = y_c.view(-1, 1) == y_c.view(1, -1)
    y_c_matrix = y_c_matrix.float()

    encoded_expanded = encoded.unsqueeze(1)
    encoded_tiled = encoded.unsqueeze(0)

    mse_matrix = F.mse_loss(encoded_expanded.expand(-1, batch_size, -1),
                            encoded_tiled.expand(batch_size, -1, -1), reduction="none").sum(dim=2)/pred2

    # Ordinal term
    y_diff_matrix = torch.abs(y_c.view(-1, 1) - y_c.view(1, -1)).float()
    ordinal_margin = ordinal_margin_scale * y_diff_matrix

    # Positive and negative terms for contrastive loss
    pos_term = y_c_matrix * mse_matrix
    neg_term = (1 - y_c_matrix) * torch.clamp(margin +
                                              ordinal_margin - mse_matrix, min=0)

    L = 0.5 * (pos_term + neg_term)

    # Set the loss for pairs with missing labels to zero
    L[missing_mask] = 0
    L_total = L.sum() / (batch_size * (batch_size - 1) - missing_mask.sum())

    return L_total


def contrastive_loss(y, encoded, margin=5):
    batch_size, ld = encoded.shape
    y_c = y.squeeze()

    y_c_matrix = y_c.view(-1, 1) == y_c.view(1, -1)
    y_c_matrix = y_c_matrix.float()

    encoded_expanded = encoded.unsqueeze(1)
    encoded_tiled = encoded.unsqueeze(0)

    mse_matrix = F.mse_loss(encoded_expanded.expand(-1, batch_size, -1),
                            encoded_tiled.expand(batch_size, -1, -1), reduction="none").sum(dim=2)

    pos_term = y_c_matrix * mse_matrix
    neg_term = (1 - y_c_matrix) * torch.clamp(margin - mse_matrix, min=0)

    L = 0.5 * (pos_term + neg_term)
    L_total = L.mean()

    return L_total


def contrastive_loss_mi_v4(y, encoded, margin=5):
    batch_size, ld = encoded.shape
    y_c = y.squeeze()

    y_c_matrix = y_c.view(-1, 1) == y_c.view(1, -1)
    y_c_matrix = y_c_matrix.float()

    encoded_expanded = encoded.unsqueeze(1)
    encoded_tiled = encoded.unsqueeze(0)

    mse_matrix = F.mse_loss(encoded_expanded.expand(-1, batch_size, -1),
                            encoded_tiled.expand(batch_size, -1, -1), reduction="none").sum(dim=2)

    pos_term = y_c_matrix * mse_matrix
    neg_term = (1 - y_c_matrix) * torch.clamp(margin - mse_matrix, min=0)

    L = 0.5 * (pos_term + neg_term)
    L_total = L.mean()

    return L_total


def _reconstruction_loss(features, decoded, distribution="gaussian"):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size = decoded.shape[0]

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(decoded, features, reduction="none").view(
            batch_size, -1).sum(axis=1)/12
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(decoded, features, reduction="none").view(
            batch_size, -1).sum(axis=1)/12
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(decoded, features, reduction="none").view(
            batch_size, -1).sum(axis=1)/12
        # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * 3
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    # loss = loss / batch_size

    return loss.mean()


def _kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    return total_kl


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


# Batch TC specific
def _get_log_pz_qz_prodzi_qzCx(latent_sample, z_mean, z_log_var, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    log_q_zCx = log_density_gaussian(
        latent_sample, z_mean, z_log_var).sum(dim=1)

    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)
    mat_log_qz = matrix_log_density_gaussian(latent_sample, z_mean, z_log_var)

    if not is_mss:
        log_qz = (torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False) -
                  math.log(batch_size * n_data))  # Ankit - modified
        log_prod_qzi = (torch.logsumexp(mat_log_qz, dim=1, keepdim=False) -
                        math.log(batch_size * n_data)).sum(1)  # Ankit - modified
    else:  # Ankit - modified
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(
            latent_sample.device)  # Ankit - modified
        log_qz = torch.logsumexp(
            log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False)  # Ankit - modified
        log_prod_qzi = torch.logsumexp(log_iw_mat.view(
            batch_size, batch_size, 1)+mat_log_qz, dim=1, keepdim=False).sum(1)  # Ankit - modified

    return log_pz, log_qz, log_prod_qzi, log_q_zCx
