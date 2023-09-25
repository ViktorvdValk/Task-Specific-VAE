import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from utils.helper_utils import set_all_seeds, init_weights
from utils.helper_evaluate import evaluate_nn, eval_epoch
from utils.helper_loss import contrastive_loss_ordinal, contrastive_loss_missing, ResettableReduceLROnPlateau, ordinal_to_binary_relevance
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from vae.encodersXAI import VAE_XAI
from vae.encoders import VAE_2D
import copy
from itertools import cycle
RANDOM_SEED = 123


class Trainer():
    """ Trainer class for training neural network. """

    def __init__(self, model_type='VAE_2D', k=3, n_layers=5, device=None, latent_dim=20, lvef=True, pred2=2,
                 contrastive=True, both=False, btc=False, alpha=1, beta=1, gamma=1, gamma2=1, gamma_m=1, dist='gaussian',
                 lr=0.001, l2=0.1, random_state=42, resample=True, reduce_overfit=1, meta=True, save=False,
                 margin=5, ordinal_margin=1, version='v2', seed=42):

        self.device = device
        self.latent_dim = latent_dim
        self.lvef = lvef
        self.k = k
        self.n_layers = n_layers
        self.model_type = model_type
        self.pred2 = pred2
        self.contrastive = contrastive
        self.both = both
        self.btc = btc
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma2 = gamma2
        self.gamma_m = gamma_m
        self.dist = dist
        self.lr = lr
        self.l2 = l2
        self.random_state = random_state
        self.resample = resample
        self.reduce_overfit = reduce_overfit
        self.save = save
        self.meta = meta
        self.margin = margin
        self.ordinal_margin = ordinal_margin
        self.create_model()
        self.prepare_model()
        set_all_seeds(seed)
        self.version = version

    def __call__(self, num_epochs=10, train_loader=None, train_loader_p=None, val_loader=None, val_loader_p=None, val_loader2=None, val_loader2_p=None, patience=15, loss_pred=None, pred_phase=True, wd=1e-4):
        self.num_epochs = num_epochs
        self.loss_fn = F.mse_loss
        self.loss_pred = loss_pred
        self.pred_phase = pred_phase
        writer = self.create_logger()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=wd)
        self.scheduler = ResettableReduceLROnPlateau(
            self.optimizer, mode='min', patience=8, factor=0.8, min_lr=1e-8)
        self.scaler = GradScaler()
        patience_counter = 0
        best_val_loss_both = np.inf
        best_val_loss_rec = np.inf
        best_prediction_loss = np.inf
        best_tc = np.inf
        halfway_switch = False
        self.log_log_var = []
        ll_var = []
        self.log_mean = []
        l_mean = []

        for epoch in range(num_epochs):
            self.epoch_nr = epoch
            self.model.train()

            (
                mean_loss,
                mean_loss_recon,
                mean_prediction,
                mean_AUC,
                mean_meta,
                mean_tc,
                reg_train_out1,
                mean_f1
            ) = self.run_epoch(train_loader, train_loader_p, train=True, pred_phase=pred_phase, inference=False, reg_train=None)
            writer = self.log(writer, [mean_loss, mean_loss_recon,
                                       mean_prediction, mean_AUC, mean_f1, mean_meta, mean_tc], epoch, phase='train')

            self.model.eval()
            with torch.no_grad():
                (
                    mean_loss,
                    mean_loss_recon,
                    mean_prediction,
                    mean_AUC,
                    mean_meta,
                    mean_tc,
                    _,
                    mean_f1
                ) = self.run_epoch(
                    val_loader,
                    val_loader_p,
                    train=False,
                    reg_train=reg_train_out1,
                    pred_phase=pred_phase,
                    inference=False,
                )
                ll_var.append(np.concatenate(self.log_log_var))
                self.log_log_var = []
                l_mean.append(np.concatenate(self.log_mean))
                self.log_mean = []
            writer = self.log(writer, [mean_loss, mean_loss_recon,
                                       mean_prediction, mean_AUC, mean_f1, mean_meta, mean_tc], epoch, phase='val')

            self.scheduler.step(mean_loss)

            if pred_phase:
                if mean_loss < best_val_loss_both:
                    best_prediction_loss = mean_prediction
                    best_recon = mean_loss_recon,
                    best_val_loss_both = mean_loss
                    best_auc = mean_AUC
                    best_tc = mean_tc
                    best_state_dict = self.model.state_dict()
                    best_reg = copy.deepcopy(reg_train_out1)
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if mean_loss < best_val_loss_rec:
                    best_recon = mean_loss_recon,
                    best_val_loss_rec = mean_loss
                    best_state_dict = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter == patience:
                print("Early stopping!")
                break
        self.model.load_state_dict(best_state_dict)
        error, final_tc = evaluate_nn(self.model, val_loader,
                                      self.model_type, self.device, meta=self.meta)
        self.model.eval()
        l_mean, ll_var = [], []
        with torch.no_grad():
            for loader, loader_p in [(train_loader, train_loader_p), (val_loader, val_loader_p), (val_loader2, val_loader2_p)]:
                (
                    _,
                    final_val_mean_loss_recon,
                    final_prediction,
                    final_val_mean_AUC,
                    _,
                    _,
                    _,
                    final_val_mean_f1,
                ) = self.run_epoch(
                    loader,
                    loader_p,
                    train=False,
                    inference=True,
                    pred_phase=True,
                    reg_train=best_reg,
                    save=True
                )
                ll_var.append(np.concatenate(self.log_log_var))
                self.log_log_var = []
                l_mean.append(np.concatenate(self.log_mean))
                self.log_mean = []

        if self.save:
            self.save_model(pred_phase, halfway_switch=halfway_switch)
        return (
            self.model,
            best_val_loss_both,
            best_prediction_loss,
            final_val_mean_AUC,
            best_tc,
            final_tc,
            final_val_mean_loss_recon,
            error,
            final_val_mean_f1,
            final_prediction,
            np.concatenate(l_mean, axis=0),
            np.concatenate(ll_var),
        )

    def create_model(self):
        """Create model based on model type."""

        model_class = eval(self.model_type)
        if self.model_type in ["VAE_XAI"]:
            model = model_class(k=self.k, l=64, latent_dim=self.latent_dim,
                                input_dim=400, lvef=self.lvef)
            init_weights(model,  init_type='xavier')

        else:
            model = model_class(k=self.k, n=self.n_layers, latent_dim=self.latent_dim,
                                input_dim=400, lvef=self.lvef, pred2=self.pred2)
            init_weights(model,  init_type='kaiming')
        self.model = model

    def prepare_model(self):
        """Prepare model for training, register hooks, move to device."""

        self.model.to(self.device)
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    def run_epoch(self, loader, loader_p, train=True, inference=False, pred_phase=False, reg_train=None, save=False):
        """Run epoch of training or evaluation."""

        loss_epoch = 0
        loss_r_epoch = 0
        loss_mse_epoch = 0
        loss_tc_epoch = 0
        loss_meta_epoch = 0
        batchsize_epoch = 0
        batchsize_epoch_pred = 0
        prediction_epoch = 0
        n_data_p = len(loader_p.dataset)
        n_data = len(loader.dataset)
        pred_inf_ins = []
        targetss = []

        for recon_batch, pred_batch in zip(loader, cycle(loader_p)):
            if self.meta:
                features, targets_recon, meta_recon = recon_batch
                features_pred, targets, meta_pred = pred_batch
            else:
                features, targets_recon = recon_batch
                features_pred, targets = pred_batch
            batchsize_pred = features_pred.shape[0]
            batchsize_recon = features.shape[0]
            if self.model_type == "VAE_XAI":
                features_comb = torch.zeros(
                    [batchsize_recon+batchsize_pred, 12, 401])
            else:
                features_comb = torch.zeros(
                    [batchsize_recon+batchsize_pred, 1, 12, 401])
            features_comb[:batchsize_recon] = features
            features_comb[batchsize_recon:] = features_pred
            if self.meta:
                meta_comb = torch.zeros([batchsize_recon+batchsize_pred])
                meta_comb[:batchsize_recon] = meta_recon
                meta_comb[batchsize_recon:] = meta_pred

            with autocast():
                n_data_comb = n_data+n_data_p
                loss_r, loss_mse, batchsize, predictions, encoded, pred_inf_input = self.run_batch_recon(
                    features_comb, inference, n_data_comb, train=train, save=save)

                if pred_phase:
                    predictions = predictions[batchsize_recon:]
                    encoded_in = encoded[batchsize_recon:, :self.pred2]
                    pred_inf_in = pred_inf_input[batchsize_recon:]
                    pred_inf_ins.append(pred_inf_in.detach().cpu().numpy())
                    targetss.append(targets[:, 0].detach().cpu().numpy())

                    prediction_loss = self.run_batch_pred(
                        targets, predictions, self.l2, encoded=encoded_in)
                    prediction_epoch += prediction_loss * batchsize_pred

                    if self.meta:
                        meta_loss = self.run_batch_meta(
                            encoded[:, :2], meta_comb)
                    else:
                        meta_loss = 0
                else:
                    meta_loss = 0
                    prediction_loss = 0

                loss = loss_r + self.gamma2*prediction_loss * \
                    batchsize_recon/batchsize_pred + self.gamma_m*meta_loss

            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            loss_epoch += loss * batchsize
            loss_r_epoch += loss_r * batchsize
            loss_mse_epoch += loss_mse * batchsize
            batchsize_epoch += batchsize
            batchsize_epoch_pred += batchsize_pred

            if self.meta:
                loss_meta_epoch += meta_loss*batchsize

        mean_loss = loss_epoch.item() / batchsize_epoch
        mean_loss_mse = loss_mse_epoch.item() / batchsize_epoch
        if self.btc:
            mean_loss_tc = loss_tc_epoch.item() / batchsize_epoch
        else:
            mean_loss_tc = 0

        if pred_phase:
            if train:
                mean_AUC, mean_f1, reg_train = eval_epoch(
                    targetss, pred_inf_ins, None)
            else:
                mean_AUC, mean_f1, reg_train = eval_epoch(
                    targetss, pred_inf_ins, reg_train)

            mean_prediction = prediction_epoch.item() / batchsize_epoch_pred

            if self.meta:
                mean_meta = loss_meta_epoch.item() / batchsize_epoch
            else:
                mean_meta = 0
        else:
            mean_f1 = 0
            mean_AUC = 0
            reg_train = None
            mean_prediction = 1000
            mean_meta = 0
        return (
            mean_loss,
            mean_loss_mse,
            mean_prediction,
            mean_AUC,
            mean_meta,
            mean_loss_tc,
            reg_train,
            mean_f1,
        )

    def run_batch_meta(self, encoded, meta_data):
        """ Calculate contrastive loss based on meta data"""

        meta_data = meta_data.to(self.device)
        loss = contrastive_loss_missing(
            meta_data, encoded=encoded, ordinal_margin_scale=0, pred2=self.pred2)
        return loss

    def run_batch_recon(self, features, inference, n_data, train=False, save=False):
        """ Calculate reconstruction loss and kl-divergence loss"""

        batchsize = features.shape[0]
        features = features.to(self.device)
        encoded, z_mean, z_log_var, decoded, predictions, pred_inf_input = self.model(
            features)
        if z_mean.isnan().any():
            z_mean = torch.nan_to_num(z_mean, posinf=10, neginf=-10)

        if inference:
            if self.model_type == 'VAE_XAI':
                decoded = self.model.decode(z_mean)[0]
            else:
                decoded = self.model.decode(z_mean)
        if features.shape[2] == 401:
            features_in = features[:, :, :400]
        else:
            features_in = features[:, :, :, :400]

        if decoded.isnan().any():
            mask_nans = torch.isnan(decoded).any(
                axis=2).any(axis=1).any(axis=1)
            decoded = decoded[~mask_nans]
            features_in = features_in[~mask_nans]

        recon_loss = (F.mse_loss(decoded, features_in, reduction="none").view(
            decoded.shape[0], -1).sum(axis=1)/12).mean()
        z_mean_min_clip = -100
        z_mean_max_clip = 100
        z_log_va_min_clip = 2e-6
        z_log_va_max_clip = 200

        z_mean = z_mean.clip(min=z_mean_min_clip, max=z_mean_max_clip)
        z_log_var = z_log_var.clip(
            min=z_log_va_min_clip, max=z_log_va_max_clip)

        kl_div = -0.5 * torch.sum(1 + z_log_var -
                                  z_mean.pow(2) - torch.exp(z_log_var), axis=1)

        loss = recon_loss + self.beta*kl_div.mean()
        if save or not (train or inference):
            self.log_log_var.append(z_log_var.detach().cpu().numpy())
            self.log_mean.append(z_mean.detach().cpu().numpy())
        return loss, recon_loss, batchsize, predictions, encoded, pred_inf_input

    def run_batch_pred(self, targets, predictions, l2, encoded):
        """ Calculate prediction loss or contrastive loss"""

        targets = targets[:, 0]
        targets = targets.to(self.device)
        if self.contrastive:
            loss = contrastive_loss_ordinal(
                targets, encoded=encoded, margin=self.margin, ordinal_margin_scale=self.ordinal_margin, pred2=self.pred2)
        elif self.both:
            loss1 = contrastive_loss_ordinal(
                targets, encoded=encoded, margin=self.margin, ordinal_margin_scale=self.ordinal_margin, pred2=self.pred2)
            reg2_loss = torch.sum(torch.square(
                self.model.regressor.weight))
            preds = predictions.squeeze()
            binary_labels = ordinal_to_binary_relevance(
                targets, 4).to(self.device)
            prediction_loss = F.binary_cross_entropy_with_logits(
                preds, binary_labels)
            loss = (2*(prediction_loss + l2*reg2_loss) + loss1)/3
        else:
            reg2_loss = torch.sum(torch.square(
                self.model.regressor.weight))
            preds = predictions.squeeze()
            binary_labels = ordinal_to_binary_relevance(
                targets, 4).to(self.device)
            prediction_loss = F.binary_cross_entropy_with_logits(
                preds, binary_labels)
            loss = prediction_loss + l2*reg2_loss
        return loss

    def log(self, writer, losses, epoch, phase):
        """ Log losses to tensorboard"""

        losses_str = ['total', 'recon',
                      'prediction', 'auc', 'f1', 'meta', 'tc']

        print(
            "{phase} epoch {epoch} of {num_epochs}; {phase}_total_loss: {mean_total}, recon: {mean_recon},"
            " pred_loss: {mean_prediction}, auc: {mean_AUC}, f1: {f1}, mi: {mean_meta}, tc: {mean_tc}".format(
                epoch=epoch,
                num_epochs=self.num_epochs,
                mean_total=np.round(losses[0], 3),
                mean_recon=np.round(losses[1], 3),
                mean_prediction=np.round(losses[2], 3),
                mean_AUC=np.round(losses[3], 3),
                f1=np.round(losses[4], 3),
                mean_meta=np.round(losses[5], 3),
                mean_tc=np.round(losses[6], 3),
                phase=phase
            )
        )

        for i, loss in enumerate(losses):
            loss_str = "Loss/"+phase+'_'+losses_str[i]
            writer.add_scalar(loss_str, loss, epoch)
        return writer

    def log_string(self, pred_phase=True):
        """ Create log string for saving model"""

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log = self.model._get_name() + self.version + "_mixed_pred_phase_" + str(pred_phase) + "_contrastive_" + str(self.contrastive) + str(self.l2) + "_btc_" + str(self.btc) + "_alpha_" + str(self.alpha) + "_beta_" + str(self.beta) + "_gamma_" + \
            str(self.gamma) + "_gamma2_" + str(self.gamma2) + "_gamma_m" + str(self.gamma_m) + "_pred2_" + \
            str(self.pred2) + "_ld_" + str(self.latent_dim) + \
            "_rs_" + str(self.random_state) + str(timestamp)
        return log

    def create_logger(self):
        """ Create tensorboard logger"""

        writer = SummaryWriter(
            log_dir="run_new2/" + self.log_string()
        )
        return writer

    def save_model(self, pred_phase, halfway_switch):
        """ Save model"""

        torch.save(
            self.model.state_dict(),
            "models/" + self.log_string(pred_phase=pred_phase)
        )
        torch.cuda.empty_cache()


def loop_loaders(loader, loader_p, meta=False):
    """ Load data from reconstruction loader and prediction loader"""

    features_all = []
    features_pred_all = []
    targets_all = []
    for recon_batch, pred_batch in zip(loader, cycle(loader_p)):
        if meta:
            features, targets_recon, meta_recon = recon_batch
            features_pred, targets, meta_pred = pred_batch
        else:
            features, targets_recon = recon_batch
            features_pred, targets = pred_batch
        batchsize_pred = features_pred.shape[0]
        batchsize_recon = features.shape[0]
        features_comb = torch.zeros(
            [batchsize_recon+batchsize_pred, 1, 12, 401])
        features_comb[:batchsize_recon] = features
        features_comb[batchsize_recon:] = features_pred
        if meta:
            meta_comb = torch.zeros([batchsize_recon+batchsize_pred])
            meta_comb[:batchsize_recon] = meta_recon
            meta_comb[batchsize_recon:] = meta_pred
        features_all.append(features_comb.detach().cpu().numpy())
        features_pred_all.append(features_pred.detach().cpu().numpy())
        targets_all.append(targets[:, 0].detach().cpu().numpy())

    features_final = np.concatenate(features_all)
    features_pred = np.concatenate(features_pred_all)
    targets_final = np.concatenate(targets_all)

    x_final = features_final
    rri_pred = features_pred[:, :, [0, 6], 400]
    x_pred = features_pred

    return x_final, x_pred, rri_pred.squeeze(), targets_final
