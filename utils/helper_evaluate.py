import torch
import torch.nn.functional as F
import numpy as np
import copy
from sklearn import linear_model
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
)
from scipy.stats import entropy
import copy
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn import metrics


def std(arr):
    """Calculate standard deviation of array, split in 20 parts."""

    nr = 20
    stds = []
    a = arr.shape[0]//nr
    for i in range(nr+1):
        arr_in = arr[i*a:(i+1)*a]
        stds.append(sum((arr_in - arr.mean())**2))
    return np.sqrt(sum(stds)/arr.shape)


def calc_cond_cor_fz(z, model, device, n_samples_z=5, n_sweeps=9):
    """Calculate conditional correlation between z and f(z)."""

    ld = z.shape[1]
    deps_z_f = np.zeros([ld, 4800])
    const_z_means = np.zeros(ld)
    for i in range(ld):
        z_in = z[:, i]
        z_fin_check = np.isfinite(z_in)
        z_in_finite = z_in[z_fin_check]
        z_in_std = std(z_in_finite)
        z_in_m = z_in_finite.mean()
        sample_3 = np.random.choice(
            range(z.shape[0]), size=n_samples_z, replace=False)
        sample_to_check = [z[sample_3[ii]] for ii in range(n_samples_z)]

        if (sum(sum(np.isinf(sample_to_check))) > 0 or sum(sum(np.isnan(sample_to_check))) > 0):
            while (sum(sum(np.isinf(sample_to_check))) > 0 or sum(sum(np.isnan(sample_to_check))) > 0):
                sample_3 = np.random.choice(
                    range(z.shape[0]), size=n_samples_z, replace=False)
                sample_to_check = [z[sample_3[ii]]
                                   for ii in range(n_samples_z)]

        f_out = np.zeros([n_samples_z, 4800])
        for j in range(n_samples_z):
            sample_1 = z[sample_3[j]]
            f_var = np.zeros([n_sweeps, 4800])
            for k in range(n_sweeps):
                sample_1[i] = z_in_m + (2 - 4/(n_sweeps-1) * k) * z_in_std
                sample_torch = torch.from_numpy(sample_1).float().to(device)
                if model.__class__.__name__ == 'VAE_XAI':
                    decoded = model.decode(
                        sample_torch.reshape(1, -1))[0].squeeze()
                else:
                    decoded = model.decode(sample_torch).squeeze()
                f_var[k, :] = decoded.detach().cpu().numpy().flatten()
            f_out[j, :] = f_var.std(axis=0)
        const_z = np.corrcoef(f_out)
        np.fill_diagonal(const_z, np.nan)
        const_z_mean = np.nanmean(const_z)
        f_out_mean = f_out.mean(axis=0).squeeze()
        assert len(f_out_mean) == 4800
        deps_z_f[i] = f_out_mean
        const_z_means[i] = const_z_mean

    H_total = 0
    for ii in range(4800):
        dep = deps_z_f[:, ii]
        dep = dep/dep.sum()
        dep = np.round(dep, 1)
        if dep.sum() < 0.2:
            H_f = entropy(np.ones(ld))
        else:
            dep = dep/dep.sum()
            # low = disentangled
            H_f = entropy(dep)
        H_total += H_f

    return const_z_means.mean(), H_total


def calc_ppl(z, model, device, ppl='corr', nr_samples=10000):
    """Calculate Perceptual path length."""

    mask_z = []
    for i in range(z.shape[1]):
        z_in = z[:, i]
        if np.mean(abs(z_in)) < 0.01:
            print('Non relevant z dim')
            mask_z.append(False)
        else:
            mask_z.append(True)

    masked_z = z[:, mask_z]
    corr_z_mat = abs(np.corrcoef(masked_z, rowvar=False))
    np.fill_diagonal(corr_z_mat, np.nan)
    corr_z = np.nanmean(corr_z_mat)
    return corr_z


def mse_pca(pca, x_pca_test, x_test, ld, nl=False):
    """Calculate MSE and correlation between original and reconstructed data with numpy."""

    corr1, corr2 = [], []
    mse2 = []
    mu = np.mean(x_test, axis=0)
    if nl:
        x_hat = pca.inverse_transform(x_pca_test).reshape(
            x_pca_test.shape[0], 12, 400)
    else:
        x_hat = np.dot(x_pca_test, pca.components_[:ld, :]).reshape(
            x_pca_test.shape[0], 12, 400)

    for i in range(len(x_test)):
        person = x_test[i][:, :400]
        person_hat = x_hat[i] + mu[:, :400]
        mse1 = 0
        for j in range(len(person)):
            lead = person[j]
            lead_hat = person_hat[j]
            corr1.append(np.corrcoef(lead, lead_hat)[0, 1])
            mse1 += np.sum((lead - lead_hat)**2)
        mse15 = mse1/12
        corr2.append(np.nanmean(corr1))
        mse2.append(np.mean(mse15))

    return np.mean(corr2), np.mean(mse2)


def mse_pca2(pca, x_pca_test, x_test, ld, nl=False):
    """Calculate MSE and correlation between original and reconstructed data with torch loss function."""

    mu = np.mean(x_test, axis=0)
    loss_fn = F.mse_loss

    if nl:
        x_hat = pca.inverse_transform(x_pca_test).reshape(
            x_pca_test.shape[0], 12, 400)
    else:
        x_hat = np.dot(x_pca_test, pca.components_[:ld, :]).reshape(
            x_pca_test.shape[0], 12, 400)
    for i in range(x_hat.shape[0]):
        x_hat[i] += mu[:, :400]

    x_hat_torch = torch.from_numpy(x_hat).float()
    x_test_torch = torch.from_numpy(x_test[:, :, :400]).float()

    pixelwise = loss_fn(x_hat_torch, x_test_torch, reduction="none")
    pixelwise = pixelwise.view(
        x_test.shape[0], -1).sum(axis=1) / 12  # sum over pixels

    return pixelwise.mean().numpy()


def get_pca(x_train, x_pred_train, x_test, x_test_pred, ld, nl=False, ica=False):
    """Get PCA components."""

    if ica:
        pca = FastICA(n_components=ld, random_state=0, whiten='unit-variance')
    elif nl:
        pca = KernelPCA(n_components=ld, kernel='poly',
                        fit_inverse_transform=True)
    else:
        pca = PCA(n_components=ld)

    pca.fit(x_train[:, :, :, :400].reshape(x_train.shape[0], 4800))
    x_pca_train = pca.transform(
        x_train[:, :, :, :400].reshape(x_train.shape[0], 4800))
    x_pca_test = pca.transform(
        x_test[:, :, :, :400].reshape(x_test.shape[0], 4800))
    x_pca_pred_train = pca.transform(
        x_pred_train[:, :, :, :400].reshape(x_pred_train.shape[0], 4800))
    x_pca_pred_test = pca.transform(
        x_test_pred[:, :, :, :400].reshape(x_test_pred.shape[0], 4800))

    return x_pca_train, x_pca_pred_train, x_pca_test, x_pca_pred_test, pca


def predict_pca(x_pca_train, x_pca_test, x_train2, x_test2, y_train, y_test):
    """Predict with PCA components."""

    x_pred_test = np.append(x_pca_test, x_test2, axis=1)
    x_pred_train = np.append(x_pca_train, x_train2, axis=1)
    reg = linear_model.LogisticRegression(
        class_weight="balanced",
    )
    reg.fit(x_pred_train, y_train > 1)
    y_pred = reg.predict_proba(x_pred_test)[:, 1]

    f1 = f1_score(y_test > 1, np.round(y_pred, 0), average='macro')
    auc = roc_auc_score(y_test > 1, y_pred)
    return auc, f1


def eval_epoch(targetss, z_means, reg_train):
    """Evaluate epoch, output prediction metrics of logistic regression."""

    targets = np.concatenate(targetss)
    z_mean = np.concatenate(z_means)

    if reg_train == None:
        reg_train_out = linear_model.LogisticRegression(
            class_weight="balanced",
        )
        reg_train_out.fit(z_mean,
                          targets > 1)
    else:
        reg_train_out = copy.deepcopy(reg_train)
    preds = reg_train_out.predict_proba(
        z_mean)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(
        targets > 1, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(targets > 1, np.round(preds, 0), average='macro')

    return auc, f1, reg_train_out


def evaluate_nn(model, test_loader, model_type, device, meta=False):
    """
    Evaluate neural network, output correlation between input and reconstruction 
    and total correlation within latent dimensions.
    """

    err1 = []
    ss = 0.0
    model.eval()
    z_means = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if meta:
                features, targets, meta = batch
            else:
                features, targets = batch
            features = features.to(device)
            z_mean = model.forward(features)[1]
            if model_type == 'VAE_XAI':
                decoded = model.decode(z_mean)[0]
            else:
                decoded = model.decode(z_mean)

            z_means.append(z_mean.cpu().numpy())

            for k in range(len(features)):
                og = features[k].to("cpu").numpy()
                dec = decoded[k].to("cpu").numpy()
                if len(og.shape) != 1:
                    for k2 in range(len(og)):
                        og2 = og[k2]
                        dec2 = dec[k2]
                        if len(og2.shape) != 1:
                            for k3 in range(len(og2)):
                                og3 = og2[k3][:400]
                                dec3 = dec2[k3]
                                if len(og3) == 400:
                                    err1.append(np.corrcoef(og3, dec3)[0, 1])
                                    ss += 1.0
                                else:
                                    print("Loop not long enough")
                        else:
                            err1.append(np.corrcoef(og2[:400], dec2)[0, 1])
                            ss += 1.0
                else:
                    err1.append(np.corrcoef(og, dec)[0, 1])
                    ss += 1.0

            err = np.nanmean(err1)
    z_means = np.concatenate(z_means)
    corr = np.corrcoef(z_means.T)
    np.fill_diagonal(corr, 0)
    tc = np.sum(np.abs(corr)) / (corr.size - corr.shape[0])
    print("Mean correlation", np.round(err, 3))
    return err, tc
