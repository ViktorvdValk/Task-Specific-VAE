import numpy as np
import torch
import subprocess
import typer
import pickle
from utils.helper_train2 import Trainer, loop_loaders
from utils.helper_evaluate import (
    predict_pca,
    get_pca,
    mse_pca,
    mse_pca2,
    calc_ppl,
    calc_cond_cor_fz,
)
from utils.helper_load import (
    load_first,
    load_rest,
)
from utils.helper_utils import git_autocommit

RANDOM_SEED = 123
CUDA_DEVICE_NUM = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_batchsize_dict = {
    "VAE_XAI": 6,
    "VAE_2D": 16,
}


def run_test(save=False, version='v1', batch_size=100, num_epochs=50):
    lvef = True
    lr = 0.002
    random_state = 0
    resample = True
    reduce_overfit = 0
    results = []
    patience = 100
    loss_pred = torch.nn.functional.binary_cross_entropy_with_logits
    k = 3
    n_layers = 5
    meta = False
    dist = 'gaussian'
    alpha, gamma = 0, 0
    seeds = 1
    sets = 4
    for model_type in [
        "VAE_2D",
        "VAE_XAI"
    ]:
        BATCH_SIZE = int(batch_size * model_batchsize_dict[model_type])
        gpu_memory = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv']).decode('utf-8').strip('memory.total [MiB]\n'))
        if gpu_memory > 30000:
            BATCH_SIZE = BATCH_SIZE * 2

        num_workers = 7
        for btc in [False]:
            for latent_dim in [50]:
                for beta in [2, 10]:
                    for gamma_m in [0]:
                        for gamma2 in [0]:
                            for pred2 in [latent_dim]:
                                for l2 in [0.1]:
                                    for direct_combined_training in [True]:
                                        for margin, ordinal_margin in [(2, 2)]:
                                            for contrastive in [False, True]:
                                                for wd in [1e-4, 1e-3]:
                                                    if not contrastive:
                                                        if contrastive == "both":
                                                            both = True
                                                        else:
                                                            both = False
                                                    else:
                                                        if contrastive == "both":
                                                            both = True
                                                        else:
                                                            both = False
                                                    torch.cuda.empty_cache()
                                                    loaders_f, loaders_p, data_f0, data_p0 = load_first(
                                                        sets, BATCH_SIZE, meta=meta, model_type=model_type, num_workers=num_workers)

                                                    train_loader, val_loader, _, _ = loaders_f
                                                    train_loader_p, val_loader_p, _, _ = loaders_p
                                                    z, z_lvar = [], []
                                                    train_loader_p = train_loader_p[1]

                                                    for seed in range(seeds):
                                                        random_state = 0
                                                        if random_state != 0:
                                                            loaders_f, loaders_p = load_rest(
                                                                data_p0, data_f0, 3, BATCH_SIZE, model_type, meta=meta)
                                                            train_loader_p, val_loader_p, _ = loaders_p
                                                            train_loader, val_loader, _ = loaders_f
                                                            train_loader_p = train_loader_p[1]

                                                        trainer = Trainer(model_type=model_type, k=k, n_layers=n_layers, device=DEVICE, latent_dim=latent_dim, lvef=lvef, pred2=pred2, contrastive=contrastive,
                                                                          both=both, btc=btc, alpha=alpha, beta=beta, gamma=gamma, gamma2=gamma2, gamma_m=gamma_m, dist=dist, lr=lr, random_state=random_state,
                                                                          resample=resample, reduce_overfit=reduce_overfit, meta=meta, save=save, margin=margin, ordinal_margin=ordinal_margin, version=version, seed=seed)

                                                        (model,
                                                            best_val_loss,
                                                            best_prediction_loss,
                                                            best_auc,
                                                            best_tc,
                                                            final_tc,
                                                            best_recon,
                                                            error,
                                                            best_f1,
                                                            best_pred_loss,
                                                            l_mean,
                                                            ll_var) = trainer(num_epochs=num_epochs, train_loader=train_loader, train_loader_p=train_loader_p, val_loader=val_loader, val_loader_p=val_loader_p,
                                                                              val_loader2=val_loader, val_loader2_p=val_loader_p, patience=patience, loss_pred=loss_pred, pred_phase=direct_combined_training, wd=wd)

                                                        corr_z = calc_ppl(
                                                            l_mean, model, DEVICE)
                                                        const_z_mean, H_total = calc_cond_cor_fz(
                                                            l_mean[:50000], model, DEVICE)

                                                        row_test = {
                                                            "model": model_type,
                                                            "contrastive": contrastive,
                                                            "btc": btc,
                                                            "wd": wd,
                                                            "pred2": pred2,
                                                            "margin": margin,
                                                            "ordinal_margin": ordinal_margin,
                                                            "latent_dim": latent_dim,
                                                            "beta": beta,
                                                            "gamma2": gamma2,
                                                            "gamma_m": gamma_m,
                                                            "l2": l2,
                                                            "auc": np.round(best_auc, 4),
                                                            "f1": np.round(best_f1, 4),
                                                            "tc": np.round(best_tc, 4),
                                                            "final_tc": np.round(final_tc, 4),
                                                            "recon": np.round(best_recon, 4),
                                                            "final_pred": np.round(best_pred_loss, 4),
                                                            "corr_z": np.round(corr_z, 4),
                                                            "cond_corr": np.round(const_z_mean, 4),
                                                            "I(x, zi|z_i)": np.round(H_total, 4),
                                                            "random_state": random_state,
                                                        }
                                                        results.append(
                                                            row_test)
                                                        print(results)
                                                        z.append(l_mean)
                                                        z_lvar.append(ll_var)


def run_PCA_baseline(lds=[2, 10], nl=False, val2=True):
    random_state = 0
    results = []
    meta = False
    model_type = "VAE_2D"
    sets = 4
    BATCH_SIZE = 2000
    num_workers = 0

    loaders_f, loaders_p, data_f0, data_p0 = load_first(
        sets, BATCH_SIZE, False, meta=meta, model_type=model_type,  num_workers=num_workers)
    train_loader, val_loader, _, _ = loaders_f
    train_loader_p, val_loader_p, _, _ = loaders_p

    for random_state in range(5):
        if random_state != 0:
            loaders_f, loaders_p = load_rest(
                data_p0, data_f0, 3, BATCH_SIZE, model_type, meta=meta)
            train_loader_p, val_loader_p, _ = loaders_p
            train_loader, val_loader, _ = loaders_f

        train_loader_p = train_loader_p[0]

        x_train, x_train_pred, rri_train_pred, y_train = loop_loaders(
            train_loader, train_loader_p, meta=meta)
        x_val, x_val_pred, rri_val_pred, y_val = loop_loaders(
            val_loader, val_loader_p, meta=meta)
        x_pca_train, x_pca_pred_train, x_pca_test, x_pca_pred_test, pca = get_pca(
            x_train, x_train_pred, x_val, x_val_pred, 100)

        with open('pca2.p', 'wb') as m:
            pickle.dump(pca, m)

        for ld in lds:

            x_pca_test_ld = x_pca_test[:, :ld]
            x_pca_pred_train_ld = x_pca_pred_train[:, :ld]
            x_pca_pred_test_ld = x_pca_pred_test[:, :ld]

            auc, f1 = predict_pca(
                x_pca_pred_train_ld, x_pca_pred_test_ld, rri_train_pred, rri_val_pred, y_train, y_val)
            corr, _ = mse_pca(pca, x_pca_test_ld,
                              x_val.squeeze(), ld, nl=nl)
            mse = mse_pca2(pca, x_pca_test_ld, x_val.squeeze(), ld, nl=nl)

            row = {"ld": ld,
                   "random_state": random_state,
                   "auc": auc,
                   "f1": f1,
                   "mse": mse,
                   "corr": corr}
            results.append(row)
        print(results)


def main(pca: bool = False, test: bool = True, save: bool = False, version: str = 'Hello world'):
    num_epochs = 500
    batch_size = 100

    if pca:
        run_PCA_baseline(small=False, rg=[0, 1, 2, 3, 4], lds=[
                         2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80], val2=True)
    elif test:
        git_autocommit(version)
        run_test(save=save, version=version,
                 batch_size=batch_size, num_epochs=num_epochs)


if __name__ == "__main__":
    typer.run(main)
