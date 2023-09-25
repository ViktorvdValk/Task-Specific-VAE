import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from vae.encodersXAI import VAE_XAI
from vae.encoders import VAE_2D
from collections import Counter
from utils.helper_data_split import (
    train_test_val_split,
)


class DatasetWithMeta(TensorDataset):
    """Dataset with meta data."""

    def __init__(self, inputs, targets, metas):
        # call the super constructor with inputs and targets
        super().__init__(inputs, targets)
        self.metas = metas
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return (self.inputs[index], self.targets[index], self.metas[index])

    def __len__(self):
        return len(self.inputs)


def create_x(data, mean="qrs"):
    """Create input data for model."""

    data_in = np.stack(data[mean].values)
    mean_rri = np.stack(data["mean_rri"].values)
    std_rri = np.stack(data["std_rri"].values)
    rri = np.stack([mean_rri, std_rri]).T

    x = np.zeros([data_in.shape[0], 12, 401])
    x[:, :, :400] = data_in
    x[:, :, 400] = np.repeat(rri, 6, axis=1)
    return x


def create_meta(data):
    """Create meta data for model."""

    mi = data.MI_type.values
    return mi


def create_y(data, predict=False):
    """Create output data for model."""

    y = data.index.values
    y = y.astype(int)

    if predict:
        y0 = data['lv_function'].values
        y0 = y0.astype(int)
        return (y0, y)
    else:
        return y


def createXYM(data_sets, predict=False):
    """Create input, output and meta data for model."""

    xs, ys, ms = [], [], []
    for _, data in enumerate(data_sets):
        x = create_x(data)
        xs.append(x)
        y = create_y(data, predict=predict)
        ys.append(y)
        m = create_meta(data)
        ms.append(m)
    return xs, ys, ms


def createXY(data_sets, predict=False):
    """Create input and output data for model."""

    xs, ys = [], []
    for _, data in enumerate(data_sets):
        x = create_x(data)
        xs.append(x)
        y = create_y(data, predict=predict)
        ys.append(y)
    return xs, ys


def drop_dups(df):
    """Drop duplicates in data."""

    for i in df.index:
        df.at[i, 'qrs1'] = df.loc[i, 'qrs'][0][0]
    return df.drop_duplicates(subset=['PatientID', 'qrs1'])


def print_PI(df):
    print(df['PatientID'].nunique())


def load_data(meta=False):
    """Load data for model."""

    path = '/exports/lkeb-hpc/vovandervalk/vae-for-ecg/data/'
    end = '6_aug_surv_MI.p'

    train_pc = pd.read_pickle(path + 'training_set_predict' + end)
    train_f = pd.read_pickle(path + 'training_set_recon' + end)

    test_pc = pd.read_pickle(path + 'test_set_predict' + end)
    test_f = pd.read_pickle(path + 'test_set_recon' + end)

    val1_pc = pd.read_pickle(path + 'val_set1_predict' + end)
    val1_f = pd.read_pickle(path + 'val_set1_recon' + end)

    val2_pc = pd.read_pickle(path + 'val_set2_predict' + end)
    val2_f = pd.read_pickle(path + 'val_set2_recon' + end)

    if meta:
        x_preds, y_preds, m_preds = createXYM(
            [train_pc, val1_pc, val2_pc, test_pc], predict=True)
        x_f, y_f, m_f = createXYM([train_f, val1_f, val2_f, test_f])
        data_f = pd.concat([train_f, val1_f, val2_f])
        data_p = pd.concat([train_pc, val1_pc, val2_pc])
        return x_f, y_f, m_f, x_preds, y_preds, m_preds, test_f, test_pc, data_f, data_p

    else:
        x_preds, y_preds = createXY(
            [train_pc, val1_pc, val2_pc, test_pc], predict=True)
        x_f, y_f = createXY([train_f, val1_f, val2_f, test_f])
        data_f = pd.concat([train_f, val1_f, val2_f])
        data_p = pd.concat([train_pc, val1_pc, val2_pc])
        return x_f, y_f, x_preds, y_preds, test_f, test_pc, data_f, data_p


def get_dataloader(x, y, m=None, batch_size=100, val2=False, model_type='VAE_XAI', num_workers=0):
    """Get dataloader for model."""

    c = 12
    if model_type in ["VAE_2D"]:
        x_torch = torch.from_numpy(
            x.reshape(x.shape[0], 1, c, -1)).float()
    else:
        x_torch = torch.from_numpy(
            x.reshape(x.shape[0], c, -1)).float()

    if type(y) != tuple:
        y_torch = torch.from_numpy(y).int()
        if m is not None:
            m_torch = torch.from_numpy(m)
            data_set = DatasetWithMeta(x_torch, y_torch, m_torch)
        else:
            data_set = TensorDataset(x_torch, y_torch)

        loader = DataLoader(
            dataset=data_set, batch_size=batch_size, num_workers=num_workers, shuffle=val2)
    else:
        y2 = np.stack([y[0], y[1]]).T
        y_torch = torch.from_numpy(y2).float()
        if m is not None:
            m_torch = torch.from_numpy(m)
            data_set = DatasetWithMeta(x_torch, y_torch, m_torch)
        else:
            data_set = TensorDataset(x_torch, y_torch)

        loader = DataLoader(
            dataset=data_set, batch_size=batch_size, num_workers=num_workers, shuffle=val2)

        if val2:
            count = Counter(y[0])
            class_count = np.array([count[0], count[1], count[2], count[3]])

            weight = 1./class_count
            samples_weight = np.array([weight[int(t)] for t in y[0]])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                samples_weight, len(samples_weight))

            loader2 = DataLoader(
                dataset=data_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=sampler
            )
            return [loader2, loader]
    return loader


def get_dataloaders2(
        x_p, y_p, x_f, y_f, batch_size, model_type="VAE_12", sets=4, num_workers=0):
    """Get dataloaders for model."""

    val2 = True
    loaders_p, loaders_f = [], []
    batch_size_pred = int(batch_size*x_p[0].shape[0]/x_f[0].shape[0])
    for x, y, p, b in [(x_p, y_p, True, batch_size_pred), (x_f, y_f, False, batch_size)]:
        for i in range(sets):
            if i == 1:
                val2 = False
            loader = get_dataloader(
                x[i], y[i], batch_size=b, val2=val2, model_type=model_type, num_workers=num_workers)
            if p:
                loaders_p.append(loader)
            else:
                loaders_f.append(loader)
    return loaders_p, loaders_f


def get_dataloaders3(
        x_p, y_p, m_p, x_f, y_f, m_f, batch_size, model_type="VAE_12", sets=4, num_workers=0):
    """Get dataloaders with meta data for model."""

    val2 = True
    loaders_p, loaders_f = [], []
    batch_size_pred = int(batch_size*x_p[0].shape[0]/x_f[0].shape[0])
    for x, y, m, p, b in [(x_p, y_p, m_p, True, batch_size_pred), (x_f, y_f, m_f, False, batch_size)]:
        for i in range(sets):
            if i == 3:
                val2 = False
            loader = get_dataloader(
                x[i], y[i], m[i], batch_size=b, val2=val2, model_type=model_type, num_workers=num_workers)
            if p:
                loaders_p.append(loader)
            else:
                loaders_f.append(loader)
    return loaders_p, loaders_f


def load_first(sets, batch_size, meta, model_type, num_workers=0):
    """Load data for first cross validation loop."""
    
    if meta:
        x_f, y_f, m_f, x_preds, y_preds, m_preds, test_f, test_pc, data_f, data_p = load_data(meta=meta)
        loaders_p, loaders_f = get_dataloaders3(
            x_preds, y_preds, m_preds, x_f, y_f, m_f, batch_size=batch_size, model_type=model_type, sets=sets, num_workers=num_workers)
    else:
        x_f, y_f, x_preds, y_preds, test_f, test_pc, data_f, data_p = load_data(meta=meta)
        loaders_p, loaders_f = get_dataloaders2(
            x_preds, y_preds, x_f, y_f, batch_size=batch_size, model_type=model_type, sets=sets)

    return loaders_f, loaders_p, data_f, data_p


def load_rest(data_p0, data_f0, sets, batch_size, model_type, meta=False):
    """Load data for optional further cross validation loops."""
    
    train_p, train_f, test_sets_p, test_sets_f = train_test_val_split(
        data_p0, data_f0, nr_splits=2, complete=False)
    if meta:
        x_preds, y_preds, m_preds = createXYM(
            [train_p, test_sets_p[0], test_sets_p[1]], predict=True)
        x_f, y_f, m_f = createXYM([train_f, test_sets_f[0], test_sets_f[1]])
        loaders_p, loaders_f = get_dataloaders3(
            x_preds, y_preds, m_preds, x_f, y_f, m_f, batch_size=batch_size, model_type=model_type, sets=sets)
    else:
        x_preds, y_preds = createXY(
            [train_p, test_sets_p[0], test_sets_p[1]], predict=True)
        x_f, y_f = createXY([train_f, test_sets_f[0], test_sets_f[1]])
        loaders_p, loaders_f = get_dataloaders2(
            x_preds, y_preds, x_f, y_f, batch_size=batch_size, model_type=model_type, sets=sets)
    return loaders_f, loaders_p
