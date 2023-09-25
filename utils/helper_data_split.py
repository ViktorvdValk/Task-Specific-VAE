import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import copy
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def check_pi_overlap(train, test):
    """Check if there is overlap in PatientID between train and test set"""

    pi_train = train.PatientID.unique()
    pi_test = test.PatientID.unique()
    for pi in pi_train:
        assert pi not in pi_test, "PatientID in training set, should not be in test/validation set"


def split_pred(data_in, random_state=42):
    """Split data into train and test set, stratified on lv_function"""

    cv = StratifiedGroupKFold(
        n_splits=6, random_state=random_state, shuffle=True)
    train_idxs1, _, _, _, _, _ = cv.split(
        data_in.index.values, data_in.lv_function.values, data_in.PatientID)
    train = data_in.iloc[train_idxs1[0]]
    test = data_in.iloc[train_idxs1[1]]
    check_pi_overlap(train, test)
    return train, test


def split_full(data, train_size, random_state=42):
    """Split data into train and test set, stratified on PatientID"""

    gss = GroupShuffleSplit(
        n_splits=2, train_size=train_size, random_state=random_state)
    train_idx, _ = gss.split(data.index, data.index, data.PatientID)
    train_f = data.iloc[train_idx[0]]
    test_f = data.iloc[train_idx[1]]
    check_pi_overlap(train_f, test_f)
    return train_f, test_f


def split_full_ex_prediction(data, test_p, random_state=42):
    """Split data into train and test set, stratified on PatientID, taken into account the split in the prediction subsets."""

    pi_test = test_p.PatientID.unique()
    p_m = len(test_p.PatientID) / len(data.PatientID)
    test1_f = copy.deepcopy(data.query('PatientID in @pi_test'))
    data.drop(test1_f.index, inplace=True)
    train_f, test2_f = split_full(data, train_size=0.85 + p_m)

    test_f = pd.concat([test1_f, test2_f])
    check_pi_overlap(train_f, test_f)
    return train_f, test_f


def load_labels():
    """Load labels for MI type"""

    df_labels = pd.read_csv('data/csv_ptids_labels.csv',
                            delimiter=';').drop(columns='Unnamed: 0')
    df_labels.drop_duplicates(subset='PatientUniqueID', inplace=True)
    df_labels.set_index('PatientUniqueID', inplace=True)
    for pi in df_labels.index:
        if df_labels.loc[pi, 'label_LAD']:
            df_labels.at[pi, 'label'] = 'LAD'
        elif df_labels.loc[pi, 'label_LCx']:
            df_labels.at[pi, 'label'] = 'LCx'
        elif df_labels.loc[pi, 'label_RCA']:
            df_labels.at[pi, 'label'] = 'RCA'
        else:
            df_labels.at[pi, 'label'] = 'None'
    return df_labels


def complete_pred_lvf2(df_lvf, df):
    """Complete lvf-function and other features in prediction set with values from full set."""

    for ind in df.index:
        pi = df.loc[ind, 'PatientID']
        date = df.loc[ind, 'Date']
        if date in df_lvf.query('PatientID == @pi').Date.values:
            row = df_lvf.query(
                '((PatientID == @pi) and (Date == @date))')
            df.at[ind, 'lv_function'] = row.lv_function.values[0]
            df.at[ind, 't'] = row.t.values[0]
            df.at[ind, 'time'] = row.time.values[0]
            df.at[ind, 'sex'] = row.sex.values[0]
            df.at[ind, 'age'] = row.age.values[0]
        else:
            df.drop(ind, inplace=True)

    df.dropna(how='any', subset=['lv_function'], inplace=True)
    return df


def complete_pred_lvf(df, data_full):
    """Complete lvf-function and other features,  including survival data in prediction set with values from full set."""

    df.reset_index(drop=True, inplace=True)
    for m in ['qrs', 'std_qrs', 'mean_rri', 'std_rri']:
        df[m] = ''
        df[m] = df[m].astype(object)

    for ind in df.index:
        pi = df.loc[ind, 'PatientID']
        date = df.loc[ind, 'Date']
        if date in data_full.query('PatientID == @pi').Date.values:
            value = data_full.query(
                '((PatientID == @pi) and (Date == @date))')
            row = df.loc[ind]
            for v in value.iterrows():
                ind = v[0]
                df.at[ind, 'PatientID'] = pi
                df.at[ind, 'Date'] = date
                df.at[ind, 'sex'] = row.sex
                df.at[ind, 'age'] = row.age
                df.at[ind, 'lv_function'] = row.lv_function
                df.at[ind, 't'] = row.t
                df.at[ind, 'dead_int'] = row.dead_int
                df.at[ind, 'time'] = row.time

                for m in ['qrs', 'std_qrs', 'mean_rri', 'std_rri']:
                    df.at[ind, m] = v[1][m]

    df = df[df['qrs'] != '']
    return df


def complete_pred_mi(df_labels, df):
    """Complete MI type in prediction set with values from labels. """

    for ind in df.index:
        pi = df.loc[ind, 'PatientID']
        if pi in df_labels.index:
            if df_labels.loc[pi, 'label'] == 'LAD':
                df.loc[ind, 'MI_type'] = 1
            elif df_labels.loc[pi, 'label'] == 'LCx':
                df.loc[ind, 'MI_type'] = 2
            elif df_labels.loc[pi, 'label'] == 'RCA':
                df.loc[ind, 'MI_type'] = 3
            else:
                df.loc[ind, 'MI_type'] = 0
        else:
            df.loc[ind, 'MI_type'] = np.nan
    return df


def train_test_val_split(data_pred, data_full, nr_splits=3, complete=True):
    """Full train-test-val split, including completing missing values in prediction set. """

    test_sets_f, test_sets_p = [], []
    data_in = copy.deepcopy(data_full[data_full['qrs'] != ''])
    df_labels = load_labels()
    if complete:
        data_pred = complete_pred_lvf2(data_pred, data_full)
        data_pred = complete_pred_mi(df_labels, data_pred)
        data_in = complete_pred_mi(df_labels, data_in)

    for _ in range(nr_splits):
        train_p, test_p = split_pred(data_pred)
        train_f, test_f = split_full_ex_prediction(data_in, test_p=test_p)
        data_pred = copy.deepcopy(train_p)
        data_in = copy.deepcopy(train_f)
        test_sets_f.append(test_f)
        test_sets_p.append(test_p)

    print('------------ Check PI Overlap ------------')
    for i in range(nr_splits):
        check_pi_overlap(train_f, test_sets_f[i])

    return train_p, train_f, test_sets_p, test_sets_f


def scale_data(data, column, scaler=None):
    """Scale data using StandardScaler."""

    if column == 'mean_rri_w':
        data_in = data[column].values
    else:
        data_in = np.stack(data[column].values)
    _shape = data_in.shape
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data_in.reshape(-1, 1))

    data_out = scaler.transform(data_in.reshape(-1, 1)).reshape(
        _shape
    )
    return data_out, scaler


def scale_all(train, tests, nr_splits, predict=False, withings=False):
    """Scale all features in train and test sets."""

    columns = ['qrs', 'std_qrs', 'mean_rri', 'std_rri']
    if predict:
        columns.append('age')

    for m in columns:
        train_scaled, scaler = scale_data(train, m)
        for i, j in enumerate(train.index):
            train.at[j, m] = train_scaled[i]

        for i in range(nr_splits):
            test_scaled, _ = scale_data(tests[i], m, scaler=scaler)
            for j, k in enumerate(tests[i].index):
                tests[i].at[k, m] = test_scaled[j]
    return train, tests


def load_full(filename):
    """Load full data set."""

    path = os.path.join(Path(os.getcwd()).parent, filename)
    df = pd.read_pickle(path)
    df = df[df['qrs'] != '']
    return df.reset_index(drop=True)


def run_splits(nr_splits, save=False, test=False):
    """Load, split and scale data."""

    end = '6_aug_surv_MI.p'
    data_pred = pd.read_pickle("data/df_ecg_dicom_lvrv_dbc2.p")
    data_pred.reset_index(drop=True, inplace=True)
    data_full = load_full('ecg12_400_all_embed8_augmented.p')

    train_pc, train_f, test_sets_pc, test_sets_f = train_test_val_split(
        data_pred, data_full, nr_splits=nr_splits)

    train_pc, test_sets_pc = scale_all(
        train_pc, test_sets_pc, nr_splits=nr_splits, predict=True)
    train_f, test_sets_f = scale_all(
        train_f, test_sets_f, nr_splits=nr_splits, predict=False)

    if save:
        train_pc.to_pickle('data/training_set_predict' + end)
        train_f.to_pickle('data/training_set_recon' + end)
        for i in range(nr_splits):
            if test and i == 0:
                test_sets_pc[i].to_pickle('data/test_set_predict' + end)
                test_sets_f[i].to_pickle('data/test_set_recon' + end)
            else:
                test_sets_pc[i].to_pickle(
                    'data/val_set' + str(i) + '_predict' + end)
                test_sets_f[i].to_pickle(
                    'data/val_set' + str(i) + '_recon' + end)
