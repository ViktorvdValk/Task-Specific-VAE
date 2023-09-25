import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy.signal as ssig
from ecg_RPNet import detector_RP
import matplotlib.pyplot as plt
import tqdm
import copy
import multiprocessing
import torch
import os
import threading

# from joblib import Parallel, delayed


class ECG_Preprocessor():

    def __init__(self, fs):
        self.fs = fs
        self.detector = detector_RP.Detector_RPNet()

    def eval_peaks(self, p, x):
        xs = np.zeros([len(p[0])-2, 400])
        delete_first = False
        delete_last = False
        for jj, peak in enumerate(p[0][1:-1]):
            qrs = x[peak-200:peak+200]
            if len(qrs) != 400:
                if jj == 0:
                    delete_first = True
                else:
                    delete_last = True
            else:
                xs[jj] = qrs

        if delete_first:
            xs = xs[1:]
        if delete_last:
            xs = xs[:-1]
        cc = np.corrcoef(xs)

        if type(cc) != np.ndarray:
            return cc, cc
        elif len(cc) < 2:
            print('potential error')
            return cc, 0
        return cc, np.mean(cc)

    def improve_peaks(self, ps, x):
        cc_og, mean_og = self.eval_peaks(ps, x)
        temp_ps = copy.deepcopy(ps)
        best_mean = mean_og
        if cc_og.ndim < 2:
            ccm = mean_og
        else:
            ccm = np.mean(cc_og, axis=1)
        for wp in np.argsort(ccm):
            wp += 1
            x_wp = ps[0][wp]

            ap = ssig.find_peaks(x)[0]
            ap = ap[ap < (x_wp+100)]
            ap = ap[ap > (x_wp-100)]
            best_pp = x_wp

            for pp in ap:
                temp_ps[0][wp] = pp
                cc, mean = self.eval_peaks(temp_ps, x)
                if mean > best_mean:
                    best_mean = mean
                    best_pp = pp

            temp_ps[0][wp] = best_pp

        return temp_ps

    def split_hb2(self, x_ecg, first=True):
        ecg, peaks, y = self.detector.predict(
            x_ecg.reshape(1, -1), calibrate=True)

        if len(peaks[0]) < 3:
            print('Too little peaks')
            return 0, 0, 0
        peaks = self.improve_peaks(peaks, x_ecg)
        xs = np.zeros([len(peaks[0])-2, 400])
        skip = 0
        for i, peak in enumerate(peaks[0][1:-1]):
            qrs = x_ecg[peak-200:peak+200]
            if len(qrs) != 400:
                if i == 0:
                    skip = 1
                else:
                    rri = peaks[0][1:-1] - peaks[0][:-2]
                    begin = peaks[0][0]
                    end = 5000 - sum(rri) - begin
                    return xs[:-1], rri, [begin, end]
            else:
                xs[i] = qrs

        rri = peaks[0][(1+skip):] - peaks[0][skip:-1]
        begin = peaks[0][skip]
        end = 5000 - sum(rri) - begin
        return xs[skip:], rri, [begin, end]

    def return_qrs(self, qrs_in, rri_in, label, aggregate=True):
        if aggregate:
            return np.median(qrs_in, axis=0), np.mean(qrs_in, axis=0), np.std(qrs_in, axis=0), np.mean(rri_in), np.std(rri_in), label
        else:
            return qrs_in, np.std(qrs_in, axis=0), np.mean(rri_in), np.std(rri_in), label

    def _checkConsecutive(self, arr, max=5):
        if len(arr) < max:
            return False
        _count = 0
        _max_count = 0
        _end = 0
        for i, d in enumerate(np.diff(arr)):
            if d == 1:
                _count += 1
                if _count > _max_count:
                    _max_count = _count
                    _end = i + 2
            else:
                _count = 0
        if _max_count > 4:
            return arr[_end-5:_end]
        else:
            return False

    def check_overlap(self, begin1, begin2, rri1, rri2, start2):
        l = begin1 + sum(rri1) - start2
        i = 0
        while begin2 < l:
            begin2 += rri2[i]
            i += 1
        return i

    def embed_sig2(self, sig_unfilt, fs, aggregate=True):
        if aggregate:
            empty = np.zeros(400), np.zeros(400), np.zeros(400), 0, 0, 'empty'
        else:
            empty = np.zeros(400), np.zeros(400), 0, 0, 'empty'

        if fs == 1000:
            sig_unfilt = ssig.resample(sig_unfilt, 5000)
            fs = 500
        elif fs == 300:
            sig_unfilt = ssig.resample(sig_unfilt, 15000)
            fs = 500

        filter = ssig.firwin(len(sig_unfilt), cutoff=[0.5, 45],
                             window='blackmanharris', pass_zero='bandpass',
                             fs=fs)
        sig = ssig.convolve(sig_unfilt, filter, mode='same', method='auto')

        sig1 = sig[:5000]
        if len(sig) > 6000:
            print(len(sig))
            sig2 = sig[-5000:]

            qrs1, rri1, ex1 = self.split_hb2(sig1)
            qrs2, rri2, ex2 = self.split_hb2(sig2)

            if type(qrs1) == int:
                if type(qrs2) == int:
                    return empty
                else:
                    qrs = qrs2
            else:
                if type(qrs2) == int:
                    qrs = qrs1
                elif len(sig) >= 10000:
                    qrs = np.concatenate([qrs1, qrs2])
                    rri = np.concatenate([rri1, rri2])
                else:
                    count = ex2[1] + 10
                    i = 0
                    while count < (len(sig) - 5000 + ex1[1]):
                        i += 1
                        if i == len(rri2):
                            break
                        else:
                            count += rri2[-i]
                    qrs = np.concatenate([qrs1[3:], qrs2[-i:]])
                    rri = np.concatenate([rri1[4:], rri2[-i:]])

        else:
            qrs, rri, ex = self.split_hb2(sig1)
            if type(qrs) == int:
                return empty

        qrs_arg = []
        for i, hb in enumerate(qrs):
            if np.mean(abs(hb)) > 1:
                qrs_arg.append(i)

        qrs = qrs[qrs_arg]
        rri = rri[qrs_arg]

        if qrs.shape[0] == 0:
            return empty
        elif qrs.shape[0] == 1:
            return empty

        cc = np.corrcoef(qrs)

        try:
            filt = cc.mean(axis=1) > 0.85  # was 0.85
        except:
            print(cc)
            print(type(cc))
            print(qrs)
            return empty

        qrs_filt = qrs[filt]

        if qrs_filt.shape[0] == 0:
            return empty
        elif qrs_filt.shape[0] == 1:
            return empty

        return self.return_qrs(qrs_filt, rri, "normal", aggregate=aggregate)

    def calc_embed_sep(self, ecg, ecg_labeled, leads=12, ecg_str='ecg12', date_str="Date", pi_str="PatientID", aggregate=False):
        max_ind = 0
        ecg_no_ag = pd.DataFrame(columns=ecg_labeled.columns)

        for j in tqdm.tqdm(ecg_labeled.index):
            qrs_stds = np.zeros([leads, 400])
            rri_means = np.zeros(leads)
            rri_stds = np.zeros(leads)
            labels = []
            sigs = ecg.loc[j, ecg_str]
            pi = ecg.loc[j, pi_str]
            date = ecg.loc[j, date_str]

            if aggregate:
                qrs_means = np.zeros([leads, 400])
                qrs_medians = np.zeros([leads, 400])

            if leads == 1:
                qrs, qrs_std, rri_mean, rri_std, label = self.embed_sig2(
                    np.squeeze(sigs), fs=300, aggregate=aggregate)
                if qrs.ndim == 2:
                    qrs_len = qrs.shape[0]
                else:
                    qrs_len = 1
                for k in range(int(qrs_len)):
                    s = pd.Series([pi, date, qrs[k], qrs_std, rri_mean,
                                  rri_std, label], index=ecg_labeled.columns.values)
                    ecg_labeled = ecg_labeled.append(s, ignore_index=True)

            else:
                if sigs.shape[1] != leads:
                    continue
                qrss = []
                qrs_len = np.zeros(12)
                for i in range(leads):
                    sig = sigs[:, i]
                    if aggregate:
                        qrs_med, qrs_mean, qrs_std, rri_mean, rri_std, label = self.embed_sig2(
                            sig, fs=ecg.loc[j, 'fs'], aggregate=aggregate)
                        qrs_means[i] = qrs_mean
                        qrs_medians[i] = qrs_med
                        qrs_len[i] = 1
                    else:
                        qrs, qrs_std, rri_mean, rri_std, label = self.embed_sig2(
                            sig, fs=ecg.loc[j, 'fs'], aggregate=aggregate)
                        if label == 'empty':
                            break
                        qrss.append(qrs)
                        if qrs.ndim == 2:
                            qrs_len[i] = qrs.shape[0]
                        else:
                            qrs_len[i] = 1
                    qrs_stds[i] = qrs_std
                    rri_means[i] = rri_mean
                    rri_stds[i] = rri_std
                    labels.append(label)
                if (qrs_len == qrs_len[0]).all() and not aggregate and label != 'empty':
                    qrss = np.stack(qrss)
                    for k in range(int(qrs_len[0]))[:20]:
                        max_ind += 1
                        new_row = pd.DataFrame(
                            {pi_str: pi, date_str: date}, index=[0])
                        for (col, data) in [('qrs', qrss[:, k]), ('std_qrs', qrs_stds), ('mean_rri', rri_means), ('std_rri', rri_stds), ('labels', labels)]:
                            new_row[col] = ""
                            new_row[col] = new_row[col].astype(object)
                            new_row.at[0, col] = data
                        ecg_no_ag = pd.concat(
                            [new_row, ecg_no_ag.loc[:]]).reset_index(drop=True)

                        # for data, column in [(pi, pi_str), (date, date_str), (qrss[:,k],'qrs'), (qrs_stds,'std_qrs'), (rri_means, 'mean_rri'), (rri_stds, 'std_rri'), (labels, 'labels')]:

                        #     if column == pi_str or column == date_str:
                        #         print(column)
                        #     else:
                        #         print(column)
                        #         print(data.shape)
                        #         print(ecg_labeled.loc[column].type())
                        #     ecg_labeled.at[max_ind, column] = data
                elif aggregate:
                    for data, column in [(qrs_medians, 'median_qrs'), (qrs_means, 'qrs'), (qrs_stds, 'std_qrs'), (rri_means, 'mean_rri'), (rri_stds, 'std_rri'), (labels, 'labels')]:
                        ecg_labeled.at[j, column] = data
        if aggregate:
            return ecg_labeled
        else:
            return ecg_no_ag

    def calc_embed_file(self, ecg, aggregate=True, leads=12, filename='ecg_withings_embed_sep.p'):
        print('Reading pickle file......')
        ecg.reset_index(drop=True, inplace=True)
        ecg_labeled = ecg[['PatientID', 'Date']].copy()
        if aggregate:
            columns = ['median_qrs', 'qrs',
                       'std_qrs', 'mean_rri', 'std_rri', 'labels']
        else:
            columns = ['qrs', 'std_qrs', 'mean_rri', 'std_rri', 'labels']

        for column in columns:
            ecg_labeled[column] = ""
            ecg_labeled[column] = ecg_labeled[column].astype(object)

        ecg_labeled = self.calc_embed_sep(
            ecg, ecg_labeled, leads=leads, aggregate=aggregate)
        ecg_labeled.to_pickle('output/'+filename)

    def run(self, withings=False, aggregate=True, _range=10):

        if withings:
            ecg = pd.read_pickle("vae-for-ecg/data/withingsECG.p")
            self.calc_embed_file(ecg, aggregate=aggregate, leads=1)

        else:
            for k in [8]:
                # for k in [8]:
                for r in [1, 2]:
                    # for r in [2]:
                    ecg = pd.read_pickle(
                        'ecg_in/ecg12_'+str(k) + str(r) + '.p')
                    self.calc_embed_file(
                        ecg, aggregate=aggregate, filename='ecg12_400_'+str(k)+str(r)+'_embed_sep.p')


if __name__ == '__main__':
    preprocessor = ECG_Preprocessor(fs=300)
    preprocessor.run(withings=False, aggregate=False)
