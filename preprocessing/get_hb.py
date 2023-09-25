import numpy as np
import numpy.ma as ma
import scipy.signal as ssig
from ecg_RPNet import detector_RP


def split_hb2(x_ecg, first=True):
    Detector = detector_RP.Detector_RPNet()
    ecg, peaks, y = Detector.predict(x_ecg.reshape(1, -1), calibrate=True)

    xs = np.zeros([len(peaks[0]) - 2, 225])

    for i, peak in enumerate(peaks[0][1:-1]):
        qrs = x_ecg[peak - 75 : peak + 150]
        xs[i] = qrs

    rri = peaks[0][1:] - peaks[0][:-1]
    begin = peaks[0][0]
    end = 5000 - sum(rri) - begin
    # mean_qrs = np.mean(xs, axis=0)
    # std_qrs = np.std(xs, axis=0)
    return xs, rri, [begin, end]


def embed_sig2(sig_unfilt, fs=300):
    filt = ssig.firwin(
        len(sig_unfilt),
        cutoff=[0.5, 45],
        window="blackmanharris",
        pass_zero="bandpass",
        fs=fs,
    )
    sig = ssig.convolve(sig_unfilt, filt, mode="same", method="auto")

    sig1 = sig[:5000]
    sig2 = sig[-5000:]

    qrs1, rri1, ex1 = split_hb2(sig1)
    qrs2, rri2, ex2 = split_hb2(sig2)

    count = ex2[1] + 10
    i = 0

    while count < (len(sig) - 5000 + ex1[1]):
        i += 1
        count += rri2[-i]

    qrs = np.concatenate([qrs1[3:], qrs2[-i:]])
    rri = np.concatenate([rri1[4:], rri2[-i:]])

    cc = np.corrcoef(qrs)

    filt1 = cc.mean(axis=1) > 0.85
    filt2 = ma.masked_array(cc, np.identity(cc.shape[0])).max(axis=1) > 0.90

    qrs_filt = qrs[np.logical_or(filt1, filt2)]

    if cc.mean() > 0.8:
        label = "normal"
    elif sum(filt1) / len(filt1) < 0.5:
        if sum(filt2) / len(filt2) == 0:
            label = "too noisy"
            return np.zeros(225), np.zeros(225), 0, 0, label
        if sum(filt2) / len(filt2) < 0.9:
            label = "noisy"
        else:
            label = "multi rhythm"
    else:
        label = "normal2"

    return (
        np.mean(qrs_filt, axis=0),
        np.std(qrs_filt, axis=0),
        np.mean(rri),
        np.std(rri),
        label,
    )


def run():
    # ecg = pd.read_csv('data\withingsECG.csv').dropna(how='any')
    for j in tqdm.tqdm(ecg.index[:10]):
        sig = ecg.loc[j, "SIGNAL"]
        qrs_mean, qrs_std, rri_mean, rri_std, label = embed_sig2(sig)
        ecg.at[j, "mean qrs"] = qrs_mean
        ecg.at[j, "std qrs"] = qrs_std
        ecg.at[j, "mean rri"] = rri_mean
        ecg.at[j, "std rri"] = rri_std

    # ecg.to_pickle('data\ecg_embedding.p')


if __name__ == "__main__":
    run()
