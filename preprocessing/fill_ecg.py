import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path


def load_data():

    df_mortality = pd.read_pickle("../data/df_1y_mortality.p")
    df_mortality["ecg"] = ""
    df_mortality["ecg"] = df_mortality["ecg"].astype(object)

    for ecg_path in ["ecg12_400_all_embed2.p", "ecg12_400_all_embed2_noisy.p"]:
        path = os.path.join(Path(os.getcwd()).parent, ecg_path)
        ecg = pd.read_pickle(path)
        ecg.reset_index(drop=True, inplace=True)

        for pi in df_mortality.index:
            date = df_mortality.loc[pi, "Cath date"]
            if pi in ecg.index.values:
                if date in ecg["Date"].values:
                    mean_qrs = ecg.query("(PatientID == @pi) and (Date == @date)")[
                        "mean qrs"
                    ].values
                    df_mortality.loc[pi, "ecg"] = mean_qrs

    return df_mortality


if __name__ == "__main__":

    df_m = load_data()
    df_m.to_pickle("data\df_1y_mortality_ecg.p")
