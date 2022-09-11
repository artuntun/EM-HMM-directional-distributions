import pandas as pd
import numpy as np
import scipy.io as sio
from em_hmm_directional.von_mises_em import vonCV_EM, EM


def main():
    input_path = "data/inputs/"
    output_path = "data/outputs/"
    X = load_fake_data(input_path)
    N, p = X.shape
    K = 6
    data, conditions = load_real_data(input_path)
    Xr = normalize_data(data)
    # phi_EM,ll = EM(Xr,K)
    K_list = [2, 3]
    df = vonCV_EM(Xr[1:, :], K_list, conditions[1:], output_path)
    print(df)


def load_real_data(path="data/", file_name="face_scrambling_ERP.mat"):
    mat = sio.loadmat(path + file_name)
    data = mat["X_erp"]
    conditions = [x[0] for x in mat["CONDITIONS"][0]]
    return data, conditions


def load_fake_data(path="data/"):
    from os import listdir

    csv = [x for x in listdir(path) if x.find(".csv") != -1]
    data = [np.array(pd.read_csv(path + X, sep=",")) for X in csv]
    X = np.concatenate(data, axis=0)
    return X


def normalize_subject(data):
    _, N = data.shape
    for i in range(N):
        data[:, i] = data[:, i] / np.linalg.norm(data[:, i])
    return data.T


def normalize_data(data):
    for i, row in enumerate(data):
        for j, x in enumerate(row):
            data[i, j] = normalize_subject(x)
    return data


if __name__ == "__main__":
    main()
