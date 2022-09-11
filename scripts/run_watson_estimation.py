from pathlib import Path
import pandas as pd
import numpy as np

from em_hmm_directional.watson_em import watson_EM

DATA_PATH = Path("./data/inputs")

# get data from 3 Watson distributions
path = DATA_PATH / "clust1.csv"
path2 = DATA_PATH / "clust2.csv"
path3 = DATA_PATH / "clust3.csv"
X1 = np.array(pd.read_csv(path, sep=","))
X2 = np.array(pd.read_csv(path2, sep=","))
X3 = np.array(pd.read_csv(path3, sep=","))
X = np.concatenate((X1, X2, X3[0:1400, :]), axis=0)

print("Estimating parameters for 3 Watson distributions")
mu, kappa, pi = watson_EM(X, 3)
print("Estimated parameters for 3 Watson distributions:")
print("mu = \n", mu)
print("kappa = \n", kappa)
print("pi = \n", pi)
