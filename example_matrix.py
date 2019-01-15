from collections import defaultdict
import statistics as stat
import numpy as np
import pickle
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

from MatrixModels import Identity, Logistic, ReLU, LeakyReLU
from MatrixModels import MatrixModel
from MatrixModels import MatrixModelWeighted
from MatrixModels import MatrixModelFriends

# Load preprocessed sparse ratings matrix from lastfm dataset, values in interval [0, 1]
with open('R_lastfm.pickle', 'rb') as handle:
    R = pickle.load(handle)
# Load preprocessed sparse friendship matrix, Fr[u, v] = 1 iff u and v are friends
with open('Fr_lastfm.pickle', 'rb') as handle:
    Fr = pickle.load(handle)


test_rmse = []

lrate = 0.01
reg = 0.02
reg_b = 0.01
lrate_decay = 0.94
n_factors = 9
seed = 2
split_size = 0.25

R = R.tocoo()
Fr = Fr.tocoo()

kf = KFold(n_splits=4, random_state=seed, shuffle=True)
# Each rating is one training example
for train_val_ind, test_ind in kf.split(R.data):
    R_train_val = sparse.coo_matrix((R.data[train_val_ind], (R.row[train_val_ind], R.col[train_val_ind])), shape=R.shape)
    R_test = sparse.coo_matrix((R.data[test_ind], (R.row[test_ind], R.col[test_ind])), shape=R.shape)

    ss = ShuffleSplit(n_splits=1, test_size=split_size, random_state=seed)
    train_ind, val_ind = next(ss.split(R_train_val.data))
    R_train = sparse.coo_matrix((R_train_val.data[train_ind], (R_train_val.row[train_ind], R_train_val.col[train_ind])), shape=R.shape)
    R_val = sparse.coo_matrix((R_train_val.data[val_ind], (R_train_val.row[val_ind], R_train_val.col[val_ind])), shape=R.shape)

    # Basic matrix model
    #"""
    mm = MatrixModel(lrate=lrate, reg=reg, reg_b=reg_b, activation=Identity(), n_factors=n_factors,
                     lrate_decay=lrate_decay, seed=seed, verbose=True)
    mm = mm.fit(R_train, R_val)
    #"""

    # Matrix model with weights on latent factors
    """
    mm = MatrixModelWeighted(lrate=lrate, reg=reg, reg_b=reg_b, activation=Identity(), n_factors=n_factors,
                             lrate_decay=lrate_decay, seed=seed, verbose=True)
    mm = mm.fit(R_train, R_val)
    """

    # Matrix model using user's friends network to improve predictions and potential weights on latent factors
    """
    mm = MatrixModelFriends(lrate=lrate, reg=reg_b, reg_b=reg, activation=Identity(), n_factors=n_factors,
                            lrate_decay=lrate_decay, seed=seed, verbose=True, use_weights=False)
    mm = mm.fit(R_train, Fr, R_val)
    """
    #print(mm.w)

    # Prediction for potential user and item
    # prediction = mm.predict(u_new, i_new)

    # Calculate RMSE on test data
    rmse = mm.rmse(R_test)
    test_rmse.append(rmse)
    print("test\t\trmse: %f" % rmse)
print("4-fold mean rmse: %f" % stat.mean(test_rmse))


