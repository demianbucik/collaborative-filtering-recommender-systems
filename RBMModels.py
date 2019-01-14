from collections import defaultdict
import math
import numpy as np
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class ScaleException(Exception):
    def __init__(self, name):
        super().__init__("Rescale %s ratings to interval [0, 1]." % name)

class AERBMModel:
    """
    Collaborative filtering recommender model used to predict rating / views / clicks...
    It consists of a mixture of restricted Boltzmann machine (RBM) model: contractive divergence learning algorithm,
    and shallow autoencoder: we optimize for reconstructions over non-missing rating, without sampling in training / prediction.
    In RBM 1 training example = 1 user ratings profile.
    """
    def __init__(self, lrate_W=0.01, lrate_b_v=0.01, lrate_b_h=0.01, reg_W=0.001, reg_b_v=0.001, reg_b_h=0.001,
                 n_components=20, lrate_decay=1., momentum=0.9, batch_size=100, gibbs_steps=3, n_epochs=40,
                 verbose=True, seed=None):
        """
        Initializes the model parameters
        :param lrate_W: Learning rate for W matrix.
        :param lrate_b_v: Learning rate for b_v visible biases vector.
        :param lrate_b_h: Learning rate for b_h hidden biases vector.
        :param reg_W:
        :param reg_b_v: Regularization parameter for b_v.
        :param reg_b_h: Regularization parameter for b_h.
        :param n_components: Number of hidden units / components.
        :param lrate_decay: Exponential learning rate decay.
        :param momentum: Momentum parameter for mini-batch gradient ascent.
        :param batch_size: Batch size for mini-batch gradient ascent.
        :param gibbs_steps: Number of Gibbs probabilities calculating steps in training and prediction.
        :param n_epochs: Maximum number of epochs.
        :param verbose: Whether to print information while training.
        :param seed: random state used.
        """
        # ratings are real values between 0 and 1
        # K is np.ndarray of shape = [nusers, nitems]
        self.lrate_W = lrate_W
        self.lrate_b_v = lrate_b_v
        self.lrate_b_h = lrate_b_h
        self.reg_W = reg_W
        self.reg_b_v = reg_b_v
        self.reg_b_h = reg_b_h
        self.n_components = n_components
        self.lrate_decay = lrate_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.gibbs_steps = gibbs_steps
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed

    def rmse(self, index, non_zero=None, W=None, b_v=None, b_h=None):
        """
        Calculates RMSE with given model parameters for given users
        :param index: User index to calculate RMSE with.
        :param non_zero: Number of non_zero entries in self.R[index], will be calculated if not given.
        :param W: RBM model parameters.
        :param b_v: RBM model parameters.
        :param b_h: RBM model parameters.
        :return: RMSE
        """
        if non_zero is None:
            non_zero = self.R_bin[index].getnnz()
        r_all = self.predict(self.R[index], W=W, b_v=b_v, b_h=b_h)
        r_all = self.R_bin[index].multiply(r_all)
        r_diff = self.R[index] - r_all
        r_diff.data **= 2
        rmse = np.sqrt(r_diff.sum() / non_zero)
        return rmse

    def _logistic(self, x):
        """
        Activation function for hidden and visible RBM units.
        :param x: input matrix/vector/number.
        :return: Logistic function applied to x element-wise.
        """
        return 1 / (1 + np.exp(-x))

    def _sample_hiddens(self, ph):
        """
        Sample h ~ p(h|v)
        Included for completeness, not used in this particular RBM model.
        :param ph: Probabilities of hiddens given visibles = p(h|v)
        :return: samples from p(h|v)
        """
        rand = self.rng.random_sample(size=ph.shape)
        return (rand < ph).astype(int)

    def _sample_visibles(self, pv):
        """
        Sample v ~ p(v|h)
        Included for completeness, not used in this particular RBM model.
        :param pv: Probabilities of visibles given hiddens = p(v|h)
        :return: samples from p(v|h)
        """
        rand = self.rng.random_sample(size=pv.shape)
        return (rand < pv).astype(int)

    def _prob_hiddens(self, v, W=None, b_h=None):
        """
        Calculates probabilities of hiddens given visibles = p(h|v).
        :param v: Values (or probabilities) of hidden units, sparse or numpy array.
        :param W: RBM model parameters.
        :param b_h: RBM model parameters.
        :return: p(h|v)
        """
        # p(h|v)
        # Probabilities of h given v
        if W is None:
            W = self.W
        if b_h is None:
            b_h = self.b_h
        # W.shape = (nitems, F)
        # v.shape = (nusers, nitems)
        # ph.shape = (nusers, F)
        #ph = np.einsum("if,ui->uf", W, v)
        ph = v.dot(W)  # Works for sparse and dense v
        ph += b_h
        return self._logistic(ph)

    def _prob_visibles(self, h, W=None, b_v=None):
        """
        Calculates probabilities of visibles given hiddens = p(v|h).
        :param h: Values (or probabilites) of hidden units, numpy array.
        :param W: RBM model parameters.
        :param b_v: RBM model parameters.
        :return: p(v|h)
        """
        if W is None:
            W = self.W
        if b_v is None:
            b_v = self.b_v
        # W.shape = (nitems, F)
        # h.shape = (nusers, F)
        # pv.shape = (nusers, nitems)
        pv = np.einsum("if,uf->ui", W, h)
        pv += b_v
        return self._logistic(pv)

    def _init_b_v(self, train_index, eps_bound=0.0001):
        """
        Initialization for b_v[i,k] = log( p_i / (1-p_i) ) [G. Hinton: A Practical Guide to Training RBMs]
        :param eps_bound: Bound to avoid division with 0.
        :return: b_v initializatin.
        """
        # R.shape = [nusers, nitems]
        # p = np.zeros((self.nitems,))
        #p = np.einsum("ui->i", self.R[self.train_ind])
        p = self.R[train_index].sum(axis=0).toarray()
        p /= p.sum()
        p[p < eps_bound] = eps_bound
        p[p > (1-eps_bound)] = 1 - eps_bound
        b_v = np.log(p / (1-p))
        return b_v

    def _check_R(self, R):
        """
        Check if given R elements are in interval [0, 1] and potentially converts to csc_matrix format
        :param R: Sparse ratings matrix.
        :return: valid R.
        """
        if R.min() < 0 or R.max() > 1:
            raise ScaleException("R")
        if not isinstance(R, sparse.csc_matrix):
            R = R.tocsc()
        return R

    def predict(self, v, W=None, b_v=None, b_h=None):
        """
        Predict all ratings for given user(s) by approximating p(v_all|v)
        :param v: Visible units - users profiles
        :param W: RBM model parameters.
        :param b_v: RBM model parameters.
        :param b_h:  RBM model parameters.
        :return: Predictions r = p(v_all|v)
        """
        # Predict all ratings r = v_all for user v
        # Approximate p(v_all|v)
        # v is a user vector (or matrix of multiple users)
        # If parameters stay None, we use class parameters (self._)
        # v.shape = (nusers, nitems, r_dim)
        # r.shape = (nusers, nitems, )
        v_origshape = v.shape
        if len(v_origshape) == 1:
            v = v.reshape((1, v.shape[0]))
        pv = v
        for t in range(self.gibbs_steps):
            ph = self._prob_hiddens(pv, W, b_h)
            pv = self._prob_visibles(ph, W, b_v)
        if len(v_origshape) == 1:
            pv = pv[0]
        return pv

    def fit(self, R, train_index, val_index=None):
        """
        Fits the RBM parameters to training data in R_train = R[train_index].
        Parameters are estimated by maximizing log posterior probability of parameters given the data.
        W, b_v, b_h = argmax log p(W, b_v, b_h | R_train) = argmax log p(R_train | W, b_v, b_h)
                                                            + lambda_W/2*norm(W) + lambda_b_v*norm(b_v) + lambda_b_h*norm(b_h)
        The likelihood is approximated by contrastive divergence function (G. Hinton) and its gradient is approximated
        with Gibbs steps (positive and negative phase) and used for learning.

        :param R: Sparse ratings matrix with elements in interval [0, 1]. Ideally in csc_matrix format.
        :param train_index: User index used for training.
        :param val_index: User index used for validation, if not given part of the training set will be used.
        :return: self
        """
        self.R = self._check_R(R)
        self.R_bin = (self.R > 0).astype(int)
        self.n_users, self.n_items = self.R.shape
        if val_index is None:
            _, val_index = train_test_split(train_index, test_size=0.25, random_state=self.seed, shuffle=True)
        val_index = val_index
        val_non_zero = self.R[val_index].getnnz()
        # To split training data in batches
        n_splits = self.n_users // self.batch_size
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        # Initialize RBM parameters
        self.W = W = self.rng.normal(loc=0., scale=0.01, size=(self.n_items, self.n_components))
        self.b_v = b_v = self._init_b_v(train_index)
        self.b_h = b_h = np.zeros((self.n_components,))

        # Parameters for optimization with momentum
        last_delta_W = np.zeros((self.n_items, self.n_components))
        last_delta_b_v = np.zeros((self.n_items,))
        last_delta_b_h = np.zeros((self.n_components,))

        rmse = self.rmse(val_index, val_non_zero)
        if self.verbose:
            print("Epoch: %3d\trmse: %f" % (0, rmse))
        train_len = len(train_index)
        lrate_W = self.lrate_W
        lrate_b_v = self.lrate_b_v
        lrate_b_h = self.lrate_b_h
        epoch = 1
        while epoch <= self.n_epochs:
            # Mini-batch gradient ascent
            for _, batch_ind in kf.split(train_index):

                curr_batch_size = len(batch_ind)
                batch_bin = self.R_bin[batch_ind]
                item_sum = batch_bin.sum(axis=0)
                # Handling 0 sums for division
                item_sum[item_sum == 0] = -1
                item_frac = 1 / item_sum  # .shape = (1, nitems)
                item_frac[item_frac < 0] = 0

                pv = pv0 = self.R[batch_ind]
                ph = ph0 = self._prob_hiddens(pv0, W, b_h)
                for t in range(self.gibbs_steps):
                    # Probability of visibles given hiddens
                    pv = self._prob_visibles(ph, W, b_v)
                    # Only reconstruct over non-missing ratings
                    pv = batch_bin.multiply(pv).toarray()
                    # Probability of hiddens given visibles
                    ph = self._prob_hiddens(pv, W, b_h)
                # Collect estimates of <v>, <h>, <v*h>, average over all users / observed ratings
                v_pos = np.einsum("ki,ui->i", item_frac, pv0)
                v_neg = np.einsum("ki,ui->i", item_frac, pv)
                h_pos = np.einsum("uf->f", ph0) / curr_batch_size
                h_neg = np.einsum("uf->f", ph) / curr_batch_size
                vh_pos = np.einsum("i,f->if", v_pos, h_pos)
                vh_neg = np.einsum("i,f->if", v_neg, h_neg)
                # Define parameter updates with momentum and regularization
                delta_b_v = (1 - self.momentum) * (v_pos - v_neg - self.reg_b_v * b_v) + self.momentum * last_delta_b_v
                delta_b_h = (1 - self.momentum) * (h_pos - h_neg - self.reg_b_h * b_h) + self.momentum * last_delta_b_h
                delta_W = (1 - self.momentum) * (vh_pos - vh_neg - self.reg_W * W) + self.momentum * last_delta_W
                last_delta_b_v = delta_b_v
                last_delta_b_h = delta_b_h
                last_delta_W = delta_W
                # Update parameters: mini-batch gradient ascent
                b_v += (self.lrate_b_v * curr_batch_size / train_len) * delta_b_v
                b_h += (self.lrate_b_h * curr_batch_size / train_len) * delta_b_h
                W += (self.lrate_W * curr_batch_size / train_len) * delta_W
            last_rmse = rmse
            rmse = self.rmse(val_index, val_non_zero)
            if self.verbose:
                print("Epoch: %3d\trmse: %f" % (epoch, rmse))
            if rmse > last_rmse:
                break
            self.W = W
            self.b_v = b_v
            self.b_h = b_h
            lrate_W *= self.lrate_decay
            lrate_b_v *= self.lrate_decay
            lrate_b_h *= self.lrate_decay
            epoch += 1
        return self

class ConditionalRBMModel:
    # Work in progress.
    # Calculating probabilities while conditioning on current user's friends.
    def __init(self):
        pass