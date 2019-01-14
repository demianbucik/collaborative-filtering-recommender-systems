from abc import abstractmethod
from collections import defaultdict
import numpy as np
from statistics import median
from scipy import sparse

class Logistic:
    """
    Activation function to be used with MatrixModels.
    Clip the evaluation output to interval [0, 1].

    """
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def d(self, g):
        #g = self(x)
        return g * (1 - g)

    def inv(self, g):
        return np.log(g / (1 - g))

class ReLU:
    """
    Activation function to be used with MatrixModels.
    Clip the evaluation output to interval [0, 1].

    """
    def __call__(self, x):
        g = np.maximum(0, x)
        g = np.minimum(g, 1)
        return g

    def d(self, g):
        if g == 0 or g == 1:
            return 0
        else:
            return 1

    def inv(self, g):
        return g

class LeakyReLU:
    """
    Activation function to be used with MatrixModels.
    Clip the evaluation output to interval [0, 1].

    """
    def __call__(self, x):
        g = np.maximum(0, x)
        g = np.minimum(g, 1)
        return g

    def d(self, g):
        if g == 0 or g == 1:
            return 0.01
        else:
            return 1

    def inv(self, g):
        return g

class Identity:
    """
    Activation function to be used with MatrixModels.
    Clip the evaluation output to interval [0, 1].

    """
    def __call__(self, x):
        g = np.maximum(0, x)
        g = np.minimum(g, 1)
        return g

    def d(self, g):
        return 1

    def inv(self, g):
        return g

class ScaleException(Exception):
    def __init__(self, name):
        super().__init__("Rescale %s ratings to interval [0, 1]." % name)


class MatrixModelBase:
    """
    Base class matrix models inherit from.

    """
    def __init__(self, lrate=0.01, reg_b=0.01, reg=0.02, activation=Identity(), n_epochs=35, n_factors=9, lrate_decay=0.92, seed=None, verbose=True):
        """
        Initializes the model.

        :param lrate: Learning rate.
        :param reg_b: Regularization parameter for biases.
        :param reg: Regularization parameter for user and item matrices.
        :param activation: Used when predicting ratings.
        :param n_epochs: Maximum number of epochs (training iterations).
        :param n_factors: Dimensionality of latent embedding space (user P matrix and item Q matrix).
        :param lrate_decay: Exponential learning rate decay.
        :param seed: Random state.
        :param verbose: Print information while training.
        """
        self.lrate = lrate
        self.reg_b = reg_b
        self.reg = reg
        self.g = activation
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.lrate_decay = lrate_decay
        self.rng = np.random.seed(seed)
        self.verbose = verbose

    def _known(self, what, index):
        """
        Returns whether user or item index is known.

        :param what: 'user' or 'item'.
        :param index: Index to check.
        :return: Boolean.
        """
        if what == 'user':
            known = 0 <= index < self.n_users
        else:  # what == 'item'
            known = 0 <= index < self.n_items
        return known

    def rmse(self, R, *args, **kwargs):
        """
        Calculates root mean squared error for ratings in R
        If no additional arguments are passed, predict function will use parameters from self.

        :param R: Sparse coo_matrix of ratings used to calculate RMSE.
        :param args: Parameters passed to the predict function.
        :param kwargs: Parameters passed to the predict function.
        :return: RMSE.
        """
        rmse = 0
        for (u, i, r) in zip(R.row, R.col, R.data):
            r_hat = self.predict(u, i, *args, **kwargs)
            rmse += (r - r_hat) ** 2
        rmse = np.sqrt(rmse / R.getnnz())
        return rmse

    def fit(self, R, R_val):
        """
        Base fitting function.

        :param R: Sparse training ratings matrix in coo_format.
        :param R_val: Validation set for early stopping. If not given the training data is used.
        :return: Nothing. Next fit function returns self.
        """
        if R.min() < 0 or R.max() > 1:
            raise ScaleException("R")
        if not isinstance(R, sparse.coo_matrix):
            R = R.tocoo()
        if R_val is None:
            R_val = R
        elif R_val.min() < 0 or R_val.max() > 1:
            raise ScaleException("R_val")
        elif not isinstance(R_val, sparse.coo_matrix):
            R_val = R_val.tocoo()
        self.R = R
        self.R_val = R_val
        self.mean = self.g.inv(np.mean(R.data))
        self.n_users = len(self.R.row)
        self.n_items = len(self.R.col)
        return

    @abstractmethod
    def predict(self, u, i, *args, **kwargs):
        pass


class MatrixModel(MatrixModelBase):
    """
    Basic matrix factorization model.

    Predicts rating rui as
        rui_hat = g(mean + user_bias[u] + item_bias[i] + P[u, :].dot(Q[:, i]))

    where   P and Q are learned user and item embeddings,
            user_bias and item_bias are learned user/item biases,
            mean is calculated from data as g_inverse(mean(rui)),
            g() is an activation function: Identity, Logistic, ReLU, LeakyReLU,
                or custom class that implements evaluation, inverse, derivative (input g(x), output g'(x)),
                all activation functions (including Identity) clip evaluation output to interval [0, 1].

    Parameters are set by optimizing J(P, Q, user_bias, item_bias) with stochastic gradient descent.
    J = sum((rui - rui_hat)**2) + reg * (norm(P) + norm(Q)) + reg_b * (norm(user_bias) + norm(item_bias))

    where   norm() is 2-norm for vectors and Frobenius norm for matrices,
            reg and reg_b are regularization parameters.
    """
    def __init__(self, lrate=0.01, reg_b=0.01, reg=0.02, activation=Identity(), n_epochs=35, n_factors=9, lrate_decay=0.92, seed=None, verbose=True):
        """
        Initializes the model.


        :param lrate: Learning rate.
        :param reg_b: Regularization parameter for biases.
        :param reg: Regularization parameter for user and item matrices.
        :param activation: Used when predicting ratings.
        :param n_epochs: Maximum number of epochs (training iterations).
        :param n_factors: Dimensionality of latent embedding space (user P matrix and item Q matrix).
        :param lrate_decay: Exponential learning rate decay.
        :param seed: Random state.
        :param verbose: Print information while training.
        """
        super().__init__(lrate, reg_b, reg, activation, n_epochs, n_factors, lrate_decay, seed, verbose)

    def fit(self, R, R_val=None):
        """
        Main function to fit the model to the data in self.R.
        Initial matrix P, item matrix Q sampled from Gaussian, biases are initialized with 0 values.
        Uses early stopping: terminates learning process when RMSE on set R_val increases.

        :param R: Sparse training ratings matrix in coo_format.
        :param R_val: Validation set for early stopping. If not given the training data is used.
        :return: Returns object with fitted parameters (self).
        """
        super().fit(R, R_val)
        P = self.rng.normal(loc=0, scale=0.01, size=(self.n_users, self.n_factors))
        Q = self.rng.normal(loc=0, scale=0.01, size=(self.n_factors, self.n_items))
        user_bias = np.zeros(self.n_users)
        item_bias = np.zeros(self.n_items)

        rmse = self.rmse(R_val, P, Q, user_bias, item_bias)
        if self.verbose:
            print("Epoch: %3d\trmse: %f" % (0, rmse))
        lrate = self.lrate
        epoch = 1
        while epoch <= self.n_epochs:
            # Training with stochastic gradient descent (without sampling training examples)
            for (u, i, r) in zip(self.R.row, self.R.col, self.R.data):
                r_hat = self.predict(u, i, P, Q, user_bias, item_bias)
                eui = (r - r_hat) * self.g.d(r_hat)  # derivative of activation function
                P[u, :] += lrate * (eui * Q[:, i] - self.reg * P[u, :])
                Q[:, i] += lrate * (eui * P[u, :] - self.reg * Q[:, i])
                user_bias[u] += lrate * (eui - self.reg_b * user_bias[u])
                item_bias[i] += lrate * (eui - self.reg_b * item_bias[i])
            last_rmse = rmse
            rmse = self.rmse(R_val, P, Q, user_bias, item_bias)
            if self.verbose:
                print("Epoch: %3d\trmse: %f" % (epoch, rmse))
            if last_rmse < rmse:
                break
            self.P = P
            self.Q = Q
            self.user_bias = user_bias
            self.item_bias = item_bias
            lrate *= self.lrate_decay
            epoch += 1
        # Set P and Q profiles to zero for unknown users and items
        unknown_users = list(set(self.R_val.row) - set(self.R.row))
        unknown_items = list(set(self.R_val.col) - set(self.R.col))
        self.P[unknown_users, :] = 0.
        self.Q[:, unknown_items] = 0.
        return self

    def predict(self, u, i, P=None, Q=None, user_bias=None, item_bias=None):
        """
        Predict the rating user u gives to item i.

        :param u: User index.
        :param i: Item index.
        :param P: User embedding matrix.
        :param Q: Item embedding matrix.
        :param user_bias: User biases vector.
        :param item_bias: Item biases vector.
        :return: Predicted rating.
        """
        if P is None:
            P = self.P
        if Q is None:
            Q = self.Q
        if user_bias is None:
            user_bias = self.user_bias
        if item_bias is None:
            item_bias = self.item_bias

        known_user = self._known('user', u)
        known_item = self._known('item', i)
        rui_hat = self.mean
        if known_user:
            rui_hat += user_bias[u]
        if known_item:
            rui_hat += item_bias[i]
        if known_user and known_item:
            rui_hat += P[u, :].dot(Q[:, i])
        # Apply potential non-linearity (activation) g
        rui_hat = self.g(rui_hat)
        return rui_hat


class MatrixModelWeighted(MatrixModelBase):
    """
    Weighted matrix factorization model.

    Predicts rating rui as
        rui_hat = g(mean + user_bias[u] + item_bias[i] + (w * P[u, :]).dot(Q[:, i]))

    where   P and Q are learned user and item embeddings,
            w is learned weight vector, * is Hadamard (element-wise) product,
            user_bias and item_bias are learned user/item biases,
            mean is calculated from data as g_inverse(mean(rui)),
            g() is an activation function: Identity, Logistic, ReLU, LeakyReLU,
                or custom class that implements evaluation, inverse, derivative (input g(x), output g'(x)),
                all activation functions (including Identity) clip evaluation output to interval [0, 1].

    Parameters are set by optimizing J(P, Q, w, user_bias, item_bias) with stochastic gradient descent.
    J = sum((rui - rui_hat)**2) + reg * (norm(P) + norm(Q) + norm(w)) + reg_b * (norm(user_bias) + norm(item_bias))

    where   norm() is 2-norm for vectors and Frobenius norm for matrices,
            reg and reg_b are regularization parameters.
    """
    def __init__(self, lrate=0.01, reg_b=0.01, reg=0.02, activation=Identity(), n_epochs=35, n_factors=9, lrate_decay=0.92, seed=None, verbose=True):
        """
        Initializes the model.


        :param lrate: Learning rate.
        :param reg_b: Regularization parameter for biases.
        :param reg: Regularization parameter for user, item matrices and w vector.
        :param activation: Used when predicting ratings.
        :param n_epochs: Maximum number of epochs (training iterations).
        :param n_factors: Dimensionality of latent embedding space (user P matrix and item Q matrix).
        :param lrate_decay: Exponential learning rate decay.
        :param seed: Random state.
        :param verbose: Print information while training.
        """
        super().__init__(lrate, reg_b, reg, activation, n_epochs, n_factors, lrate_decay, seed, verbose)

    def fit(self, R, R_val=None):
        """
        Main function to fit the model to the data in self.R.
        Initial matrix P, item matrix Q sampled from Gaussian, biases are initialized with 0 values.
        Uses early stopping: terminates learning process when RMSE on set R_val increases.

        :param R: Sparse training ratings matrix in coo_format.
        :param R_val: Validation set for early stopping. If not given the training data is used.
        :return: Returns object with fitted parameters (self).
        """
        super().fit(R, R_val)
        P = self.rng.normal(loc=0, scale=0.01, size=(self.n_users, self.n_factors))
        Q = self.rng.normal(loc=0, scale=0.01, size=(self.n_factors, self.n_items))
        w = 0.01 * np.ones(self.n_factors)
        user_bias = np.zeros(self.n_users)
        item_bias = np.zeros(self.n_items)

        rmse = self.rmse(R_val, P, Q, w, user_bias, item_bias)
        if self.verbose:
            print("Epoch: %3d\trmse: %f" % (0, rmse))
        lrate = self.lrate
        epoch = 1
        while epoch <= self.n_epochs:
            # Training with stochastic gradient descent (without sampling training examples)
            for (u, i, r) in zip(self.R.row, self.R.col, self.R.data):
                r_hat = self.predict(u, i, P, Q, w, user_bias, item_bias)
                eui = (r - r_hat) * self.g.d(r_hat)  # derivative of activation function
                P[u, :] += lrate * (eui * w * Q[:, i] - self.reg * P[u, :])
                Q[:, i] += lrate * (eui * w * P[u, :] - self.reg * Q[:, i])
                w += lrate * (eui * (P[u, :] * Q[:, i]) - self.reg * w)
                user_bias[u] += lrate * (eui - self.reg_b * user_bias[u])
                item_bias[i] += lrate * (eui - self.reg_b * item_bias[i])
            last_rmse = rmse
            rmse = self.rmse(R_val, P, Q, w, user_bias, item_bias)
            if self.verbose:
                print("Epoch: %3d\trmse: %f" % (epoch, rmse))
            if last_rmse < rmse:
                break
            self.P = P
            self.Q = Q
            self.w = w
            self.user_bias = user_bias
            self.item_bias = item_bias
            lrate *= self.lrate_decay
            epoch += 1
        # Set P and Q profiles to zero for unknown users and items
        unknown_users = list(set(self.R_val.row) - set(self.R.row))
        unknown_items = list(set(self.R_val.col) - set(self.R.col))
        self.P[unknown_users, :] = 0.
        self.Q[:, unknown_items] = 0.
        return self

    def predict(self, u, i, P=None, Q=None, w=None, user_bias=None, item_bias=None):
        """
        Predict the rating user u gives to item i.

        :param u: User index.
        :param i: Item index.
        :param P: User embedding matrix.
        :param Q: Item embedding matrix.
        :param w: Latent factor weights.
        :param user_bias: User biases vector.
        :param item_bias: Item biases vector.
        :return: Predicted rating.
        """
        if P is None:
            P = self.P
        if Q is None:
            Q = self.Q
        if user_bias is None:
            user_bias = self.user_bias
        if item_bias is None:
            item_bias = self.item_bias
        if w is None:
            w = self.w

        known_user = self._known('user', u)
        known_item = self._known('item', i)
        rui_hat = self.mean
        if known_user:
            rui_hat += user_bias[u]
        if known_item:
            rui_hat += item_bias[i]
        if known_user and known_item:
            rui_hat += (w * P[u, :]).dot(Q[:, i])
        # Apply potential non-linearity g
        rui_hat = self.g(rui_hat)
        return rui_hat


class MatrixModelFriends(MatrixModelBase):
    """
    Matrix factorization model that uses user's friends network to improve predictions.
    Also has an option to use weights like in previous model.

    Predicts rating rui as
        rui_hat = g(mean + user_bias[u] + item_bias[i] + (sum(F[j, :])/sqrt(|Fu|) + w * P[u, :]).dot(Q[:, i]))

    where   P and Q are learned user and item embeddings,
            w is learned weight vector, * is Hadamard (element-wise) product, used only if use_weights=True,
            F are learned friends embeddings, added to P[u, :] for all friends j of user u,
            |Fu| is number of friends user u has,
            user_bias and item_bias are learned user/item biases,
            mean is calculated from data as g_inverse(mean(rui)),
            g() is an activation function: Identity, Logistic, ReLU, LeakyReLU,
                or custom class that implements evaluation, inverse, derivative (input g(x), output g'(x)),
                all activation functions (including Identity) clip evaluation output to interval [0, 1].

    Parameters are set by optimizing J(P, Q, F, wuser_bias, item_bias) with stochastic gradient descent.
    J = sum((rui - rui_hat)**2) + reg * (norm(P) + norm(Q) + norm(F) + norm(w)) + reg_b * (norm(user_bias) + norm(item_bias))

    where   norm() is 2-norm for vectors and Frobenius norm for matrices,
            reg and reg_b are regularization parameters.
    """

    def __init__(self, lrate=0.01, reg_b=0.01, reg=0.02, activation=Identity(), n_epochs=35, n_factors=9, lrate_decay=0.92, use_weights=False, seed=None, verbose=True):
        """
        Initializes the model.

        :param lrate: Learning rate.
        :param reg_b: Regularization parameter for biases.
        :param reg: Regularization parameter for user, item, implicit (friends) matrices.
        :param activation: Used when predicting ratings.
        :param n_epochs: Maximum number of epochs (training iterations).
        :param n_factors: Dimensionality of latent embedding space (user P matrix and item Q matrix).
        :param lrate_decay: Exponential learning rate decay.
        :param use_weights: Whether the algorithm should use weight vector w when predicting ratings
        :param seed: Random state.
        :param verbose: Print information while training.
        """
        super().__init__(lrate, reg_b, reg, activation, n_epochs, n_factors, lrate_decay, seed, verbose)
        self.use_weights = use_weights

    def _init_Fr(self, Fr):
        """
        Initialize friends Fr as defaultdict(lambda: np.array([])) from sparse binary relation matrix:
        {u: np.array[v1, ..., vn]), ...} where v1, ..., vn are indices of user u's friends.
        Used in training for faster indexing.

        :param Fr: Sparse matrix where Fr[u, v] = 1 iff u and v are friends.
        :return: defaultdict Fr initialization.
        """
        if not isinstance(Fr, sparse.coo_matrix):
            Fr = Fr.tocoo()
        if isinstance(Fr, sparse.coo_matrix):
            d = defaultdict(list)
            for (u, v) in zip(Fr.row, Fr.col):
                d[u].append(v)
            Fr = defaultdict(lambda: np.array([]))
            for u in d.keys():
                Fr[u] = np.array(d[u])
        # Check if wrong format was passed
        if not isinstance(Fr, defaultdict):
            raise ValueError("Wrong format of argument Fr was passed. Check fit or _init_Fr function docstring.")
        return Fr

    def fit(self, R, Fr, R_val=None):
        """
        Main function to fit the model to the data in self.R.
        Initial matrix P, item matrix Q sampled from Gaussian, biases are initialized with 0 values.
        Uses early stopping: terminates learning process when RMSE on set R_val increases.

        :param R: Sparse training ratings in coo_matrix format.
        :param Fr: User friends network in coo_matrix (or any sparse) format. Fr[u, v] = 1 iff u and v are friends.
                The Fr relation is symmetric. Another accepted input is defaultdict: {u: np.array([v1, ..., vn])}.
        :param R_val: Validation set for early stopping. If not given the training data is used.
        :return: Returns object with fitted parameters (self).
        """
        super().fit(R, R_val)
        self.Fr = self._init_Fr(Fr)

        P = self.rng.normal(loc=0, scale=0.01, size=(self.n_users, self.n_factors))
        Q = self.rng.normal(loc=0, scale=0.01, size=(self.n_factors, self.n_items))
        F = np.zeros((self.n_users, self.n_factors))
        user_bias = np.zeros(self.n_users)
        item_bias = np.zeros(self.n_items)
        w = np.ones(self.n_factors)
        if self.use_weights:
            # Initialization if used in training
            w *= 0.01

        rmse = self.rmse(R_val, P, Q, F, w, user_bias, item_bias)
        if self.verbose:
            print("Epoch: %3d\trmse: %f" % (0, rmse))
        lrate = self.lrate
        epoch = 1
        while epoch <= self.n_epochs:
            # Training with stochastic gradient descent (without sampling training examples)
            for (u, i, r) in zip(self.R.row, self.R.col, self.R.data):
                F_sum = np.sum(self.F[self.Fr[u], :], axis=0)
                F_num = np.sqrt(self.Fr[u].shape[0])  # Try without sqrt
                if F_num > 0:
                    F_sum /= F_num
                r_hat = self.predict(u, i, P, Q, F, w, user_bias, item_bias)
                eui = (r - r_hat) * self.g.d(r_hat)  # Derivative of activation function
                P[u, :] += lrate * (eui * w * Q[:, i] - self.reg * P[u, :])
                Q[:, i] += lrate * (eui * (F_sum + w * P[u, :]) - self.reg * Q[:, i])
                if self.use_weights:
                    w += lrate * (eui * P[u, :] * Q[:, i] - self.reg * w)
                if F_num > 0:
                    F[self.Fr[u], :] += lrate * (eui * Q[:, i] / F_num - self.reg * F[self.Fr[u], :])
                user_bias[u] += lrate * (eui - self.reg_b * user_bias[u])
                item_bias[i] += lrate * (eui - self.reg_b * item_bias[i])
            last_rmse = rmse
            rmse = self.rmse(R_val, P, Q, F, w, user_bias, item_bias)
            if self.verbose:
                print("Epoch: %3d\trmse: %f" % (epoch, rmse))
            if last_rmse < rmse:
                break
            self.P = P
            self.Q = Q
            self.F = F
            self.w = w
            self.user_bias = user_bias
            self.item_bias = item_bias
            lrate *= self.lrate_decay
            epoch += 1
        # Set P and Q profiles to zero for unknown users and items
        unknown_users = list(set(self.R_val.row) - set(self.R.row))
        unknown_items = list(set(self.R_val.col) - set(self.R.col))
        self.P[unknown_users, :] = 0.
        self.Q[:, unknown_items] = 0.
        return self

    def predict(self, u, i, P=None, Q=None, F=None, w=None, user_bias=None, item_bias=None):
        """
        Predict the rating user u gives to item i.

        :param u: User index.
        :param i: Item index.
        :param P: User embedding matrix.
        :param Q: Item embedding matrix.
        :param F: User friendship embeddings matrix.
        :param user_bias: User biases vector.
        :param item_bias: Item biases vector.
        :return: Predicted rating.
        """
        if P is None:
            P = self.P
        if Q is None:
            Q = self.Q
        if F is None:
            F = self.F
        if user_bias is None:
            user_bias = self.user_bias
        if item_bias is None:
            item_bias = self.item_bias

        known_user = self._known('user', u)
        known_item = self._known('item', i)
        rui_hat = self.mean
        if known_user:
            rui_hat += user_bias[u]
        if known_item:
            rui_hat += item_bias[i]
        if known_user and known_item:
            F_sum = np.sum(F[self.Fr[u], :], axis=0)
            F_num = np.sqrt(self.Fr[u].shape[0])  # Try without sqrt
            if F_num > 0:
                F_sum /= F_num
            rui_hat += (F_sum + w * P[u, :]).dot(Q[:, i])
        # Apply potential non-linearity (activation) g
        rui_hat = self.g(rui_hat)
        return rui_hat
