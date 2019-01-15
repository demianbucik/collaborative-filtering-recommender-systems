# Collaborative Filtering Recommender Systems
Implementations of matrix factorization and restricted Boltzmann machine based algorithms for collaborative filtering in Python using Numpy, Scipy.

### The basics
Matrix factorization models minimize regularized sum of squared error (equivalent to maximizing log posterior of parameters assuming normal distributions), where rating is predicted as biases plus interaction between user and item. Interaction is modelled as inner product of learned user and item latent feature vectors. 

Weighted model uses weights on latent factors, model idea is described in https://arxiv.org/pdf/1710.00482.pdf. 

Friends model uses user's friends network to improve predictions, each user has an additional learned latent feature vector representing his average impact on friends. The idea of implicit feedback used to models friendships is explained in previous article (SVD++).


Restricted Boltzmann machine (RBM) model maximizes log posterior probability of the parameters given the data. The likelihood is approximated with contrastive divergence function (G. Hinton) and its gradient is approximated with Gibbs steps (positive and negative phase) and used for learning. We try to reconstruct non-missing ratings (not generate new examples from data distribution) thus no sampling is used in training or prediction. This gist of using RBMs in collaborative filtering is explained in https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf, but we model ratings as real values between 0 and 1.


### Using models
User indices are mapped to [0, ..., n_users] before constructing R ratings matrix, same for items.
Matrix factorization models (example in example_matrix.py)
```python
mm = MatrixModel(*args)
mm = mm.fit(R_train, R_val)
rui_hat = mm.predict(u_new, i_new)
```
RBM model (example coming soon)
```python
rbm = AERBMModel(*args)
rbm = rbm.fit(R_train, R_val)
R_hat = rbm.predict(R_new)
```

### Example data
Number of time user plays an artist's song on last.fm The data is clipped if it passes a certan threshold and then rescaled to [0,1].
Histogram with 20 bins of rescaled data
<img src="lastfm_ratings_hist.png">
