import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import rankdata
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.linalg import eigh
import plotly.express as px
from sklearn.mixture import BayesianGaussianMixture
from numpy.linalg import svd
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

hk40 = pd.read_csv("hk40.csv")

hk40 = hk40.loc[:, ~hk40.columns.str.contains('^Unnamed')]

hk40 = hk40.apply(pd.to_numeric, errors='coerce')

hk40 = hk40.fillna(method='ffill').fillna(method='bfill')

hk40_array = hk40.values.reshape(1304, 40, 1)


class Embedding:
    """
    :param d: Embedding Dimension
    :type d: int

    :param y: The training data. Must be in the form of a numpy array of size (T,N,Q)
    :type y: numpy.ndarray

    :param embedding_method: The possible embedding methods are: 'Pearson Correlation' (Default), 'Precision Matrix', 'Spearman Correlation', 'Kendall Correlation', 'Covariance Matrix"
    :type embedding_method: str

    :param cutoff_feature: The feature on which to calculate dhat via MP of embedding method
    :type cutoff_feature: int

    :return: None
    :rtype: None
    """

    def __init__(
        self,
        y: np.ndarray,
        d: int = None,
        embedding_method: str = 'Pearson Correlation',
        cutoff_feature: int = 1
    ) -> None:
        self.y = y
        self.embedding_method = embedding_method
        self.cutoff_feature = cutoff_feature
        self.d = d if d is not None else self.marchenko_pastur_estimate()
        if self.d <= 0:
            print("WARNING: Embedding dimension (d) is invalid. Setting to default value of 1.")
            self.d = 1

    @property
    def N(self):
        N = self.y.shape[1]
        return N

    @property
    def Q(self):
        Q = self.y.shape[2]
        return Q

    @property
    def T(self):
        T = self.y.shape[0]
        return T

    def pearson_correlations(self) -> np.ndarray:
        """
        :return: A row concatenated array of the (N x N) Pearson correlation matrix for each feature. Size = (Q,N,N)
        :rtype: numpy.ndarray
        """
        corr_mat = np.zeros((self.Q, self.N, self.N))
        for q in range(self.Q):
            p_corr = np.corrcoef(self.y[:, :, q].T)
            corr_mat[q] = p_corr
        return corr_mat

    def covariance_matrix(self) -> np.ndarray:
        """
        :return: A row concatenated array of the (N x N) covariance matrix for each feature. Size = (Q,N,N)
        :rtype: numpy.ndarray
        """
        cov_mat = np.zeros((self.Q, self.N, self.N))
        for q in range(self.Q):
            cov_mat[q] = np.cov(self.y[:, :, q].T)
        return cov_mat

    def precision_matrix(self) -> np.ndarray:
        """
        :return: A row concatenated array of the (N x N) precision matrix for each feature. Size = (Q,N,N)
        :rtype: numpy.ndarray
        """
        prec_mat = np.zeros((self.Q, self.N, self.N))
        for q in range(self.Q):
            p_corr = np.corrcoef(self.y[:, :, q].T)
            prec_mat[q] = np.linalg.inv(p_corr)
        return prec_mat

    def marchenko_pastur_estimate(self) -> int:
        """
        :return: Estimated number of "significant" dimensions
        :rtype: int
        """
        if self.embedding_method == 'Pearson Correlation':
            Sigma = self.pearson_correlations()
            eigenvalues = np.linalg.eigvals(Sigma[self.cutoff_feature])
            cutoff = (1 + np.sqrt(self.N / self.T)) ** 2
            d_hat = np.count_nonzero(eigenvalues > cutoff)
        elif self.embedding_method == 'Precision Matrix':
            Sigma = self.precision_matrix()
            eigenvalues = np.linalg.eigvals(Sigma[self.cutoff_feature])
            ratio_limit = self.N / self.T
            cutoff = ((1 - np.sqrt(ratio_limit)) / (1 - ratio_limit)) ** 2
            d_hat = np.count_nonzero(eigenvalues < cutoff)
        elif self.embedding_method == 'Full UASE Correlation':
            flat_X = np.reshape(self.y[:, :, :2], (self.T, self.N * 2))
            Sigma = np.corrcoef(flat_X.T)
            print(f"Sigma shape: {Sigma.shape}")
            ratio_limit = 2 * self.N / self.T
            cutoff = (1 + np.sqrt(ratio_limit)) ** 2
            eigenvalues = np.linalg.eigvals(Sigma)
            d_hat = np.count_nonzero(eigenvalues > cutoff)
        else:
            print("ERROR : Embedding method must be one of Pearson Correlation or Precision Matrix")

        if d_hat <= 0:
            print("WARNING: Estimated embedding dimension is 0. Setting to default value of 1.")
            d_hat = 1

        return d_hat

    def marcenko_pastur_denoised_correlation(self) -> np.ndarray:
        """
        :return: A row concatenated array of the (N x N) Pearson correlation matrix for each feature.
        :rtype: np.ndarray
        """
        corr_mat = np.zeros((self.Q, self.N, self.N))
        for q in range(self.Q):
            p_corr = np.corrcoef(self.y[:, :, q].T)
            cutoff = (1 + np.sqrt(self.N / self.T)) ** 2
            signal_eigs_all, signal_eigv_all = eigh(p_corr, subset_by_value=(cutoff, np.inf))
            signal_corr = signal_eigv_all @ np.diag(signal_eigs_all) @ signal_eigv_all.T
            if np.array_equal(signal_corr, np.zeros((self.N, self.N))):
                print("WARNING: All eigenvalues are less than Marcenko-Pastur cutoff")
                corr_mat[q] = p_corr
            else:
                corr_mat[q] = signal_corr
        return corr_mat

    def embed_corr_matrix(self, corr_matrix: np.ndarray, n_iter: int, random_state: int) -> np.ndarray:
        """
        :param corr_matrix: Correlation matrix to be embedded. Size = (Q,N,N). Note you can also pass the precision matrix in here.
        :type corr_matrix: numpy.ndarray

        :param n_iter: Number of iterations to run randomized SVD solver.
        :type n_iter: int

        :param random_state: Used during randomized svd. Pass an int for reproducible results across multiple function calls.
        :type random_state: int

        :return: embedded_array. UASE for each stock-feature vector. Size = (Q,N,d)
        :rtype: numpy.ndarray
        """
        flat_dist_corr = np.reshape(corr_matrix, (self.N * self.Q, self.N))
        svd = TruncatedSVD(n_components=self.d, n_iter=n_iter, random_state=random_state)
        embedded_array = svd.fit_transform(flat_dist_corr)
        embedded_array = np.reshape(embedded_array, (self.Q, self.N, self.d))
        return embedded_array


class fit:
    """
    Compute the neighbours of stock i, compute the corresponding covariates for stock i at time t and do regression against the
    actual value of stock i at time t. The estimated regression coefficients are computed using OLS.

    :param embedded_array: UASE for each stock-feature vector. Size = (Q,N,d)
    :type embedded_array: np.ndarray

    :param training_set: Array containing the training data. Size = (T_train,N,Q)
    :type training_set: np.ndarray

    :param target_feature: The feature that you are trying to predict e.g. pvCLCL returns
    :type target_feature: int

    :param cutoff_distance: The radius within which stocks are included in the VAR predictive model. It is the same for all stock-features
    :type cutoff_distance: float

    :param UASE_dim: Embedding Dimension
    :type UASE_dim: int

    :param alpha: Exponential parameter
    :type alpha: float

    :param lookback_window: How many previous days to use to predict tomorrow's returns
    :type lookback_window: int

    :param weights: The EMA weights. They must sum to 1. Shape = (lookback_window)
    :type weights: np.ndarray

    :return: None
    :rtype: None
    """

    def __init__(self,
                 embedded_array: np.ndarray,
                 training_set: np.ndarray,
                 target_feature: int,
                 cutoff_distance: float = 1,
                 UASE_dim: int = 3,
                 alpha: float = 0.4,
                 lookback_window: int = 1,
                 weights: np.ndarray = None,
                 kmeans_random: int = 1,
                 ) -> None:
        self.embedded_array = embedded_array
        self.training_set = training_set
        self.target_feature = target_feature
        self.cutoff_distance = cutoff_distance
        self.UASE_dim = UASE_dim
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.weights = weights if weights is not None else self.compute_weights()
        self.kmeans_random = kmeans_random

    @property
    def N(self):
        N = self.training_set.shape[1]
        return N

    @property
    def Q(self):
        Q = self.training_set.shape[2]
        return Q

    @property
    def T_train(self):
        T_train = self.training_set.shape[0]
        return T_train

    def compute_weights(self) -> np.ndarray:
        weights = np.zeros((self.lookback_window))
        for t in range(self.lookback_window):
            weights[t] = self.alpha ** t
        return weights

    def gmm(self, k: int) -> np.ndarray:
        """
        GMM clustering. Number of clusters must be pre-specified. EM algorithm is then run.

        :param k: The number of clusters.
        :type k: int

        :return: A binary array with value 1 for the neighboring stocks in the same cluster and 0 otherwise.
                    Shape = (Q, N, N)
        :rtype: np.ndarray

        :return: Array of integers where each integer labels a k-means cluster. Shape = (Q, N)
        :rtype: np.ndarray
        """
        constrained_distances = np.zeros((self.Q, self.N, self.N))
        all_labels = np.zeros((self.Q, self.N))
        for q in range(self.Q):
            feature_embedding = self.embedded_array[q, :, :]
            gmm_labels = GaussianMixture(n_components=k, random_state=self.kmeans_random, init_params='k-means++').fit_predict(feature_embedding)
            labels = gmm_labels
            all_labels[q] = labels
            similarity_matrix = self.groupings_to_2D(labels)
            constrained_distances[q] = similarity_matrix

        return constrained_distances, all_labels

    def full_UASE_gmm(self, k: int) -> np.ndarray:
        """
        GMM clustering. Multiple features are clustered at once. Number of clusters must be pre specified. EM algorithm is then run.

        :param k: The number of clusters
        :type k: int

        :return: A binary array with value 1 for the neighbouring stocks in the same cluster and 0 otherwise (Shape = (Q,N,N))
        :rtype: np.ndarray

        :return: Array of integers where each integer labels a kmeans cluster (Shape = (Q,N))
        :rtype: np.ndarray
        """
        feature_embedding = self.embedded_array
        feature_embedding = np.reshape(feature_embedding, (self.Q * self.N, self.UASE_dim))
        gmm_labels = GaussianMixture(n_components=k, random_state=self.kmeans_random, init_params='k-means++').fit_predict(feature_embedding)
        labels = gmm_labels
        similarity_matrix = self.groupings_to_2D(labels)
        all_labels = np.reshape(labels, (self.Q, self.N))
        target_similarity = similarity_matrix[self.target_feature * self.N:self.target_feature * self.N + self.N, :]
        constrained_distances = np.reshape(target_similarity, (self.N, self.Q, self.N)).transpose(1, 0, 2)
        return constrained_distances, all_labels

    def covariates(self, constrained_array: np.ndarray) -> np.ndarray:
        """
        :param constrained_array: Shape = (Q,N,N) Some constraint on which neighbours to sum up to get a predictor along that feature.
        :type: np.ndarray

        :rtype: np.ndarray
        :return: Shape = (N,N,Q,T_train) For each stock, we have a maximum (this max is not reached do to clustering regularisation) of NQ predictors. There are T_train training values for each predictor.
        :rtype: np.ndarray
        """
        covariates = np.zeros((self.N, self.N, self.Q, self.T_train), dtype=np.float32)
        for i in range(self.N):
            c = constrained_array[:, i, :, None] * (self.training_set.transpose(2, 1, 0))
            c = c.transpose(1, 0, 2)
            covariates[i] = c
        return covariates

    def ols_parameters(self, constrained_array: np.ndarray) -> np.ndarray:
        """
        :param constrained_array: Some constraint on which neighbours to sum up to get a predictor along that feature. Shape= (Q,N,N)
        :type: np.ndarray

        :return: ols_params Shape = (N,N,Q)
        :rtype: np.ndarray
        """
        ols_params = np.zeros((self.N, self.N * self.Q))
        covariates = self.covariates(constrained_array=constrained_array)
        targets = self.training_set[:, :, self.target_feature]
        for i in range(self.N):
            ols_reg_object = LinearRegression(fit_intercept=False)
            x = covariates[i].reshape(-1, covariates[i].shape[-1], order='F').T[:-1, :]
            non_zero_col_indices = np.where(x.any(axis=0))[0]
            x_reg = x[:, non_zero_col_indices]
            y = targets[1:, i]
            ols_fit = ols_reg_object.fit(x_reg, y)
            ols_params[i, non_zero_col_indices] = ols_fit.coef_
        ols_params = np.reshape(ols_params, (self.N, self.N, self.Q), order='F')
        return ols_params

    def groupings_to_2D(self, labels: np.ndarray) -> np.ndarray:
        N = len(labels)
        similarity_matrix = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:
                    similarity_matrix[i, j] = 1
        return similarity_matrix



embedding = Embedding(
    y=hk40_array,
    embedding_method='Pearson Correlation',
    cutoff_feature=0
)

corr_matrix = embedding.pearson_correlations()

embedded_array = embedding.embed_corr_matrix(
    corr_matrix=corr_matrix,
    n_iter=5,
    random_state=42
)



fit_model = fit(
    embedded_array=embedded_array,
    training_set=hk40_array,
    target_feature=0,
    cutoff_distance=1.0,
    UASE_dim=embedded_array.shape[2],
    alpha=0.4,
    lookback_window=5
)

constrained_distances, labels = fit_model.gmm(k=6)

ols_params = fit_model.ols_parameters(constrained_array=constrained_distances)
print("OLS :")
print(ols_params)


def moving_average_predictors(X_test: np.ndarray, alpha: float) -> np.ndarray:
    """
    Utility function to compute the exponentially weighted average stock returns over the past l days.

    :param X_test: Last l days of stock values. Shape = (Q,N,l)
    :type X_test: np.ndarray

    :param alpha: Exponential parameter
    :type alpha: float

    :return weighted_average: Exponential smoothed predictors. Shape = (Q,N)
    :rtype weighted_average: np.ndarray
    """
    l = X_test.shape[2]
    w = np.flip(np.array([alpha ** t for t in range(l)]))
    normalisation = np.sum(w)
    weighted_y = w * X_test
    weighted_average = np.sum(weighted_y, axis=2)
    weighted_average = weighted_average / normalisation
    return weighted_average


class predict:
    """
    Given the set of neighbouring stocks and the coefficients for each feature, predict the next day returns of a stock using the prediction model:

    $$X_{i,q,t+1} = \sum_{q'=1}^{Q}  \sum_{j = 1}^{N} \hat{A}_{ij}^{(qq')} \hat{\Phi}_{ij}^{(qq')} X_{j,q',t}$$

    :param ols_params: Phi coefficients. Should contain zeros for stocks not in neighbourhood. Shape = (N,N,Q)
    :type ols_params: np.ndarray

    :param todays_Xs: The value of X_{i,q} for each feature of each stock today. Shape = (N,Q)
    :type todays_Xs: np.ndarray
    """

    def __init__(self,
                 ols_params: np.ndarray,
                 todays_Xs: np.ndarray,
                 ) -> None:
        self.ols_params = ols_params
        self.todays_Xs = todays_Xs

    def next_day_prediction(self) -> np.ndarray:
        """
        :return: s
            Next day predictions. Shape = (N)
        :rtype: np.ndarray
        """
        if self.ols_params.shape[0] != self.todays_Xs.shape[0]:
            raise ValueError("ols_params and todays_Xs dimensions do not match!")

        s = np.dot(self.ols_params, self.todays_Xs).flatten()
        return s

    def next_day_truth(self, phi: np.ndarray) -> np.ndarray:
        """
        :param phi: Ground truth VAR coefficients. Shape = (N,N,Q)
        :type phi: np.ndarray

        :return s: Next day values before noise is added. Shape = (N)
        :rtype s: np.ndarray
        """

        s = np.sum(np.sum(phi * self.todays_Xs, axis=1), axis=1)
        return s


class benchmarking:

    """
    Class to compute daily benchmarking statistics when doing backtesting.

    :param predictions: Predicted returns for each day. Shape = (N)
    :type predictions: np.ndarray

    :param market_excess_returns: Excess returns on the prediction day. Equal to Raw returns minus SPY returns. Shape = (N)
    :type market_excess_returns: np.ndarray

    :param yesterdays_predictions: Predictions from the previous day. Shape = (N). Used to determine if a transaction occurred.
    :type yesterdays_predictions: np.ndarray

    :return: None
    :rtype: None
    """
    def __init__(self,
                 predictions: np.ndarray,
                 market_excess_returns: np.ndarray,
                 yesterdays_predictions: np.ndarray,
                 ) -> None:
        self.predictions = predictions
        self.market_excess_returns = market_excess_returns
        self.yesterdays_predictions = yesterdays_predictions

    @property
    def n_stocks(self):
        n_stocks = self.predictions.shape[0]
        return n_stocks

    def hit_ratio(self) -> float:
        """
        :return: ratio
            The fraction of predictions with the same sign as market excess returns
        :rtype: float
        """
        is_correct_sign = np.sign(self.predictions) * np.sign(self.market_excess_returns)
        is_corr_ones = np.where(is_correct_sign == 1, 1, 0)
        ratio = np.sum(is_corr_ones) / (self.n_stocks)
        return ratio

    def long_ratio(self) -> float:
        """
        :return: long_ratio
            The fraction of predictions with sign +1
        :rtype: float
        """
        prediction_sign = np.sign(self.predictions)
        is_corr_ones = np.where(prediction_sign == 1, 1, 0)
        long_ratio = np.sum(is_corr_ones) / (self.n_stocks)
        return long_ratio

    def corr_SP(self) -> float:
        """
        :return: corr_SP
            The Spearman correlation between your predictions and the target market excess returns
        :rtype: float
        """
        rho_sp, p = spearmanr(self.predictions, self.market_excess_returns)
        return rho_sp

    def PnL(self, quantile: float) -> float:
        """
        :param quantile: The top x% largest (in magnitude) predictions where x ∈ [0,1].
        :type quantile: float

        :return PnL: The quantile PnL where PnL is defined as \sum_{all_stocks} sign(predictions)*market_excess_returns
        :rtype PnL: float
        """

        prediction_ranks = rankdata(np.abs(self.predictions), method='min')
        cutoff_rank = self.n_stocks * (1 - quantile)
        quantile_predictions = np.where(prediction_ranks >= cutoff_rank, self.predictions, 0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(signed_predictions * self.market_excess_returns)
        return PnL

    def transaction_indicator(self):
        """
        :return: transaction_indicator
            1 if a transaction occured, 0 otherwise. Shape = (N)
        :rtype: np.ndarray
        """
        transaction_indicator = np.where(np.sign(self.predictions) - np.sign(self.yesterdays_predictions) == 0, 0, 1)
        return transaction_indicator

    def weighted_PnL_transactions(self, weights: np.ndarray, quantile: float) -> float:
        """
        :param quantile: The top x% largest (in magnitude) predictions where x ∈ [0,1].
        :type quantile: float

        :param weights: The portfolio weightings for each stock. Shape = (N)
        :type weights: np.ndarray

        :return PnL: The quantile PnL where PnL is defined as ∑_{all_stocks} sign(predictions)*market_excess_returns
        :rtype PnL: float
        """

        prediction_ranks = rankdata(np.abs(self.predictions), method='min')
        cutoff_rank = self.n_stocks * (1 - quantile)
        quantile_predictions = np.where(prediction_ranks >= cutoff_rank, self.predictions, 0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(weights * (signed_predictions * self.market_excess_returns - (0) * self.transaction_indicator()))
        portfolio_size = np.sum(weights)
        PnL = PnL / portfolio_size
        return PnL


pred = pd.read_csv("pred.csv")

pred = pred.iloc[:, 1:]

todays_Xs = pred.values



todays_Xs = todays_Xs[-1, :].reshape(40, 1)

ols_params = np.squeeze(ols_params)

np.random.seed(2)

ols_params = np.nan_to_num(ols_params)
todays_Xs = np.nan_to_num(todays_Xs)

predict_model = predict(
    ols_params=ols_params,
    todays_Xs=todays_Xs
)

next_day_predictions = predict_model.next_day_prediction()


pd.DataFrame(next_day_predictions, columns=["Predictions"]).to_csv("predictions.csv", index=False)

predictions = next_day_predictions
market_excess_returns = np.random.randn(40)
yesterdays_predictions = np.random.randn(40)

benchmark = benchmarking(
    predictions=predictions,
    market_excess_returns=market_excess_returns,
    yesterdays_predictions=yesterdays_predictions
)

quantile = 1
pnl = benchmark.PnL(quantile=quantile)
#np.random.seed(40)#print("PnL:", pnl)

weights = np.random.rand(40)
weights /= np.sum(weights)

weighted_pnl = benchmark.weighted_PnL_transactions(weights=weights, quantile=quantile)

total_investment = 100000
pnl_percentage = (pnl / total_investment) * 100

weighted_pnl_percentage = (weighted_pnl / total_investment) * 100


def get_R(A: np.ndarray):
    """
    :param A: The adjacency matrix with 1s on the diagonals. Shape = (N,N)
    :type A: np.ndarray

    :return R: Shape = (N**2,M)
    :rtype R: np.ndarray
    """
    N = A.shape[0]
    M = np.sum(A)
    R = np.identity(M)
    alpha = A.flatten('F')
    zero_rows = []
    counter = 0
    for i in range(N ** 2):
        if alpha[i] == 0:
            zero_rows.append(counter)
        elif alpha[i] == 1:
            counter += 1
    R = np.insert(R, zero_rows, 0, axis=0)
    return R


def masked_design(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    :param X: Design matrix. Shape = (T-1,N)
    :type X: np.ndarray

    :param A: Some adjacency matrix. Shape = (N,N)
    :type A: np.ndarray

    :return X_mask: Shape = (N,T-1,N)
    :rtype X_mask: np.ndarray
    """
    X_mask = A[:, np.newaxis, :] * X[np.newaxis, :, :]
    return X_mask


def bias_factor(X: np.ndarray, A: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
    """
    :param X: Design matrix. Shape = (T-1,N)
    :type X: np.ndarray

    :param A: Some true adjacency matrix. Shape = (N,N)
    :type A: np.ndarray

    :param A_hat: Some reconstructed adjacency matrix. Shape = (N,N)
    :type A_hat: np.ndarray

    :return inverse_bias: Shape = (N,N,N)
    :rtype inverse_bias: np.ndarray
    """
    tolerance = 1e-8
    X_A = masked_design(X, A)
    X_A_hat = masked_design(X, A_hat)
    N = A.shape[0]
    inverse_bias = np.zeros((N, N, N))
    for i in range(N):
        X_A_non_zero_col_indices = np.where(X_A[i].any(axis=0))[0]
        X_A_hat_non_zero_col_indices = np.where(X_A_hat[i].any(axis=0))[0]
        X_A_i = X_A[i, :, X_A_non_zero_col_indices].T
        X_A_hat_i = X_A_hat[i, :, X_A_hat_non_zero_col_indices].T
        D = np.linalg.inv(X_A_hat_i.T @ X_A_hat_i) @ (X_A_hat_i.T @ X_A_i)
        close_to_zero = np.isclose(D, 0, atol=tolerance, rtol=tolerance)
        D[close_to_zero] = 0
        inverse_bias[i][np.ix_(X_A_hat_non_zero_col_indices, X_A_non_zero_col_indices)] = D
    return inverse_bias


def correct_for_bias(X: np.ndarray, A: np.ndarray, A_hat: np.ndarray, phi_estimate: np.ndarray, phi_true: np.ndarray) -> np.ndarray:
    """
    :param X: Design matrix. Shape = (T-1,N)
    :type X: np.ndarray

    :param A: Some true adjacency matrix. Shape = (N,N)
    :type A: np.ndarray

    :param A_hat: Some reconstructed adjacency matrix. Shape = (N,N)
    :type A_hat: np.ndarray

    :param phi_estimate: Shape = (N,N)
    :type phi_estimate: np.ndarray

    :param phi_true: Shape = (N,N)
    :type phi_true: np.ndarray

    :return phi_corrected: Shape = (N,N)
    :rtype phi_corrected: np.ndarray
    """
    D = bias_factor(X, A, A_hat)
    N = A.shape[0]
    phi_corrected = np.zeros((N, N))
    for i in range(N):
        phi_corrected[i] = phi_estimate[i] + phi_true[i] - D[i] @ phi_true[i]
    return phi_corrected


def drawSBM(sizes: list, p_in: float, p_out: float, random_state: int = 42):
    G = nx.random_partition_graph(sizes=sizes, p_in=p_in, p_out=p_out, directed=False, seed=random_state)

    positions = nx.spring_layout(G, seed=random_state)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.size = node_adjacencies
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Stochastic Block Model Graph",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def zhu(d: np.ndarray):
    """
    :param d: An array of eigenvalues (or another measure of importance) sorted in descending order of importance
    :type d: np.ndarray

    :return profile_likelihood: Array of the profile likelihood values, one for each dimension, q
    :rtype profile_likelihood: np.ndarray

    :return np.argmax(profile_likelihood): The dimension, q, at which the profile log-likelihood is maximum
    :rtype np.argmax(profile_likelihood): int
    """
    p = len(d)
    profile_likelihood = np.zeros(p)
    for q in range(1, p - 1):
        mu1 = np.mean(d[:q])
        mu2 = np.mean(d[q:])
        sd = np.sqrt(((q - 1) * (np.std(d[:q]) ** 2) + (p - q - 1) * (np.std(d[q:]) ** 2)) / (p - 2))
        profile_likelihood[q] = norm.logpdf(d[:q], loc=mu1, scale=sd).sum() + norm.logpdf(d[q:], loc=mu2, scale=sd).sum()
    return profile_likelihood[1:p - 1], np.argmax(profile_likelihood[1:p - 1]) + 1


def iterate_zhu(d: np.ndarray, x: int = 3):
    """
    Find the dimension, q1, of the 1st largest gap in the scree plot, then the dimension, q2 > q1, of the
    second largest gap in the scree plot given q1, and so on ... up to dimension x

    :param d: An array of eigenvalues (or another measure of importance) sorted in descending order of importance
    :type d: np.ndarray

    :return results: Array of dimensions [q1, q2, ..., qx], where q1 is the dimension of the 1st largest gap in the scree plot, q2 is the dimension of the second largest gap given q1, and so on up to dimension x
    :rtype results: np.ndarray

    """
    results = np.zeros(x, dtype=int)
    results[0] = zhu(d)[1]
    for i in range(x - 1):
        results[i + 1] = results[i] + zhu(d[results[i]:])[1]
    return results


def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar', 'band', 'bars', 'bands', None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar', 'bars', None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band', 'bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg, val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig




def get_gamma(phi: np.ndarray, noise_variance: float) -> np.ndarray:
    """
    Compute the covariance matrix corresponding to a VAR(1) process with VAR coefficient matrix phi and
    noise covariance matrix noise_variance*Identity.

    :param: phi: numpy array of dimension (N,N)
    :type phi: np.ndarray

    :param noise_variance: scalar noise covariance
    :type noise_variance: float
    """
    n = phi.shape[0]
    sigma = noise_variance * np.identity(n)
    vec_sigma = np.reshape(sigma, (n ** 2), order='F')
    phi_kron = -np.kron(phi, phi)
    np.fill_diagonal(phi_kron, phi_kron.diagonal() + 1)
    vec_gamma = np.linalg.inv(phi_kron) @ vec_sigma
    gamma = np.reshape(vec_gamma, (n, n), order='F')
    return gamma


def groupings_to_2D(input_array: np.ndarray) -> np.ndarray:
    """
    Turn a 1d array of integers (groupings) into a 2d binary array, A, where
    A[i,j] = 1 iff i and j have the same integer value in the 1d groupings array.

    :param input_array: 1d array of integers.
    :type input_array: np.ndarray

    :return: 2d Representation. Shape = (len(input_array),len(input_array))
    :rtype: np.ndarray
    """

    L = len(input_array)
    A = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if input_array[i] == input_array[j]:
                A[i][j] = int(1)
            else:
                continue

    return A.astype(int)

np.random.seed(48)

predictions = np.random.randn(240, 40)
market_excess_returns = np.random.randn(240, 40)

yesterdays_predictions = np.roll(predictions, shift=1, axis=0)

weights = np.random.rand(40)
weights /= np.sum(weights)

total_investment = 100000

daily_pnl = []
daily_weighted_pnl = []

for day in range(predictions.shape[0] - 1):
    benchmark = benchmarking(
        predictions=predictions[day],
        market_excess_returns=market_excess_returns[day],
        yesterdays_predictions=yesterdays_predictions[day]
    )
    quantile = 1
    pnl = benchmark.PnL(quantile=quantile)
    #weighted_pnl = benchmark.weighted_PnL_transactions(weights=weights, quantile=quantile)

    daily_pnl.append(pnl)
    #daily_weighted_pnl.append(weighted_pnl)

daily_pnl = np.array(daily_pnl)
#daily_weighted_pnl = np.array(daily_weighted_pnl)

daily_pnl_percentage = (daily_pnl / total_investment) * 100
#daily_weighted_pnl_percentage = (daily_weighted_pnl / total_investment) * 100

cumulative_pnl_percentage = np.cumsum(daily_pnl_percentage)
#cumulative_weighted_pnl_percentage = np.cumsum(daily_weighted_pnl_percentage)

time_points = np.arange(1, len(cumulative_pnl_percentage) + 1)
plt.figure(figsize=(10, 6))
plt.plot(time_points, cumulative_pnl_percentage, label="Cumulative PnL (%)", color="blue", marker="o")
plt.title("Cumulative PnL Percentage ")
plt.xlabel("Time Points")
plt.ylabel("Cumulative Percentage (%)")
plt.legend()
plt.grid(True)
plt.show()
np.random.seed(48)

time_points = np.arange(1, len(cumulative_pnl_percentage) + 1)

plt.figure(figsize=(12, 8))
plt.plot(time_points, cumulative_pnl_percentage, label="Cumulative PnL (%)", color="blue", marker="o", linewidth=2)

max_index = np.argmax(cumulative_pnl_percentage)
min_index = np.argmin(cumulative_pnl_percentage)

plt.scatter(time_points[max_index], cumulative_pnl_percentage[max_index], color="green", label=f"Max: {cumulative_pnl_percentage[max_index]:.2f}%")
plt.scatter(time_points[min_index], cumulative_pnl_percentage[min_index], color="red", label=f"Min: {cumulative_pnl_percentage[min_index]:.2f}%")

plt.annotate(f"Max: {cumulative_pnl_percentage[max_index]:.2f}%",
             xy=(time_points[max_index], cumulative_pnl_percentage[max_index]),
             xytext=(time_points[max_index] + 10, cumulative_pnl_percentage[max_index] + 5),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=10, color="green")

plt.annotate(f"Min: {cumulative_pnl_percentage[min_index]:.2f}%",
             xy=(time_points[min_index], cumulative_pnl_percentage[min_index]),
             xytext=(time_points[min_index] - 10, cumulative_pnl_percentage[min_index] - 5),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color="red")

plt.title("Cumulative PnL Percentage Over Time", fontsize=16, fontweight="bold")
plt.xlabel("Time Points", fontsize=12)
plt.ylabel("Cumulative Percentage (%)", fontsize=12)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.legend(fontsize=12)

plt.show()

print(labels)

def calculate_sharpe_ratio(predictions: np.ndarray, market_excess_returns: np.ndarray = None) -> float:
    if market_excess_returns is None:
        market_excess_returns = predictions
    excess_returns = predictions - market_excess_returns
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    if std_excess_return == 0:
        return 0
    sharpe_ratio = mean_excess_return / std_excess_return
    return sharpe_ratio


def calculate_mae(predictions: np.ndarray, actual_values: np.ndarray) -> float:
    mae = np.mean(np.abs(predictions - actual_values))
    return mae


predictions = next_day_predictions
actual_values = market_excess_returns

sharpe_ratio = calculate_sharpe_ratio(predictions)
print("Sharpe Ratio:", sharpe_ratio)

predictions = next_day_predictions
actual_values = market_excess_returns
epsilon = 1e-8
predictions += np.random.normal(0, epsilon, size=predictions.shape)
market_excess_returns += np.random.normal(0, epsilon, size=market_excess_returns.shape)

def calculate_sharpe_ratio(predictions: np.ndarray, market_excess_returns: np.ndarray = None) -> float:
    if market_excess_returns is None:
        market_excess_returns = predictions

    excess_returns = predictions - market_excess_returns
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)

    if std_excess_return == 0:
        print("WARNING: Standard deviation of excess returns is 0, cannot calculate Sharpe Ratio.")
        return 0

    sharpe_ratio = mean_excess_return / std_excess_return
    return sharpe_ratio

sharpe_ratio = calculate_sharpe_ratio(predictions, market_excess_returns)
print("Sharpe Ratio:", sharpe_ratio)

mae = calculate_mae(predictions, actual_values)
print("Mean Absolute Error (MAE):", mae)

print(labels)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_cluster_distribution_plot():
    


    
    stock_cluster_map = {
        "0001.HK": 4, "0002.HK": 3, "0003.HK": 3, "0005.HK": 4, "0006.HK": 3,
        "0011.HK": 1, "0012.HK": 5, "0016.HK": 1, "0017.HK": 1, "0027.HK": 1,
        "0066.HK": 2, "0101.HK": 5, "0144.HK": 3, "0175.HK": 1, "0267.HK": 3,
        "0293.HK": 1, "0386.HK": 0, "0388.HK": 4, "0700.HK": 3, "0762.HK": 4,
        "0836.HK": 3, "0857.HK": 5, "0883.HK": 2, "0939.HK": 3, "0941.HK": 3,
        "0968.HK": 2, "0981.HK": 1, "1038.HK": 3, "1044.HK": 2, "1093.HK": 1,
        "1109.HK": 1, "1113.HK": 1, "1299.HK": 3, "1928.HK": 3, "1997.HK": 2,
        "2318.HK": 1, "2319.HK": 3, "2388.HK": 3, "2628.HK": 2, "3888.HK": 1
    }

    
    industry_map = {
        # -------- FIN --------
        '0005.HK': 'FIN', '0011.HK': 'FIN', '0388.HK': 'FIN', '0939.HK': 'FIN',
        '1299.HK': 'FIN', '2318.HK': 'FIN', '2388.HK': 'FIN', '2628.HK': 'FIN',
        # -------- PRO --------
        '0001.HK': 'PRO', '0012.HK': 'PRO', '0016.HK': 'PRO', '0017.HK': 'PRO',
        '0101.HK': 'PRO', '1109.HK': 'PRO', '1113.HK': 'PRO', '1997.HK': 'PRO',
        # -------- UNE --------
        '0002.HK': 'UNE', '0003.HK': 'UNE', '0006.HK': 'UNE', '0066.HK': 'UNE',
        '0386.HK': 'UNE', '0836.HK': 'UNE', '0857.HK': 'UNE', '0883.HK': 'UNE',
        '1038.HK': 'UNE',
        # -------- TEC --------
        '0700.HK': 'TEC', '0762.HK': 'TEC', '0941.HK': 'TEC', '0981.HK': 'TEC',
        '3888.HK': 'TEC',
        # -------- CON --------
        '0027.HK': 'CON', '0175.HK': 'CON', '0293.HK': 'CON', '1044.HK': 'CON',
        '1093.HK': 'CON', '1928.HK': 'CON', '2319.HK': 'CON',
        # -------- IMM --------
        '0144.HK': 'IMM', '0267.HK': 'IMM', '0968.HK': 'IMM'
    }

   
    data = []
   
    common_stocks = set(stock_cluster_map.keys()) & set(industry_map.keys())

    for stock in common_stocks:
        data.append({
            'Stock': stock,
            'EstimatedCluster': stock_cluster_map[stock],
            'TrueIndustry': industry_map[stock]
        })

 
    df = pd.DataFrame(data)

 
    contingency_table = pd.crosstab(df['EstimatedCluster'], df['TrueIndustry'])
    
    all_industries = sorted(df['TrueIndustry'].unique())
    contingency_table = contingency_table.reindex(columns=all_industries, fill_value=0)
    

    all_clusters = sorted(df['EstimatedCluster'].unique())
    contingency_table = contingency_table.reindex(all_clusters, fill_value=0)


    sns.set_theme(style="whitegrid")
    
   
    colors = sns.color_palette("tab10", len(all_industries))
    
    
    ax = contingency_table.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 8),
        color=colors,
        width=0.8 
    )

    
    plt.title('Distribution of True Industries within Estimated Clusters', fontsize=16)
    plt.xlabel('Estimated Cluster', fontsize=12)
    plt.ylabel('Count', fontsize=12)

  
    plt.xticks(rotation=0)


    plt.legend(title='True Industry', bbox_to_anchor=(1.02, 1), loc='upper left')


    plt.figtext(0.5, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=16, weight='bold')

    plt.tight_layout(rect=[0, 0.05, 0.9, 1]) 


    plt.savefig('cluster_industry_distribution.png', dpi=300)
    




industry_map = {
    '0005.HK': 'FIN', '0011.HK': 'FIN', '0388.HK': 'FIN', '0939.HK': 'FIN',
    '1299.HK': 'FIN', '2318.HK': 'FIN', '2388.HK': 'FIN', '2628.HK': 'FIN',
    '0001.HK': 'PRO', '0012.HK': 'PRO', '0016.HK': 'PRO', '0017.HK': 'PRO',
    '0101.HK': 'PRO', '1109.HK': 'PRO', '1113.HK': 'PRO', '1997.HK': 'PRO',
    '0002.HK': 'UNE', '0003.HK': 'UNE', '0006.HK': 'UNE', '0066.HK': 'UNE',
    '0386.HK': 'UNE', '0836.HK': 'UNE', '0857.HK': 'UNE', '0883.HK': 'UNE',
    '1038.HK': 'UNE',
    '0700.HK': 'TEC', '0762.HK': 'TEC', '0941.HK': 'TEC', '0981.HK': 'TEC',
    '3888.HK': 'TEC',
    '0027.HK': 'CON', '0175.HK': 'CON', '0293.HK': 'CON', '1044.HK': 'CON',
    '1093.HK': 'CON', '1928.HK': 'CON', '2319.HK': 'CON',
    '0144.HK': 'IMM', '0267.HK': 'IMM', '0968.HK': 'IMM'
}

tickers = list(industry_map.keys())
n = len(tickers)
R = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        if industry_map[tickers[i]] == industry_map[tickers[j]]:
            R[i, j] = 1

R_df = pd.DataFrame(R, index=tickers, columns=tickers)
print(R_df)

    




import logging
import os
import time

default_log_path = './logs'

def formatted_logger(label, level=None, format=None, date_format=None, file_path=None):
    log = logging.getLogger(label)
    if level is None:
        level = logging.INFO
    elif level.lower() == 'debug':
        level = logging.DEBUG
    elif level.lower() == 'info':
        level = logging.INFO
    elif level.lower() == 'warn':
        level = logging.WARN
    elif level.lower() == 'error':
        level = logging.ERROR
    elif level.lower() == 'critical':
        level = logging.CRITICAL
    log.setLevel(level)

    if format is None:
        format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
    if date_format is None:
        date_format = '%Y-%m-%d %H:%M:%S'
    if file_path is None:
        if not os.path.exists(default_log_path):
            os.makedirs(default_log_path)
        file_path = '%s/%s.%s.log.txt' % (default_log_path, label, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    formatter = logging.Formatter(format, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    return log


import numpy as np
from scipy.special import psi
#from formatted_logger import formatted_logger

log = formatted_logger('MMSB', 'info')

class MMSB:
    """
    Mixed-membership stochastic block models, Airoldi et al., 2008

    """

    def __init__(self, Y, K, alpha = 1):
        """ follows the notations in the original NIPS paper

        :param Y: node by node interaction matrix, row=sources, col=destinations
        :param K: number of mixtures
        :param alpha: Dirichlet parameter
        :param rho: sparsity parameter
        """
        self.N = int(Y.shape[0])    # number of nodes
        self.K = K
        self.alpha = np.ones(self.K)
        self.Y = Y

        self.optimize_rho = False
        self.max_iter = 500
        #self.ll = 0.0

        #variational parameters
        self.phi = np.random.dirichlet(self.alpha, size=(self.N, self.N))

        self.gamma = np.random.dirichlet([1]*self.K, size=self.N)

        self.B = np.random.random(size=(self.K,self.K))

        self.rho = (1.-np.sum(self.Y)/(self.N*self.N))     # 1 - density of matrix

    def variational_inference(self, converge_ll_fraction=1e-3):
        """ run variational inference for this model
        maximize the evidence variational lower bound

        :param converge_ll_fraction: terminate variational inference when the fractional change of the lower bound falls below this
        """
        converge = False
        old_ll = 0.
        iteration = 0

        while not converge:
            ll = self.run_e_step()
            ll += self.run_m_step()

            iteration += 1

            if iteration >= self.max_iter:
                converge = True

            log.info('iteration %d, lower bound %.2f' %(iteration, ll))
    
    
        return ll 
    
    def run_e_step(self):
        """ compute variational expectations 
        """
        ll = 0.    

        for p in range(self.N):
            for q in range(self.N):
                new_phi = np.zeros(self.K)

                for g in range(self.K):
                    new_phi[g] = np.exp(psi(self.gamma[p,g])-psi(np.sum(self.gamma[p,:]))) * np.prod(( (self.B[g,:]**self.Y[p,q]) 
                        * ((1.-self.B[g,:])**(1.-self.Y[p,q])) ) 
                        ** self.phi[q,p,:] )
                self.phi[p,q,:] = new_phi/np.sum(new_phi)

                new_phi = np.zeros(self.K)
                for h in range(self.K):
                    new_phi[h] = np.exp(psi(self.gamma[q,h])-psi(np.sum(self.gamma[q,:]))) * np.prod(( (self.B[:,h]**self.Y[p,q]) 
                        * ((1.-self.B[:,h])**(1.-self.Y[p,q])) ) 
                        ** self.phi[p,q,:] )
                self.phi[q,p,:] = new_phi/np.sum(new_phi)

                for k in range(self.K):
                    self.gamma[p,k] = self.alpha[k] + np.sum(self.phi[p,:,k]) + np.sum(self.phi[:,p,k])
                    self.gamma[q,k] = self.alpha[k] + np.sum(self.phi[q,:,k]) + np.sum(self.phi[:,q,k])

        return ll
    
    

    def run_m_step(self):
        """ maximize the hyper parameters
        """
        ll = 0.

        self.optimize_alpha()
        self.optimize_B()
        if self.optimize_rho:
            self.update_rho()

        return ll

    def optimize_alpha(self):
        return

    def optimize_B(self):
        for g in range(self.K):
            for h in range(self.K):
                tmp1=0.
                tmp2=0.
                for p in range(self.N):
                    for q in range(self.N):
                        tmp = self.phi[p,q,g]*self.phi[q,p,h]
                        tmp1 += self.Y[p,q]*tmp
                        tmp2 += tmp
                self.B[g,h] = tmp1/tmp2
        return

    def update_rho(self):
        return


from numpy.random import dirichlet, multinomial, binomial

def generate_mmsb_graph(N, K, alpha, B, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

   
    pi = np.array([dirichlet(alpha) for _ in range(N)])

   
    R = np.zeros((N, N))

    for p in range(N):
        for q in range(N):
            
            z_pq = multinomial(1, pi[p])  # One-hot 
           
            z_qp = multinomial(1, pi[q])  

           
            prob = z_pq @ B @ z_qp.T  
            R[p, q] = binomial(1, prob)

    return R, pi



N = 40                
K = 6                
alpha = np.ones(K)     





R_np = R_df.values  
model = MMSB(R_np, 6)
model.variational_inference()

B_mmsb = model.B

print(model.gamma)
print(B_mmsb)
RR, pi = generate_mmsb_graph(N, K, alpha, B_mmsb, seed=42)
R_full = pd.DataFrame(RR)
R_full.to_csv("lon_matrix_R.csv", index=False)
print("R_full:",R_full)
print("loncency Matrix R:\n", R)
print("\nMembership matrix π (each row is a node):\n", pi)




print("restart:")


def estimate_A_mle_3d(hk40_array: np.ndarray, B_mmsb: np.ndarray):
    
    T, d, _ = hk40_array.shape
    r = B_mmsb.shape[0]

    # reshape (T, d, 1) → (T, d)
    X_t = hk40_array[1:].reshape(T - 1, d).T         # shape: (d, T-1)
    X_t_minus1 = hk40_array[:-1].reshape(T - 1, d).T # shape: (d, T-1)

    # 计算 Z = B_mmsb @ X_{t-1}
    Z = B_mmsb @ X_t_minus1                          # shape: (r, T-1)

    # MLE / OLS 公式
    ZZT_inv = np.linalg.inv(Z @ Z.T)                # shape: (r, r)
    A_hat = X_t @ Z.T @ ZZT_inv                     # shape: (d, r)

    return A_hat




A_hat = estimate_A_mle_3d(hk40_array, RR)
print("MLE estimate of A has shape:", A_hat.shape)  

ols_params = A_hat


print("OLS :")
print(ols_params)

ols_params1 = np.squeeze(ols_params)
#print("should stop here")

np.random.seed(2)




predict_model = predict(
    ols_params=ols_params1,
    todays_Xs=todays_Xs
)

next_day_predictions1 = predict_model.next_day_prediction()

print( next_day_predictions1)

pd.DataFrame(next_day_predictions1, columns=["Predictions"]).to_csv("predictions.csv", index=False)

predictions1 = next_day_predictions1
market_excess_returns = np.random.randn(40)
yesterdays_predictions = np.random.randn(40)

benchmark = benchmarking(
    predictions=predictions1,
    market_excess_returns=market_excess_returns,
    yesterdays_predictions=yesterdays_predictions
)

quantile = 1
pnl1 = benchmark.PnL(quantile=quantile)
print("PnL:", pnl1)



print("PnL:", pnl1)


print(pnl==pnl1)













total_investment = 100000
pnl_percentage1 = (pnl1 / total_investment) * 100
print("PnL 百分比:", pnl_percentage1, "%")


np.random.seed(40)

predictions1 = np.random.randn(240, 40)
market_excess_returns1 = np.random.randn(240, 40)

yesterdays_predictions1= np.roll(predictions1, shift=1, axis=0)

weights = np.random.rand(40)

weights /= np.sum(weights)

total_investment = 100000

daily_pnl1 = []
#daily_weighted_pnl = []




for day in range(predictions1.shape[0] - 1):
    benchmark = benchmarking(
        predictions=predictions1[day],
        market_excess_returns=market_excess_returns1[day],
        yesterdays_predictions=yesterdays_predictions1[day]
    )
    quantile = 1
    pnl1 = benchmark.PnL(quantile=quantile)
    daily_pnl1.append(pnl1)
    

daily_pnl1 = np.array(daily_pnl1)

daily_pnl_percentage1 = (daily_pnl1 / total_investment) * 100


cumulative_pnl_percentage1 = np.cumsum(daily_pnl_percentage1)


time_points1 = np.arange(1, len(cumulative_pnl_percentage1) + 1)
plt.figure(figsize=(10, 6))
plt.plot(time_points1, cumulative_pnl_percentage1, label="Cumulative PnL (%)", color="blue", marker="o")
plt.title("Cumulative PnL Percentage ")
plt.xlabel("Time Points")
plt.ylabel("Cumulative Percentage (%)")
plt.legend()
plt.grid(True)
plt.show()


time_points = np.arange(1, len(cumulative_pnl_percentage1) + 1)

plt.figure(figsize=(12, 8))
plt.plot(time_points, cumulative_pnl_percentage1, label="Cumulative PnL (%)", color="blue", marker="o", linewidth=2)

max_index1 = np.argmax(cumulative_pnl_percentage1)
min_index1 = np.argmin(cumulative_pnl_percentage1)

plt.scatter(time_points[max_index1], cumulative_pnl_percentage1[max_index1], color="green", label=f"Max: {cumulative_pnl_percentage1[max_index1]:.4f}%")
plt.scatter(time_points[min_index1], cumulative_pnl_percentage1[min_index1], color="red", label=f"Min: {cumulative_pnl_percentage1[min_index1]:4f}%")

plt.annotate(f"Max: {cumulative_pnl_percentage1[max_index1]:.2f}%",
             xy=(time_points[max_index1], cumulative_pnl_percentage1[max_index1]),
             xytext=(time_points[max_index1] + 10, cumulative_pnl_percentage1[max_index1] + 5),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=10, color="green")

plt.annotate(f"Min: {cumulative_pnl_percentage1[min_index1]:.2f}%",
             xy=(time_points[min_index1], cumulative_pnl_percentage1[min_index1]),
             xytext=(time_points[min_index1] - 10, cumulative_pnl_percentage1[min_index1] - 5),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color="red")

plt.title("Cumulative PnL Percentage Over Time", fontsize=16, fontweight="bold")
plt.xlabel("Time Points", fontsize=12)
plt.ylabel("Cumulative Percentage (%)", fontsize=12)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.legend(fontsize=12)

plt.show()



predictions1 = next_day_predictions1
actual_values = market_excess_returns
epsilon = 1e-8
predictions1 += np.random.normal(0, epsilon, size=predictions.shape)
market_excess_returns += np.random.normal(0, epsilon, size=market_excess_returns.shape)



sharpe_ratio1 = calculate_sharpe_ratio(predictions1, market_excess_returns)
print("Sharpe Ratio:", sharpe_ratio1)

mae1 = calculate_mae(predictions, actual_values)
print("Mean Absolute Error (MAE):", mae1)


print("Sharpe Ratio:", sharpe_ratio)
print("Mean Absolute Error (MAE):", mae)


def generate_cluster_distribution_plot():
    


    
    stock_cluster_map = {
        "0001.HK": 4, "0002.HK": 3, "0003.HK": 3, "0005.HK": 4, "0006.HK": 3,
        "0011.HK": 1, "0012.HK": 5, "0016.HK": 1, "0017.HK": 1, "0027.HK": 1,
        "0066.HK": 2, "0101.HK": 5, "0144.HK": 3, "0175.HK": 1, "0267.HK": 3,
        "0293.HK": 1, "0386.HK": 0, "0388.HK": 4, "0700.HK": 3, "0762.HK": 4,
        "0836.HK": 3, "0857.HK": 5, "0883.HK": 2, "0939.HK": 3, "0941.HK": 3,
        "0968.HK": 2, "0981.HK": 1, "1038.HK": 3, "1044.HK": 2, "1093.HK": 1,
        "1109.HK": 1, "1113.HK": 1, "1299.HK": 3, "1928.HK": 3, "1997.HK": 2,
        "2318.HK": 1, "2319.HK": 3, "2388.HK": 3, "2628.HK": 2, "3888.HK": 1
    }

    
    industry_map = {
        # -------- FIN --------
        '0005.HK': 'FIN', '0011.HK': 'FIN', '0388.HK': 'FIN', '0939.HK': 'FIN',
        '1299.HK': 'FIN', '2318.HK': 'FIN', '2388.HK': 'FIN', '2628.HK': 'FIN',
        # -------- PRO --------
        '0001.HK': 'PRO', '0012.HK': 'PRO', '0016.HK': 'PRO', '0017.HK': 'PRO',
        '0101.HK': 'PRO', '1109.HK': 'PRO', '1113.HK': 'PRO', '1997.HK': 'PRO',
        # -------- UNE --------
        '0002.HK': 'UNE', '0003.HK': 'UNE', '0006.HK': 'UNE', '0066.HK': 'UNE',
        '0386.HK': 'UNE', '0836.HK': 'UNE', '0857.HK': 'UNE', '0883.HK': 'UNE',
        '1038.HK': 'UNE',
        # -------- TEC --------
        '0700.HK': 'TEC', '0762.HK': 'TEC', '0941.HK': 'TEC', '0981.HK': 'TEC',
        '3888.HK': 'TEC',
        # -------- CON --------
        '0027.HK': 'CON', '0175.HK': 'CON', '0293.HK': 'CON', '1044.HK': 'CON',
        '1093.HK': 'CON', '1928.HK': 'CON', '2319.HK': 'CON',
        # -------- IMM --------
        '0144.HK': 'IMM', '0267.HK': 'IMM', '0968.HK': 'IMM'
    }

   
    data = []
   
    common_stocks = set(stock_cluster_map.keys()) & set(industry_map.keys())

    for stock in common_stocks:
        data.append({
            'Stock': stock,
            'EstimatedCluster': stock_cluster_map[stock],
            'TrueIndustry': industry_map[stock]
        })

 
    df = pd.DataFrame(data)

 
    contingency_table = pd.crosstab(df['EstimatedCluster'], df['TrueIndustry'])
    
    all_industries = sorted(df['TrueIndustry'].unique())
    contingency_table = contingency_table.reindex(columns=all_industries, fill_value=0)
    

    all_clusters = sorted(df['EstimatedCluster'].unique())
    contingency_table = contingency_table.reindex(all_clusters, fill_value=0)


    sns.set_theme(style="whitegrid")
    
   
    colors = sns.color_palette("tab10", len(all_industries))
    
    
    ax = contingency_table.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 8),
        color=colors,
        width=0.8 
    )

    
    plt.title('Distribution of True Industries within Estimated Clusters', fontsize=16)
    plt.xlabel('Estimated Cluster', fontsize=12)
    plt.ylabel('Count', fontsize=12)

  
    plt.xticks(rotation=0)


    plt.legend(title='True Industry', bbox_to_anchor=(1.02, 1), loc='upper left')


    plt.figtext(0.5, 0.01, '(a)', wrap=True, horizontalalignment='center', fontsize=16, weight='bold')

    plt.tight_layout(rect=[0, 0.05, 0.9, 1]) 


    plt.savefig('cluster_industry_distribution.png', dpi=300)



B_mmsb_df = pd.DataFrame(B_mmsb)


B_mmsb_df.to_csv("B_mmsb.csv", index=False)


print(B_mmsb)























