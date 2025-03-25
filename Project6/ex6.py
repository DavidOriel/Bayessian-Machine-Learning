import numpy as np
from matplotlib import pyplot as plt
from ex6_utils import (plot_ims, load_MNIST, outlier_data, gmm_data, plot_2D_gmm, load_dogs_vs_frogs,
                       BayesianLinearRegression, poly_kernel, cluster_purity)
from scipy.special import logsumexp
from typing import Tuple
from scipy.stats import norm
from scipy.stats import multivariate_normal

from scipy.stats import wishart
import matplotlib
matplotlib.use('TkAgg')

def outlier_regression(model: BayesianLinearRegression, X: np.ndarray, y: np.ndarray, p_out: float, T: int,
                       mu_o: float=0, sig_o: float=10) -> Tuple[BayesianLinearRegression, np.ndarray]:
    """
    Gibbs sampling algorithm for robust regression (i.e. regression assuming there are outliers in the data)
    :param model: the Bayesian linear regression that will be used to fit the data
    :param X: the training data, as a numpy array of shape [N, d] where N is the number of points and d is the dimension
    :param y: the regression targets, as a numpy array of shape [N,]
    :param p_out: the assumed probability for outliers in the data
    :param T: number of Gibbs sampling iterations to use in order to fit the model
    :param mu_o: the assumed mean of the outlier points
    :param sig_o: the assumed variance of the outlier points
    :return: the fitted model assuming outliers, as a BayesianLinearRegression model, as well as a numpy array of the
             indices of points which were considered as outliers
    """
    k_t = np.zeros(X.shape[0]) # the probability for each value to be an outlier
    model.fit(X,y)
    for t in range(T):
        for i,sample  in enumerate(X):
            p_y_given_outlier = norm.pdf(y[i], loc=mu_o, scale=sig_o)
            y_i = np.array([y[i]])
            if X[i].ndim == 0:
                x_i = np.array([X[i]])  # Reshape to (1, d)
            else:
                x_i = np.array([np.ravel(X[i])])   # Keep as is
            yi_likelihood = np.exp(model.log_likelihood(x_i,y_i))
            p = p_out*p_y_given_outlier/(p_out*p_y_given_outlier+ (1-p_out)*yi_likelihood)
            k_t[i]  = np.random.choice([0, 1], p=[1-p[0], p[0]])
        inliers_mask = (k_t == 0)
        inliers_x = X[inliers_mask]
        inliers_y = y[inliers_mask]
        model.fit(inliers_x,inliers_y)
    outliers = np.where(k_t == 1)[0]
    return model, outliers


class BayesianGMM:
    def __init__(self, k: int, alpha: float, mu_0: np.ndarray, sig_0: float, nu: float, beta: float,
                 learn_cov: bool=True):
        """
        Initialize a Bayesian GMM model
        :param k: the number of clusters to use
        :param alpha: the value of alpha to use for the Dirichlet prior over the mixture probabilities
        :param mu_0: the mean of the prior over the means of each Gaussian in the GMM
        :param sig_0: the variance of the prior over the means of each Gaussian in the GMM
        :param nu: the nu parameter of the inverse-Wishart distribution used as a prior for the Gaussian covariances
        :param beta: the variance of the inverse-Wishart distribution used as a prior for the covariances
        :param learn_cov: a boolean indicating whether the cluster covariances should be learned or not
        """
        # constant
        self.k = k
        self.alpha = alpha
        self.d = mu_0.shape[0]
        self.nu = nu
        self.beta = beta
        #priors
        self.mu_0 = mu_0
        self.sig_0 = sig_0

        #updating values
        self.learn_cov = learn_cov
        Psi = np.linalg.inv(nu * np.eye(self.d)*self.beta)
        self.cov_k  = np.array([np.eye(self.d) * self.beta for _ in range(self.k)])
        self.pi = np.random.dirichlet([self.alpha] * self.k)
        self.mu_k = np.random.multivariate_normal(self.mu_0, self.sig_0 * np.eye(self.d), self.k)

        self.z = None
        self.N_k = None

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the log-likelihood of each data point under each Gaussian in the GMM
        :param X: the data points whose log-likelihood should be calculated, as a numpy array of shape [N, d]
        :return: the log-likelihood of each point under each Gaussian in the GMM
        """
        l_likelihood = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            if self.learn_cov:
                l_likelihood[:, k] = multivariate_normal.logpdf(X, mean=self.mu_k[k], cov=self.cov_k[k])
            else:
                l_likelihood[:, k] = -0.5 * (
                        self.d * np.log(2 * np.pi * self.beta) +
                        (1 / self.beta) * np.sum((X - self.mu_k[k]) ** 2, axis=1)
                )
        return l_likelihood

    def cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Clusters the data according to the learned GMM
        :param X: the data points to be clustered, as a numpy array of shape [N, d]
        :return: a numpy array containing the indices of the Gaussians the data points are most likely to belong to,
                 as a numpy array with shape [N,]
        """
        N, d = X.shape
        # Pre-compute log-likelihood for all clusters and all points
        log_probs = np.zeros((N, self.k))
        for k in range(self.k):
            log_probs[:, k] = multivariate_normal.logpdf(X, mean=self.mu_k[k], cov=self.cov_k[k])

        # Assign each point to the cluster with the highest log-likelihood
        clustered_data = np.argmax(log_probs, axis=1)

        return clustered_data

    def gibbs_fit(self, X:np.ndarray, T: int) -> 'BayesianGMM':
        """
        Fits the Bayesian GMM model using a Gibbs sampling algorithm
        :param X: the training data, as a numpy array of shape [N, d] where N is the number of points
        :param T: the number of sampling iterations to run the algorithm
        :return: the fitted model
        """
        N, d = X.shape

        for t in range(T):
            print("gibbs fit iteration number: ",t)
            q = self.calculate_q(X)

            self.z = self.calculate_z(X, q)
            self.N_k = self.calculate_N()
            self.pi = self.calculate_pi()
            self.learn_mu(X)
            if self.learn_cov:
                self.learn_covariance(X)
        return self

    def calculate_q(self, X):
        """
        Calculates the responsibility matrix q, where q[i, k] is the probability of data point i
        belonging to cluster k.
        :param X: The data points as a numpy array of shape [N, d]
        :return: A numpy array of shape [N, K], where each row sums to 1
        """
        log_likelihoods = self.log_likelihood(X)  # Shape: (N, K)

        # Compute log of unnormalized responsibilities
        log_probs = log_likelihoods + np.log(self.pi)  # Add log(pi) for each cluster

        # Normalize using logsumexp for numerical stability
        log_q = log_probs - logsumexp(log_probs, axis=1, keepdims=True)  # Subtract row-wise log-sum-exp

        # Convert back to probabilities (responsibilities)
        q = np.exp(log_q)  # Shape: (N, K)

        return q

    def calculate_z(self, X, q):
        z = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            z[i] = np.random.choice(len(q[i]),p=q[i])
        return z

    def calculate_N(self):
        N = np.zeros(self.k)
        for k in range(self.k):
            N[k]  = np.sum(self.z == k)
        return N
    def calculate_pi(self):
        return np.random.dirichlet(self.alpha+self.N_k)

    def learn_covariance(self, X: np.ndarray):
        for k in range(self.k):
            x_diff = X[self.z == k] - self.mu_k[k]
            S_k = x_diff.T @ x_diff
            self.cov_k[k] = (self.nu * np.eye(self.d) * self.beta + S_k) / (self.nu + self.N_k[k])

    def learn_mu(self, X: np.ndarray):
        for k in range(self.k):
            x_sum = X[self.z == k].sum(axis=0)
            cov_inv = np.linalg.inv(self.cov_k[k])
            precision = self.N_k[k] * cov_inv + np.eye(self.d) / self.sig_0
            mean = np.linalg.solve(precision, cov_inv @ x_sum + self.mu_0 / self.sig_0)
            self.mu_k[k] = np.random.multivariate_normal(mean, np.linalg.inv(precision))


if __name__ == '__main__':
    # ------------------------------------------------------ section 2 - Robust Regression
    # ---------------------- question 2
    # load the outlier data
    x, y = outlier_data(50)
    # init BLR model that will be used to fit the data
    mdl = BayesianLinearRegression(theta_mean=np.zeros(2), theta_cov=np.eye(2), sample_noise=0.15)

    # sample using the Gibbs sampling algorithm and plot the results
    plt.figure()
    plt.scatter(x, y, 15, 'k', alpha=.75)
    xx = np.linspace(-0.2, 5.2, 100)
    for t in [0, 1, 5, 10, 25]:
        samp, outliers = outlier_regression(mdl, x, y, T=t, p_out=0.1, mu_o=4, sig_o=2)
        plt.plot(xx, samp.predict(xx), lw=2, label=f'T={t}')
    plt.xlim([np.min(xx), np.max(xx)])
    plt.legend()
    plt.show()
    #
    # ---------------------- question 3
    # load the images to use for classification
    N = 1000
    ims, labs = load_dogs_vs_frogs(N)
    # define BLR model that should be used to fit the data
    mdl = BayesianLinearRegression(sample_noise=0.001, kernel_function=poly_kernel(2))
    # use Gibbs sampling to sample model and outliers
    samp, outliers = outlier_regression(mdl, ims, labs, p_out=0.01, T=50, mu_o=0, sig_o=.5)
    # plot the outliers
    plot_ims(ims[outliers], title='outliers')

    # ------------------------------------------------------ section 3 - Bayesian GMM
    # ---------------------- question 5
    # load 2D GMM data
    k, N = 5, 1000
    X = gmm_data(N, k)
    threshold = 1e-4
    clusters_above_threshold = []
    for i in range(5):
        gmm = BayesianGMM(k=50, alpha=.01, mu_0=np.zeros(2), sig_0=.5, nu=5, beta=.5)
        gmm.gibbs_fit(X, T=100)

        # plot a histogram of the mixture probabilities (in descending order)
        pi = gmm.pi  # mixture probabilities from the fitted GMM
        num_clusters = np.sum(pi > threshold)
        clusters_above_threshold.append(num_clusters)
        plt.figure()
        plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
        plt.ylabel(r'$\pi_k$')
        plt.xlabel('cluster number')
        plt.show()

        # plot the fitted 2D GMM
        plot_2D_gmm(X, gmm.mu_k, gmm.cov_k, gmm.cluster(X))  # the second input are the means and the third are the covariances
    average_clusters = np.mean(clusters_above_threshold)
    print(f"On average, {average_clusters:.2f} clusters have $\pi_k > {threshold}$ across 5 runs.")



    # ---------------------- questions 6-7
    # load image data
    MNIST, labs = load_MNIST()
    # flatten the images
    ims = MNIST.copy().reshape(MNIST.shape[0], -1)
    gmm = BayesianGMM(k=500, alpha=1, mu_0=0.5*np.ones(ims.shape[1]), sig_0=.1, nu=1, beta=.25, learn_cov=False)
    gmm.gibbs_fit(ims, 100)

    # plot a histogram of the mixture probabilities (in descending order)
    pi = gmm.pi  # mixture probabilities from the fitted GMM
    plt.figure()
    plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
    plt.ylabel(r'$\pi_k$')
    plt.xlabel('cluster number')
    plt.show()
    num_clusters = np.sum(pi > threshold)
    print("num of clusters over threshhold:",num_clusters )
    # find the clustering of the images to different Gaussians
    cl = gmm.cluster(ims)
    clusters = np.unique(cl)
    print(f'{len(clusters)} clusters used')
    # calculate the purity of each of the clusters
    purities = np.array([cluster_purity(labs[cl == k]) for k in clusters])
    purity_inds = np.argsort(purities)
    pi = gmm.pi  # mixture probabilities from the fitted GMM
    # plot 25 images from each of the clusters with the top 5 purities
    for ind in purity_inds[-5:]:
        clust = clusters[ind]
        plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')

    # plot 25 images from each of the clusters with the bottom 5 purities
    for ind in purity_inds[:5]:
        clust = clusters[ind]
        plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')

