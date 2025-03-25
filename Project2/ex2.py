import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
import matplotlib
matplotlib.use('TkAgg')

def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        pol = np.vander(x/degree, degree+1, increasing=True)
        pol[:,0] = 1
        return pol
    return pbf


def fourier_basis_functions(num_freqs: int) -> Callable:
    """
    Create a function that calculates the fourier basis functions up to a certain frequency
    :param num_freqs: the number of frequencies to use
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Fourier basis functions, a numpy array of shape [N, 2*num_freqs + 1]
    """
    def fbf(x: np.ndarray):
        x = np.asarray(x).reshape(-1, 1)  # Reshape to (N, 1) for broadcasting

        # Precompute the frequency terms
        const = 2 * np.pi / 24
        freqs = np.arange(1, num_freqs + 1).reshape(1, -1)  # (1, num_freqs)

        # Compute cosine and sine values for each frequency
        cos_vals = np.cos(const * freqs * x)  # Shape: (N, num_freqs)
        sin_vals = np.sin(const * freqs * x)  # Shape: (N, num_freqs)
        return  np.hstack([np.ones((x.shape[0], 1)), cos_vals, sin_vals])
    return fbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        poly_terms = np.vander(x, 4, increasing=True)
        diff = x[:, None] - knots[None, :]
        spline_terms = np.maximum(diff, 0) ** 3
        H = np.hstack([poly_terms, spline_terms ])
        return H
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.fitted_model)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]

    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # sigma_post = np.linalg.inv(np.linalg.inv(self.theta_cov)+ 1/np.power(self.sig, 2) @ self.basis_functions(X))
        H = self.basis_functions(X)
        H_sig_HT = H @ self.theta_cov @ H.T
        M = self.sig * np.eye(H.shape[0]) + H_sig_HT
        self.sigma_post = self.theta_cov-self.theta_cov@ H.T @ np.linalg.inv(M)@ H @ self.theta_cov
        self.mu_post = self.theta_mean+self.theta_cov @ H.T @ np.linalg.inv(M) @ (y-H @ self.theta_mean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        y_pred = H @ self.mu_post
        return y_pred

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.basis_functions(X)
        cov = H @ self.sigma_post @ H.T + np.eye(H.shape[0])* self.sig
        posterior_cov = H @ self.sigma_post @ H.T
        std = np.sqrt(np.diagonal(posterior_cov) + self.sig )
        return std

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        predict = np.random.multivariate_normal(self.mu_post,cov = self.sigma_post )
        sampled_functions = predict @ H.T  # Project sampled thetas onto the design matrix
        return sampled_functions

class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_functions = basis_functions
        self.fitted_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H= self.basis_functions(X)
        self.fitted_model = np.linalg.inv(H.T @ H) @ H.T @ y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        y_pred =  H @ self.fitted_model
        return y_pred

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def plot_prior(x: np.ndarray, mu: np.ndarray, cov: np.ndarray, basis_func: Callable, label_param, label_type="degree"):
    """
    Plot the prior mean, confidence interval, and sampled functions.

    :param x: Range of x-values for plotting.
    :param mu: Prior mean vector.
    :param cov: Prior covariance matrix.
    :param basis_func: Basis function generator.
    :param label_param: Parameter to describe the model (e.g., degree for polynomial, frequency for Fourier).
    :param label_type: Type of label, 'degree' for polynomial or 'frequency' for Fourier.
    """
    # Design matrix for the input range x
    H = basis_func(x)

    # Compute the mean and standard deviation of the prior
    mean_prior = H @ mu
    std_prior = np.sqrt(np.diagonal(H @ cov @ H.T))

    # Perform Cholesky decomposition for sampling
    chol = np.linalg.cholesky(cov)

    # Plot the prior mean and confidence interval
    plt.figure()
    plt.fill_between(x, mean_prior - std_prior, mean_prior + std_prior, color='lightblue', alpha=0.5,
                     label='Confidence Interval')
    plt.plot(x, mean_prior, 'k', lw=2, label='Prior Mean')  # Prior mean as a solid black line

    # Sample and plot 5 functions from the prior
    for i in range(5):
        sampled_theta = mu + chol @ np.random.randn(chol.shape[-1])  # Sample from N(mu, cov)
        sampled_function = H @ sampled_theta  # Project sampled theta onto the design matrix
        plt.plot(x, sampled_function, linestyle='--', alpha=0.7, label='Sampled Function' if i == 0 else None)

    # Plot styling
    plt.title(f'Prior Plot - {label_type.capitalize()} = {label_param}')
    plt.xlabel('hour')
    plt.ylabel('temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_posterior(test_hours, test: np.ndarray, blr, basis_func, train_hours, train, label_param, label_type="degree"):
    """
    Compute and plot the posterior mean, confidence interval, and sampled functions
    alongside training and test points.

    :param test_hours: Hours corresponding to the test set.
    :param test: Test set values.
    :param blr: Bayesian Linear Regression model.
    :param basis_func: Basis function generator.
    :param train_hours: Training hours.
    :param train: Training values.
    :param label_param: Parameter to describe the model (e.g., degree for polynomial, frequency for Fourier).
    :param label_type: Type of label, 'degree' for polynomial or 'frequency' for Fourier.
    """
    # Design matrix for test points
    H_test = basis_func(test_hours)

    # Compute posterior mean and standard deviation
    mean_post = H_test @ blr.mu_post
    std_post = np.sqrt(np.diagonal(H_test @ blr.sigma_post @ H_test.T) + blr.sig**2)

    # Generate samples from the posterior
    samples_post = blr.posterior_sample(test_hours)  # Use posterior_sample method

    # Compute MMSE error
    mse_mmse = np.mean((test - mean_post) ** 2)
    print(f'Average squared error with Bayesian LR (MMSE) and {label_type} {label_param}: {mse_mmse:.2f}')

    # Plot posterior results
    plt.figure()

    # Plot training and test points
    # plt.scatter(train_hours, train, color='blue', label='Train Points', alpha=0.7)
    plt.scatter(test_hours, test, color='orange', label='Test Points', alpha=0.7)

    # Plot confidence interval

    plt.fill_between(test_hours, mean_post - std_post, mean_post + std_post, alpha=0.5,
                     color='lightblue', label='Confidence Interval')

    # Plot posterior mean
    plt.plot(test_hours, mean_post, 'k', lw=2, label='Posterior Mean')

    # Plot sampled functions
    for i in range(5):
        samples_post = blr.posterior_sample(test_hours)
        plt.plot(test_hours, samples_post, alpha=0.7, label='Sampled Function' if i == 0 else None)

    # Plot styling
    plt.text(0.05, 0.95, f'MMSE: {mse_mmse:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title(f'Bayesian Linear Regression - {label_type.capitalize()} = {label_param}')
    plt.xlabel('hour')
    plt.ylabel('temperature')
    plt.legend()
    plt.grid(True)
    plt.show()
def main():
    # load the data for November 16 2024
    nov16 = np.load('nov162024.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)
        # print average squared error performance
        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours))**2):.2f}')

        # plot graphs for linear regression part
        plt.scatter(test_hours, test, label=f'Test Points (d={d})', alpha=0.7)  # Test points
        plt.plot(test_hours, ln.predict(test_hours), label=f'Predicted Curve (d={d})', linewidth=2)  # Predicted curve

    # Add labels, legend, and show the plot
    plt.xlabel('Hours')
    plt.ylabel('Target')
    plt.title('Test Points and Predicted Curves for Different Polynomial Degrees')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees

    # frequencies for Fourier basis
    freqs = [1, 2, 3]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        # Learn prior
        prior_mu, prior_cov = learn_prior(hours, temps, pbf)
        # Fit Bayesian Linear Regression
        blr = BayesianLinearRegression(prior_mu, prior_cov, sigma, pbf)
        blr.fit(train_hours, train)
        # Prior plot
        plot_prior(x, prior_mu, prior_cov, pbf, deg, label_type="degree")

        # Posterior plot
        plot_posterior(test_hours, test, blr, pbf, train_hours, train, deg, label_type="degree")

        # ---------------------- fourier basis functions
    for K in freqs:
        rbf = fourier_basis_functions(K)
        # Learn prior
        prior_mu, prior_cov = learn_prior(hours, temps, rbf)
        # Fit Bayesian Linear Regression
        blr = BayesianLinearRegression(prior_mu, prior_cov, sigma, rbf)
        blr.fit(train_hours, train)

        # Plot Prior
        plot_prior(x, prior_mu, prior_cov, rbf, K, label_type="frequency")

        # Plot Posterior
        plot_posterior(test_hours, test, blr, rbf, train_hours, train, K, label_type="frequency")

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)
        blr.fit(train_hours, train)
        # plot prior graphs
        plot_prior(x, mu, cov, spline, (ind,k), label_type="ind, k ")

        # plot posterior graphs
        plot_posterior(test_hours, test, blr, spline, train_hours, train, (ind,k), label_type="ind, k ")


if __name__ == '__main__':
    main()
