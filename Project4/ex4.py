import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Spectral': r'Spectral, $\alpha={}$, $\beta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def average_error(pred: np.ndarray, vals: np.ndarray):
    """
    Calculates the average squared error of the given predictions
    :param pred: the predicted values
    :param vals: the true values
    :return: the average squared error between the predictions and the true values
    """
    return np.mean((pred - vals)**2)


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        x = np.array(x)
        y = np.array(y)
        ker = alpha* np.exp(-beta*np.linalg.norm(x-y)**2)
        return ker
    return kern


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        ker = alpha* np.exp(-beta*np.abs(x-y))
        return ker
    return kern


def Spectral_kernel(alpha: float, beta: float, gamma: float) -> Callable:
    """
    An implementation of the Spectral kernel (see https://arxiv.org/pdf/1302.4245.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        x = np.array(x)
        y = np.array(y)
        ker = alpha * np.exp(-beta * np.linalg.norm(x - y)**2) * np.cos(np.pi * np.linalg.norm(x - y) / gamma)
        return ker
    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        x = np.array(x)
        y = np.array(y)
        numerator = 2 * beta * (np.dot(x, y) + 1)
        denominator = np.sqrt((1 + 2 * beta * (np.dot(x, x) + 1)) * (1 + 2 * beta * (np.dot(y, y) + 1)))
        ker = alpha * (2 / np.pi) * np.arcsin(numerator / denominator)
        return ker
    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        self.kernel = kernel
        self.noise = noise
        self.K = None

    def calculate_k(self,X):
        n  = X.shape[0]
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n)] for i in range(n)])
        K+= self.noise * np.eye(X.shape[0])
        return K
    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        n = X.shape[0]
        self.K = np.zeros((n, n))
        self.X_train = X
        self.y_train = y
        for i in range(n):
            for j in range(n):
                val = self.kernel(X[i], X[j])
                self.K[i, j] = val
        self.K = self.K + self.noise*np.eye(n)
        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        n_train = self.X_train.shape[0]
        n_test = X.shape[0]
        K_star = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_star[i, j] = self.kernel(X[i], self.X_train[j])
        y_hat = K_star @ self.alpha
        return y_hat

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        n_train = self.X_train.shape[0]  # Number of training points
        n_test = X.shape[0]  # Number of test points
        K_star = np.array([[self.kernel(X[i], self.X_train[j]) for j in range(n_train)] for i in range(n_test)])
        k_test = np.array([self.kernel(X[i], X[i]) for i in range(n_test)])
        K_inv = np.linalg.inv(self.K)  # Assumes self.K was precomputed during training
        mu = K_star @ K_inv @ self.y_train
        cov = k_test - K_star @ K_inv @ K_star.T
        samples = np.random.multivariate_normal(mu, cov)
        return samples

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        n_test = X.shape[0]
        K_star = np.array([[self.kernel(X[i], self.X_train[j]) for j in range(self.X_train.shape[0])] for i in range(n_test)])
        k_test = np.array([self.kernel(X[i], X[i]) for i in range(n_test)])
        v = np.linalg.solve(self.L,K_star.T)
        V_f = k_test - v.T@v
        std = np.sqrt(np.diagonal(V_f))
        return std

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        n = X.shape[0]
        ev = -.5*y.T@self.alpha -np.sum(np.log(np.diag(self.L)))-n/2*np.log(np.pi*2)
        return ev


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([2.4, .9, 2.8, -2.9, -1.5])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],           # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 0.5, 0.5],        # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 0.1, 1],        # insert your parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],                       # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 0.5, 0.5],                    # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 0.1, 1],                    # insert your parameters, order: alpha, beta

        # Gibbs kernels
        ['Spectral', Spectral_kernel, 1, .25, 1],            # insert your parameters, order: alpha, beta, gamma
        ['Spectral', Spectral_kernel, 0.5, 0.5, 2],    # insert your parameters, order: alpha, beta, gamma
        ['Spectral', Spectral_kernel, 0.25, 1, 3],    # insert your parameters, order: alpha, beta, gamma

        # Neurel network kernels
        ['NN', NN_kernel, 1, 0.25],                         # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, .5, 0.5],                      # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 0.25, 1],                      # insert your parameters, order: alpha, beta
    ]
    noise_var = 0.05

    # plot all of the chosen parameter settings
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])    # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise_var)
        # plot prior variance and samples from the priors
        mean_prior = np.zeros(len(xx))
        k = gp.calculate_k(xx)
        samples = np.random.multivariate_normal(mean_prior, k, size=5)

        # Plot prior variance and samples
        plt.figure()
        plt.fill_between(xx, -2 * np.sqrt(np.diag(k)), 2 * np.sqrt(np.diag(k)), alpha=0.3)
        plt.plot(xx, mean_prior, 'k--', lw=2, label="Prior Mean")  # Mean is zero
        for i, sample in enumerate(samples):
            plt.plot(xx, sample, lw=1, label=f"Sample {i + 1}" if i == 0 else None)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])

        # fit the GP to the data and calculate the posterior mean and confidence interval
        gp.fit(x, y)
        m, s = gp.predict(xx), 2*gp.predict_std(xx)

        # plot posterior mean, confidence intervals and samples from the posterior
        plt.figure()
        plt.fill_between(xx, m-s, m+s, alpha=.3)
        plt.plot(xx, m, lw=2)
        for i in range(6): plt.plot(xx, gp.sample(xx), lw=1)
        plt.scatter(x, y, 30, 'k', zorder=10)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        plt.show()

    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(0.1, 7, 101)
    noise_var = .27

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise_var).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    # plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    min_ev, median_ev, max_ev = betas[srt[0]], betas[srt[(len(evidence)+1)//2]], betas[srt[-1]]
    print(f"Beta with highest evidence: {max_ev}")
    print(f"Beta with lowest evidence: {min_ev}")
    print(f"Beta with median evidence: {median_ev}")
    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure()
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=min_ev), noise_var).fit(x, y).predict(xx), lw=2, label='min evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=median_ev), noise_var).fit(x, y).predict(xx), lw=2, label='median evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=max_ev), noise_var).fit(x, y).predict(xx), lw=2, label='max evidence')
    plt.scatter(x, y, 30, 'k', alpha=.5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()



