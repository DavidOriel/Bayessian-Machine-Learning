import numpy as np
from matplotlib import pyplot as plt
from ex5_utils import load_im_data, BayesianLinearRegression, gaussian_basis_functions, accuracy, Gaussian, plot_ims
import matplotlib
matplotlib.use('TkAgg')
def calculate_post_mu(x,mu_0,sig_0,sig):
    N = len(x)
    var = 1 /(N*1/sig + 1/sig_0)
    mean = (1/sig*np.sum(x,axis=0)+1/sig_0*mu_0)*var
    return np.random.multivariate_normal(mean, var * np.eye(x.shape[1]))


def calculate_decision_boundary(x, mu_pos, mu_neg):
    # Compute the squared norms of the means
    norm_mu_pos_squared = np.linalg.norm(mu_pos) ** 2
    norm_mu_neg_squared = np.linalg.norm(mu_neg) ** 2

    # Compute the difference between means
    mu_diff = mu_pos - mu_neg

    # Compute the components of the decision boundary
    constant_term = (norm_mu_pos_squared - norm_mu_neg_squared) / (2 * mu_diff[1])
    slope = -mu_diff[0] / mu_diff[1]

    # Compute the decision boundary (y-value)
    y = constant_term + slope * x
    return y
def main():
    # ------------------------------------------------------ section 1
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_p, mu_m = np.array([1, 1]), np.array([-1, -1])

    # sample 5 points from each class
    np.random.seed(0)
    x_p = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_m = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)

    # <your code here>
    mu_pos = calculate_post_mu(x_p,mu_p,sig,sig)
    mu_neg = calculate_post_mu(x_m,mu_m,sig,sig)

    x_range = np.linspace(-2, 2, 100)

    # Calculate and plot the MMSE decision boundary
    mmse_boundary = calculate_decision_boundary(x_range, mu_pos, mu_neg)

    # Plot sampled points and MMSE decision boundary
    plt.figure(figsize=(10, 6))
    plt.scatter(x_p[:, 0], x_p[:, 1], color='blue', label='Class + (5 points)')
    plt.scatter(x_m[:, 0], x_m[:, 1], color='red', label='Class - (5 points)')
    plt.plot(x_range, mmse_boundary, label='MMSE Decision Boundary', color='green', linestyle='--')

    # Sample and plot 10 decision boundaries from posterior
    for _ in range(10):
        sampled_mu_pos = calculate_post_mu(x_p, mu_p, sig_0, sig)
        sampled_mu_neg = calculate_post_mu(x_m, mu_m, sig_0, sig)
        sampled_boundary = calculate_decision_boundary(x_range, sampled_mu_pos, sampled_mu_neg)
        plt.plot(x_range, sampled_boundary, color='gray', linestyle=':', alpha=0.5)

    # Plot settings
    plt.title('MMSE and Sampled Decision Boundaries')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


    # ------------------------------------------------------ section 2
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()

    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])

    # ------------------------------------------------------ section 2.1
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)
        dogs_gaus = Gaussian(beta = beta,nu = nu)
        dogs_gaus.fit(dogs)
        frogs_gaus = Gaussian(beta=beta, nu = nu)
        frogs_gaus.fit(frogs)
        # Compute log-likelihoods and predictions for training set
        log_likelihood_train_dogs = dogs_gaus.log_likelihood(train)
        log_likelihood_train_frogs = frogs_gaus.log_likelihood(train)
        y_hat_train = np.argmax([log_likelihood_train_dogs, log_likelihood_train_frogs], axis=0)

        # Compute log-likelihoods and predictions for test set
        log_likelihood_test_dogs = dogs_gaus.log_likelihood(test)
        log_likelihood_test_frogs = frogs_gaus.log_likelihood(test)
        y_hat_test = np.argmax([log_likelihood_test_dogs, log_likelihood_test_frogs], axis=0)
        y_hat_train = np.where(y_hat_train == 0, 1, -1)  # Maps 0 -> 1, 1 -> -1
        y_hat_test = np.where(y_hat_test == 0, 1, -1)
        # Compute accuracies
        train_score[i] = accuracy(y_hat_train , labels)
        test_score[i] = accuracy(y_hat_test , labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(r'value of $\nu$')
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define question variables
    beta = .02
    sigma = .1
    Ms = [250, 500, 750, 1000, 2000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ms)), np.zeros(len(Ms))

    blr = None
    for i, M in enumerate(Ms):
        print(f'Gaussian basis functions using {M} samples', end='', flush=True)
        dogs_samples = dogs[np.random.choice(dogs.shape[0], M, replace=False)]
        frogs_samples = frogs[np.random.choice(frogs.shape[0], M, replace=False)]
        label_samples = np.concatenate([np.ones(M), -np.ones(M)])
        centers = np.concatenate([dogs_samples, frogs_samples], axis=0)
        bf = gaussian_basis_functions(centers = centers, beta=beta)
        blr = BayesianLinearRegression(theta_mean=np.zeros(2*M), theta_cov=np.eye(2*M), sig=sigma,
                                       basis_functions=bf).fit(train, labels)
        train_score[i]= accuracy(blr.predict(train), labels)
        test_score[i] = accuracy(blr.predict(test), labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ms, train_score, lw=2, label='train')
    plt.plot(Ms, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('# of samples')
    plt.xscale('log')
    plt.show()

    # calculate how certain the model is about the predictions
    d = np.abs(blr.predict(dogs_t) / blr.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')


if __name__ == '__main__':
    main()







