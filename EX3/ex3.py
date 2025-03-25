import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior
plt.interactive(False)

def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    H = model.h(X)
    N, P = H.shape
    log_det_ratio = np.log(np.linalg.det(map_cov) / np.linalg.det(sig))
    quadratic_term = (map - mu).T @ np.linalg.inv(sig) @ (map - mu)
    norm_term = np.linalg.norm(y - H @ map) ** 2
    log_p = (
        0.5 * log_det_ratio
        - 0.5 * (quadratic_term + norm_term/n + N * np.log(n))
        - 0.5 * P * np.log(2 * np.pi)
    )
    return log_p

def plot_model(x, y, f, y_hat, model_std, label, color, title):
    # Plot observed data and true function
    plt.plot(x, y, ".", label="Data", alpha=0.5)
    plt.plot(x, f(x), "k--", label="True Function", linewidth=1)
    # Plot model mean and confidence interval
    plt.plot(x, y_hat, f"{color}", label=f"{label}")
    plt.fill_between(
        x,
        y_hat - model_std,
        y_hat + model_std,
        color=color,
        alpha=0.2,
        label=f"{label} CI",
    )

    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: (-x ** 2 + 10 * x ** 3 + 50 * np.sin(x / 6) + 10) / 100
    f3 = lambda x: (.5 * x ** 6 - .75 * x ** 4 + 2.75 * x ** 2) / 50
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: 1 * (np.cos(x * 4) + 4 * np.abs(x - 2))
    functions = [f1, f2, f3, f4, f5]

    noise_var = .25
    x = np.linspace(-3, 3, 500)

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alpha = 1
    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        evs = []
        models = []
        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            model = BayesianLinearRegression(mean, cov, noise_var, pbf)
            ev = log_evidence(model, x, y)
            evs.append(ev)
            models.append(model)

        plt.figure(figsize=(12, 8))
        plt.plot(degrees, evs, label=f'Function {i + 1}')
        plt.xlabel('Degree of Polynomial Basis', fontsize=14)
        plt.ylabel('Log-Evidence', fontsize=14)
        plt.title('Log-Evidence vs. Degree of Polynomial Basis', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.show()

        # Find best and worst models
        best_model = models[np.argmax(evs)].fit(x, y)
        worst_model = models[np.argmin(evs)].fit(x, y)

        best_degree = degrees[np.argmax(evs)]
        worst_degree = degrees[np.argmin(evs)]

        best_y_hat = best_model.predict(x)
        best_std = best_model.predict_std(x)

        worst_y_hat = worst_model.predict(x)
        worst_std = worst_model.predict_std(x)

        # Plot both models in the same figure
        plt.figure(figsize=(12, 8))

        # Plot data and true function
        plt.plot(x, y, ".", label="Data", alpha=0.5)
        plt.plot(x, f(x), "k--", label="True Function", linewidth=1)

        # Plot best model
        plt.plot(x, best_y_hat, "g", label=f"Best Model (Degree {best_degree})")
        plt.fill_between(
            x,
            best_y_hat - best_std,
            best_y_hat + best_std,
            color="green",
            alpha=0.2,
            label="Best Model CI",
        )

        # Plot worst model
        plt.plot(x, worst_y_hat, "r", label=f"Worst Model (Degree {worst_degree})")
        plt.fill_between(
            x,
            worst_y_hat - worst_std,
            worst_y_hat + worst_std,
            color="red",
            alpha=0.2,
            label="Worst Model CI",
        )

        plt.xlabel("x", fontsize=14)
        plt.ylabel("y", fontsize=14)
        plt.title(f"Fitted Points for Function {i + 1} (Best & Worst Models)", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162024.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        evs[i] = ev

    # plot log-evidence versus amount of sample noise
    plt.figure(figsize=(8, 6))
    plt.plot(noise_vars, evs, label="Log Evidence", color="blue", linewidth=2)
    plt.title("Log Evidence vs. Noise Variance", fontsize=14)
    plt.xlabel("Noise Variance", fontsize=12)
    plt.ylabel("Log Evidence", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    #print the sample noise with the highest evidence:
    highest = np.argmax(evs)
    print("Sample noise with the highest evidence: ", noise_vars[highest])

if __name__ == '__main__':
    main()



