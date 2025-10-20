import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import powerlaw as pwl
import pycop
import seaborn as sns
import openturns as ot
import openturns.viewer as viewer
from scipy.stats import entropy
import scipy.stats as st
 

def histogram(data, num_bins=31, log_bin=True, density=True):
     
    """
    Compute a histogram with configurable binning.

    Parameters:
    - data: array-like, input data
    - num_bins: int, number of bins (default: 31)
    - log_bin: bool, use logarithmic binning (default: True)
    - density: bool, normalize histogram to density (default: True)

    Returns:
    - centers: array, bin centers
    - hist: array, histogram counts
    - widths: array, bin widths
    - edges: array, bin edges
    """
    
    if len(data) == 0:
        raise ValueError("Input data must not be empty.")

    min_val, max_val = min(data), max(data)
    
    if log_bin:
        if min_val <= 0:
            # raise ValueError("Logarithmic binning requires strictly positive data.")
            data = data[data>0]
            print('Removed negative and zero values from data.')
        bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
    else:
        bins = np.linspace(min_val, max_val, num_bins)

    # Compute histogram
    hist, edges = np.histogram(data, bins=bins, density=density,)

    # Compute bin centers and widths
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]

    # Ensure consistent filtering
    non_zero = hist > 0
    centers, hist, widths = centers[non_zero], hist[non_zero], widths[non_zero]


    return centers, hist, widths, edges



def empirical_extremal_function(data, link_formula, survival=False, conf_level=0.95, points=100):
    n = len(data)
    ranks = (st.rankdata(data, axis=0)) / (n + 1)
    u_grid = np.linspace(1/(points+1), points/(points+1), points)
    
    values = np.zeros(points)
    ci_low = np.zeros(points)
    ci_up = np.zeros(points)
    
    z = st.norm.ppf(0.5 + 0.5 * conf_level)
    
    for i, u in enumerate(u_grid):
        if survival:
            cuu = np.mean((u < np.minimum(ranks[:, 0], ranks[:, 1])))
        else:
            cuu = np.mean((u > np.maximum(ranks[:, 0], ranks[:, 1])))

        # Adjust cuu to avoid log issues
        epsilon = np.finfo(float).eps
        cuu = np.clip(cuu, epsilon, 1 - epsilon)

        # Estimate
        values[i] = link_formula(u, cuu)

        # Confidence interval for cuu
        se = np.sqrt(cuu * (1 - cuu) / n)
        lb = cuu - z * se
        ub = cuu + z * se
        lb = np.clip(lb, epsilon, 1 - epsilon)
        ub = np.clip(ub, epsilon, 1 - epsilon)

        # CI for function
        ci_low[i] = np.clip(link_formula(u, lb), -1, 1)
        ci_up[i] = np.clip(link_formula(u, ub), -1, 1)

    return u_grid, values, ci_low, ci_up

# Link formulas translated to Python
def chi(u, cuu):
    return 2 - np.log(cuu) / np.log(u)

def chi_bar(u, cuu):
    return 2 * np.log(1 - u) / np.log(cuu) - 1

def chi_L(u, cuu):
    return np.log(1 - cuu) / np.log(1 - u)

def chi_bar_L(u, cuu):
    return 2 * np.log(u) / np.log(cuu) - 1

# Interface functions for clarity
def draw_upper_tail(data):
    return empirical_extremal_function(data, chi, survival=False)

def draw_upper_extremal(data):
    return empirical_extremal_function(data, chi_bar, survival=True)

def draw_lower_tail(data):
    return empirical_extremal_function(data, chi_L, survival=False)

def draw_lower_extremal(data):
    return empirical_extremal_function(data, chi_bar_L, survival=False)

# # Example usage (assuming your data is available as x,y):
# # data = agents[['activity','attractiveness']].values
# data=agents[['tx_out','tx_in']].values

# u, val, low, up = draw_upper_tail(data)

# # Example plotting
# plt.figure(figsize=(8, 5))
# plt.plot(u, val, label='Estimate', color='red')
# plt.fill_between(u, low, up, color='blue', alpha=0.2, label='CI')
# plt.xlabel('$u$',rotation = 0,labelpad=20)
# plt.ylabel('$\\chi(u)$',rotation = 0,labelpad=20)
# plt.title('Upper Tail Dependence Function')
# # plt.grid()
# plt.legend()
# plt.show()

    

def entropy_goodness_of_fit(observed: np.ndarray, fitted: np.ndarray, bins='auto', epsilon=1e-10) -> dict:
    """
    Evaluates goodness of fit between an observed and fitted distribution using entropy-based metrics.
    Fixes KL divergence by adding ε to avoid division by zero and includes normalized JS divergence.

    Parameters:
    - observed (np.ndarray): Empirical data.
    - fitted (np.ndarray): Samples from the fitted model.
    - bins (int or str): Number of bins for probability estimation (default: 'auto').
    - epsilon (float): Small value to prevent zero probabilities.

    Returns:
    - dict: Dictionary with entropy values and divergence metrics.
    """
    # Compute histogram-based probability distributions
    observed_hist, bin_edges = np.histogram(observed, bins=bins, density=True)
    fitted_hist, _ = np.histogram(fitted, bins=bin_edges, density=True)

    # Normalize to get probability distributions, add epsilon to avoid zero probabilities
    observed_prob = (observed_hist + epsilon) / np.sum(observed_hist + epsilon)
    fitted_prob = (fitted_hist + epsilon) / np.sum(fitted_hist + epsilon)

    # Compute entropy
    entropy_observed = entropy(observed_prob)  # Entropy of empirical data
    entropy_fitted = entropy(fitted_prob)  # Entropy of fitted model

    # Compute KL divergence (small values mean better fit)
    kl_div = entropy(observed_prob, fitted_prob)

    # Compute Jensen-Shannon divergence (symmetric & bounded)
    m = 0.5 * (observed_prob + fitted_prob)
    js_div = 0.5 * (entropy(observed_prob, m) + entropy(fitted_prob, m))

    # Normalize JS divergence between 0 and 1
    js_div_normalized = js_div / np.log(2)

    return {
        "entropy_observed": entropy_observed,
        "entropy_fitted": entropy_fitted,
        "kl_divergence": kl_div,
        "js_divergence": js_div,
        "js_divergence_normalized": js_div_normalized
    }


def copula_ks_test(empirical_data, copula_samples):
    from scipy.stats import ks_2samp
    from statsmodels.distributions.empirical_distribution import ECDF
    """
    Computes Kolmogorov-Smirnov distance for each marginal of a bivariate copula.
    
    Parameters:
    - empirical_data: numpy array of shape (n,2), empirical rank-transformed data.
    - copula_samples: numpy array of shape (n,2), samples from fitted copula.

    Returns:
    - KS statistic and p-value for each marginal.
    """
    assert empirical_data.shape == copula_samples.shape, "Data and copula samples must have the same shape"
    
    # Extract marginals
    U_empirical, V_empirical = empirical_data[:, 0], empirical_data[:, 1]
    U_copula, V_copula = copula_samples[:, 0], copula_samples[:, 1]

    # Compute KS test for each dimension
    ks_stat_U, p_value_U = ks_2samp(U_empirical, U_copula)
    ks_stat_V, p_value_V = ks_2samp(V_empirical, V_copula)

    return (ks_stat_U, p_value_U), (ks_stat_V, p_value_V)


def ks_distance(empirical_data, copula_samples):
    from statsmodels.distributions.empirical_distribution import ECDF

    """
    Computes the normalized Kolmogorov-Smirnov distance for bivariate copula data.

    Parameters:
    - empirical_data: (n,2) array of empirical copula samples.
    - copula_samples: (n,2) array of copula model samples.

    Returns:
    - KS distance for each marginal.
    """
    n = empirical_data.shape[0]  # Sample size

    # Compute empirical CDFs
    ecdf_emp_U = ECDF(empirical_data[:, 0])
    ecdf_emp_V = ECDF(empirical_data[:, 1])
    
    # Compute theoretical CDFs
    ecdf_cop_U = ECDF(copula_samples[:, 0])
    ecdf_cop_V = ECDF(copula_samples[:, 1])

    # Compute max absolute difference (KS statistic)
    ks_U = np.max(np.abs(ecdf_emp_U(empirical_data[:, 0]) - ecdf_cop_U(empirical_data[:, 0])))
    ks_V = np.max(np.abs(ecdf_emp_V(empirical_data[:, 1]) - ecdf_cop_V(empirical_data[:, 1])))

    # Apply normalization
    ks_distance_U = np.sqrt(n) * ks_U
    ks_distance_V = np.sqrt(n) * ks_V

    return ks_distance_U, ks_distance_V


def mle_power_law_alpha(data):
    from scipy.optimize import minimize
    """ Estimate power-law exponent 'a' using Maximum Likelihood Estimation (MLE). """
    data = data.dropna().values  # Remove NaNs
    data = data[data > 0]  # Ensure all values are positive
    
    n_min = np.min(data)  # Smallest positive value
    N = len(data)  # Number of observations

    # MLE formula for power-law exponent a
    a_mle = -1 - (np.sum(np.log(data / n_min)) / N) ** -1

    return a_mle



import scipy.stats as stats
import scipy.spatial.distance as dist
from scipy.stats import wasserstein_distance, energy_distance, ks_2samp, cramervonmises_2samp
from sklearn.metrics.pairwise import rbf_kernel

def run_metrics(X=np.random.normal(loc=[0, 0], scale=[1, 1], size=(1000, 2)),Y=np.random.normal(loc=[0, 0], scale=[1, 1], size=(1000, 2))):
    # np.random.seed(42)
    
    print("Wasserstein Distance:", compute_wasserstein(X, Y))
    print("Energy Distance:", compute_energy(X, Y))
    # print("MMD:", compute_mmd(X, Y))
    print("KL Divergence:", compute_kl_divergence(X, Y))
    print("JS Divergence:", compute_js_divergence(X, Y))

    print("Kolmogorov-Smirnov Distance:", compute_ks_distance(X, Y))
    print("Cramér-von Mises Distance:", compute_cramer_von_mises(X, Y))

    return X, Y

def compute_wasserstein(X, Y):
    """Computes the Wasserstein distance between two empirical bivariate distributions."""
    return wasserstein_distance(X[:, 0], Y[:, 0]) + wasserstein_distance(X[:, 1], Y[:, 1])

def compute_energy(X, Y):
    """Computes the energy distance between two bivariate distributions."""
    return energy_distance(X.ravel(), Y.ravel())

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """Computes Maximum Mean Discrepancy (MMD) between two distributions using an RBF kernel."""
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

def compute_kl_divergence(X, Y):
    """Computes KL divergence using KDE-based density estimation."""
    kde_X = stats.gaussian_kde(X.T)
    kde_Y = stats.gaussian_kde(Y.T)
    
    pts = np.vstack([X, Y])
    p_X = kde_X(pts.T) + 1e-10
    p_Y = kde_Y(pts.T) + 1e-10
    
    return np.sum(p_X * np.log2(p_X / p_Y)) / len(pts)

def compute_js_divergence(X, Y):
    """Computes Jensen-Shannon divergence using KDE-based density estimation."""
    kde_X = stats.gaussian_kde(X.T)
    kde_Y = stats.gaussian_kde(Y.T)
    
    pts = np.vstack([X, Y])
    p_X = kde_X(pts.T) + 1e-10
    p_Y = kde_Y(pts.T) + 1e-10
    M = 0.5 * (p_X + p_Y)
    
    return 0.5 * np.sum(p_X * np.log2(p_X / M)) / len(pts) + 0.5 * np.sum(p_Y * np.log2(p_Y / M)) / len(pts)

def compute_ks_distance(X, Y):
    """Computes Kolmogorov-Smirnov distance for each dimension and returns the maximum."""
    ks_x = ks_2samp(X[:, 0], Y[:, 0]).statistic
    ks_y = ks_2samp(X[:, 1], Y[:, 1]).statistic
    return max(ks_x, ks_y)

def compute_cramer_von_mises(X, Y):
    """Computes Cramér-von Mises distance for each dimension and returns the average."""
    cvm_x = cramervonmises_2samp(X[:, 0], Y[:, 0]).statistic
    cvm_y = cramervonmises_2samp(X[:, 1], Y[:, 1]).statistic
    return (cvm_x + cvm_y) / 2
