import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import powerlaw as pwl
# from myutils.utilities import *
import pycop
import seaborn as sns
import openturns as ot
import openturns.viewer as viewer
import powerlaw as pwl
import numpy as np
from scipy.stats import entropy

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

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


def histogram(data, num_bins=31, log_bin=True, density=True):
    import numpy as np
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


def balance_constructor(txdf):
    """
    Constructs a balance record DataFrame from transaction data.

    Parameters:
    txdf (DataFrame): Input DataFrame containing at least:
                      ['date', 'source', 'target', 'source_bal_post', 'target_bal_post']

    Returns:
    DataFrame: A DataFrame with columns ['date', 'crid', 'balance'] tracking balance changes.
    """
    # Ensure input has the required columns
    required_cols = ['date', 'source', 'target', 'source_bal_post', 'target_bal_post','weight']
    if not all(col in txdf.columns for col in required_cols):
        raise ValueError(f"Missing required columns in input DataFrame: {set(required_cols) - set(txdf.columns)}")

    # Construct balance records for source and target
    balances = pd.DataFrame({
        'date': txdf['date'].repeat(2).values,  # Repeat each transaction twice for source and target
        'crid': pd.concat([txdf['source'], txdf['target']], ignore_index=True),  # User identifiers
        'balance': pd.concat([txdf['source_bal_post'], txdf['target_bal_post']], ignore_index=True),  # Balances
        'type' : txdf['type'].repeat(2).values,
        'tx_id': txdf['id'].repeat(2).values,
        'weight': txdf.weight.repeat(2).values,
        # 'period': txdf['period'].repeat(2).values
    })

    return balances

    

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
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF

def copula_ks_test(empirical_data, copula_samples):
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
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

def ks_distance(empirical_data, copula_samples):
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
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mle_power_law_alpha(data):
    """ Estimate power-law exponent 'a' using Maximum Likelihood Estimation (MLE). """
    data = data.dropna().values  # Remove NaNs
    data = data[data > 0]  # Ensure all values are positive
    
    n_min = np.min(data)  # Smallest positive value
    N = len(data)  # Number of observations

    # MLE formula for power-law exponent a
    a_mle = -1 - (np.sum(np.log(data / n_min)) / N) ** -1

    return a_mle
import numpy as np
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

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as dist
from scipy.spatial.distance import jensenshannon

def compute_js_distance(X, Y):
    """Computes Jensen-Shannon distance using KDE-based density estimation."""
    kde_X = stats.gaussian_kde(X.dropna())
    kde_Y = stats.gaussian_kde(Y.dropna())
    
    grid = np.linspace(min(X.min(), Y.min()), max(X.max(), Y.max()), 1000)
    p_X = kde_X(grid) + 1e-10
    p_Y = kde_Y(grid) + 1e-10
    M = 0.5 * (p_X + p_Y)

    div = 0.5 * np.sum(p_X * np.log2(p_X / M)) + 0.5 * np.sum(p_Y * np.log2(p_Y / M))
    
    return div,np.sqrt(div)

def compute_js_distance_hist(X, Y, bins=50):
    """Computes JS distance using histograms instead of KDE."""
    p_X, _ = np.histogram(X.dropna(), bins=bins, density=True)
    p_Y, _ = np.histogram(Y.dropna(), bins=bins, density=True)

    p_X = p_X / np.sum(p_X) + 1e-10
    p_Y = p_Y / np.sum(p_Y) + 1e-10
    M = 0.5 * (p_X + p_Y)

    return 0.5 * np.sum(p_X * np.log2(p_X / M)) + 0.5 * np.sum(p_Y * np.log2(p_Y / M))

import numpy as np
from scipy.stats import entropy

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
        "js_divergence_normalized": js_div_normalized,
        "sqrt_js_div_normalized": np.sqrt(js_div_normalized)
    }
def compute_sequential_js(pivot_table):
    """Computes Jensen-Shannon distances sequentially between adjacent columns."""
    columns = pivot_table.columns[1:]  # Exclude index column 'crid'
    results = {}
    
    for i in range(len(columns) - 1):
        col1, col2 = columns[i], columns[i + 1]
        div, dist = compute_js_distance(pivot_table[col1], pivot_table[col2])
        results[f"{col1} vs {col2}"] = dist
    
    return results

def powlaw_ppf(a, xmin=1, xmax=np.inf):
    if a <= 1:
        raise ValueError("Parameter 'a' must be greater than 1 for a valid power-law distribution.")

    # Precompute terms to use in the PPF calculation
    b = 1 - a
    c = (xmax**b - xmin**b) if xmax < np.inf else -xmin**b

    # Define the PPF as a function of the quantile r
    def ppf(r):
        if np.any((r < 0) | (r > 1)):
            raise ValueError("Input 'r' must be between 0 and 1.")
        return (r * c + xmin**b)**(1 / b)

    return ppf
def round_df(df, precision = 3, includes='float'):
    df[df.select_dtypes(include=includes).columns] = df.select_dtypes(include='float').apply(
        lambda x: x.round(precision)    )
    return df

def vect_range(data):
    return min(data),max(data)

def scale_vect(arr, a=0, b=1):
    arr = np.array(arr)  # Ensure arr is a numpy array
    return ((arr - arr.min()) / (arr.max() - arr.min())) * (b - a) + a

def plot_2d_histogram(data, ax, **kwargs):
    ax.hist2d(data[:, 0], data[:, 1], bins=30, cmap='Blues')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

def plot_scatter_kde(data, ax=None, x_label='X', y_label='Y', log_scale=False,cmap='plasma', **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    """
    Creates a scatter plot with KDE-based coloring for density, compatible with `make_gif_arr`.

    Parameters:
    - data: A 2D array or dataframe with two columns to be plotted.
    - ax: The matplotlib axes object to plot on.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - log_scale: If True, both axes will use a log scale.
    - kwargs: Additional keyword arguments for customization.
    """
    if ax is None:
        fig,ax=plt.subplots()
    # Perform a kernel density estimate on the data
    xy = np.vstack([data[:, 0], data[:, 1]])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted on top
    idx = z.argsort()
    x, y, z = data[:, 0][idx], data[:, 1][idx], z[idx]

    s = kwargs.get('s', 50)  # Default size of the scatter points
    # Create the scatter plot
    scatter = ax.scatter(x, y, c=z, s=s, edgecolor='face', cmap=cmap)
    

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density')

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Apply log scale if specified
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        

    # Additional plotting customizations via kwargs
    # ax.set(**kwargs)



def agents_constructor(transaction_dataframe,users_dataframe=None, standard=True, end=None, begin=None, how='outer'):
    if users_dataframe is None:
        try:
            users_dataframe = USERS
        except NameError:
            raise ValueError("USERS is not defined in this environment. Please provide a value for 'users_dataframe'.")
    # Check if end and begin are provided, otherwise use END and START if they exist
    if end is None:
        try:
            end = END
        except NameError:
            raise ValueError("END is not defined in this environment. Please provide a value for 'end'.")
    
    if begin is None:
        try:
            begin = START
        except NameError:
            raise ValueError("START is not defined in this environment. Please provide a value for 'begin'.")

    # Aggregation for sources
    sources = transaction_dataframe.groupby("source").agg({
                                    'weight':['count','sum','mean','median','max','min'],
                                    'frac_out':['mean','median','max','min'], # I added this line
                                    'target':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    sources.columns = [
        "total_txns_out", "total_exp", "mean_exp", "median_exp", "max_exp", "min_exp",  # weight columns
        "mean_frac_out", "median_frac_out", "max_frac_out", "min_frac_out",            # frac_out columns
        "num_targets", "first_txns_out"                                                # target and date columns
    ]
    sources.reset_index(inplace=True)
    
    sources['opening'] = sources['source'].map(users_dataframe.set_index('crid')['start'])
    sources['frac_from_1st_out'] = abs(end - (sources.first_txns_out)) / abs(end - begin)
    sources['frac_from_op'] = abs(end - (sources.opening.where(sources.opening > begin, begin))) / abs(end - begin)
    sources['eff_txns_out'] = sources.total_txns_out / sources.frac_from_op
    sources['eff_outdegree'] = sources.num_targets / sources.frac_from_op

    # Aggregation for targets
    targets = transaction_dataframe.groupby("target").agg({
                                    'weight': ['count','sum','mean','median','max','min'],
                                    'frac_in': ['mean','median','max','min'], # I added this line
                                    'source':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    targets.columns = [
        "total_txns_in", "total_earn", "mean_earn", "median_earn", "max_earn", "min_earn",  # weight columns
        "mean_frac_in", "median_frac_in", "max_frac_in", "min_frac_in",                    # frac_in columns
        "num_sources", "first_txns_in"                                                    # source and date columns
    ]
    targets.reset_index(inplace=True)

    targets['opening'] = targets['target'].map(users_dataframe.set_index('crid')['start'])
    targets['frac_from_1st_in'] = abs(end - (targets.first_txns_in)) / abs(end - begin)
    targets['frac_from_op'] = abs(end - (targets.opening.where(targets.opening > begin, begin))) / abs(end - begin)
    targets['eff_txns_in'] = targets.total_txns_in / targets.frac_from_op
    targets['eff_indegree'] = targets.num_sources / targets.frac_from_op

    targets.rename(columns={'target': 'crid'}, inplace=True)
    sources.rename(columns={'source': 'crid'}, inplace=True)

    agents = pd.merge(targets, sources, on=['crid','opening','frac_from_op'], how=how)
    
    if how == 'outer':
        agents = agents.fillna(0)
    
    if standard:
        agents = agents.rename(columns={
            'total_txns_in':'attractiveness',
            'total_txns_out':'activity',
            'total_earn':'vol_in',
            'total_exp':'vol_out',
            'num_sources':'in_deg',
            'num_targets':'out_deg',
            'eff_txns_in': 'eff_attr',
            'eff_txns_out': 'eff_act',
        })
        agents = agents[['crid', 'frac_from_op',
                         'attractiveness','activity', 
                         'vol_in','vol_out',
                         'in_deg','out_deg',
                         'eff_attr', 'eff_act',
                         'eff_indegree', 'eff_outdegree',
                         'first_txns_in', 'first_txns_out',
                         'frac_from_1st_in','frac_from_1st_out',
                         'mean_earn', 'mean_exp',
                         'median_earn', 'median_exp',
                         'max_earn', 'max_exp',
                         'min_earn', 'min_exp'
                    ]]
    else:
        agents = agents[['crid', 'opening','frac_from_op',
                         'total_txns_in','total_txns_out', 
                         'total_earn','total_exp',
                         'num_sources','num_targets',
                         'eff_txns_in', 'eff_txns_out',
                         'eff_indegree', 'eff_outdegree',
                         'first_txns_in', 'first_txns_out',
                         'frac_from_1st_in','frac_from_1st_out',
                         'mean_earn', 'mean_exp',
                         'median_earn', 'median_exp', 
                         'max_earn', 'max_exp',
                         'min_earn', 'min_exp'
                    ]]
    agents = agents.merge(users_dataframe,how='left',on='crid')

    return agents



def make_gif_from_figures(figures, gif_filename="output.gif", duration=500):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
    """
    Create a GIF from a list of matplotlib figures.
    
    Parameters:
    figures (list): A list of matplotlib.figure.Figure objects.
    gif_filename (str): The filename for the output GIF.
    duration (int): Time duration between frames in milliseconds.
    """
    frames = []
    
    # Store image bytes in memory until we save the GIF
    image_data = []
    
    for fig in figures:
        # Save each figure to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image_data.append(buf.getvalue())  # Store the raw image bytes
        buf.close()

    # Convert the stored image bytes into frames
    frames = [Image.open(io.BytesIO(img_bytes)) for img_bytes in image_data]

    # Save as GIF
    frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)



def basic_agents(transaction_dataframe,user_df = None,tx_type = 'STANDARD',how='outer'):
    # if user_df is None:
    #     try:
    #         user_df = USERS.copy(deep=True)
    #     except exception as e:
    #         print(e)

    sources = transaction_dataframe.groupby("source").agg({
                                    'weight':['count','sum','mean','median','max','min'],
                                    'frac_out':['mean','median','max','min'], # I added this line
                                    'target':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    sources.columns = [
        "total_txns_out", "total_exp", "mean_exp", "median_exp", "max_exp", "min_exp",  # weight columns
        "mean_frac_out", "median_frac_out", "max_frac_out", "min_frac_out",            # frac_out columns
        "num_targets", "first_txns_out"                                                # target and date columns
    ]
    sources.reset_index(inplace=True)

    targets = transaction_dataframe.groupby("target").agg({
                                    'weight': ['count','sum','mean','median','max','min'],
                                    'frac_in': ['mean','median','max','min'], # I added this line
                                    'source':'nunique',
                                    'date':'first'})
    
    # Correct column names after aggregation
    targets.columns = [
        "total_txns_in", "total_earn", "mean_earn", "median_earn", "max_earn", "min_earn",  # weight columns
        "mean_frac_in", "median_frac_in", "max_frac_in", "min_frac_in",                    # frac_in columns
        "num_sources", "first_txns_in"                                                    # source and date columns
    ]
    targets.reset_index(inplace=True)

    targets.rename(columns={'target': 'crid'}, inplace=True)
    sources.rename(columns={'source': 'crid'}, inplace=True)

    agents = pd.merge(targets, sources, on=['crid'], how=how)
    # agents['activity']
    
    if how == 'outer':
        agents = agents.fillna(0)
    agents = agents.rename(columns={
            'total_txns_in':'tx_in',
            'total_txns_out':'tx_out',
            'total_earn':'vol_in',
            'total_exp':'vol_out',
            'num_sources':'in_deg',
            'num_targets':'out_deg',
            
        })
    
    # agents = agents[['crid', 'frac_from_op',
    #                     'attractiveness','activity', 
    #                     'vol_in','vol_out',
    #                     'in_deg','out_deg',
    #                     'eff_attr', 'eff_act',
    #                     'eff_indegree', 'eff_outdegree',
    #                     'first_txns_in', 'first_txns_out',
    #                     'frac_from_1st_in','frac_from_1st_out',
    #                     'mean_earn', 'mean_exp',
    #                     'median_earn', 'median_exp',
    #                     'max_earn', 'max_exp',
    #                     'min_earn', 'min_exp'
    #             ]]
    return agents


def empirical_copula(u, v, u_eval, v_eval):
    import numpy as np
    """
    Computes empirical copula C_n(u_eval, v_eval) from data u,v (uniform margins).
    
    Parameters:
        u, v : Uniform marginals of original data (numpy arrays).
        u_eval, v_eval : points at which to evaluate empirical copula (floats).

    Returns:
        Empirical copula value C_n(u_eval,v_eval)
    """
    n = len(u)
    ranks_u = (np.argsort(np.argsort(u)) + 1) / (n + 1)
    ranks_v = (np.argsort(np.argsort(v)) + 1) / (n + 1)

    C_n = np.mean((ranks_u <= u_eval) & (ranks_v <= v_eval))
    return C_n



def to_uniform(data):
    from scipy.stats import rankdata
    """Convert raw data to pseudo-uniform margins."""
    n = len(data)
    return rankdata(data) / (n + 1.0)


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def standard_scaler_shift(series: pd.Series) -> pd.Series:
    """
    Applies StandardScaler to a pandas Series and shifts values to keep them non-negative.
    
    Parameters:
        series (pd.Series): Input series with non-negative values.
    
    Returns:
        pd.Series: Scaled and shifted series.
    """
    # Controlla che i dati siano tutti >= 0
    if (series < 0).any():
        raise ValueError("All values must be non-negative.")
    
    # Reshape per StandardScaler (richiede 2D input)
    series_reshaped = series.values.reshape(-1, 1)
    
    # Applica StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(series_reshaped).flatten()  # Torna a 1D
    
    # Calcola il minimo per lo shift positivo
    min_shift = abs(scaled_data.min())
    scaled_data += min_shift  # Shift verso valori positivi
    
    return pd.Series(scaled_data, index=series.index)


# Function to compute power-law PDF
def powerlaw_function(x, alpha, xmin, C):
    """
    Computes the probability density function (PDF) of a power-law distribution.
    Now includes normalization factor C.
    """
    return C * (x / xmin) ** -alpha

# Function to recalculate normalization constant C
def recalculate_constant(alpha, xmin, xmax):
    """
    Recalculate the normalization constant C for the power-law PDF to fit the entire range.
    """
    if alpha == 1:
        raise ValueError("Alpha cannot be 1 for a valid power-law distribution.")
    C_new = (alpha - 1) / (xmin ** (1 - alpha) - xmax ** (1 - alpha))
    return C_new

# Function to determine xmin for a given quantile
def find_xmin_by_quantile(data, quantile=0.2):
    """
    Finds xmin such that exactly `quantile` fraction of data is above xmin.
    """
    return np.percentile(data, 100 * quantile)  # 80th percentile for 20% quantile

import numpy as np
from collections import Counter

class PMF:
    def __init__(self, data):
        self.data = np.array(data)
        self.total_count = len(self.data)
        self.counts = Counter(self.data)  # Count occurrences
        self.pmf_dict = {k: v / self.total_count for k, v in self.counts.items()}  # PMF dictionary

    def __call__(self, x):
        return self.pmf_dict.get(x, 0)  # Return probability if exists, else 0

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

class MultivariateMultimodal:
    """
    A class to sample from a mixture of multivariate distributions.

    Parameters:
    ----------
    distributions : list of tuples
        Each tuple should be (sampling_function, args, kwargs).
        - sampling_function: Function to generate samples (e.g., np.random.multivariate_normal).
        - args: Positional arguments for the sampling function.
        - kwargs: Keyword arguments for the sampling function.

    weights : list of floats
        Mixing proportions for each distribution. Should sum to 1.

    Methods:
    -------
    sample(size=1)
        Generates 'size' samples from the mixture distribution.

    Example Usage:
    --------------
    >>> distributions = [
    >>>     (np.random.multivariate_normal, [[0, 0], [[1, 0.5], [0.5, 1]]], {}),
    >>>     (np.random.multivariate_normal, [[5, 5], [[1, -0.3], [-0.3, 1]]], {}),
    >>>     (np.random.multivariate_normal, [[-5, 5], [[1, 0.2], [0.2, 1]]], {})
    >>> ]
    >>> weights = [0.3, 0.4, 0.3]
    >>> multi_modal_mv = MultivariateMultimodal(distributions, weights)
    >>> samples = multi_modal_mv.sample(5000)
    """

    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.weights = np.array(weights) / np.sum(weights)  # Normalize weights

    def sample(self, size=1):
        """
        Generates samples from the multimodal multivariate distribution.

        Parameters:
        ----------
        size : int
            Number of samples to generate.

        Returns:
        -------
        np.ndarray
            Array of shape (size, num_features) containing the sampled data.
        """
        chosen_indices = np.random.choice(len(self.distributions), size=size, p=self.weights)
        samples = np.array([self.distributions[i][0](*self.distributions[i][1], **self.distributions[i][2]) for i in chosen_indices])
        return np.vstack(samples)  # Stack samples into a single array


 


# def balance_constructor(tx_with_bal, users, bins = 20, index = None):

#     if 'period' not in tx_with_bal.columns:
#         print('Adding period column to txns dataframe...')
#         if index is None:
#             tx_with_bal['period'] = pd.cut(tx_with_bal.index, bins=bins,labels=False)
#         if index is not None:
#             tx_with_bal['period'] = pd.cut(tx_with_bal[index], bins=bins,labels=False)
#     else: 
#         print('period already in txns dataframe')

#     balances_pre = pd.DataFrame({
#         'id': tx_with_bal['id'].repeat(2).values,
#         'date': tx_with_bal['date'].repeat(2).values,  # Repeat each time twice
#         'balance': tx_with_bal[['source_bal_pre', 'target_bal_pre']].values.flatten(),
#         'xdai': tx_with_bal[['source_xdai', 'target_xdai']].values.flatten(),  # Stack feature1 and feature2 in an alternating pattern
#         'type': tx_with_bal['type'].repeat(2).values,
#         'period':tx_with_bal['period'].repeat(2).values,
#         'weight': tx_with_bal['weight'].repeat(2) * np.tile([-1,1],len(tx_with_bal)),
#         'kind': np.tile(['source','target'],len(tx_with_bal))
#     })

#     balances_post = pd.DataFrame({
#         'id': tx_with_bal['id'].repeat(2).values,
#         'date': tx_with_bal['date'].repeat(2).values,  # Repeat each time twice
#         'balance': tx_with_bal[['source_bal_post', 'target_bal_post']].values.flatten(),
#         'xdai': tx_with_bal[['source_xdai', 'target_xdai']].values.flatten(),  # Stack feature1 and feature2 in an alternating pattern
#         'type': tx_with_bal['type'].repeat(2).values,
#         'period':tx_with_bal['period'].repeat(2).values,
#         'weight': tx_with_bal['weight'].repeat(2) * np.tile([-1,1],len(tx_with_bal)),
#         'kind': np.tile(['source','target'],len(tx_with_bal))
#     })

#     balances_pre.balance = balances_pre.balance.apply(lambda x: 0 if abs(x) < 1e-3 else x)
#     balances_post.balance = balances_post.balance.apply(lambda x: 0 if abs(x) < 1e-3 else x)
    
#     balances_pre = balances_pre.merge(
#         users[['xdai', 'role', 'area_type', 'area_name','svol_in', 'svol_out','stxns_in', 'stxns_out', 'sunique_in', 'sunique_out','initial_bal','final_bal','net','otxns_in', 'otxns_out','crid']],  # Ensure the right columns are selected
#         on='xdai',  # Use 'xdai' as the merge key in users
#         how='left'
#     )
#     balances_post = balances_post.merge(
#         users[['xdai', 'role', 'area_type', 'area_name','svol_in', 'svol_out','stxns_in', 'stxns_out', 'sunique_in', 'sunique_out','initial_bal','final_bal','net','otxns_in', 'otxns_out','crid']],  # Ensure the right columns are selected
#         on='xdai',  # Use 'xdai' as the merge key in users
#         how='left'
#     )

#     return balances_pre, balances_post