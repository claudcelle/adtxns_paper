import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 

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