from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union


 

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





@dataclass(frozen=True)
class KDEScatterData:
    """Container for reuse: x/y data + per-point density (z)."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def kde_scatter_estimate(
    data: Union[np.ndarray, Any],
    *,
    x: Union[int, str] = 0,
    y: Union[int, str] = 1,
    log_space: bool = False,
    bw_method: Optional[Union[str, float]] = None,
    drop_nonfinite: bool = True,
) -> KDEScatterData:
    """
    Compute per-point KDE density once (z), to be reused for plotting with different styles.

    Parameters
    ----------
    data:
        Nx2 ndarray or pandas-like (DataFrame/series columns).
    x, y:
        Column selectors: int (ndarray) or str (DataFrame-like).
    log_space:
        If True, KDE is computed in log10-space (requires x>0 and y>0).
        Returned x/y are still in original space (not logged).
    bw_method:
        Passed to scipy.stats.gaussian_kde (None uses scipy default).
    drop_nonfinite:
        If True, remove rows with non-finite x/y (and non-positive when log_space=True).

    Returns
    -------
    KDEScatterData(x, y, z)
    """
    from scipy.stats import gaussian_kde

    # Extract columns without depending on pandas
    if isinstance(data, np.ndarray):
        xv = np.asarray(data[:, x], dtype=float) if isinstance(x, int) else np.asarray(data[:, 0], dtype=float)
        yv = np.asarray(data[:, y], dtype=float) if isinstance(y, int) else np.asarray(data[:, 1], dtype=float)
    else:
        xv = np.asarray(data[x], dtype=float)
        yv = np.asarray(data[y], dtype=float)

    if drop_nonfinite:
        m = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[m], yv[m]

    if log_space:
        if drop_nonfinite:
            pos = (xv > 0) & (yv > 0)
            xv, yv = xv[pos], yv[pos]
        # If not dropping, gaussian_kde will fail on <=0 after log; keep it strict.
        xv_d, yv_d = np.log10(xv), np.log10(yv)
    else:
        xv_d, yv_d = xv, yv

    xy = np.vstack([xv_d, yv_d])
    kde = gaussian_kde(xy, bw_method=bw_method)
    z = kde(xy)

    return KDEScatterData(x=xv, y=yv, z=z)






def plot_scatter_from_kde(
    kde_data: KDEScatterData,
    ax=None,
    *,
    x_label: Optional[str] = "X",
    y_label: Optional[str] = "Y",
    log_scale: bool = False,
    sort_by_density: bool = True,
    colorbar: bool = True,
    colorbar_kw: Optional[dict] = None,
    scatter_kw: Optional[dict] = None,
    cmap: str = "plasma",
    norm=None,
    rasterized: bool = True,
):
    """
    Plot precomputed KDE scatter; tweak styles freely without recomputing KDE.

    Returns
    -------
    ax, PathCollection, Colorbar|None
    """
    import matplotlib.pyplot as plt

    if scatter_kw is None:
        scatter_kw = {}
    if colorbar_kw is None:
        colorbar_kw = {}

    x = np.asarray(kde_data.x)
    y = np.asarray(kde_data.y)
    z = np.asarray(kde_data.z)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if sort_by_density:
        order = np.argsort(z)  # low first, dense last
        x, y, z = x[order], y[order], z[order]

    skw = dict(s=10, linewidths=0, alpha=1.0)
    skw.update(scatter_kw)

    sc = ax.scatter(
        x, y,
        c=z,
        cmap=cmap,
        norm=norm,
        rasterized=rasterized,
        **skw,
    )

    cbar = None
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, **colorbar_kw)
        cbar.set_label(colorbar_kw.get("label", "Density"))

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    return ax, sc, cbar

def plot_scatter_density(
    data: Union[np.ndarray, "Any"],
    ax=None,
    *,
    x: Union[int, str] = 0,
    y: Union[int, str] = 1,
    x_label: Optional[str] = "X",
    y_label: Optional[str] = "Y",
    log_scale: bool = False,
    density: Literal["hist", "kde"] = "hist",
    bins: Union[int, Tuple[int, int]] = 200,
    bandwidth: Optional[float] = None,
    smooth_sigma: float = 1.0,
    cmap: str = "plasma",
    norm=None,
    sort: bool = True,
    colorbar: bool = True,
    colorbar_kw: Optional[dict] = None,
    rasterized: bool = True,
    scatter_kw: Optional[dict] = None,
):
    """
    Fast scatter plot colored by estimated point density.

    Why this is faster:
      - default density="hist" uses a 2D histogram + Gaussian smoothing (O(N)ish),
        instead of gaussian_kde at every point (can be O(N^2) for large N).

    Parameters
    ----------
    data:
        Nx2 ndarray, or a pandas-like object (DataFrame/structured) supporting column access.
    ax:
        Matplotlib Axes. If None, creates a new figure+axes.
    x, y:
        Column selectors: integer index for ndarray, or column name for DataFrame-like.
    log_scale:
        If True, axes are log-scaled. Density estimation is done in log-space for positive data.
    density:
        "hist" (fast, default) or "kde" (slower, closer to true KDE).
    bins:
        Histogram bins for "hist" density.
    bandwidth:
        KDE bandwidth for density="kde". If None, scipy chooses default.
    smooth_sigma:
        Gaussian smoothing sigma (in bin units) for density="hist".
    cmap, norm:
        Passed to scatter for colormapping.
    sort:
        If True, plot low-density first so dense regions show on top.
    colorbar:
        Add a colorbar attached to the axes' figure (better matplotlib integration).
    colorbar_kw:
        Dict passed to fig.colorbar(...).
    rasterized:
        Rasterize scatter points in vector outputs (PDF/SVG) for much smaller files.
    scatter_kw:
        Dict forwarded to ax.scatter (size, alpha, marker, linewidths, etc.).

    Returns
    -------
    ax, scatter, cbar, density_values
    """
    import matplotlib.pyplot as plt

    if scatter_kw is None:
        scatter_kw = {}
    if colorbar_kw is None:
        colorbar_kw = {}

    # --- get x/y arrays without forcing pandas dependency ---
    if isinstance(data, np.ndarray):
        arr = data
        xv = np.asarray(arr[:, x], dtype=float) if isinstance(x, int) else np.asarray(arr[:, 0], dtype=float)
        yv = np.asarray(arr[:, y], dtype=float) if isinstance(y, int) else np.asarray(arr[:, 1], dtype=float)
    else:
        # pandas-like: data[x] gives a column
        xv = np.asarray(data[x], dtype=float)
        yv = np.asarray(data[y], dtype=float)

    # drop non-finite
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # --- density estimation (optionally in log-space) ---
    if log_scale:
        # density in log-space only makes sense for strictly positive values
        pos = (xv > 0) & (yv > 0)
        xv_d, yv_d = np.log10(xv[pos]), np.log10(yv[pos])
        map_back = pos
    else:
        xv_d, yv_d = xv, yv
        map_back = slice(None)

    z = np.full_like(xv, np.nan, dtype=float)

    if density == "hist":
        from scipy.ndimage import gaussian_filter

        # histogram over density-space coordinates
        H, xedges, yedges = np.histogram2d(xv_d, yv_d, bins=bins)
        if smooth_sigma and smooth_sigma > 0:
            H = gaussian_filter(H, sigma=smooth_sigma)

        # map each point to its bin value (fast)
        # note: histogram2d returns shape (nx, ny) for x then y
        xi = np.searchsorted(xedges, xv_d, side="right") - 1
        yi = np.searchsorted(yedges, yv_d, side="right") - 1
        nx, ny = H.shape
        inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        z_sub = np.full_like(xv_d, np.nan, dtype=float)
        z_sub[inside] = H[xi[inside], yi[inside]]
        z[map_back] = z_sub

    elif density == "kde":
        from scipy.stats import gaussian_kde

        xy = np.vstack([xv_d, yv_d])
        kde = gaussian_kde(xy, bw_method=bandwidth)
        z[map_back] = kde(xy)
    else:
        raise ValueError("density must be 'hist' or 'kde'")

    # if log_scale, points with non-positive values were excluded from density;
    # keep them but color them with the minimum finite density so they still show.
    finite_z = z[np.isfinite(z)]
    if finite_z.size:
        z_min = float(np.nanmin(finite_z))
        z[~np.isfinite(z)] = z_min
    else:
        z[:] = 0.0

    # --- sort so dense points are drawn last ---
    if sort:
        order = np.argsort(z)
        xv, yv, z = xv[order], yv[order], z[order]

    # --- draw ---
    # sensible defaults; user can override via scatter_kw
    skw = dict(s=10, linewidths=0, alpha=1.0)
    skw.update(scatter_kw)

    sc = ax.scatter(
        xv,
        yv,
        c=z,
        cmap=cmap,
        norm=norm,
        rasterized=rasterized,
        **skw,
    )

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, **colorbar_kw)
        cbar.set_label("Density")
    else:
        cbar = None

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    return ax, sc, cbar, z
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


def plot_distribution(data, n_bins = 20, log_bin=True, log_scale=False, ax=None, return_dist = False,density = True, c=1,**kwargs):
    """
    Plots the distribution of a given dataset using a specified number of bins and optional logarithmic binning and scaling.

    Parameters:
        data (array-like): The dataset to plot the distribution for.
        n_bins (int, optional): The number of bins to use for the histogram. Defaults to 20.
        log_bin (bool, optional): Whether to use logarithmic binning. Defaults to True.
        log_scale (bool, optional): Whether to use logarithmic scaling for the x-axis and y-axis. Defaults to False.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot the distribution. If None, a new figure and axes are created. Defaults to None.
        return_dist (bool, optional): Whether to return the distribution data. Defaults to False.
        density (bool, optional): Whether to plot the normalized histogram. Defaults to True.
        c (float, optional): The scaling factor for the histogram plot. Defaults to 1.
        **kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        dict: If return_dist is True, a dictionary containing the following keys:
            - 'bin_centers' (array-like): The centers of the bins.
            - 'bin_edges' (array-like): The edges of the bins.
            - 'bin_widths' (array-like): The widths of the bins.
            - 'hist_normalized' (array-like): The normalized histogram values.
            - 'hist_raw' (array-like): The raw histogram values.

    """
    
    plot_arguments = [
            'linestyle', 'linewidth', 'color',
            'marker', 'markersize', 'markeredgecolor', 'markeredgewidth',
            'markerfacecolor', 'markerfacecoloralt', 'fillstyle', 'label',
            'drawstyle', 'alpha', 'solid_capstyle', 'solid_joinstyle',
            'dash_capstyle', 'dash_joinstyle', 'linestyle', 'antialiased',
            'dash_dot_phase', 'dashes', 'pickradius', 'zorder', 'scalex',
            'scaley', 'gid', 'snap', 'url', 'visible', 'xdata', 'ydata',
            'path_effects'
        ]
    subplots_arguments = [
        'nrows', 'ncols', 'sharex', 'sharey', 'squeeze', 'subplot_kw',
        'gridspec_kw', 'fig_kw', 'constrained_layout', 'figsize', 'dpi',
        'facecolor', 'edgecolor', 'num', 'clear', 'tight_layout'
    ]
    
    plot_arguments = {key:kwargs[key] for key in kwargs.keys() if key in plot_arguments}
    subplots_arguments = {key:kwargs[key] for key in kwargs.keys() if key in subplots_arguments}
    if ax is None:
        fig, ax = plt.subplots(**subplots_arguments)
        own_figure = True
    else:
        own_figure = False
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    data = data.dropna()
    if log_bin or log_scale:
        data = data[data > 0]
    hist = distr_bin_updated(data, n_bin=n_bins, logbin=log_bin)
    if density:        
        ax.plot(hist['bin_centers'], c*hist['hist_normalized'],**plot_arguments)
    else:
        ax.plot(hist['bin_centers'], hist['hist_raw'],**plot_arguments)

    if log_bin:
        log_scale = True
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if 'title' in kwargs.keys() and own_figure:  # Only set the title if this function created the figure
        ax.set_title(kwargs['title'])
    if 'label' in plot_arguments.keys():
        ax.legend()

    if own_figure:  # Show the plot only if this function created the figure
        # plt.title(title)
        plt.show()
    if return_dist:
        return hist



def distr_bin_updated(data, n_bin=30, logbin=True, var_type='cont'):
    """
    Calculates the distribution of a given dataset using a specified type of binning.

    Parameters:
        data (array-like): The dataset to calculate the distribution for.
        n_bin (int, optional): The number of bins to use for the histogram. Defaults to 30.
        logbin (bool, optional): Whether to use logarithmic binning. Defaults to True.
        var_type (str, optional): The type of binning to use. Must be either 'cont' or 'disc'. Defaults to 'cont'.

    Returns:
        dict: A dictionary containing the following keys:
            - 'bin_centers' (array-like): The centers of the bins.
            - 'bin_edges' (array-like): The edges of the bins.
            - 'bin_widths' (array-like): The widths of the bins.
            - 'hist_normalized' (array-like): The normalized histogram values.
            - 'hist_raw' (array-like): The raw histogram values.

    Raises:
        ValueError: If the input data is empty or if the data contains non-positive values for logarithmic binning.
        ValueError: If the var_type is not 'cont' or 'disc'.
    """
    if len(data) == 0:
        raise ValueError("Error: Empty data")
    
    if logbin:
        if np.any(data <= 0):
            raise ValueError("Error: Nonpositive data for logarithmic binning")
        bins_edges = np.logspace(np.log10(min(data)), np.log10(max(data)), n_bin+1)
    else:
        bins_edges = np.linspace(min(data), max(data), n_bin+1)
    
    hist_raw, edges = np.histogram(data, bins_edges)
    # Remove empty bins by filtering where hist_raw is nonzero
    nonzero_indices = np.nonzero(hist_raw)[0]
    hist_raw = hist_raw[nonzero_indices]
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_centers = bin_centers[nonzero_indices]
    bin_widths = np.diff(edges)
    bin_widths = bin_widths[nonzero_indices]
    
    if var_type == 'cont':
        hist_normalized = hist_raw / (bin_widths * hist_raw.sum())
    elif var_type == 'disc':
        hist_normalized = hist_raw / hist_raw.sum()
    else:
        raise ValueError("Invalid var_type. Use 'cont' or 'disc'.")

    result = {
        'bin_centers': bin_centers,
        'bin_edges': edges[:-1][nonzero_indices],  # exclude last appended max(data) value for edges, filter non-zero
        'bin_widths': bin_widths,
        'hist_normalized': hist_normalized,
        'hist_raw': hist_raw
    }
    
    return result
