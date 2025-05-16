"""
Visualization functions for model evaluation and analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import uncertainty_toolbox as uct # For calibration plot
from typing import List, Optional, Tuple

def plot_scatter_with_uncertainty(x_true, y_pred_mean, y_pred_std, color_metric=None,
                                xlabel=r'$\tau_{\mathrm{experiment}}$ (s)',
                                ylabel=r'Predicted $\tau_{\mathrm{BNN}}$ (s)',
                                title=None, xlim=None, ylim=None,
                                space_label="Mahalanobis Distance",
                                uncertainty_label='Total Uncertainty ($\pm 2\sigma$)',
                                vmin=None, vmax=None, extend_cbar=None,
                                show_trendline=True, trendline_color='blue',
                                save_path=None):
    """
    Create a scatter plot of true vs. predicted values with uncertainty bands, 
    colored by an additional metric (e.g., Mahalanobis distance).

    Args:
        x_true: Array of true values.
        y_pred_mean: Array of predicted mean values.
        y_pred_std: Array of predicted standard deviations.
        color_metric: Array of values to use for coloring points (optional).
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title of the plot.
        xlim: Tuple specifying x-axis limits (optional).
        ylim: Tuple specifying y-axis limits (optional).
        space_label: Label for the color bar.
        uncertainty_label: Label for the uncertainty bands.
        vmin: Minimum value for the color scale (optional).
        vmax: Maximum value for the color scale (optional).
        extend_cbar: Which extend arrows to draw on the colorbar ('neither', 'both', 'min', 'max') (optional).
        show_trendline: Whether to show the linear regression trendline (default True).
        trendline_color: Color of the trendline (default 'blue').
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=(7, 6))
    
    # Sort all data by x_true for proper fill_between
    sort_idx = np.argsort(x_true)
    x_sorted = np.array(x_true)[sort_idx]
    y_mean_sorted = np.array(y_pred_mean)[sort_idx]
    y_std_sorted = np.array(y_pred_std)[sort_idx]
    
    # Use fill_between for uncertainty (2-sigma)
    plt.fill_between(x_sorted, 
                    y_mean_sorted - 2 * y_std_sorted, 
                    y_mean_sorted + 2 * y_std_sorted, 
                    color='lightgray', alpha=0.6, label=uncertainty_label)

    # Scatter plot
    if color_metric is not None:
        color_metric_sorted = np.array(color_metric)[sort_idx] if color_metric is not None else None
        sc = plt.scatter(x_sorted, y_mean_sorted, c=color_metric_sorted, cmap='viridis', 
                         vmin=vmin, vmax=vmax,
                         s=20, alpha=0.7, edgecolors='k', linewidths=0.1)
        cbar = plt.colorbar(sc, extend=extend_cbar)
        cbar.set_label(space_label)
    else:
        plt.scatter(x_sorted, y_mean_sorted, s=20, alpha=0.7, edgecolors='k', linewidths=0.1)

    # Add trendline (linear regression)
    if show_trendline:
        # Calculate linear regression
        slope, intercept = np.polyfit(x_true, y_pred_mean, 1)
        # Create trendline points
        if xlim is None:
            xlim = plt.xlim()
        x_trend = np.array([xlim[0], xlim[1]])
        y_trend = slope * x_trend + intercept
        # Plot trendline
        plt.plot(x_trend, y_trend, color=trendline_color, linestyle='-', 
                 label=f'Trendline')

    # Ideal line (y=x)
    if xlim is None:
        xlim = plt.xlim()
    if ylim is None:
        ylim = plt.ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show() # Avoid showing plot immediately, let the caller decide

def plot_performance_comparison(model_names, metric_values, metric='rmse',
                                title=None, ylim=None, txt_h=0.01, colors=None, save_path=None):
    """
    Create a bar plot comparing performance metrics (RMSE or R2) of different models.

    Args:
        model_names: List of model names.
        metric_values: List of corresponding metric values.
        metric: Type of metric ('rmse' or 'r2').
        title: Title of the plot.
        ylim: Tuple specifying y-axis limits (optional).
        txt_h: Vertical offset for text labels on bars.
        colors: List of colors for the bars (optional).
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=(6, 5))
    bars = plt.bar(model_names, metric_values, color=colors if colors else plt.cm.viridis(np.linspace(0, 1, len(model_names))))
    
    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + txt_h, 
                 f'{yval:.3f}', va='bottom' if yval >= 0 else 'top', ha='center') 

    ylabel = 'RMSE (s)' if metric == 'rmse' else 'R2 Score'
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    else:
        # Adjust ylim slightly for text visibility
        current_ylim = plt.ylim()
        if metric == 'rmse':
             plt.ylim(0, current_ylim[1] * 1.15)
        # For R2, ensure reasonable bounds if not specified
        elif metric == 'r2' and ylim is None:
             plt.ylim(min(metric_values)-0.1 if min(metric_values) < 0 else 0, 
                      max(metric_values)+0.1 if max(metric_values) > 0 else 0.1)
                 
    plt.xticks(rotation=15, ha='right') # Rotate labels slightly
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()

def plot_iter_prediction(tau_mean, tau_std, mahalanobis_dist, iter_params, save_path=None):
    """
    Visualize ITER prediction with uncertainty and Mahalanobis distance.

    Args:
        tau_mean: Predicted mean confinement time for ITER.
        tau_std: Predicted standard deviation of confinement time for ITER.
        mahalanobis_dist: Mahalanobis distance of ITER point from training data.
        iter_params: Dictionary of ITER parameters used for the prediction.
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=(6, 6))
    
    # Create a pseudo x-axis
    x = [0]
    y = [tau_mean]
    yerr = [2 * tau_std] # 2-sigma uncertainty

    plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=8, capsize=5, 
                 label=f'$\tau = {tau_mean:.2f} \pm {2*tau_std:.2f}$ s (2$\sigma$)')

    # Add text annotations for parameters and Mahalanobis distance
    param_text = 'ITER Parameters:\n' + '\n'.join([f'{k}: {v}' for k, v in iter_params.items() if k in ['IP', 'BT', 'NEL', 'PLTH', 'RGEO', 'AMIN', 'TAU98Y2']])
    param_text += f'\n\nMahalanobis Dist (Eng): {mahalanobis_dist:.2f}'
    
    # Place text box
    plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.ylabel('Predicted Confinement Time $\tau$ (s)')
    plt.title('ITER Confinement Time Prediction (Normalized Model)')
    # Hide x-axis ticks and labels as it's just a single point visualization
    plt.xticks([]) 
    plt.xlim(-0.5, 0.5)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()


def plot_simple_scatter(x_true, y_pred, color='blue', label='Model',
                         xlabel=r'$\tau_{\mathrm{experiment}}$ (s)', ylabel='Predicted $\tau$ (s)',
                         title=None, xlim=None, ylim=None, 
                         scatter_alpha=0.7, scatter_edgecolors='k', scatter_linewidths=0.1,
                         show_trendline=True, trendline_color='blue',
                         save_path=None):
    """
    Create a simple scatter plot of true vs. predicted values without uncertainty.

    Args:
        x_true: Array of true values.
        y_pred: Array of predicted values.
        color: Color for the scatter points.
        label: Label for the scatter points.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title of the plot.
        xlim: Tuple specifying x-axis limits (optional).
        ylim: Tuple specifying y-axis limits (optional).
        scatter_alpha: Alpha transparency for scatter points.
        scatter_edgecolors: Edge color for scatter points.
        scatter_linewidths: Edge line width for scatter points.
        show_trendline: Whether to show the linear regression trendline (default True).
        trendline_color: Color of the trendline (default 'blue').
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(x_true, y_pred, color=color, label=label, 
                alpha=scatter_alpha, edgecolors=scatter_edgecolors, linewidths=scatter_linewidths, s=20)

    # Add trendline (linear regression)
    if show_trendline:
        # Calculate linear regression
        slope, intercept = np.polyfit(x_true, y_pred, 1)
        # Create trendline points
        if xlim is None:
            xlim = plt.xlim()
        x_trend = np.array([xlim[0], xlim[1]])
        y_trend = slope * x_trend + intercept
        # Plot trendline
        plt.plot(x_trend, y_trend, color=trendline_color, linestyle='-', 
                 label=f'Trendline (y={slope:.3f}x+{intercept:.3f})')

    # Ideal line (y=x)
    if xlim is None:
        xlim = plt.xlim()
    if ylim is None:
        ylim = plt.ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()

# Placeholder for plot_ipb98_prediction if needed, or integrate into plot_simple_scatter/other plots.
# def plot_ipb98_prediction(...): 
#     pass

# Add the calibration plot from uncertainty_toolbox if needed directly here
# or call it from the analysis scripts.
def plot_calibration_uct(mean, std, true_y, title=None, save_path=None):
    """
    Plot calibration metrics using uncertainty_toolbox.
    
    Args:
        mean: Array of predicted mean values.
        std: Array of predicted standard deviations.
        true_y: Array of true values.
        title: Optional title for the plot.
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=(8, 6))
    uct.plot_calibration(mean, std, true_y)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show() 