"""
Calculates the Mahalanobis distance.
"""
import numpy as np
from scipy.linalg import pinvh

def calculate_mahalanobis_distance(X_train, X_eval, exclude_binary=False):
    """
    Calculate Mahalanobis distance between evaluation set and training set.

    Args:
        X_train: Training set features (NumPy array).
        X_eval: Evaluation set features (NumPy array).
        exclude_binary: If True, exclude columns with only 0s and 1s before calculation.

    Returns:
        NumPy array of Mahalanobis distances for each point in X_eval.
    """
    if exclude_binary:
        binary_cols_mask = np.all((X_train == 0) | (X_train == 1), axis=0)

        X_train_non_binary = X_train[:, ~binary_cols_mask]
        X_eval_non_binary = X_eval[:, ~binary_cols_mask]
        
        if X_train_non_binary.shape[1] == 0:
            return np.zeros(X_eval.shape[0]) 
        
        X_train_proc = X_train_non_binary
        X_eval_proc = X_eval_non_binary
    else:
        X_train_proc = X_train
        X_eval_proc = X_eval
        
    mean = np.mean(X_train_proc, axis=0)
    # Handle potential singular matrix with pseudo-inverse
    try:
        cov = np.cov(X_train_proc, rowvar=False)
        inv_cov = pinvh(cov) 
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix calculation failed. Using identity matrix.")
        inv_cov = np.eye(X_train_proc.shape[1])
        
    # Calculate Mahalanobis distance
    diff = X_eval_proc - mean
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)
        
    mahalanobis_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    mahalanobis_sq[mahalanobis_sq < 0] = 0
    
    return np.sqrt(mahalanobis_sq) 