"""
Data loading, preprocessing, and utility functions.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import jax.random as jr
import os

class DataLoader:
    """Simple DataLoader for JAX.
    Manages batching and shuffling of data.
    """
    def __init__(self, features, targets, batch_size, shuffle=True, seed=None):
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = jr.PRNGKey(seed) if seed is not None else jr.PRNGKey(0)
        self.n_samples = features.shape[0]
        self.indices = np.arange(self.n_samples)
        if shuffle:
            self.key, subkey = jr.split(self.key)
            self.indices = jr.permutation(subkey, self.indices)

    def __iter__(self):
        self.current_pos = 0
        if self.shuffle:
            self.key, subkey = jr.split(self.key)
            self.indices = jr.permutation(subkey, self.indices)
        return self

    def __next__(self):
        if self.current_pos >= self.n_samples:
            raise StopIteration
        
        start = self.current_pos
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        self.current_pos = end
        
        return self.features[batch_indices], self.targets[batch_indices]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size # Number of batches

def load_data(file_path, filter_query=None):
    """
    Load data from a CSV file and optionally apply a filter.

    Args:
        file_path (str): Path to the CSV file.
        filter_query (str, optional): Pandas query string to filter the data. Defaults to None.

    Returns:
        pandas.DataFrame: Loaded and filtered DataFrame.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded data from {file_path}. Original size: {len(df)}")
        if filter_query:
            df_filtered = df.query(filter_query)
            print(f"Applied filter: '{filter_query}'. Filtered size: {len(df_filtered)}")
            return df_filtered
        else:
            return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading or filtering data: {e}")
        raise

def prepare_features(df, input_cols, output_col, eval_cols=None):
    """
    Extract features (X), target (y), and evaluation columns from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        input_cols (list): List of column names for input features (X).
        output_col (str): Column name for the target variable (y).
        eval_cols (list, optional): List of column names for additional evaluation data. Defaults to None.

    Returns:
        tuple: Contains X (np.ndarray), y (np.ndarray), and optionally eval_data (np.ndarray).
    """
    X = df[input_cols].to_numpy()
    y = df[[output_col]].to_numpy() # Keep as 2D array
    
    if eval_cols:
        eval_data = df[eval_cols].to_numpy()
        return X, y, eval_data
    else:
        return X, y

def scale_data(X_train, y_train, X_val=None, X_test=None, exclude_cols_indices=None):
    """
    Scale features (X) and target (y) using StandardScaler.
    Optionally applies scaling to validation and test sets using the scaler fitted on the training data.
    Allows excluding specific columns from feature scaling (e.g., binary features).

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray, optional): Validation features. Defaults to None.
        X_test (np.ndarray, optional): Test features. Defaults to None.
        exclude_cols_indices (list, optional): List of column indices in X to exclude from scaling. Defaults to None.

    Returns:
        tuple: Contains scaled data (X_train_scaled, y_train_scaled, X_val_scaled, X_test_scaled) 
               and the fitted scalers (scaler_X, scaler_y).
               Scaled validation/test sets are None if the corresponding inputs were None.
    """
    # Scale target variable (y)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Scale features (X)
    scaler_X = StandardScaler()
    X_train_scaled = X_train.copy()
    
    cols_to_scale_mask = np.ones(X_train.shape[1], dtype=bool)
    if exclude_cols_indices:
        cols_to_scale_mask[exclude_cols_indices] = False
        
    # Fit scaler only on the columns to be scaled
    if np.any(cols_to_scale_mask):
        scaler_X.fit(X_train[:, cols_to_scale_mask])
        X_train_scaled[:, cols_to_scale_mask] = scaler_X.transform(X_train[:, cols_to_scale_mask])
    else:
         print("Warning: No columns selected for feature scaling.")

    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = X_val.copy()
        if np.any(cols_to_scale_mask):
            X_val_scaled[:, cols_to_scale_mask] = scaler_X.transform(X_val[:, cols_to_scale_mask])

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        if np.any(cols_to_scale_mask):
             X_test_scaled[:, cols_to_scale_mask] = scaler_X.transform(X_test[:, cols_to_scale_mask])

    return X_train_scaled, y_train_scaled, X_val_scaled, X_test_scaled, scaler_X, scaler_y

def split_data(X, y, indices, test_size=0.1, random_state=42):
    """
    Split data into training and validation sets based on pre-defined indices.
    Ensures y is split correctly alongside indices.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        indices (np.ndarray): Array of indices corresponding to X and y.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for splitting.

    Returns:
        tuple: train_indices, val_indices, y_train_split, y_val_split
    """
    train_indices, val_indices, y_train_split, y_val_split = train_test_split(
        indices, y, test_size=test_size, random_state=random_state
    )
    print(f"Split data: Train size={len(train_indices)}, Validation size={len(val_indices)}")
    return train_indices, val_indices, y_train_split, y_val_split

def pad_data(X_list, y_list, eval_list, max_len):
    padded_X_list = []
    padded_y_list = []
    padded_eval_list = []
    mask_list = []

    for X, y, e in zip(X_list, y_list, eval_list):
        n_samples = X.shape[0]
        pad_size = max_len - n_samples

        mask = jnp.ones(n_samples)

        if pad_size > 0:
            X_pad = jnp.zeros((pad_size, X.shape[1]))
            y_pad = jnp.zeros((pad_size, y.shape[1]))
            e_pad = jnp.zeros((pad_size, e.shape[1]))
            mask_pad = jnp.zeros(pad_size)

            X = jnp.vstack([X, X_pad])
            y = jnp.vstack([y, y_pad])
            e = jnp.vstack([e, e_pad])
            mask = jnp.concatenate([mask, mask_pad])

        padded_X_list.append(X)
        padded_y_list.append(y)
        padded_eval_list.append(e)
        mask_list.append(mask)

    padded_X = jnp.stack(padded_X_list, axis=0)
    padded_y = jnp.stack(padded_y_list, axis=0)
    padded_eval = jnp.stack(padded_eval_list, axis=0)
    masks = jnp.stack(mask_list, axis=0)

    return padded_X, padded_y, padded_eval, masks