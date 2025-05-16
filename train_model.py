import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler

from utils.data_utils import (load_data, prepare_features, scale_data, 
                             split_data, DataLoader)
from utils.model_utils import (create_models, create_optimizers, train_models,
                              save_scaler, save_metadata)

SEED = 42
np.random.seed(SEED)

DATA_FILE = 'data/processed/normalized.csv'
BASE_OUTPUT_DIR = './trained_models' 

FILTER_CONFIGS = {
    'std5': "(SELDB5 == 1)",
}

MODEL_CONFIGS = {
    'std5_norm': {
        'filter_id': 'std5',
        'model_type': 'normalized',
        'input_cols': ["qCYL", "GWF", "DELTA", "PFIN", "PRFN",  "HIGHZ", "COAT"],
        'output_col': "HIPB98Y2",
        'eval_cols': ["TAUTH", "TAU98Y2", "TAU20"],
        'exclude_scaling_indices': [-2, -1],
    }
}

ENSEMBLE_SIZE = 20
HIDDEN_LAYERS = [200, 200, 200, 200, 200]
ACTIVATIONS = ['tanh'] * 10 + ['swish'] * 10 
if len(ACTIVATIONS) != ENSEMBLE_SIZE:
    raise ValueError(f"Number of activations ({len(ACTIVATIONS)}) must match ENSEMBLE_SIZE ({ENSEMBLE_SIZE})")

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100 
VALIDATION_SPLIT_SIZE = 0.1 
BATCH_SIZE_TRAIN = None

def train_single_model(model_name: str):
    config = MODEL_CONFIGS[model_name]
    filter_id = config['filter_id']
    filter_query = FILTER_CONFIGS.get(filter_id)
        
    input_cols = config['input_cols']
    output_col = config['output_col']
    eval_cols = config['eval_cols']
    exclude_scaling_indices = config['exclude_scaling_indices']
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)

    train_val_df = load_data(DATA_FILE, filter_query=filter_query)

    X_train_val, y_train_val, eval_train_val = prepare_features(train_val_df, input_cols, output_col, eval_cols)
    indices_train_val = np.arange(X_train_val.shape[0])

    train_indices, val_indices, y_train_scaled_split, y_val_scaled_split = split_data(
        X_train_val, y_train_val, indices_train_val, 
        test_size=VALIDATION_SPLIT_SIZE, random_state=SEED
    )
    X_train_split = X_train_val[train_indices]
    y_train_split = y_train_val[train_indices]
    X_val_split = X_train_val[val_indices]
    y_val_split = y_train_val[val_indices]
    eval_train_split = eval_train_val[train_indices]
    eval_val_split = eval_train_val[val_indices]
    
    X_train_scaled, y_train_scaled, X_val_scaled, _, scaler_X, scaler_y = scale_data(
        X_train_split, y_train_split, X_val=X_val_split, 
        exclude_cols_indices=exclude_scaling_indices
    )
    y_val_scaled = scaler_y.transform(y_val_split)

    batch_size_train_actual = BATCH_SIZE_TRAIN if BATCH_SIZE_TRAIN is not None else X_train_scaled.shape[0]
    data_train_loader = DataLoader(X_train_scaled, y_train_scaled, batch_size=batch_size_train_actual, shuffle=True, seed=SEED)
    data_val_loader = DataLoader(X_val_scaled, y_val_scaled, batch_size=X_val_scaled.shape[0], shuffle=False, seed=SEED)

    input_dim = X_train_scaled.shape[1]
    output_dim = y_train_scaled.shape[1]
    models = create_models(input_dim, output_dim, HIDDEN_LAYERS, ACTIVATIONS, ENSEMBLE_SIZE, SEED)
    optimizers = create_optimizers(models, LEARNING_RATE, WEIGHT_DECAY)

    trained_models = train_models(
        models, optimizers, data_train_loader, data_val_loader, 
        EPOCHS, ENSEMBLE_SIZE, output_dir 
    )

    scaler_X_path = os.path.join(output_dir, 'scaler_X.joblib')
    scaler_y_path = os.path.join(output_dir, 'scaler_y.joblib')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    save_scaler(scaler_X, scaler_X_path)
    save_scaler(scaler_y, scaler_y_path)
    
    metadata = {
        'model_name': model_name,
        'filter_id': filter_id,
        'filter_query': filter_query,
        'input_cols': input_cols,
        'input_cols_latex': config.get('input_cols_latex', input_cols),
        'output_col': output_col,
        'eval_cols': eval_cols,
        'exclude_scaling_indices': exclude_scaling_indices,
        'ensemble_size': ENSEMBLE_SIZE,
        'hidden_layers': HIDDEN_LAYERS,
        'activations': ACTIVATIONS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'epochs': EPOCHS,
        'seed': SEED,
        'training_data_size': len(X_train_split),
        'validation_data_size': len(X_val_split),
    }

    save_metadata(metadata, metadata_path)
    
    np.save(os.path.join(output_dir, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_val_scaled.npy'), X_val_scaled)
    np.save(os.path.join(output_dir, 'eval_val_split.npy'), eval_val_split)

if __name__ == "__main__":
    train_single_model('std5_norm')