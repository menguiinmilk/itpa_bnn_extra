"""
Utility functions for model creation, training, saving, and loading.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax
import orbax.checkpoint as ocp # For saving/loading models
import joblib # For saving/loading scalers
import json # For saving/loading metadata
import os
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler # Import StandardScaler
import numpy as np

# Assuming GaussianMLP is defined elsewhere
from models.MLP import GaussianMLP 

def create_models(input_dim: int, output_dim: int, 
                  hidden_layers: List[int], activations: List[str],
                  ensemble_size: int, seed: int) -> List[GaussianMLP]:
    """
    Create an ensemble of GaussianMLP models.

    Args:
        input_dim: Input dimension for the models.
        output_dim: Output dimension for the models.
        hidden_layers: List of hidden layer sizes.
        activations: List of activation function names (one per model).
        ensemble_size: Number of models in the ensemble.
        seed: Base random seed for reproducibility.

    Returns:
        A list of initialized GaussianMLP model instances.
    """
    key = jr.PRNGKey(seed)
    models = []
    if len(activations) != ensemble_size:
        raise ValueError(f"Length of activations ({len(activations)}) must match ensemble_size ({ensemble_size})")
        
    for i in range(ensemble_size):
        key, subkey = jr.split(key)
        # Ensure activation list matches ensemble size or handle appropriately
        activation = activations[i]
        model = GaussianMLP(
            din=input_dim,
            hidden_layers=hidden_layers,
            dout=output_dim,
            activation=activation, # Pass activation string
            rngs=nnx.Rngs(subkey)
        )
        models.append(model)
    print(f"Created {ensemble_size} models.")
    return models

def create_optimizers(models: List[nnx.Module], learning_rate: float, weight_decay: float) -> List[nnx.Optimizer]:
    """
    Create AdamW optimizers for each model in the ensemble.

    Args:
        models: List of model instances.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay (L2 regularization) for the optimizer.

    Returns:
        A list of Optimizer instances.
    """
    optimizers = [
        nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay)) 
        for model in models
    ]
    print(f"Created {len(optimizers)} optimizers with LR={learning_rate}, WD={weight_decay}.")
    return optimizers

# Define the loss function globally or pass it if it varies
@nnx.jit
def gaussian_nll_loss(model: nnx.Module, batch: tuple) -> jnp.ndarray:
    """
    Gaussian negative log-likelihood loss.
    Clips log variance to prevent numerical instability.
    """
    X, y = batch
    mu, logvar = model(X) 
    # Clip logvar: Avoid very small variances (exp(logvar) -> 0) -> large loss 
    # and very large variances (typically less problematic but good practice)
    logvar = jnp.clip(logvar, -10, 10) # Adjust clip range if needed

    loss = 0.5 * jnp.mean(logvar + (y - mu)**2 / jnp.exp(logvar))
    return loss

@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.metrics.Average, batch: tuple):
    """
    Performs a single training step: computes loss, gradients, and updates model.
    Uses the globally defined gaussian_nll_loss.
    """
    def loss_fn_wrapper(m, b):
        return gaussian_nll_loss(m, b)
        
    grad_fn = nnx.value_and_grad(loss_fn_wrapper)
    loss, grads = grad_fn(model, batch)
    
    metrics.update(loss=loss) # Log the loss for the epoch average
    optimizer.update(grads)

@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.metrics.Average, batch: tuple):
    """
    Performs a single evaluation step: computes loss.
    Uses the globally defined gaussian_nll_loss.
    """
    loss = gaussian_nll_loss(model, batch)
    metrics.update(loss=loss) # Log the loss for the epoch average

def train_models(models: List[nnx.Module], optimizers: List[nnx.Optimizer],
                 data_train: Any, data_val: Any, # Use DataLoader instances 
                 epochs: int, ensemble_size: int,
                 model_save_dir: str):
    """
    Train all models in the ensemble, saving the best state for each.

    Args:
        models: List of initialized model instances.
        optimizers: List of corresponding optimizers.
        data_train: DataLoader for the training set.
        data_val: DataLoader for the validation set.
        epochs: Number of training epochs.
        ensemble_size: Number of models (for logging).
        model_save_dir: Directory to save the best model checkpoints.

    Returns:
        A list of trained model instances (restored to their best validation state).
    """
    trained_models = []
    metrics_tracker = nnx.metrics.Average('loss') # Reusable metrics tracker
    # Removed overall CheckpointManager as we handle per-model saving/restoring
    # checkpointer = ocp.CheckpointManager(os.path.join(model_save_dir, 'checkpoints'), ocp.Checkpointer(ocp.args.StandardSave(nnx.State({}))), options=ocp.CheckpointManagerOptions(max_to_keep=1))

    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        print(f"\n--- Training Model {i+1}/{ensemble_size} ---")
        best_val_loss = float('inf')
        best_model_state_saved = False # Flag to track if a best state was ever saved
        graphdef, initial_state = nnx.split(model) # Get graphdef and initial state structure
        
        # Per-model checkpoint directory
        model_ckpt_dir = os.path.join(model_save_dir, f'model_{i}')
        os.makedirs(model_ckpt_dir, exist_ok=True)
        # Use StandardCheckpointHandler to handle saving/restoring state
        model_checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())

        for epoch in range(epochs):
            # Training Phase
            metrics_tracker.reset()
            model.train()
            for train_batch in data_train:
                train_step(model, optimizer, metrics_tracker, train_batch)
            computed_train_loss = metrics_tracker.compute()
            train_loss = np.array(computed_train_loss['loss'] if isinstance(computed_train_loss, dict) else computed_train_loss).item()
            
            # Validation Phase
            metrics_tracker.reset()
            model.eval()
            for val_batch in data_val:
                eval_step(model, metrics_tracker, val_batch)
            computed_val_loss = metrics_tracker.compute()
            val_loss = np.array(computed_val_loss['loss'] if isinstance(computed_val_loss, dict) else computed_val_loss).item()
            
            # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model state based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Capture the current best state
                _, current_best_state = nnx.split(model) 
                # Save the best state immediately using Orbax
                save_path = os.path.join(model_ckpt_dir, 'best_model')
                # 기존 모델 디렉토리가 존재하면 삭제
                if os.path.exists(os.path.abspath(save_path)):
                    import shutil
                    shutil.rmtree(os.path.abspath(save_path))
                    
                # Save the state dict (Pytree)
                model_checkpointer.save(os.path.abspath(save_path), args=ocp.args.StandardSave(current_best_state))
                best_model_state_saved = True
                # print(f"  -> New best validation loss: {best_val_loss:.4f}. Checkpoint saved.")
        
        # Restore the best model state after training completes
        if best_model_state_saved:
            # Load the best checkpoint state
            print(f"Restoring best model state for model {i+1} from checkpoint...")
            save_path = os.path.join(model_ckpt_dir, 'best_model')
            # Restore the state, providing the initial_state structure as target
            # restored_state = model_checkpointer.restore(save_path, args=ocp.args.StandardRestore(initial_state))
            restored_state = model_checkpointer.restore(os.path.abspath(save_path), args=ocp.args.StandardRestore(initial_state)) # Use absolute path
            # Merge the restored state with the original graph definition
            model = nnx.merge(graphdef, restored_state)
            print(f"Model {i+1} state restored to best validation performance.")
                 
        else:
             print(f"Warning: No best model state was saved for model {i+1} (validation loss might not have improved). Using final state.")

        trained_models.append(model)
        print(f"Model {i+1} training finished. Best Validation Loss: {best_val_loss:.4f}")
    
    return trained_models

# --- Saving/Loading Utilities --- 

def save_model_ensemble(models: List[nnx.Module], save_dir: str):
    """
    Save the state of each model in the ensemble to a separate subdirectory.
    Note: This saves the *final* state, not necessarily the best one found during training.
          The train_models function already saves the best state.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, model in enumerate(models):
        model_path = os.path.join(save_dir, f'model_{i}', 'final_state') # Save final state separately
        _, state = nnx.split(model)
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        # checkpointer.save(model_path, args=ocp.args.StandardSave(state))
        checkpointer.save(os.path.abspath(model_path), args=ocp.args.StandardSave(state)) # Use absolute path
        print(f"Saved final state of model {i} to {model_path}")

def load_model_ensemble(model_template: GaussianMLP, # Need a template model with correct structure/rngs
                        load_dir: str, ensemble_size: int) -> List[GaussianMLP]:
    """
    Load an ensemble of models from Orbax checkpoints (restoring their states).
    Loads the 'best_model' checkpoint saved by train_models.
    Requires a template model instance to restore into.
    """
    loaded_models = []
    print(f"Loading model ensemble from: {load_dir}")
    # Get graph definition and initial state structure from the template
    graphdef, initial_state = nnx.split(model_template) 
    
    for i in range(ensemble_size):
        # Path to the best model state checkpoint
        model_state_path = os.path.join(load_dir, f'model_{i}', 'best_model') 
        # Convert to absolute path before checking existence and restoring
        abs_model_state_path = os.path.abspath(model_state_path)
        if not os.path.exists(abs_model_state_path):
             raise FileNotFoundError(f"Best model checkpoint directory not found for model {i} at {abs_model_state_path}")
        
        print(f"Loading model {i} state from {abs_model_state_path}...")
        # Create a checkpointer with the standard handler
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        # Restore the state, providing the initial state structure as the target Pytree
        # restored_state = checkpointer.restore(model_state_path, args=ocp.args.StandardRestore(initial_state))
        restored_state = checkpointer.restore(abs_model_state_path, args=ocp.args.StandardRestore(initial_state)) # Use absolute path
        
        # Merge the graph definition from the template with the restored state
        loaded_model = nnx.merge(graphdef, restored_state)
        loaded_models.append(loaded_model)
            
    print(f"Loaded {len(loaded_models)} models.")
    return loaded_models

def save_scaler(scaler: StandardScaler, path: str):
    """
    Save a scikit-learn scaler object to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

def load_scaler(path: str) -> StandardScaler:
    """
    Load a scikit-learn scaler object from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found at {path}")
    scaler = joblib.load(path)
    print(f"Scaler loaded from {path}")
    return scaler

def save_metadata(metadata: Dict[str, Any], path: str):
    """
    Save metadata dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {path}")

def load_metadata(path: str) -> Dict[str, Any]:
    """
    Load metadata dictionary from a JSON file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    with open(path, 'r') as f:
        metadata = json.load(f)
    print(f"Metadata loaded from {path}")
    return metadata 

def load_model_ensemble_members(model_dir: str, metadata: dict) -> list:
    """
    Loads individual members of a GaussianMLP model ensemble.

    Args:
        model_dir (str): Path to the base directory of the saved model.
        metadata (dict): Loaded metadata dictionary containing model configuration.

    Returns:
        list: A list of loaded NNX model objects (ensemble members).

    Raises:
        FileNotFoundError: If metadata, scalers, or model checkpoints are missing.
        KeyError: If required keys are missing in the metadata.
        ValueError: If activation function cannot be determined or other config issues arise.
        Exception: For other unexpected errors during loading.
    """
    print(f"Loading model ensemble members individually for {metadata.get('model_name', os.path.basename(model_dir))}...")
    trained_models_loaded = []

    # Ensure necessary keys are present before creating template
    required_keys = ['input_cols', 'hidden_layers', 'activations', 'seed', 'ensemble_size']
    if not all(key in metadata for key in required_keys):
        missing_keys = [key for key in required_keys if key not in metadata]
        raise KeyError(f"Missing required keys {missing_keys} in metadata for model in {model_dir}")

    input_dim = len(metadata['input_cols'])
    output_dim = 1 # Assuming scalar target (e.g., H-factor or TAUTH)
    base_hidden_layers = metadata['hidden_layers']
    base_seed = metadata['seed']
    actual_ensemble_size = metadata['ensemble_size']
    activations_list = metadata.get('activations', [])

    # Validate or determine activations list
    if not isinstance(activations_list, list) or len(activations_list) != actual_ensemble_size:
        print(f"Warning: Activations list length mismatch or invalid format in metadata for {model_dir}. Expected {actual_ensemble_size}, got {len(activations_list) if isinstance(activations_list, list) else 'invalid'}.")
        # Try fallback: Use first activation if available
        if activations_list and isinstance(activations_list, list) and activations_list[0]:
            print(f"Using first activation '{activations_list[0]}' for all members.")
            activations_list = [activations_list[0]] * actual_ensemble_size
        # Try another fallback: Assume 10 tanh + 10 swish if size is 20
        elif actual_ensemble_size == 20:
             print("Assuming 10 tanh + 10 swish structure due to ensemble size 20 and invalid/missing activation list.")
             # Determine activations within the loop below
             activations_list = None # Signal to determine in loop
        else:
            raise ValueError(f"Cannot determine activations for ensemble in {model_dir}. Provide a valid 'activations' list in metadata or ensure ensemble size is 20 for tanh/swish split.")

    # Determine activation outside loop if list is valid
    get_activation = None
    if activations_list:
        get_activation = lambda i: activations_list[i]
    else: # Must be size 20, determine in loop based on index
        get_activation = lambda i: 'tanh' if i < actual_ensemble_size // 2 else 'swish'

    try:
        for i in range(actual_ensemble_size):
            current_activation = get_activation(i)
            print(f"  Loading model {i} with activation: {current_activation}")

            # Create specific template for this member
            model_template_i = GaussianMLP(
                din=input_dim,
                hidden_layers=base_hidden_layers,
                dout=output_dim,
                activation=current_activation,
                rngs=nnx.Rngs(base_seed + i) # Use unique seed per member
            )

            graphdef_i, initial_state_i = nnx.split(model_template_i)
            model_state_path = os.path.join(model_dir, f'model_{i}', 'best_model')
            abs_model_state_path = os.path.abspath(model_state_path)

            if not os.path.exists(abs_model_state_path):
                raise FileNotFoundError(f"Best model checkpoint not found for member {i} at {abs_model_state_path}.")

            # Load state using Orbax Checkpointer
            checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
            restored_state = checkpointer.restore(abs_model_state_path, args=ocp.args.StandardRestore(initial_state_i))
            loaded_model_i = nnx.merge(graphdef_i, restored_state)
            trained_models_loaded.append(loaded_model_i)

        print(f"--- Ensemble loaded successfully from {model_dir} ---")
        return trained_models_loaded

    except FileNotFoundError as e:
        print(f"Error loading ensemble member state from {model_dir}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred loading ensemble from {model_dir}: {e}")
        raise

def eng2norm(ip: float, bt: float, nel: float, plth: float, rgeo: float, amin: float, karea: float, meff: float, delta: float, surfform: float,
             pnbi: float, palpha: float, pecrh: float, picrh: float, highz: int, coat: int, 
             ) -> dict:
    """
    Calculate engineering parameters based on input plasma parameters.
    Derived from logic in normalize.py and iter_utils.py.

    Args:
        ip (float): Plasma current (MA).
        bt (float): Toroidal magnetic field (T).
        nel (float): Line-averaged electron density (1e19 m^-3).
        plth (float): Total loss power (MW) = POHM + PNBI + PECRH + PICRH - dW/dt - PSHI - PRADCORE.
                      In HDB5, PLTH = POHM + PAUX - PRADCORE - DWMHD.
        rgeo (float): Major radius (m).
        amin (float): Minor radius (m).
        karea (float): Elongation (kappa_a).
        meff (float): Effective atomic mass (amu).
        delta (float): Triangularity.
        pnbi (float): Neutral Beam Injection power (MW).
        palpha (float): Alpha heating power (MW).
        pecrh (float): Electron Cyclotron Resonance Heating power (MW).
        picrh (float): Ion Cyclotron Resonance Heating power (MW).
        highz (int): High-Z material flag (0 or 1). Input directly.
        coat (int): Coating flag (0 or 1). Input directly.
    Returns:
        dict: Dictionary containing calculated engineering parameters:
              'qCYL', 'GWF', 'DELTA_OUT', 'PNBIN', 'PRFN', 'HIGHZ_OUT', 'COAT_OUT'.
    """
    n20 = nel / 10.0  # Convert NEL from 1e19 m^-3 to 1e20 m^-3

    # Calculate PLH (L-H threshold power)
    plh = 2.15 * (n20**0.782) * (bt**0.772) * (amin**0.975) * (rgeo) * (2.0 / meff)
    
        
    # Use a very small number instead of 0 for division to avoid Inf/NaN
    safe_plh = plh if plh != 0 else np.finfo(float).eps

    # Calculate qCYL
    # qCYL = 5 * (BT / IP) * (AMIN^2 / RGEO) * KAREA
    qcyl = (5.0 * (bt / ip) * (amin**2 / rgeo) * karea) if ip != 0 and rgeo !=0 else np.inf
    
    # Calculate GWF
    # GWF = N20 / (IP / (pi * AMIN^2))
    gw_denominator = (ip / (np.pi * amin**2)) if ip != 0 and amin != 0 else 0
    gwf = (n20 / gw_denominator) if gw_denominator != 0 else np.inf

    # Calculate PNBIN
    # PNBIN = (PNBI + PALPHA) / PLH
    pfin = (pnbi + palpha) / safe_plh
    
    # Calculate PRFN
    # PRFN = (PECRH + PICRH) / PLH
    prfn = (pecrh + picrh) / safe_plh

    # DELTA is a direct input, so we return it as DELTA_OUT for clarity.
    delta_out = delta
    
    highz_out = highz
    coat_out = coat

    return {
        'qCYL': qcyl,
        'GWF': gwf,
        'DELTA': delta_out, # Renamed to avoid confusion with input 'delta'
        'PFIN': pfin,
        'PRFN': prfn,
        'HIGHZ': highz_out, # Renamed
        'COAT': coat_out    # Renamed
    }