"""
Evaluation metrics and prediction functions.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import jax
import jax.numpy as jnp
from flax import nnx

# Assuming GaussianMLP is defined elsewhere (e.g., in models.MLP)
# from models.MLP import GaussianMLP 

def evaluate_performance(y_true, y_pred):
    """
    Evaluate model performance using R2 score and RMSE.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing R2 score and RMSE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'r2': r2, 'rmse': rmse}

def predict_with_uncertainty(models, X):
    """
    Perform prediction with uncertainty estimation using an ensemble of models.

    Args:
        models: List of trained model instances.
        X: Input data (JAX array or NumPy array).

    Returns:
        Tuple containing:
        - mean_prediction: Mean prediction over the ensemble.
        - epistemic_uncertainty: Epistemic uncertainty (variance).
        - aleatoric_uncertainty: Aleatoric uncertainty (variance).
        - total_uncertainty: Total uncertainty (variance).
    """
    if not isinstance(X, jnp.ndarray):
        X = jnp.array(X) # Ensure X is a JAX array
        
    predictions = []
    logvars = []

    for model in models:
        model.eval() # Set model to evaluation mode (important if using dropout etc.)
        # 직접 모델 호출 (JIT 없이)
        mean, logvar = model(X)
        predictions.append(mean)
        logvars.append(logvar)

    # 모든 예측을 jnp.array로 변환
    predictions = [jnp.asarray(pred) for pred in predictions]
    logvars = [jnp.asarray(logvar) for logvar in logvars]
    
    predictions = jnp.stack(predictions)
    logvars = jnp.stack(logvars)

    # Calculate mean prediction
    mean_prediction = jnp.mean(predictions, axis=0)

    # Calculate epistemic uncertainty (variance of ensemble means)
    epistemic_uncertainty = jnp.var(predictions, axis=0)

    # Calculate aleatoric uncertainty (mean of predicted variances)
    # variance = exp(logvar)
    aleatoric_uncertainty = jnp.mean(jnp.exp(logvars), axis=0)

    # Calculate total uncertainty
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

    return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty

def convert_to_original_scale(mean_scaled, uncertainties_scaled, scaler_y):
    """
    Convert scaled predictions and uncertainties back to the original data scale.

    Args:
        mean_scaled: Scaled mean predictions.
        uncertainties_scaled: Tuple of scaled uncertainties (epistemic, aleatoric, total).
        scaler_y: Fitted StandardScaler object for the target variable.

    Returns:
        Tuple containing:
        - mean_original: Mean predictions in the original scale.
        - epistemic_std_original: Epistemic uncertainty (standard deviation) in the original scale.
        - aleatoric_std_original: Aleatoric uncertainty (standard deviation) in the original scale.
        - total_std_original: Total uncertainty (standard deviation) in the original scale.
    """
    # Inverse transform the mean prediction
    mean_original = scaler_y.inverse_transform(mean_scaled.reshape(-1, 1))

    # Scale of the target variable (standard deviation)
    scale_y = scaler_y.scale_[0]

    # Convert variances to standard deviations and scale back
    # std_dev = sqrt(variance) * scale_y
    epistemic_std_original = jnp.sqrt(uncertainties_scaled[0]) * scale_y
    aleatoric_std_original = jnp.sqrt(uncertainties_scaled[1]) * scale_y
    total_std_original = jnp.sqrt(uncertainties_scaled[2]) * scale_y
    
    # Ensure output shapes are consistent (flatten if necessary)
    mean_original = mean_original.flatten()
    epistemic_std_original = epistemic_std_original.flatten()
    aleatoric_std_original = aleatoric_std_original.flatten()
    total_std_original = total_std_original.flatten()

    return mean_original, epistemic_std_original, aleatoric_std_original, total_std_original

def calculate_confinement_time(h_factor_mean, h_factor_std, tau_scaling_law):
    """
    Calculate confinement time and its uncertainty from H-factor predictions.

    Args:
        h_factor_mean: Predicted mean H-factor (e.g., HIPB98Y2).
        h_factor_std: Predicted standard deviation of H-factor.
        tau_scaling_law: Confinement time from the corresponding scaling law (e.g., TAU98Y2).

    Returns:
        Tuple containing:
        - tau_mean: Predicted mean confinement time (tau_experiment).
        - tau_std: Predicted standard deviation of confinement time.
    """
    # Ensure inputs are NumPy arrays for calculation
    h_factor_mean = np.asarray(h_factor_mean)
    h_factor_std = np.asarray(h_factor_std)
    tau_scaling_law = np.asarray(tau_scaling_law)
    
    # Calculate mean confinement time: tau_mean = H * tau_scaling
    tau_mean = h_factor_mean * tau_scaling_law

    # Calculate standard deviation using error propagation (assuming no correlation)
    # Variance(tau) = (d(tau)/d(H))^2 * Var(H) = (tau_scaling)^2 * Var(H)
    # StdDev(tau) = sqrt( (tau_scaling)^2 * StdDev(H)^2 ) = tau_scaling * StdDev(H)
    tau_std = tau_scaling_law * h_factor_std

    return tau_mean, tau_std 