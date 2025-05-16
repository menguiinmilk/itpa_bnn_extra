import jax
import jax.numpy as jnp
from flax import nnx

class GaussianOutputLayer(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.mu = nnx.Linear(din, dout, rngs=rngs)
        self.logvar = nnx.Linear(din, dout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class GaussianOutputLayer2(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.mu = nnx.Linear(din, dout, rngs=rngs)
        self.var = nnx.Linear(din, dout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        mu = self.mu(x)
        var = jax.nn.softplus(self.var(x))
        return mu, var

class MLP(nnx.Module):
    """
    Basic MLP (Multi-Layer Perceptron) model.
    """
    def __init__(self, din, hidden_layers, dout, rngs, activation='tanh'):
        """
        Initialize MLP model.
        
        Args:
            din: Input dimension
            hidden_layers: List of hidden layer sizes
            dout: Output dimension
            rngs: Random number generator state
            activation: Activation function ('tanh' or 'swish')
        """
        super().__init__()
        self.layers = []
        
        # Input layer
        self.layers.append(nnx.Linear(din, hidden_layers[0], rngs=rngs))
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = jnp.tanh
        elif activation == 'swish':
            self.activation = jax.nn.swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nnx.Linear(hidden_layers[i], hidden_layers[i+1], rngs=rngs))
        
        # Output layer
        self.layers.append(nnx.Linear(hidden_layers[-1], dout, rngs=rngs))
    
    def __call__(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        return self.layers[-1](x)

class GaussianMLP(nnx.Module):
    """
    Multi-Layer Perceptron that outputs parameters for a Gaussian distribution.
    """
    def __init__(self, din: int, hidden_layers: list[int], dout: int, activation: str, *, rngs: nnx.Rngs):
        """
        Initializes the GaussianMLP model.

        Args:
            din: Input dimension.
            hidden_layers: List of integers specifying the number of neurons in each hidden layer.
            dout: Output dimension (should be 1 for scalar target).
            activation: Activation function name (e.g., 'relu', 'tanh', 'swish').
            rngs: NNX random number generators.
        """
        self.layers = []
        sizes = [din] + hidden_layers
        
        # Define activation function based on input string
        if activation == 'relu':
            act_fn = nnx.relu
        elif activation == 'tanh':
            act_fn = nnx.tanh
        elif activation == 'swish':
            act_fn = nnx.swish
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        for i, (nin, nout) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(nnx.Linear(nin, nout, rngs=rngs))
            self.layers.append(act_fn)
            
        self.mean_layer = nnx.Linear(sizes[-1], dout, rngs=rngs)
        self.logvar_layer = nnx.Linear(sizes[-1], dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        """
        Forward pass through the network.

        Args:
            x: Input data.

        Returns:
            A tuple containing the mean (mu) and log variance (logvar) 
            of the predicted Gaussian distribution.
        """
        for layer in self.layers:
            x = layer(x)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class GaussianMLP2(nnx.Module):
    def __init__(self, din: int, hidden_layers: list[int], dout: int, rngs: nnx.Rngs):
        self.mlp = MLP(din, hidden_layers[:-1], hidden_layers[-1], rngs)
        self.output_layer = GaussianOutputLayer2(hidden_layers[-1], dout, rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        y = self.mlp(x)
        return self.output_layer(y)

if __name__ == "__main__":
    model = MLP(din=12, hidden_layers=[10, 10, 10], dout=1, rngs=nnx.Rngs(0))
    nnx.display(model)
