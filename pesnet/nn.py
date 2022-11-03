import functools
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np


Activation = Callable[[jnp.ndarray], jnp.ndarray]
ParamTree = Union[jnp.ndarray,
                  Iterable['ParamTree'],
                  Mapping[Any, 'ParamTree']]


def dsilu(x):
    sig_x = jnn.sigmoid(x)
    return sig_x * (1 + x * (1 - sig_x))


def dtanh(x):
    return dsilu(x)*2-1

def s_func(x):
    return x * (1 - nn.tanh(nn.tanh(x) * x))


ACTIVATION_GAINS = {
    nn.silu: 1.7868129431578026,
    nn.tanh: 1.5927812698663606,
    nn.sigmoid: 4.801203511726151,
    dsilu: 2.7785038406086646,
    dtanh: 1.3892519203043323,
    s_func: 3.363177499295129,
}


def activation_function(fn: Union[str, Activation]):
    if callable(fn):
        return fn
    elif fn == 'dsilu':
        return dsilu
    elif fn == 'dtanh':
        return dtanh
    elif fn == 's_func':
        return s_func
    else:
        try:
            return getattr(nn, fn)
        except:
            return getattr(jnp, fn)


Dense_no_bias = functools.partial(nn.Dense, use_bias=False)
Dense = nn.Dense
Embed = nn.Embed


def residual(
    x: jnp.ndarray,
    y: jnp.ndarray
) -> jnp.ndarray:
    """Adds a residual connection between input x and output y if possible.

    Args:
        x (jnp.ndarray): input
        y (jnp.ndarray): output

    Returns:
        jnp.ndarray: new output
    """
    if x.shape == y.shape:
        return (x + y) / jnp.sqrt(2.0)
    else:
        return y


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Activation
    intermediate_bias: bool = True
    final_bias: bool = False

    @nn.compact
    def __call__(self, x):
        if len(self.hidden_dims) == 0:
            return x

        Dense_inter = Dense if self.intermediate_bias else Dense_no_bias
        Dense_out = Dense if self.final_bias else Dense_no_bias

        activation = ActivationWithGain(self.activation)
        for hidden_dim in self.hidden_dims[:-1]:
            x = activation(
                Dense_inter(hidden_dim)(x))
        x = Dense_out(self.hidden_dims[-1])(x)
        return x

    @staticmethod
    def extract_final_linear(params):
        key = list(params)[-1]
        return params[key]


class AutoMLP(nn.Module):
    out_dim: int
    n_layers: int
    activation: Activation
    scale: str = 'log'
    intermediate_bias: bool = True
    final_bias: bool = True

    @nn.compact
    def __call__(self, x):
        inp_dim = x.shape[-1]
        # We use np instead of jnp to ensure that it is static.
        if self.out_dim > 0 and inp_dim > 0:
            if self.scale == 'log':
                hidden_dims = np.round(
                    np.logspace(
                        np.log(inp_dim),
                        np.log(self.out_dim),
                        self.n_layers + 1,
                        base=np.e
                    )
                ).astype(np.int32)[1:]
            elif self.scale == 'linear':
                hidden_dims = np.round(
                    np.linspace(
                        inp_dim,
                        self.out_dim,
                        self.n_layers + 1
                    )
                ).astype(np.int32)[1:]
            else:
                raise ValueError()
        else:
            hidden_dims = [0]
        if inp_dim == 0:
            hidden_dims = [self.out_dim]

        Dense_inter = Dense if self.intermediate_bias else Dense_no_bias
        Dense_out = Dense if self.final_bias else Dense_no_bias

        activation = ActivationWithGain(self.activation)
        for hidden_dim in hidden_dims[:-1]:
            x = activation(
                Dense_inter(hidden_dim)(x))
        x = Dense_out(hidden_dims[-1])(x)
        return x


class ActivationWithGain(nn.Module):
    activation: Activation

    @nn.compact
    def __call__(self, x):
        return self.activation(x) * ACTIVATION_GAINS[self.activation]

    
def named(name, module, *args, **kwargs):
    return type(name, (module,), {})(*args, **kwargs)


def additive_softmax(x, axis=None):
    x_sum = jnp.sum(x, axis=axis, keepdims=True)
    return x / x_sum
