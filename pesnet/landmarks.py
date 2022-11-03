from typing import Sequence, Tuple, Union

import pdb
import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp
import jraph

from pesnet.nn import Activation, AutoMLP, activation_function, named, residual


class LandmarkOuput(nn.Module):
    node_out_dims: Tuple[int, ...]
    global_out_dims: Tuple[int, ...]
    landmark_size: int
    activation: Union[str, Activation] = nn.silu

    @nn.compact
    def __call__(self, node_params, global_params):
        g_params = [self.param(
                        f'gparams{i}',
                        jnn.initializers.normal(stddev=1.),
                        (self.landmark_size, _shape)
                    ) for i, _shape in enumerate(self.global_out_dims)]
        
        n_params = [self.param(
                        f'nparams{i}',
                        jnn.initializers.normal(stddev=1.),
                        (self.landmark_size, _shape)
                    ) for i, _shape in enumerate(self.node_out_dims)]
                
        for i, (g, gp) in enumerate(zip(g_params, global_params)):
            global_params[i] = gp @ jnn.softmax(g, axis=0)
        
        for i, (n, np) in enumerate(zip(n_params, node_params)):
            node_params[i] = np @ jnn.softmax(n, axis=0)
        
        return node_params, global_params

class LandmarkPlaceholder(nn.Module):
    @nn.compact
    def __call__(self, node_params, global_params):
        return node_params, global_params
