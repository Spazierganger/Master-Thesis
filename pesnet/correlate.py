from typing import Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from pesnet.nn import Activation, AutoMLP, activation_function, named, residual


class CorrelateMean(nn.Module):
    
    @nn.compact
    def __call__(self, atoms, params1, params2):
        for i, p in enumerate(params1):
            params1[i] = jnp.repeat(p.mean(0, keepdims=True), p.shape[0], axis=0)
        
        for i, p in enumerate(params2):
            params2[i] = jnp.repeat(p.mean(0, keepdims=True), p.shape[0], axis=0)
        return params1, params2


def atoms_dist(atoms):
    n_nuclei = atoms.shape[-2]
    senders, receivers = jnp.triu_indices(n_nuclei, k=1)
    dist = atoms[senders] - atoms[receivers]
    pair_wise_dist = jnp.linalg.norm(dist, keepdims=False, axis=-1)
    return pair_wise_dist

batch_atoms_dist = jax.vmap(atoms_dist, in_axes=(0,))
    
    
class CorrelateDist(nn.Module):
    @nn.compact
    def __call__(self, atoms, params1, params2):
        dist = batch_atoms_dist(atoms)
        dist = jnp.linalg.norm(dist[None] - dist[:, None], keepdims=False, axis=-1)  # n_configs x (n(n-1)/2)
        
        temp = self.param(
            'geom_corr_temp',
            jnn.initializers.ones,
            (1,)
        )
        
        scores = jax.nn.softmax(- dist * jnn.softplus(temp), axis=0)
        
        for i, p in enumerate(params1):
            params1[i] = jnp.einsum('fj,f...->j...', scores, p)
        
        for i, p in enumerate(params2):
            params2[i] = jnp.einsum('fj,f...->j...', scores, p)
        return params1, params2
