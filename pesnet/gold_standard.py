import functools
from typing import Sequence, Tuple, Union, Optional, Dict

import pdb
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp

from pesnet.nn import (MLP, Activation, ActivationWithGain,
                       activation_function, residual, AutoMLP, additive_softmax)
from pesnet.ferminet import LogSumDet, Orbitals, apply_covariance, IsotropicEnvelope, FullEnvelope
from pesnet.gnn import BesselRBF


class InvariantEncoding(nn.Module):

    @nn.compact
    def __call__(self, electrons, atoms):
        
        n_elec = electrons.shape[0]
        n_atoms = atoms.shape[0]

        r_im3 = electrons[:, None] - atoms[None]
        r_im_norm = jnp.linalg.norm(r_im3, keepdims=True, axis=-1)
        r_im4 = jnp.concatenate([r_im3, r_im_norm], axis=-1)
        
        r_ij = electrons[:, None] - electrons[None]
        r_ij_norm = jnp.linalg.norm(
            r_ij + jnp.eye(n_elec)[..., None],
            keepdims=True,
            axis=-1
        ) * (1.0 - jnp.eye(n_elec)[..., None])
        r_ij4 = jnp.concatenate([r_ij, r_ij_norm], axis=-1)
        
        h0 = r_im4.reshape(n_elec, -1)
        k0 = r_im4
        g0 = r_ij_norm
        
        nuc_embedding = self.param(
            'nuc_embedding',
            jnn.initializers.normal(stddev=1.),
            (n_atoms, 4)
        )
        
        return h0, k0, g0, nuc_embedding
    
    
def recover_g(g_same, g_diff, spins):
    n_up, n_down = spins
    g_up = jnp.concatenate([g_same[:n_up * n_up, :].reshape(n_up, n_up, -1), 
                            g_diff[:n_up * n_down, :].reshape(n_up, n_down, -1)], axis=1)
    g_down = jnp.concatenate([g_diff[n_up * n_down:, :].reshape(n_down, n_up, -1), 
                              g_same[n_down * n_down:, :].reshape(n_down, n_down, -1)], axis=1)
    g = jnp.concatenate([g_up, g_down], axis=0)
    return g
    

def break_g(g, spins):
    n_up, n_down = spins
    g_same = jnp.concatenate([g[..., :n_up, :n_up, :].reshape(n_up * n_up, -1), 
                              g[..., n_up:, n_up:, :].reshape(n_down * n_down, -1)], axis=0)
    g_diff = jnp.concatenate([g[..., :n_up, n_up:, :].reshape(n_up * n_down, -1), 
                              g[..., n_up:, :n_up, :].reshape(n_up * n_down, -1)], axis=0)
    return g_same, g_diff
    
    
class Symmetric(nn.Module):
    spins: Tuple[int]
    b_size: int = 32
    c_size: int = 32
    
    @nn.compact
    def __call__(self, g_same, g_diff, h, k, nuc_embd):
        n_up, n_dn = self.spins
        g_same = nn.Dense(self.b_size)(g_same)
        g_diff = nn.Dense(self.b_size)(g_diff)
                
        g = recover_g(g_same, g_diff, self.spins)
                
        h1 = nn.Dense(self.c_size)(h)
        s_el = (g * h1).sum(1)
                
        h2 = jnp.tile(h[:n_up, ...].mean(keepdims=True), [sum(self.spins), 1])
        h3 = jnp.tile(h[n_up:, ...].mean(keepdims=True), [sum(self.spins), 1])
                
        h4 = (nn.Dense(self.c_size)(nuc_embd[None]) * nn.Dense(self.b_size)(k)).sum(1)
                
        return jnp.concatenate([h, h1, h2, h3, h4], axis = -1)
    
    
class Layer(nn.Module):
    spins: Tuple[int]
    a_size: int = 256
    a_ion: int = 32
    activation: Activation = jnn.tanh

    @nn.compact
    def __call__(self, h, k, g_same, g_diff, nuc_embedding):
        act = self.activation
        
        h_new = Symmetric(self.spins)(g_same, g_diff, h, k, nuc_embedding)
        h_new = residual(h, act(nn.Dense(self.a_size)(h_new)))
        
        k_new = residual(k, act(nn.Dense(self.a_ion)(k)))
        
        g_same = residual(g_same, act(nn.Dense(self.a_ion)(g_same)))
        g_diff = residual(g_diff, act(nn.Dense(self.a_ion)(g_diff)))
            
        return h_new, k_new, g_same, g_diff
        

class Model(nn.Module):
    charges: Tuple[int]
    spins: Tuple[int, int]
    full_det: bool = False
    envelope_type: str = 'isotropic'
    hidden_dims: Sequence[Tuple[int]] = (256, 256, 256, 256)
    determinants: int = 16

    def setup(self):

        self.input_construction = InvariantEncoding()
        
        self.layers = [
            Layer(spins=self.spins)
            for d in self.hidden_dims
        ]

        self.to_orbitals = Orbitals(self.spins, self.determinants, 
                                    full_det=self.full_det,
                                    envelope_type=self.envelope_type)
        self.logsumdet = LogSumDet()
    
    def encode(self, electrons, atoms):
        atoms = atoms.reshape(-1, 3)
        electrons = electrons.reshape(-1, 3)
        
        h, k, g, nuc_embedding = self.input_construction(electrons, atoms)
        r_im4 = k
        g_same, g_diff = break_g(g, self.spins)
        
        for i, layer in enumerate(self.layers):
            h, k, g_same, g_diff = layer(h, k, g_same, g_diff, nuc_embedding)
        
        return h, r_im4

    def orbitals(self, electrons, atoms):
        h_one, r_im = self.encode(electrons, atoms)
        return self.to_orbitals(h_one, r_im)

    def signed(self, electrons, atoms):
        h_one, r_im = self.encode(electrons, atoms)
        orbitals = self.to_orbitals(h_one, r_im)
        sign, log_psi = self.logsumdet(orbitals)
        return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]