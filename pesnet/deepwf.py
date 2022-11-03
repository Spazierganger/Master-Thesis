import functools
from typing import Sequence, Tuple, Union, Optional, List

import pdb

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp

from pesnet.nn import (MLP, Activation, ActivationWithGain,
                       activation_function, residual, AutoMLP)


def construct_single_features(
    h_one: jnp.ndarray,
    h_two: jnp.ndarray,
    spins: Tuple[int, int]
) -> jnp.ndarray:
    """Construct the electron specific input to the next layer.

    Args:
        h_one (jnp.ndarray): (N, single_dim)
        h_two (jnp.ndarray): (N, N, pair_dim)
        spins (Tuple[int, int]): (spin_up, spin_down)

    Returns:
        jnp.ndarray: (N, single_dim + 2*pair_dim)
    """
    h_twos = h_two.split(spins[0:1], axis=0)
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]
    return jnp.concatenate([h_one] + g_two, axis=1)


def construct_global_features(
    h_one: jnp.ndarray,
    spins: Tuple[int, int]
) -> jnp.ndarray:
    """Construct the global input to the next layer.

    Args:
        h_one (jnp.ndarray): (N, single_dim)
        spins (Tuple[int, int]): (spin_up, spin_down)

    Returns:
        jnp.ndarray: (single_dim)
    """
    h_ones = h_one.split(spins[0:1], axis=0)
    g_one = [jnp.mean(h, axis=0, keepdims=True)
             for h in h_ones if h.size > 0]
    return jnp.concatenate(g_one, axis=-1).reshape(1, 2*h_one.shape[1])
    

def apply_covariance(x, y):
    """Equivalent to jnp.einsum('ijk,kmjn->ijmn', x, y)."""
    i, _, _ = x.shape
    k, m, j, n = y.shape
    x = x.transpose((1, 0, 2))
    y = y.transpose((2, 0, 1, 3)).reshape((j, k, m * n))
    vdot = jax.vmap(jnp.dot, (0, 0))
    return vdot(x, y).reshape((j, i, m, n)).transpose((1, 0, 2, 3))

    
class IsotropicEnvelope(nn.Module):
    out_size: int

    @nn.compact
    def __call__(self, x):
        # x is of shape n_elec, n_nuclei, 1
        _, n_nuclei, _ = x.shape
        sigma = self.param(
            'sigma',
            jnn.initializers.ones,
            (n_nuclei, self.out_size)
        )
        pi = self.param(
            'pi',
            jnn.initializers.ones,
            (n_nuclei, self.out_size)
        )
        return jnp.sum(jnp.exp(-x[..., -1:] * sigma) * pi, axis=1)
    

class FullEnvelope(nn.Module):
    out_size: int
    
    @nn.compact
    def __call__(self, x):
        _, n_nuclei, _ = x.shape
        sigma = self.param(
            'sigma',
            lambda key, natom, nparam: jnp.tile(jnp.eye(3)[..., None, None], [1, 1, natom, nparam]),
            n_nuclei, self.out_size,
        )
        pi = self.param(
            'pi',
            jnn.initializers.ones,
            (n_nuclei, self.out_size),
        )
        r_ae = apply_covariance(x[..., :-1], sigma)
        r_ae = jnp.linalg.norm(r_ae, axis=2)
        return jnp.sum(jnp.exp(-r_ae) * pi, axis=1)


class InvariantEncoding(nn.Module):
    nuclei_embedding: int
    mlp_dims: Sequence[int]
    activation: Activation

    @nn.compact
    def __call__(self, electrons, atoms):
        n_elec = electrons.shape[0]
        n_atoms = atoms.shape[0]

        r_im3 = electrons[:, None] - atoms[None]
        r_im_norm = jnp.linalg.norm(r_im3, keepdims=True, axis=-1)
        r_im4 = jnp.concatenate([r_im3, r_im_norm], axis=-1)
        h_one = r_im4

        r_ij = electrons[:, None] - electrons[None]
        r_ij_norm = jnp.linalg.norm(
            r_ij + jnp.eye(n_elec)[..., None],
            keepdims=True,
            axis=-1
        ) * (1.0 - jnp.eye(n_elec)[..., None])
        h_two = jnp.concatenate([r_ij, r_ij_norm], axis=-1)

        nuc_embedding = self.param(
            'nuc_embedding',
            jnn.initializers.normal(),
            (n_atoms, self.nuclei_embedding)
        )
        h_one = nn.Dense(self.nuclei_embedding)(h_one)
        h_one = (h_one + nuc_embedding) / 2 ** 0.5
        
        h_one = MLP(
            self.mlp_dims,
            activation=self.activation
        )(h_one).mean(1)

        return h_one, h_two, r_im4


class FermiLayer(nn.Module):
    spins: Tuple[int, int]
    single_out: int
    pair_out: int
    activation: Activation

    @nn.compact
    def __call__(self, h_one, h_two):
        activation = ActivationWithGain(self.activation)

        # Single update
        one_in = construct_single_features(h_one, h_two, self.spins)
        global_in = construct_global_features(h_one, self.spins)
        h_one_new = activation((nn.Dense(self.single_out)(
            one_in) + nn.Dense(self.single_out, use_bias=False)(global_in))/jnp.sqrt(2))
        h_one = residual(h_one, h_one_new)

        # Pairwise update
        if self.pair_out > 0:
            h_two_new = activation(nn.Dense(self.pair_out)(h_two))
            h_two = residual(h_two, h_two_new)
        else:
            h_two_new = h_two
        return h_one, h_two
    
    
class Net(nn.Module):
    spins: Tuple[int, int]
    hidden_size1: List[int]
    hidden_size2: List[int]
    activation1: Union[str, Activation] = 'silu'
    activation2: Union[str, Activation] = 'tanh'

    def setup(self):
        self.mlp1 = MLP(self.hidden_size1, activation=activation_function(self.activation1))
        self.mlp2 = MLP(self.hidden_size2, activation=activation_function(self.activation2))

    def __call__(self, h_one):
        h_ones = h_one.split(self.spins[:1], axis=0)
                
        sumlogA = 0.
        for h, n_elec in zip(h_ones, self.spins):
            idx_h1 = jnp.array([i for i in range(n_elec - 1) for j in range(n_elec - 1 - i)], dtype=jnp.int32)
            h1 = h[idx_h1, :]
            h2 = jnp.concatenate([h[i:, :] for i in range(1, n_elec)])
            
            hidden = self.mlp1(jnp.concatenate([h1, h2], axis=-1)) - self.mlp1(jnp.concatenate([h2, h1], axis=-1))
            a = jnp.prod(hidden, axis=0)
            A = self.mlp2(a)
            sumlogA += jnp.log(jnp.abs(A)).squeeze()
        
        return sumlogA


class DeepWF(nn.Module):
    n_nuclei: int
    spins: Tuple[int, int]
    hidden_dims: Sequence[Tuple[int, int]] = (
        (256, 32), (256, 32), (256, 32), (256, 32))
    input_mlp_dims: Sequence[int] = (32, 32)
    nuclei_embedding: int = 32
    input_activation: Union[str, Activation] = nn.tanh
    fermi_activation: Union[str, Activation] = nn.silu
    jastrow_config: Optional[dict] = None

    def setup(self):
        self.axes = self.variable(
            'constants',
            'axes',
            jnp.eye,
            3
        )
        self.input_construction = InvariantEncoding(
            nuclei_embedding=self.nuclei_embedding,
            mlp_dims=self.input_mlp_dims,
            activation=activation_function(self.input_activation),
        )
        # Do not compute an update for the last pairwise layer
        hidden_dims = [list(h) for h in self.hidden_dims]
        hidden_dims[-1][1] = 0
            
        self.fermi_layers = [
            FermiLayer(
                spins=self.spins,
                single_out=d[0],
                pair_out=d[1],
                activation=activation_function(self.fermi_activation)
            )
            for d in hidden_dims
        ]
        
        self.anti_sym = Net(self.spins, 
                            hidden_size1=[256, 256], 
                            hidden_size2=[64, 1], 
                            activation1='silu', 
                            activation2='tanh')

        if self.jastrow_config is not None:
            self.jastrow = AutoMLP(1, 
                                   self.jastrow_config['n_layers'], 
                                   activation_function(self.jastrow_config['activation']))

    def encode(self, electrons, atoms):
        # Prepare input
        atoms = atoms.reshape(-1, 3)
        electrons = electrons.reshape(-1, 3)
        h_one, h_two, r_im = self.input_construction(electrons, atoms)

        for fermi_layer in self.fermi_layers:
            h_one, h_two = fermi_layer(h_one, h_two)

        return h_one, r_im

    def orbitals(self, electrons, atoms):
        # h_one, r_im = self.encode(electrons, atoms)
        # return self.to_orbitals(h_one, r_im)
        return None

    def signed(self, electrons, atoms):
        h_one, r_im = self.encode(electrons, atoms)
        
        sumlogA = self.anti_sym(h_one)
        
        return None, sumlogA
        
#         orbitals = self.to_orbitals(h_one, r_im)
#         sign, log_psi = self.logsumdet(orbitals)
        
#         if self.jastrow_config is not None:
#             log_psi += self.jastrow(h_one).sum()

#         return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]
