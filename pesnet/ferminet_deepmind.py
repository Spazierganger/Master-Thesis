import functools
from typing import Sequence, Tuple, Union, Optional

import pdb
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp

from pesnet.nn import Activation, activation_function, residual, AutoMLP, MLP
from pesnet.ferminet import Orbitals, IsotropicEnvelope, FullEnvelope, InvariantEncoding


batch_cross = jax.vmap(
                jax.vmap(
                    jax.vmap(
                        lambda a, b: jnp.cross(a, b).squeeze(), in_axes=(None, 1)
                    ), in_axes=(0, None)
                ), in_axes=(0, 0)
            )


def construct_symmetric_features(h_one: jnp.ndarray, h_two: jnp.ndarray,
                                 spins: Tuple[int, int]) -> jnp.ndarray:
    # Split features into spin up and spin down electrons
        
    h_ones = h_one.split(spins[:1], axis=0)
    h_twos = h_two.split(spins[:1], axis=0)

    # Construct inputs to next layer
    # h.size == 0 corresponds to unoccupied spin channels.
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]

    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]

    return jnp.concatenate([h_one] + g_one + g_two, axis=1)


def construct_input_features(electrons, atoms):
    n_elec = electrons.shape[0]
    n_atoms = atoms.shape[0]

    r_im3 = electrons[:, None] - atoms[None]
    r_im_norm = jnp.linalg.norm(r_im3, keepdims=True, axis=-1)
    r_im4 = jnp.concatenate([r_im3, r_im_norm], axis=-1)
    h_one = r_im4.reshape(n_elec, -1)

    r_ij = electrons[:, None] - electrons[None]
    r_ij_norm = jnp.linalg.norm(
        r_ij + jnp.eye(n_elec)[..., None],
        keepdims=True,
        axis=-1
    ) * (1.0 - jnp.eye(n_elec)[..., None])
    h_two = jnp.concatenate([r_ij, r_ij_norm], axis=-1)

    return h_one, h_two, r_im4


class FermiLayer(nn.Module):
    spins: Tuple[int, int]
    single_out: int
    pair_out: int
    activation: Activation = jnp.tanh

    @nn.compact
    def __call__(self, h_one, h_two):
        
        h_one_new = construct_symmetric_features(h_one, h_two, self.spins)
        h_one_new = self.activation(nn.Dense(self.single_out, bias_init=jnn.initializers.normal(stddev=1.0))(h_one_new))
        h_one = residual(h_one, h_one_new)

        # Pairwise update
        if self.pair_out > 0:
            h_two_new = self.activation(nn.Dense(self.pair_out, bias_init=jnn.initializers.normal(stddev=1.0))(h_two))
            h_two = residual(h_two, h_two_new)
        else:
            h_two_new = h_two
        return h_one, h_two


class LogSumDet(nn.Module):
    @nn.compact
    def __call__(self, xs):
        det1 = functools.reduce(
            lambda a, b: a*b,
            [x.reshape(-1) for x in xs if x.shape[-1] == 1],
            1
        )

        sign_in, logdet = functools.reduce(
            lambda a, b: (a[0]*b[0], a[1]+b[1]),
            [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
            (1, 0)
        )

        maxlogdet = jax.lax.stop_gradient(jnp.max(logdet))
        det = sign_in * det1 * jnp.exp(logdet - maxlogdet)

        result = jnp.sum(det)

        sign_out = jnp.sign(result)
        log_out = jnp.log(jnp.abs(result)) + maxlogdet
        return sign_out, log_out


class FermiNet(nn.Module):
    spins: Tuple[int, int]
    full_det: bool
    envelope_type: str
    hidden_dims: Sequence[Tuple[int, int]] = (
        (256, 32), (256, 32), (256, 32), (256, 32))
    determinants: int = 16
    jastrow_config: Optional[dict] = None

    def setup(self):
        hidden_dims = [list(h) for h in self.hidden_dims]
        hidden_dims[-1][1] = 0
        
        # self.input_construction = InvariantEncoding(
        #     nuclei_embedding=32,
        #     mlp_dims=[32, 32],
        #     activation=activation_function(nn.tanh),
        #     encoder_geometrics=False,
        # )
        
        self.fermi_layers = [
            FermiLayer(
                spins=self.spins,
                single_out=d[0],
                pair_out=d[1]
            )
            for d in hidden_dims
        ]

        self.to_orbitals = Orbitals(self.spins, self.determinants, full_det=self.full_det, envelope_type=self.envelope_type)
        self.logsumdet = LogSumDet()
        
        if self.jastrow_config is not None:
            self.jastrow = AutoMLP(1, 
                                   self.jastrow_config['n_layers'], 
                                   activation_function(self.jastrow_config['activation']))
            self.jastrow_weight = self.param(
                'jastrow_weight',
                lambda _, val: val,
                0.,
            )

    def encode(self, electrons, atoms):
        # Prepare input
        atoms = atoms.reshape(-1, 3)
        electrons = electrons.reshape(-1, 3)
        h_one, h_two, r_im = construct_input_features(electrons, atoms)

        for fermi_layer in self.fermi_layers:
            h_one, h_two = fermi_layer(h_one, h_two)

        return h_one, r_im

    def orbitals(self, electrons, atoms):
        h_one, r_im = self.encode(electrons, atoms)
        return self.to_orbitals(h_one, r_im)

    def signed(self, electrons, atoms):
        h_one, r_im = self.encode(electrons, atoms)
        orbitals = self.to_orbitals(h_one, r_im)
        sign, log_psi = self.logsumdet(orbitals)
        
        if self.jastrow_config is not None:
            log_psi += self.jastrow(h_one).sum() * self.jastrow_weight
        
        return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]
