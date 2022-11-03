import functools
from typing import Sequence, Tuple, Union, Optional

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp

from pesnet.nn import (MLP, Activation, ActivationWithGain,
                       activation_function, residual, AutoMLP)

batch_cross = jax.vmap(
                jax.vmap(
                    jax.vmap(
                        lambda a, b: jnp.cross(a, b).squeeze(), in_axes=(None, 1)
                    ), in_axes=(0, None)
                ), in_axes=(0, 0)
            )

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


class Orbitals(nn.Module):
    spins: Tuple[int, int]
    determinants: int
    bias_orbitals: bool = False
    full_det: bool = True
    envelope_type: str = 'full'

    @nn.compact
    def __call__(self, h_one, r_im):
        envelope = IsotropicEnvelope if self.envelope_type == 'isotropic' else FullEnvelope
        h_by_spin = h_one.split(self.spins[0:1], axis=0)
        r_im_by_spin = r_im.split(self.spins[0:1], axis=0)
        
        orbitals = []
        for h, r, s in zip(h_by_spin, r_im_by_spin, self.spins):
            out_size = self.determinants * sum(self.spins) if self.full_det else self.determinants * s
            orb = nn.Dense(out_size, use_bias=self.bias_orbitals)(h) * envelope(out_size)(r)
            orb = jnp.reshape(orb, [s, -1, sum(self.spins) if self.full_det else s])
            orb = jnp.transpose(orb, (1, 0, 2))
            orbitals.append(orb)
            
        if self.full_det:
            orbitals = [jnp.concatenate(orbitals, axis=1)]
        return orbitals

    
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
    encoder_geometrics: bool

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
            jnn.initializers.normal(stddev=1.0),
            (n_atoms, self.nuclei_embedding)
        )
        h_one = nn.Dense(self.nuclei_embedding)(h_one)
        h_one = (h_one + nuc_embedding) / 2 ** 0.5
        
        h_one = MLP(
            self.mlp_dims,
            activation=self.activation
        )(h_one).mean(1)
        
        if self.encoder_geometrics:
            # (ne, na, 1, 3), (ne, 1, na, 3) -> (ne, na, na, 3)
            geo_emb = batch_cross(r_im3[..., None, :], r_im3[:, None, ...])
            geo_emb_norm = jnp.linalg.norm(
                geo_emb + jnp.eye(n_atoms)[None, ..., None],
                keepdims=True,
                axis=-1
            ) * (1.0 - jnp.eye(n_atoms)[None, ..., None])
            geo_emb = jnp.concatenate([geo_emb, geo_emb_norm], axis=-1)
            geo_emb = nn.Dense(self.nuclei_embedding * 2)(geo_emb)
            
            # (W @ ij_features) + (nuc_meb_i || nuc_emb_j)
            geo_emb += jnp.concatenate((
                jnp.broadcast_to(nuc_embedding[None, None, ...], (1, n_atoms, n_atoms, self.nuclei_embedding)),
                jnp.broadcast_to(nuc_embedding[None, :, None, ...], (1, n_atoms, n_atoms, self.nuclei_embedding)),
            ), axis=-1)
            
            geo_emb = geo_emb / 2 ** 0.5
            
            geo_emb = MLP(
                            self.mlp_dims,
                            activation=self.activation
                        )(geo_emb).mean((1, 2,))
            h_one = jnp.concatenate((h_one, geo_emb), axis=-1)

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

        w = self.param(
            'w',
            jnn.initializers.ones,
            (det.size,)
        )
        result = jnp.vdot(w, det)

        sign_out = jnp.sign(result)
        log_out = jnp.log(jnp.abs(result)) + maxlogdet
        return sign_out, log_out


class FermiNet(nn.Module):
    n_nuclei: int
    spins: Tuple[int, int]
    full_det: bool
    envelope_type: str
    hidden_dims: Sequence[Tuple[int, int]] = (
        (256, 32), (256, 32), (256, 32), (256, 32))
    determinants: int = 16
    input_mlp_dims: Sequence[int] = (32, 32)
    nuclei_embedding: int = 32
    input_activation: Union[str, Activation] = nn.tanh
    fermi_activation: Union[str, Activation] = nn.silu
    encoder_geometrics: bool = False
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
            encoder_geometrics=self.encoder_geometrics,
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

        self.to_orbitals = Orbitals(self.spins, self.determinants, full_det=self.full_det, envelope_type=self.envelope_type)
        self.logsumdet = LogSumDet()
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
        h_one, r_im = self.encode(electrons, atoms)
        return self.to_orbitals(h_one, r_im)

    def signed(self, electrons, atoms):
        h_one, r_im = self.encode(electrons, atoms)
        orbitals = self.to_orbitals(h_one, r_im)
        sign, log_psi = self.logsumdet(orbitals)
        
        if self.jastrow_config is not None:
            log_psi += self.jastrow(h_one).sum()

        return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]
