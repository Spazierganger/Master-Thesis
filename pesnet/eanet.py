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


class NonParamBesselRBF(nn.Module):
    dim: int
    out_dim: int
    cutoff: float
    
    @nn.compact
    def __call__(self, keys, atoms, n_atoms):
        f = (jnp.arange(self.dim, dtype=jnp.float32) + 1) * jnp.pi
        x = jnp.linalg.norm(atoms[None] - atoms[:, None, :], keepdims=False, axis=-1)
        x_ext = x[..., None] + 1e-8
        result = jnp.sqrt(2. / self.cutoff) * jnp.sin(f * x_ext / self.cutoff) / x_ext
        result = result.reshape(*x.shape, -1)
        initializer = jax.nn.initializers.glorot_normal()
        w1 = initializer(keys, (self.dim, 4), jnp.float32)
        w2 = initializer(keys, (4, self.out_dim), jnp.float32)
        return result @ w1 @ w2


class InvariantEncoding(nn.Module):
    charges: Tuple[int]
    spins: Tuple[int, int]
    hone_dense_dim: int
    htwo_dense_dim: int
    activation: Activation

    @nn.compact
    def __call__(self, electrons, atoms):
        activation = ActivationWithGain(self.activation)
        
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
        h_two = jnp.concatenate([r_ij, r_ij_norm], axis=-1)
        
        h_two = activation(nn.Dense(self.htwo_dense_dim)(h_two))
        
        h_two = jnp.concatenate([h.mean(1, keepdims=True) for h in h_two.split(self.spins[:1], axis=1)], axis=-1)
        h_two = jnp.tile(h_two, [1, n_atoms, 1])
        
        nuc_embedding1 = self.param(
            'nuc_embedding1',
            jnn.initializers.normal(stddev=0.01),
            (1, n_atoms, 4)
        )
        h_one = activation(nn.Dense(self.hone_dense_dim)(r_im4 + nuc_embedding1))
        
        h_one_two = jnp.concatenate([h_one, h_two], axis=-1)
        
        return h_one_two, r_im4
    
    
class MessagePassingPreproc(nn.Module):
    message_passing: Dict
    out_dim: int
    
    @nn.compact
    def __call__(self, atoms):
        n_atoms = atoms.shape[0]
        
        mp_weight = None
        rbf = None
        if n_atoms > 1:
            if self.message_passing['method'] == 'softmax':
                atom_distance = -jnp.linalg.norm(atoms[None] - atoms[:, None, :], keepdims=False, axis=-1)
                temperature = self.param(
                    'temperature',
                    jnn.initializers.ones,
                    (n_atoms, 1,)
                )
                atom_distance *= temperature
                atom_distance -= jnp.eye(n_atoms) * 10000
                mp_weight = jax.nn.softmax(atom_distance, axis=1)
            elif self.message_passing['method'] == 'rbf':
                rbf = self.param(
                    'MP_rbf',
                    NonParamBesselRBF(self.message_passing['rbf_dim'], self.out_dim, self.message_passing['rbf_cutoff']),
                    atoms, n_atoms
                )
            elif self.message_passing['method'] == 'mean':
                mp_weight = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32) / n_atoms
            elif self.message_passing['method'] == 'none':
                pass
            else:
                raise ValueError
        
        return mp_weight, rbf
    
    
class EANetLayer(nn.Module):
    spins: Tuple[int, int]
    single_out: int
    activation: Activation

    @nn.compact
    def __call__(self, h_one, aggr_weight, rbf):
        activation = ActivationWithGain(self.activation)
        n_atoms = h_one.shape[0]
        
        if aggr_weight is None and rbf is None:
            h_one_new = activation(nn.Dense(self.single_out)(h_one))
        elif aggr_weight is None and rbf is not None:
            message = (h_one * rbf).mean(0, keepdims=True)
            h_one_new = activation((nn.Dense(self.single_out)(h_one) + \
                                    nn.Dense(self.single_out)(message)) / 2 ** 0.5)
        elif aggr_weight is not None and rbf is None:
            message = jnp.einsum('jnl,mn->jml', h_one, aggr_weight)
            h_one_new = activation((nn.Dense(self.single_out)(h_one) + \
                                    nn.Dense(self.single_out)(message)) / 2 ** 0.5)
        else:
            raise ValueError
            
        return residual(h_one, h_one_new)
    
    
class GaussRBF(nn.Module):
    num_rbfs: int
    
    @nn.compact
    def __call__(self, r_im):
        # 1 x 1 x 16
        mu = self.param(
            'gauss_rbf_mu',
            lambda _, num: jnp.linspace(0, num - 1, num, dtype=jnp.float32)[None, None, ...],
            self.num_rbfs
        )
        sigma = self.param(
            'gauss_rbf_sigma',
            lambda _, shape: jnp.ones(shape, dtype=jnp.float32) * 10,
            (1, 1, self.num_rbfs,)
        )
        
        rbf = jnp.exp(- (r_im - mu) ** 2 / jnn.softplus(sigma))
        
        return rbf.mean(-1, keepdims=True)
    


class Electron2MetaNodes(nn.Module):
    spins: Optional[Tuple[int, int]]
    rbf_configs: Optional[Dict]
    num_gauss_rbf: int
    dim: int
    activation: Activation
    
    @nn.compact
    def __call__(self, h_one_two, r_im_norm):
        activation = ActivationWithGain(self.activation)
        n_elec, n_atoms, _ = h_one_two.shape
        
        if self.rbf_configs is not None:
            rbf = BesselRBF(self.rbf_configs['rbf_dim'], self.rbf_configs['rbf_cutoff'])(r_im_norm)
            rbf = nn.Dense(4, use_bias=False)(rbf)
            rbf = nn.Dense(h_one_two.shape[-1], use_bias=False)(rbf)
            h_one_two = h_one_two * rbf
            
        if self.num_gauss_rbf > 0:
            rbf = GaussRBF(self.num_gauss_rbf)(r_im_norm[..., None])
            h_one_two = h_one_two * rbf
            
        if self.spins is not None:
            hs = h_one_two.split(self.spins[:1], axis=0)
            atom_emb = jnp.concatenate([h.mean(0, keepdims=True) for h in hs], axis=-1)
        else:
            atom_emb = h_one_two.mean(0, keepdims=True)
            
        atom_emb = activation(nn.Dense(self.dim)(atom_emb))
        elec_emb = h_one_two.mean(1, keepdims=True)
        
        return atom_emb, elec_emb
        

class EANet(nn.Module):
    charges: Tuple[int]
    spins: Tuple[int, int]
    full_det: bool
    envelope_type: str
    hidden_dims: Sequence[Tuple[int]] = (256, 256, 256, 256)
    determinants: int = 16
    hone_dense_dim: int = 256
    htwo_dense_dim: int = 256
    input_activation: Union[str, Activation] = nn.tanh
    fermi_activation: Union[str, Activation] = nn.silu
    jastrow_config: Optional[Dict] = None
    compress_rbf: Optional[Dict] = None
    compress_gauss_rbf: int = 0
    message_passing: Dict = None
    split_spin: bool = True

    def setup(self):
        # self.axes = self.variable(
        #     'constants',
        #     'axes',
        #     jnp.eye,
        #     3
        # )
        
        self.input_construction = InvariantEncoding(
            charges=self.charges,
            spins=self.spins,
            hone_dense_dim=self.hone_dense_dim,
            htwo_dense_dim=self.htwo_dense_dim,
            activation=activation_function(self.input_activation),
        )
        
        self.mp_preproc = MessagePassingPreproc(self.message_passing, 
                                                out_dim=self.hidden_dims[0])

        self.layers = [
            EANetLayer(
                spins=self.spins,
                single_out=d,
                activation=activation_function(self.fermi_activation),
            )
            for d in self.hidden_dims
        ]

        self.to_orbitals = Orbitals(self.spins, self.determinants, 
                                    full_det=self.full_det,
                                    envelope_type=self.envelope_type)
        self.logsumdet = LogSumDet()
        if self.jastrow_config is not None:
            self.jastrow = AutoMLP(1,
                                   self.jastrow_config['n_layers'],
                                   activation_function(self.jastrow_config['activation']),
                                   final_bias=False)
            self.jastrow_weight = self.param(
                'jastrow_weight',
                lambda _, val: val,
                0.,
            )
            
        self.linear = nn.Dense(self.hidden_dims[-1])
        self.act = ActivationWithGain(activation_function(self.fermi_activation))
        self.node_compress = Electron2MetaNodes(self.spins if self.split_spin else None, 
                                                self.compress_rbf, 
                                                self.compress_gauss_rbf, 
                                                self.hidden_dims[0],
                                                activation_function(self.fermi_activation))
    
    def encode(self, electrons, atoms):
        # atoms = atoms.reshape(-1, 3) @ self.axes.value
        # electrons = electrons.reshape(-1, 3) @ self.axes.value
        
        atoms = atoms.reshape(-1, 3)
        electrons = electrons.reshape(-1, 3)
        
        h_one_two, r_im4 = self.input_construction(electrons, atoms)
        mp_weight, rbf = self.mp_preproc(atoms)
        
        atom_emb, elec_emb = self.node_compress(h_one_two, r_im4[..., -1])

        for i, layer in enumerate(self.layers):
            atom_emb = layer(atom_emb, mp_weight, rbf)
        
        elec_emb = self.act(self.linear(elec_emb))

        h_one = elec_emb * atom_emb
        
        return h_one, h_one.mean(1), r_im4

    def orbitals(self, electrons, atoms):
        _, h_one, r_im = self.encode(electrons, atoms)
        return self.to_orbitals(h_one, r_im)

    def signed(self, electrons, atoms):
        h_one_full, h_one, r_im = self.encode(electrons, atoms)
        orbitals = self.to_orbitals(h_one, r_im)
        sign, log_psi = self.logsumdet(orbitals)
        if self.jastrow_config is not None:
            log_psi += self.jastrow(h_one_full).sum() * self.jastrow_weight
        return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]