import jax
import jax.nn as jnn
import jax.numpy as jnp                # JAX NumPy
from flax import linen as nn           # The Linen API
import numpy as np                     # Ordinary NumPy
from typing import Any, Callable, Tuple, Optional, Sequence, Dict, Union
from flax.linen.module import Module
from pesnet.ferminet import LogSumDet, Orbitals, apply_covariance, IsotropicEnvelope, FullEnvelope
from pesnet.nn import MLP, Activation, ActivationWithGain, activation_function, residual
from pesnet.transformer_attns import FastSelfAttention, FlaxBertSelfAttention


class TransformerEncoder(nn.Module):
    encodertype: str
    config: Dict
    spins: Tuple
    last_act: bool
    ee_attention: bool
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.activation = ActivationWithGain(activation_function(self.config['hidden_act']))
        
        if self.encodertype == 'vanilla':
            self.attn_layer = FlaxBertSelfAttention(self.config['att_hidden_size_v'], 
                                              self.config['att_hidden_size_qk'], 
                                              self.config['ee_readout_size'],
                                              self.config['num_attention_heads'],
                                              self.config['QK_init_scale'],
                                              self.ee_attention,
                                              self.spins, 
                                              dtype=self.dtype)
        elif self.encodertype == 'fastformer':
            self.attn_layer = FastSelfAttention(self.spins,
                                                self.config['attn_hidden_size'], 
                                                self.config['num_attention_heads'], 
                                                ee_information=self.ee_attention,
                                                ee_readout_size=self.config['ee_readout_size'],
                                                dtype=self.dtype)
        else:
            raise ValueError
        
        self.layers = [
                        nn.Dense(self.config['intermediate_size'], dtype=self.dtype,),
                        nn.Dense(self.config['intermediate_size'], use_bias=False, dtype=self.dtype,),
                        self.activation,
                        nn.Dense(self.config['intermediate_size'], dtype=self.dtype,),
                        self.activation,
                       ]
        
        self.ee_layers = [
                            nn.Dense(self.config['ee_intermediate_size'], dtype=self.dtype,),
                            self.activation,
                         ]
        
    def __call__(self, h_one, h_two, r_ij=None):
        one_in, global_in = self.attn_layer(h_one, h_two, r_ij)
        
        # common linear layers
        lin1_1, lin1_2, act1, lin2, act2 = self.layers
        h_one_new = act1((lin1_1(one_in) + lin1_2(global_in)) / 2 ** 0.5)
        h_one_new = lin2(h_one_new)
        h_one_new = act2(h_one_new) if act2 is not None else h_one_new

        # residual
        h_one = residual(h_one, h_one_new)
        
        # update h-two embeddings
        lin, act = self.ee_layers
        h_two_new = act(lin(h_two))
        h_two = residual(h_two, h_two_new)
            
        return h_one, h_two
    
    
class InvariantEncoding(nn.Module):
    nuclei_embedding: int
    mlp_dims: Sequence[int]
    h_one_activation: Activation
    h_two_activation: Activation

    @nn.compact
    def __call__(self, electrons, atoms):
        n_elec = electrons.shape[0]
        n_atoms = atoms.shape[0]

        r_im = electrons[:, None] - atoms[None]
        r_im_norm = jnp.linalg.norm(r_im, keepdims=True, axis=-1)
        r_im = jnp.concatenate([r_im, r_im_norm], axis=-1)
        h_one = r_im

        r_ij = electrons[:, None] - electrons[None]
        r_ij_norm = jnp.linalg.norm(
            r_ij + jnp.eye(n_elec)[..., None],
            keepdims=True,
            axis=-1
        ) * (1.0 - jnp.eye(n_elec)[..., None])
        h_two = jnp.concatenate([r_ij, r_ij_norm], axis=-1)
        
        r_ij = 1 / (nn.tanh(r_ij) * r_ij + 1)

        nuc_embedding = self.param(
            'nuc_embedding',
            jnn.initializers.normal(),
            (n_atoms, self.nuclei_embedding)
        )
        h_one = nn.Dense(self.nuclei_embedding)(h_one)
        h_one = (h_one + nuc_embedding) / 2 ** 0.5

        h_one = MLP(
            self.mlp_dims,
            activation=self.h_one_activation
        )(h_one)
        
        h_one = h_one.mean(1)
        
        h_two = MLP(
            self.mlp_dims,
            activation=self.h_two_activation
        )(h_two)

        return h_one, h_two, r_im, r_ij

    
class Transformer(nn.Module):
    encodertype: str
    config: Dict
    spins: Tuple[int, int]
    full_det: bool
    envelope_type: str
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.encoder = [TransformerEncoder(self.encodertype,
                                        self.config, 
                                        self.spins, 
                                        last_act = (i != self.config['num_hidden_layers'] - 1),
                                        ee_attention = self.config['ee_attention'],
                                        dtype=self.dtype) \
                        for i in range(self.config['num_hidden_layers'])]
        self.Embedding = InvariantEncoding(nuclei_embedding=self.config['nuclei_embedding'],
                                           mlp_dims=self.config['input_mlp_dims'],
                                           h_one_activation=activation_function(self.config['h_one_act']), 
                                           h_two_activation=activation_function(self.config['h_two_act']))
        self.to_orbitals = Orbitals(self.spins, 
                                    self.config['determinants'], 
                                    full_det=self.full_det, 
                                    envelope_type=self.envelope_type)
        self.logsumdet = LogSumDet()
        
    
    def orbitals(self, electrons, atoms):
        atoms = atoms.reshape(-1, 3)
        electrons = electrons.reshape(-1, 3)
        h_one, h_two, r_im, r_ij = self.Embedding(electrons, atoms)  # (N, embd), (N, M, 4)

        for i, encoder in enumerate(self.encoder):
            h_one, h_two = encoder(h_one, h_two, r_ij)
        
        return self.to_orbitals(h_one, r_im)


    def __call__(self, electrons, atoms,):
        """
        electrons: jnp.ndarray (N, 3)
        atoms: jnp.ndarray (M, 3)
        """
        orbitals = self.orbitals(electrons, atoms,)
        return self.logsumdet(orbitals)[1]
