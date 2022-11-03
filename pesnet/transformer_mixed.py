from typing import Any, Callable, Tuple, Optional, Sequence, Dict
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from pesnet.nn import Activation, ActivationWithGain, activation_function, residual
from flax.linen.module import Module
from pesnet.ferminet import LogSumDet, FermiLayer
from pesnet.transformer_comp import TransformerEncoder, InvariantEncoding, Orbitals


class MixtureModel(nn.Module):
    encodertype: str
    config: Dict
    spins: Tuple[int, int]
    full_det: bool
    envelope_type: str
    determinants = 16
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    
    def setup(self):
        self.axes = self.variable(
            'constants',
            'axes',
            jnp.eye,
            3
        )
        self.Embedding = InvariantEncoding(nuclei_embedding=self.config['nuclei_embedding'],
                                           mlp_dims=self.config['input_mlp_dims'],
                                           h_one_activation=activation_function(self.config['h_one_act']), 
                                           h_two_activation=activation_function(self.config['h_two_act']), 
                                           ee_attention=self.config['ee_attention'])
        self.to_orbitals = Orbitals(self.spins, self.determinants, full_det=self.full_det, envelope_type=self.envelope_type)
        self.logsumdet = LogSumDet()
        
        hidden_dims = [list(self.config['fermi_hidden_dims']) for _ in range(self.config['num_hidden_layers'])]
        hidden_dims[-1][1] = 0

        self.encoder = [TransformerEncoder(
                                        self.encodertype,
                                        self.config,
                                        self.spins,
                                        last_act=(i != self.config['num_hidden_layers'] - 1),
                                        ee_attention=False,
                                        update_htwo=False,
                                        dtype=self.dtype) \
                        if i in self.config['transformer_idx'] else \
                        FermiLayer(
                                    spins=self.spins,
                                    single_out=hidden_dims[i][0],
                                    pair_out=hidden_dims[i][1],
                                    activation=activation_function(self.config['hidden_act'])
                                ) \
                        for i in range(self.config['num_hidden_layers'])]

       
    def orbitals(self, electrons, atoms):
        atoms = atoms.reshape(-1, 3) @ self.axes.value
        electrons = electrons.reshape(-1, 3) @ self.axes.value
        h_one, h_two, r_im = self.Embedding(electrons, atoms)

        for i, layer in enumerate(self.encoder):
            if isinstance(layer, TransformerEncoder):
                h_one, h_two, _ = layer(h_one, h_two, res = i >= 1)
            else:
                h_one, h_two = layer(h_one, h_two)

        return self.to_orbitals(h_one, r_im)

    def __call__(self, electrons, atoms):
        orbitals = self.orbitals(electrons, atoms)
        return self.logsumdet(orbitals)[1]
