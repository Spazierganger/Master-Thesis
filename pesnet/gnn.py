from dataclasses import field
from typing import Sequence, Tuple, Union, List

import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp
import jraph
import numpy as np

from pesnet.nn import Activation, AutoMLP, activation_function, named, residual
from pesnet.spherical_basis import make_positional_encoding


class BesselRBF(nn.Module):
    out_dim: int
    cutoff: float

    @nn.compact
    def __call__(self, x):
        f = self.param(
            'f',
            lambda *_: (jnp.arange(self.out_dim,
                        dtype=jnp.float32) + 1) * jnp.pi,
            (self.out_dim,)
        )
        x_ext = x[..., None] + 1e-8
        result = jnp.sqrt(2./self.cutoff) * jnp.sin(f*x_ext/self.cutoff)/x_ext
        return result.reshape(*x.shape, -1)


class MessagePassing(nn.Module):
    out_dim: int
    n_layers: int
    activation: Activation

    @nn.compact
    def __call__(self, n_embed, e_embed, senders, receivers):
        inp_features = jnp.concatenate([
            n_embed[senders],
            n_embed[receivers],
            e_embed
        ], axis=-1)
        return jraph.segment_mean(
            AutoMLP(self.out_dim, self.n_layers,
                    activation=self.activation)(inp_features),
            senders,
            num_segments=n_embed.shape[0],
            indices_are_sorted=True
        )


class Update(nn.Module):
    out_dim: int
    n_layers: int
    activation: Activation

    @nn.compact
    def __call__(self, n_embed, msg):
        inp_features = jnp.concatenate([n_embed, msg], axis=-1)
        return residual(n_embed, AutoMLP(self.out_dim, self.n_layers, activation=self.activation)(inp_features))


class GNNPlaceholder(nn.Module):
    @nn.compact
    def __call__(self, nuclei):
        return [], []


class NodeOut(nn.Module):
    max_charge: int
    out_dim: int
    depth: int
    activation: Activation

    @nn.compact
    def __call__(self, x, charges):
        out_bias = nn.Embed(self.max_charge, self.out_dim)(charges)
        return AutoMLP(self.out_dim, self.depth, self.activation, final_bias=False)(x) + out_bias


class GNN(nn.Module):
    charges: Tuple[int, ...]
    node_out_dims: Tuple[int, ...]
    global_out_dims: Tuple[int, ...]
    layers: Sequence[Tuple[int, int]] = ((32, 64), (32, 64))
    embedding_dim: int = 32
    msg_mlp_depth: int = 2
    update_mlp_depth: int = 2
    out_mlp_depth: int = 2
    rbf: nn.Module = BesselRBF
    rbf_dim: int = 32
    rbf_cutoff: float = 10.0
    aggregate_before_out: bool = True
    use_pos_encoding: bool = True
    pos_encoding_config: dict = field(default_factory=lambda *_: {
        'cutoff': 5.,
        'n_sph': 7,
        'n_rad': 6
    })
    activation: Union[str, Activation] = nn.silu

    @nn.compact
    def __call__(self, nuclei):
        activation = activation_function(self.activation)
        axes = self.variable(
            'constants',
            'axes',
            jnp.eye,
            3
        )
        nuclei = nuclei.reshape(-1, 3) @ axes.value
        n_nuclei = nuclei.shape[0]

        # Construct fully connected graph
        senders = np.arange(n_nuclei).repeat(n_nuclei-1)
        receivers = np.arange(n_nuclei)[None].repeat(n_nuclei-1, axis=0)
        receivers = (receivers + np.arange(n_nuclei-1)[:, None]+1) % n_nuclei
        receivers = receivers.reshape(-1)

        senders, receivers = jnp.array(senders), jnp.array(receivers)

        # Edge embedding
        e_embed = self.rbf(self.rbf_dim, self.rbf_cutoff)(
            jnp.linalg.norm(nuclei[senders] - nuclei[receivers], axis=-1))

        # Node embedding
        n_embed = nn.Embed(
            max(self.charges),
            self.embedding_dim,
            embedding_init=jnn.initializers.normal(stddev=1)
        )(self.charges)
        if self.use_pos_encoding:
            pos_embed = make_positional_encoding(
                **self.pos_encoding_config)(nuclei - nuclei.mean(-2, keepdims=True))
            n_embed = jnp.concatenate([n_embed, pos_embed], axis=-1)

        # Message passing and update
        embeddings = [n_embed]
        for layer in self.layers:
            msg = MessagePassing(layer[0], self.msg_mlp_depth, activation=activation)(
                n_embed, e_embed, senders, receivers)
            n_embed = Update(layer[1], self.update_mlp_depth,
                             activation=activation)(n_embed, msg)
            embeddings.append(n_embed)

        # Output
        if self.aggregate_before_out:
            n_embed = jnp.concatenate(embeddings, axis=-1)
        # Node
        node_output = [
            NodeOut(max(self.charges), out, self.out_mlp_depth,
                    activation)(n_embed, self.charges)
            for out in self.node_out_dims
        ]
        # Global
        if len(self.global_out_dims) > 0:
            global_inp = jnp.mean(n_embed, axis=0)
        global_output = [
            named('GlobalOut', AutoMLP, out, self.out_mlp_depth,
                  activation=activation)(global_inp)
            for out in self.global_out_dims
        ]
        return node_output, global_output

    
class GNNFoo(nn.Module):
    n_params: List
    g_params: List
    
    def setup(self):
        nparams = [jnp.concatenate(p, axis=0) for p in zip(*self.n_params)]
        self.nparams = [
            self.param(
                f'np{i}',
                lambda *_: p,
                None
            ) for i, p in enumerate(nparams)
        ]
        
        gparams = [jnp.concatenate(p, axis=0) for p in zip(*self.g_params)]
        self.gparams = [
            self.param(
                f'gp{i}',
                lambda *_: p,
                None
            ) for i, p in enumerate(gparams)
        ]
    
    def __call__(self, idx):
        return [p[idx] for p in self.nparams], [p[idx] for p in self.gparams]
