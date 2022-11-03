import jax
import jax.numpy as jnp                # JAX NumPy
from flax import linen as nn           # The Linen API
from jax.lax import stop_gradient
from typing import Any, Callable, Tuple, Optional, Sequence, Dict, Union
import pdb
import numpy as np


# https://arxiv.org/pdf/2108.09084.pdf
class FastSelfAttention(nn.Module):
    spins: Tuple[int, int]
    hidden_size: int
    num_attention_heads: int = 4
    ee_information: bool = False
    ee_readout_size: int = 128
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`\
                    : {self.config.num_attention_heads}"
            )

        self.query = [nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            # kernel_init=jax.nn.initializers.normal(self.config['initializer_range'], self.dtype),
        ) for _ in range(len(self.spins))]
        self.key = [nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            # kernel_init=jax.nn.initializers.normal(self.config['initializer_range'], self.dtype),
        ) for _ in range(len(self.spins))]
        self.value = [nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            # kernel_init=jax.nn.initializers.normal(self.config['initializer_range'], self.dtype),
        ) for _ in range(len(self.spins))]
        self.wq = nn.Dense(
            self.num_attention_heads,
            dtype=self.dtype,
            # kernel_init=jax.nn.initializers.normal(self.config['initializer_range'], self.dtype),
        )
        self.wk = nn.Dense(
            self.num_attention_heads,
            dtype=self.dtype,
            # kernel_init=jax.nn.initializers.normal(self.config['initializer_range'], self.dtype),
        )
        self.transform = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            # kernel_init=jax.nn.initializers.normal(self.config['initializer_range'], self.dtype),
        )
        if self.ee_information:
            self.ee_info_readout = nn.Dense(
                self.ee_readout_size,
                dtype=self.dtype,)

    def __call__(self, hidden_states, h_two=None, r_ij=None):
        # d
        head_dim = self.hidden_size // self.num_attention_heads
        
        hidden_states = hidden_states.split(self.spins[:1], axis=0)

        # N x (head x d)
        input_query_states = jnp.concatenate([self.query[i](hidden_states[i]) for i in range(len(self.spins))], axis=0)
        value_states = jnp.concatenate([self.value[i](hidden_states[i]) for i in range(len(self.spins))], axis=0)
        key_states = jnp.concatenate([self.key[i](hidden_states[i]) for i in range(len(self.spins))], axis=0)
        
        # N x head
        query_for_score = self.wq(input_query_states)
        # head x N
        query_weight = jax.nn.softmax(query_for_score.T)
        # head x 1 x N
        query_weight = jnp.expand_dims(query_weight, axis=1)
        # N x head x d
        query_states = input_query_states.reshape(input_query_states.shape[:-1] + (self.num_attention_heads, head_dim))
        # head x N x d
        query_states = jnp.transpose(query_states, (1, 0, 2))
        
        # head x 1 x d
        pooled_query = jnp.matmul(query_weight, query_states)
        # 1 x (head x d)
        pooled_query = pooled_query.reshape(1, self.num_attention_heads * head_dim)
        
        # N x (head x d)
        mixed_query_key_layer = key_states * pooled_query
        # mixed_query_key_layer = (key_states + pooled_query) / 2 ** 0.5
        
        # N x head
        query_key_score = self.wk(mixed_query_key_layer)
        # head x N
        query_key_weight = jax.nn.softmax(query_key_score.T)
        # head x 1 x N
        query_key_weight = jnp.expand_dims(query_key_weight, axis=1)
        # N x head x d
        key_states = key_states.reshape(key_states.shape[:-1] + (self.num_attention_heads, head_dim))
        # head x N x d
        key_states = jnp.transpose(key_states, (1, 0, 2))
        # head x 1 x d
        pooled_key = jnp.matmul(query_key_weight, key_states)
        # 1 x (head x d)
        pooled_key = pooled_key.reshape(1, self.num_attention_heads * head_dim)
        
        # N x (head x d)
        weighted_value = (pooled_key * value_states)
        # weighted_value = (pooled_key + value_states) / 2 ** 0.5
        
        weighted_value = (self.transform(weighted_value) + input_query_states) / 2 ** 0.5
        
        if self.ee_information:
            ee_readout = self.ee_info_readout(h_two)
            ee_readout = ee_readout.split(self.spins[:-1], axis=1)
            weighted_value = jnp.concatenate((weighted_value, 
                                              jnp.mean(ee_readout[0], axis=1, keepdims=False),
                                              jnp.mean(ee_readout[1], axis=1, keepdims=False)), axis=-1)
        
        return weighted_value
    
    
def dot_product_attention_weights(query: jnp.array,
                                  key: jnp.array,
                                  ee_attn: jnp.array = None,
                                  ee_attn_weight: Union[float, jnp.array, Tuple] = None):

    query = query / jnp.sqrt(query.shape[-1])
    
    # head, Nq, Nk
    attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
        
    if ee_attn is not None and ee_attn_weight is not None:
        if isinstance(ee_attn_weight, (float, jnp.float32)):
            attn_weights = attn_weights * (1 - ee_attn_weight) + ee_attn * ee_attn_weight
        else:
            attn_weights = attn_weights * ee_attn_weight[0] + ee_attn * ee_attn_weight[1]
    
    # attn_weights = (attn_weights + ee_attn) / 2 ** 0.5
    
    attn_weights = jax.nn.softmax(attn_weights)
    
    return attn_weights


class FlaxBertSelfAttention(nn.Module):
    att_hidden_size_v: int
    att_hidden_size_qk: int
    ee_readout_size: int
    num_attention_heads: int
    QK_init_scale: float
    ee_attention: bool
    spins: Tuple
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # self.ln = nn.LayerNorm(use_scale=False)
        
        self.query = [nn.Dense(
            self.att_hidden_size_qk,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.QK_init_scale, self.dtype),
        ) for _ in range(2)]
        self.key = [nn.Dense(
            self.att_hidden_size_qk,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.QK_init_scale, self.dtype),
        ) for _ in range(2)]
        self.value = [nn.Dense(
            self.att_hidden_size_v,
            dtype=self.dtype,
        ) for _ in range(2)]
        
        self.ee_info_readout = nn.Dense(self.ee_readout_size, dtype=self.dtype,)
        
        if self.ee_attention:
            self.ee_attn_readout = nn.Dense(
                self.num_attention_heads,
                dtype=self.dtype,)
            
            self.attn_ratio = self.param(
                'attn_ratio',
                lambda _, val: jnp.array([val, val], dtype=jnp.float32),
                1.0)
        else:
            self.ee_attn_readout = None
            self.attn_ratio = None
        

    def __call__(self, hidden_states, h_two=None, r_ij=None):
        # hidden_states = self.ln(hidden_states)

        h_one_spins = hidden_states.split(self.spins[0:1], axis=0)
        
        Q = [self.query[i](h_one).reshape(
            h_one.shape[:1] + (self.num_attention_heads, -1)
            ) for i, h_one in enumerate(h_one_spins)]
        K = [self.key[i](h_one).reshape(
            h_one.shape[:1] + (self.num_attention_heads, -1)
            ) for i, h_one in enumerate(h_one_spins)]
        V = [self.value[i](h_one).reshape(
            h_one.shape[:1] + (self.num_attention_heads, -1)
            ) for i, h_one in enumerate(h_one_spins)]
        
        Q = jnp.concatenate(Q, axis=0)
        K = jnp.concatenate(K, axis=0)
        V = jnp.concatenate(V, axis=0)
        
        ee_attn = self.ee_attn_readout(r_ij).transpose(2, 0, 1) if self.ee_attention else None

        attn_weights = dot_product_attention_weights(
            Q,
            K,
            ee_attn,
            self.attn_ratio,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, V)
        attn_output = attn_output.reshape(attn_output.shape[:1] + (-1,))
        
        attn_outputs = attn_output.split(self.spins[0:1], axis=0)
        global_in = jnp.concatenate([jnp.mean(attn_outputs[0], axis=0, keepdims=True), 
                                     jnp.mean(attn_outputs[1], axis=0, keepdims=True)], axis=-1)
        
        ee_readout = self.ee_info_readout(h_two)
        ee_readout = ee_readout.split(self.spins[:-1], axis=0)
        attn_output = jnp.concatenate((attn_output, 
                                  jnp.mean(ee_readout[0], axis=0, keepdims=False),
                                  jnp.mean(ee_readout[1], axis=0, keepdims=False)), axis=-1)
        
        return attn_output, global_in
