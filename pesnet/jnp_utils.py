import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def _unique_axis(ar, axis, return_index=False, return_inverse=False,
                 return_counts=False, size: int = None):
    """
    Find the unique elements of an array along a particular axis.
    """
    aux, mask, perm, out_shape = jnp.lax_numpy._unique_axis_sorted_mask(
        ar,
        axis
    )
    if size is None:
        ind = mask
        new_shape = (mask.sum() or aux.shape[1], *out_shape)
    else:
        idx = list(range(len(ar.shape)))
        idx[0], idx[axis] = axis, 0
        ind = jnp.nonzero(mask, size=size)[0]
        new_shape = (size, *[ar.shape[i] for i in idx[1:]])
    result = jnp.moveaxis(aux[:, ind].T.reshape(*new_shape), 0, axis)

    ret = (result,)
    if return_index:
        if aux.size:
            ret += (perm[ind],)
        else:
            ret += (perm,)
    if return_inverse:
        if aux.size:
            imask = jnp.cumsum(mask) - 1
            inv_idx = jnp.zeros(mask.shape, dtype=jnp.int32)
            inv_idx = inv_idx.at[perm].set(imask)
        else:
            inv_idx = jnp.zeros(ar.shape[axis], dtype=int)
        ret += (inv_idx,)
    if return_counts:
        if aux.size:
            if size is not None:
                idx = jnp.nonzero(mask, size=size+1)[0]
                idx = idx.at[1:].set(jnp.where(idx[1:], idx[1:], mask.size))
            else:
                idx = jnp.concatenate(
                    jnp.lax_numpy.nonzero(mask) +
                    (jnp.lax_numpy.array([mask.size]),)
                )
            ret += (jnp.diff(idx),)
        elif ar.shape[axis]:
            ret += (jnp.array([ar.shape[axis]]),)
        else:
            ret += (jnp.empty(0, dtype=int),)
    return ret


def unique(ar: jnp.ndarray, return_index: bool = False, return_inverse: bool = False,
           return_counts: bool = False, axis: int = None, *, size: int = None) -> jnp.ndarray:
    """jnp.unique implementation with support for setting size and axis simultaneously.

    Args:
        ar (jnp.ndarray): data
        return_index (bool, optional): Return indices. Defaults to False.
        return_inverse (bool, optional): Return inverse mapping. Defaults to False.
        return_counts (bool, optional): Return counts. Defaults to False.
        axis (int, optional): axis to reduce. Defaults to None.
        size (int, optional): size to make the function jit compatible. Defaults to None.

    Returns:
        jnp.ndarray: unique items + return_index + return_inverse + return_counts
    """
    ar = jnp.asarray(ar)

    if axis is None:
        ret = jnp.lax_numpy._unique1d(
            ar,
            return_index,
            return_inverse,
            return_counts,
            size=size
        )
    else:
        ret = _unique_axis(
            ar,
            axis,
            return_index,
            return_inverse,
            return_counts,
            size=size
        )

    return ret[0] if len(ret) == 1 else ret


def tree_mul(tree, x):
    return jax.tree_map(lambda a: a*x, tree)


def tree_dot(a, b):
    return jax.tree_util.tree_reduce(
        jnp.add, jax.tree_map(
            jnp.sum, jax.tree_multimap(jax.lax.mul, a, b))
    )
