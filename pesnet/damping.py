"""
This file contains different damping schedules. The most important
function here is `make_damping_fn` which creates any of the defined
schedules given a configuration.
"""
import jax
import jax.numpy as jnp
import optax

from pesnet.constants import pmean_if_pmap
from pesnet.train import make_schedule


def make_damping_schedule(val_and_grad, **kwargs):
    # Simple fixed schedules based on the step number
    schedule = make_schedule(kwargs)

    def eval_and_schedule(key, damping, opt_state, t, params, electrons, atoms, e_l, mcmc_width, **kwargs):
        damping = schedule(t)
        _, grads = val_and_grad(
            t,
            params,
            electrons,
            atoms,
            e_l,
            damping=damping,
            **kwargs
        )
        return grads, damping
    return eval_and_schedule


def make_adaptive_damping_fn(val_and_grad, mcmc_step, el_fn, opt_update, step_size, test_every, threshold):
    # A adaptive damping scheme where every `test_every`
    # iterations three damping values are compared in terms of
    # energy and the lowest one is taken.
    def eval_and_adapt(key, damping, opt_state, t, params, electrons, atoms, e_l, mcmc_width, **kwargs):
        def try_dampings(_):
            nonlocal key
            dampings = jnp.array(
                [damping*step_size, damping, damping/step_size])

            key, subkey = jax.random.split(key)
            tmp_electrons, _ = mcmc_step(
                params,
                electrons,
                atoms,
                subkey,
                mcmc_width
            )

            d_stds = []
            d_grads = []
            for damp in dampings:
                grads = val_and_grad(
                    t,
                    params,
                    electrons,
                    atoms,
                    e_l,
                    damping=damp,
                    **kwargs
                )[1]

                tmp_params = pmean_if_pmap(
                    optax.apply_updates(
                        params,
                        opt_update(
                            grads,
                            opt_state,
                            params
                        )[0]
                    )
                )

                std = pmean_if_pmap(
                    el_fn(
                        tmp_params,
                        tmp_electrons,
                        atoms
                    ).std(-1).mean()
                )

                d_grads.append(grads)
                d_stds.append(std)

            idx = ((d_stds[1] - d_stds[0])/d_stds[1] > threshold)*-1 + \
                ((d_stds[1] - d_stds[2])/d_stds[1] > threshold)*1
            idx = (idx+1).astype(jnp.int32)
            grads = jax.tree_multimap(
                lambda *args: jnp.stack(args, axis=0)[idx],
                *d_grads
            )
            return grads, jnp.clip(dampings[idx], 1e-5, 1e-2)

        def fixed_damping(_):
            return val_and_grad(t, params, electrons, atoms, e_l, damping=damping, **kwargs)[1], damping

        return jax.lax.cond(t % test_every == 0, try_dampings, fixed_damping, ())
    return eval_and_adapt


def make_std_based_damping_fn(val_and_grad, base, target_pow=0.5, **kwargs):
    # A simple damping scheme based on the standard deviation of the local energy.
    def data_based(key, damping, opt_state, t, params, electrons, atoms, e_l, mcmc_width, **kwargs):
        target = pmean_if_pmap(
            base * jnp.power(jnp.sqrt(e_l.var(-1).mean()), target_pow))
        damping = jnp.where(damping < target, damping/0.999, damping)
        damping = jnp.where(damping > target, 0.999*damping, damping)
        damping = jnp.clip(damping, 1e-8, 1e-3)
        _, grads = val_and_grad(
            t, params, electrons, atoms, e_l, damping=damping, **kwargs)
        return grads, damping
    return data_based


def make_damping_fn(
        method: str,
        val_and_grad,
        mcmc_step,
        el_fn,
        opt_update,
        kwargs):
    method = method.lower()
    if method == 'schedule':
        return make_damping_schedule(
            val_and_grad,
            init=kwargs['init'],
            **kwargs['schedule'])
    elif method == 'adaptive':
        return make_adaptive_damping_fn(
            val_and_grad,
            mcmc_step,
            el_fn,
            opt_update,
            **kwargs['adaptive']
        )
    elif method == 'std_based':
        return make_std_based_damping_fn(
            val_and_grad,
            # init=kwargs['init'],
            **kwargs['std_based']
        )
    else:
        raise ValueError()
