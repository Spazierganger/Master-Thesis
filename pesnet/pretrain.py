from typing import Callable, List, Optional, Tuple

import pdb
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pesnet.constants import pmean
from pesnet.nn import MLP, ParamTree
from pesnet.systems.scf import Scf
from pesnet.mcmc import mh_update


def eval_orbitals(scf_approx: List[Scf], electrons: jnp.ndarray, spins: Tuple[int, int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the molecular orbitals of Hartree Fock calculations.

    Args:
        scf_approx (List[Scf]): Hartree Fock calculations, length H
        electrons ([type]): (H, B, N, 3)
        spins ([type]): (spin_up, spin_down)

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [(H, B, spin_up, spin_up), (H, B, spin_down, spin_down)] molecular orbitals
    """
    if isinstance(scf_approx, (list, tuple)):
        n_scf = len(scf_approx)
    else:
        n_scf = 1
        scf_approx = [scf_approx]
    leading_dims = electrons.shape[:-1]
    electrons = electrons.reshape([n_scf, -1, 3])  # (batch*nelec, 3)
    # (batch*nelec, nbasis), (batch*nelec, nbasis)
    mos = [scf.eval_molecular_orbitals(e)
           for scf, e in zip(scf_approx, electrons)]
    mos = (np.stack([mo[0] for mo in mos], axis=0),
           np.stack([mo[1] for mo in mos], axis=0))
    # Reshape into (batch, nelec, nbasis) for each spin channel
    mos = [mo.reshape(leading_dims + (sum(spins), -1)) for mo in mos]

    alpha_spin = mos[0][..., :spins[0], :spins[0]]
    beta_spin = mos[1][..., spins[0]:, :spins[1]]
    return alpha_spin, beta_spin


def eval_slater(scf_approx: List[Scf], electrons: jnp.ndarray, spins: Tuple[int, int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the Slater determinants of Hartree Fock calculations

    Args:
        scf_approx (List[Scf]): Hartree Fock solutions
        electrons (jnp.ndarray): (H, B, N, 3)
        spins (Tuple[int, int]): (spin_up, spin_down)

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: (sign, log_psi)
    """
    matrices = eval_orbitals(scf_approx, electrons, spins)
    slogdets = [np.linalg.slogdet(elem) for elem in matrices]
    signs = np.array([elem[0] for elem in slogdets])
    log_wfs = np.array([elem[1] for elem in slogdets])
    log_slater = np.sum(log_wfs, axis=0)
    sign = np.prod(signs, axis=0)
    return sign, log_slater


def make_pretrain_step(
    batch_network: Callable,
    mcmc_step: Callable,
    batch_orbitals: Callable,
    batch_get_params: ParamTree,
    opt_update: Callable,
    aux_loss: Optional[Callable] = None,
    full_det: bool = False,
    train_gnn: bool = False
):
    """Returns the pretraining step function.

    Args:
        mcmc_step (Callable): Sampling function, see `mcmc.py`.
        batch_orbitals (Callable): Wave function orbital function
        batch_get_params (ParamTree): wave function parameters
        opt_update (Callable): Optimizer update function
        aux_loss (Optional[Callable], optional): Auxilliary loss function. Defaults to None.
        full_det (bool, optional): Whether the network uses `full_det=True`. Defaults to False.
        train_gnn (bool, optional): Whether to train the GNN as well. Defaults to False.
    """
    def pretrain_step(params, electrons, atoms, targets, opt_state, key, mcmc_width):
        def loss_fn(params, electrons, atoms, targets):
            fermi_params = batch_get_params(params, atoms)
            orbitals = batch_orbitals(
                fermi_params,
                electrons,
                atoms
            )
            if full_det and len(targets) == 2:
                leading_dims = targets[0].shape[:-2]
                na = targets[0].shape[-2]
                nb = targets[1].shape[-1]
                targets = [jnp.concatenate(
                    (jnp.concatenate((targets[0], jnp.zeros((*leading_dims, na, nb))), axis=-1),
                     jnp.concatenate((jnp.zeros((*leading_dims, nb, na)), targets[1]), axis=-1)),
                    axis=-2)]
            k = orbitals[0].shape[-3]
            n_devices = jax.device_count()
            configs_per_device = targets[0].shape[0]
            assert k % n_devices == 0
            if configs_per_device > 1:
                idx = jnp.around(
                    jnp.linspace(
                        0,
                        configs_per_device-1,
                        k
                    )
                ).astype(jnp.int32)
                idx2 = jnp.arange(k)
                result = jnp.array([
                    jnp.mean((t[idx] - o[idx, :, idx2])**2) for t, o in zip(targets, orbitals)
                ]).sum()
                mask = jnp.zeros((k, k), dtype=jnp.int32)
                mask = jax.ops.index_update(
                    mask,
                    jax.ops.index[idx, idx2],
                    1
                )
            else:
                result = jnp.array([
                    jnp.mean((t[..., None, :, :] - o)**2) for t, o in zip(targets, orbitals)
                ]).sum()
            if aux_loss is not None:
                result += aux_loss(params, electrons, atoms)
            return pmean(result)

        val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
        loss_val, grad = val_and_grad(
            params,
            electrons,
            atoms,
            targets
        )
        grad = pmean(grad)

        # Rescale GNN params if a MetaGNN is  used
        if 'params' in grad['gnn']:
            gnn_grad = grad['gnn']
            node_out_grad = [
                (k, gnn_grad['params'][k]['Embed_0']['embedding'])
                for k in gnn_grad['params'].keys()
                if 'NodeOut' in k
            ]
            global_out_grad = [
                (k, MLP.extract_final_linear(gnn_grad['params'][k])['bias'])
                for k in gnn_grad['params'].keys()
                if 'GlobalOut' in k
            ]

            scaling = 0.1 if train_gnn else 0

            # Rescale GNN gradients
            gnn_grad['params'] = jax.tree_map(
                lambda x: scaling * x, gnn_grad['params'])

            # Reset final biases
            for k, val in node_out_grad:
                gnn_grad['params'][k]['Embed_0']['embedding'] = val
            for k, val in global_out_grad:
                MLP.extract_final_linear(gnn_grad['params'][k])['bias'] = val

        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        params = pmean(params)
        
        electrons, pmove = mcmc_step(
            params,
            electrons,
            atoms,
            key,
            mcmc_width
        )
        
#         fermi_params = batch_get_params(params, atoms)
        
#         def logprob_fn(x): return 2. * batch_network(fermi_params, x, atoms)
    
#         if isinstance(mcmc_width, jnp.ndarray) and mcmc_width.ndim == 1:
#             mcmc_width = mcmc_width[:, None, None]
            
#         batch_per_device = electrons.shape[-2]
#         logprob = logprob_fn(electrons)
#         num_accepts = jnp.zeros(1)
            
#         electrons, key, _, num_accepts = mh_update(logprob_fn, electrons, key, logprob, num_accepts, stddev=mcmc_width)
        
#         pmove = num_accepts / batch_per_device
#         if pmove.ndim == 0:
#             pmove = pmean_if_pmap(pmove)

        return params, electrons, opt_state, loss_val, pmove
    return pretrain_step
