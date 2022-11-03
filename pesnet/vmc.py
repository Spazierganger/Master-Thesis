import functools
import logging
import os
from typing import Dict, List, Tuple

import pdb
import jax
import jax.experimental.host_callback
import jax.numpy as jnp
import numpy as np
import optax
import tqdm.auto as tqdm

from pesnet import jax_utils
from pesnet.constants import pmap
from pesnet.damping import make_damping_fn
from pesnet.hamiltonian import make_local_energy_function
from pesnet.mcmc import make_mcmc
from pesnet.nn import ParamTree
from pesnet.optim import accumulate, scale_by_key
from pesnet.pesnet import make_pesnet
from pesnet.pretrain import eval_orbitals, make_pretrain_step
from pesnet.systems.scf import Scf
from pesnet.train import (make_loss, make_loss_and_natural_gradient_fn, make_schedule,
                          make_training_step)
from pesnet.utils import (EMAPyTree, MCMCStepSizeScheduler,
                          OnlineMean, make_fisher_trace)


class PesVmc:
    """This class contains a lot of utility functions to ease
    the training of models on the potential energy surface.
    It takes care of initializing the network, sampler,
    energy computation and optimization. Further, it includes
    utility functions to thermalize samples, evaluate the
    energies and checkpointing.
    """

    def __init__(
        self,
        key: jnp.ndarray,
        charges: jnp.ndarray,
        spins: Tuple[int, int],
        pesnet_config: Dict,
        sampler_config: Dict,
        training_config: Dict,
        train_state_deacy: float = 0.0
    ) -> None:
        """Constructor.

        Args:
            key (jnp.ndarray): jax.random.PRNGKey
            charges (jnp.ndarray): (M) charges of the system.
            spins (Tuple[int, int]): spins of the system.
            pesnet_config (Dict): Wfnet parameters, see 'pesnet.py'.
            sampler_config (Dict): Sampler parameters, see 'mcmc.py'.
            training_config (Dict): Optimization configuration.
            train_state_deacy (float, optional): EWM decay factor for the training state. Defaults to 0.0.
        """

        self.key = key

        self.num_devices = jax.device_count()

        self.charges = charges
        self.spins = spins

        self.pesnet_config = pesnet_config
        self.sampler_config = sampler_config
        self.training_config = training_config
        self.train_state_deacy = train_state_deacy

        self.key, subkey = jax.random.split(key)
        
        self.aux_loss_weight = training_config['aux_loss_weight']
        
        modeltype = pesnet_config['model_type']
        logging.info(f"\n Using {modeltype} \n")
        config = pesnet_config[f'{modeltype}_params']
        
        self.params, self.pesnet_fns = make_pesnet(
                modeltype,
                subkey,
                charges,
                spins,
                ferminet_params=config,
                gnn_params=self.pesnet_config['gnn_params'],
                node_filter=self.pesnet_config['node_filter'],
                global_filter=self.pesnet_config['global_filter'],
                include_default_filter=self.pesnet_config['include_default_filter'],
                full_det=self.pesnet_config['full_det'],
                envelope_type=self.pesnet_config['envelope_type'],
            )        
        self.params = jax_utils.replicate(self.params)
        
        # Vmap are all over the number of configurations
        self.batch_gnn = jax.vmap(
            self.pesnet_fns.gnn.apply, in_axes=(None, 0))
        # For FermiNet first vmap over electrons then vmap over configurations!
        self.batch_fermi = jax.vmap(
            jax.vmap(self.pesnet_fns.ferminet.apply, in_axes=(None, 0, None)), in_axes=(0, 0, 0))
        
        self.batch_get_fermi_params = self.pesnet_fns.batch_get_fermi_params
        self.pm_get_fermi_params = pmap(self.batch_get_fermi_params)

        # WfNet already uses a vmapped ferminet interallly, so we don't need to vmap over the electrons here
        # self.batch_pesnet = jax.vmap(
        #     self.pesnet_fns.pesnet_fwd, in_axes=(None, 0, 0))
        self.batch_pesnet = self.pesnet_fns.batch_pesnet
        self.pm_pesnet = pmap(self.batch_pesnet)
        
        # unused
        # self.batch_pesnet_orbitals = jax.vmap(
        #     self.pesnet_fns.pesnet_orbitals, in_axes=(None, 0, 0)
        # )
        
        self.batch_fermi_orbitals = jax.vmap(jax.vmap(
            functools.partial(
                self.pesnet_fns.ferminet.apply, 
                method=self.pesnet_fns.ferminet.orbitals
            ), 
            in_axes=(None, 0, None)
        ), in_axes=(0, 0, 0))
        self.pm_gnn = pmap(self.batch_gnn)

        # Sampling
        # Here we need a seperate width per atom (since the atom configuration changes)
        self.width = jax_utils.broadcast(
            jnp.ones((self.num_devices,)) *
            sampler_config['init_width']
        )
        self.width_scheduler = MCMCStepSizeScheduler(self.width)
        self.key, subkey = jax.random.split(self.key)

        self.sampler = make_mcmc(self.batch_fermi, **sampler_config)
        self.pm_sampler = pmap(self.sampler)
        # We need to wrap the sampler to first produce the parameters of the ferminet
        # otherwise we would have to execute the GNN for every sampling iteration

        def pesnet_sampler(params, electrons, atoms, key, width):
            fermi_params = self.batch_get_fermi_params(params, atoms)
            return self.sampler(fermi_params, electrons, atoms, key, width)
        self.pesnet_sampler = pesnet_sampler
        self.pm_pesnet_sampler = pmap(self.pesnet_sampler)

        # Prepare random keys
        self.key, *subkeys = jax.random.split(self.key, self.num_devices+1)
        subkeys = jnp.stack(subkeys)
        self.shared_key = jax_utils.broadcast(subkeys)

        # Prepare energy computation
        # We first need to compute the parameters and feed them into the energy computation
        self.local_energy_fn = make_local_energy_function(
            self.pesnet_fns.ferminet.apply,
            atoms=None,
            charges=charges
        )
        self.batch_local_energy = jax.vmap(jax.vmap(
            self.local_energy_fn, in_axes=(None, 0, None)
        ), in_axes=(0, 0, 0))
        self.pm_local_energy = pmap(self.batch_local_energy)

        def local_energy(params, electrons, atoms):
            fermi_params = self.batch_get_fermi_params(params, atoms)
            return self.batch_local_energy(fermi_params, electrons, atoms)
        self.batch_pesnet_local_energy = local_energy
        self.pm_pesnet_local_energy = pmap(local_energy)

        # Prepare some util functions
        self.fisher_trace = pmap(make_fisher_trace(self.batch_pesnet))

        # Prepare optimizer
        self.lr_gnn_prefactor = training_config['lr']['gnn_prefactor'] if 'gnn_prefactor' in training_config else 1.
        self.lr_schedule = make_schedule(training_config['lr'])

        ###########################################
        # Prepare VMC loss and gradient function
        self.use_cg = training_config['gradient'] == 'natural'
        self.opt_alg = training_config['optimizer']
        self.initialize_optimizer()

        self.train_state = EMAPyTree()

        ###########################################
        # Pretraining
        # self.initialize_pretrain_optimizer()

    def compute_init_damping(self, electrons: jnp.ndarray, atoms: jnp.ndarray, scaling: float = 1e-4):
        """Computes the initial damping based on ther trace of the fisher matrix

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)
            scaling (float, optional): Multiplied by the Fisher trace as initial damping. Defaults to 1e-4.
        """
        damping = self.fisher_trace(
            self.params, electrons, atoms)[0]*scaling
        self.training_config['cg']['damping']['init'] = damping

    def initialize_optimizer(self, optimizer: str = None, atoms: jnp.ndarray = None):
        """Initializes the optimizer and training step.

        Args:
            optimizer (str, optional): Overwrites the optimizer in the training config. Defaults to None.
            atoms (jnp.ndarray, optional): (..., M, 3) if specified an auxilliary loss is added
                which forces the parameters to stay close to the initial distribution. Defaults to None.
        """
        if optimizer is None:
            optimizer = self.opt_alg

        # Init optimizer
        lr_schedule = [
            optax.scale_by_schedule(self.lr_schedule),
            # scale_by_key({
            #     'gnn': self.lr_gnn_prefactor,
            #     'ferminet': 1.,
            #     'landmarks': 1.,
            # }),
            optax.scale(-1.)
        ]
        if optimizer == 'adam':
            self.optimizer = optax.chain(
                optax.scale_by_adam(),
                *lr_schedule
            )
        elif optimizer == 'sgd':
            self.optimizer = optax.chain(
                *lr_schedule
            )
        elif optimizer == 'sgd+clip':
            if self.training_config['accumulate_n'] > 1:
                self.optimizer = optax.chain(
                    optax.clip_by_global_norm(
                        self.training_config['max_norm']),
                    accumulate(self.training_config['accumulate_n']),
                    *lr_schedule
                )
            else:
                self.optimizer = optax.chain(
                    optax.clip_by_global_norm(
                        self.training_config['max_norm']),
                    *lr_schedule
                )
        elif optimizer == 'rmsprop':
            self.optimizer = optax.chain(
                optax.scale_by_rms(),
                *lr_schedule
            )
        elif optimizer == 'lamb':
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(eps=1e-7),
                optax.scale_by_trust_ratio(),
                *lr_schedule
            )
        else:
            raise ValueError()

        # Initialize loss function
        if self.training_config['gradient'] == 'euclidean':
            self.loss = make_loss(
                self.batch_pesnet,
                normalize_gradient=True,
                **self.training_config)
            self.loss_and_grads = jax.value_and_grad(self.loss)
        elif self.training_config['gradient'] == 'natural':
            self.loss_and_grads = make_loss_and_natural_gradient_fn(
                self.batch_pesnet,
                **self.training_config,
                **self.training_config['cg'],
                aux_loss=self.get_aux_loss(atoms, self.aux_loss_weight)
            )
            # TODO: This is hacky, it would be better to use a new variable
            self.loss_and_grads = make_damping_fn(
                self.training_config['cg']['damping']['method'],
                self.loss_and_grads,
                self.pesnet_sampler,
                self.batch_pesnet_local_energy,
                self.optimizer.update,
                self.training_config['cg']['damping']
            )
        else:
            raise ValueError(self.training_config['gradient'])

        # Initialize training step
        self.opt_state = jax.pmap(self.optimizer.init)(self.params)
        self.train_step = make_training_step(
            self.pesnet_sampler,
            self.loss_and_grads,
            self.batch_pesnet_local_energy,
            self.optimizer.update,
            uses_cg=self.use_cg,
        )

        # Initialize EMA and epoch counter
        self.epoch = jax_utils.replicate(jnp.zeros([]))
        self.train_state = EMAPyTree()
        # Let's initialize it exactly like cg in scipy:
        # https://github.com/scipy/scipy/blob/edb50b09e92cc1b493242076b3f63e89397032e4/scipy/sparse/linalg/isolve/utils.py#L95
        if self.training_config['gradient'] == 'natural':
            self.train_state.update({
                'last_grad': jax.tree_map(lambda x: jnp.zeros_like(x), self.params),
                'damping': jax_utils.replicate(jnp.ones([])*self.training_config['cg']['damping']['init'])
            }, 0)

    def initialize_pretrain_optimizer(self, lr: float, atoms: jnp.ndarray = None, train_gnn: bool = False):
        """Initializes the pretraining optimizer and update function.

        Args:
            atoms (jnp.ndarray, optional): (..., M, 3) If specified an auxilliary loss is added
                that forces the parameters to stay close the initialization. Defaults to None.
            train_gnn (bool, optional): Whether to train the GNN. Defaults to False.
        """
        self.pre_opt_init, self.pre_opt_update = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale_by_trust_ratio(),
            scale_by_key({
                'gnn': self.lr_gnn_prefactor,
                'ferminet': 1.
            }),
            optax.scale(-lr),
        )
        # self.pre_opt_init, self.pre_opt_update = optax.adam(lr)
        self.pre_opt_state = jax.pmap(self.pre_opt_init)(self.params)

        self.pretrain_step = pmap(make_pretrain_step(
            self.batch_fermi,
            self.pesnet_sampler,
            self.batch_fermi_orbitals,
            self.batch_get_fermi_params,
            self.pre_opt_update,
            aux_loss=self.get_aux_loss(atoms, self.aux_loss_weight),
            train_gnn=train_gnn,
            full_det=self.pesnet_config['full_det'],
        ))

    def get_aux_loss(self, atoms: jnp.ndarray, weight: float = 1e-3):
        """Creates an auxilliary loss function that forces parameters to
        stay close the value computed at call of this function.

        Args:
            atoms (jnp.ndarray): (..., M, 3) atom coordinates
            weight (float, optional): Loss weighting factor. Defaults to 1e-3.

        Returns:
            Callable: Auxilliary loss
        """
        if atoms is None:
            return None
        init_statistics = jax.tree_map(
            lambda x: jnp.array([x.mean(), x.var()]),
            self.get_fermi_params(atoms)
        )

        def aux_loss(params, electrons, atoms):
            f_params = self.batch_get_fermi_params(params, atoms)
            f_stats = jax.tree_map(
                lambda x: jnp.array([x.mean(), x.var()]),
                f_params
            )
            loss = jax.tree_multimap(
                lambda x, y: ((x-y)**2).mean(),
                f_stats,
                init_statistics
            )
            loss = jax.tree_util.tree_reduce(jnp.add, loss)
            return weight * loss
        return aux_loss

    def get_fermi_params(self, atoms: jnp.ndarray) -> ParamTree:
        """Returns the full Ansatz parameters for the given molecular structures.

        Args:
            atoms (jnp.ndarray): (..., M, 3)

        Returns:
            ParamTree: Ansatz parameters
        """
        return self.pm_get_fermi_params(self.params, atoms)

    def thermalize_samples(
            self,
            electrons: jnp.ndarray,
            atoms: jnp.ndarray,
            n_iter: int,
            show_progress: bool = True,
            adapt_step_width: bool = False) -> jnp.ndarray:
        """Thermalize electrons.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)
            n_iter (int): Number of thermalizing steps to take.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.
            adapt_step_width (bool, optional): Whether to adapt the step width. Defaults to False.

        Returns:
            jnp.ndarray: thermalized electrons
        """
        iters = tqdm.trange(n_iter, desc='Thermalizing',
                            leave=False) if show_progress else range(n_iter)
        fermi_params = self.pm_get_fermi_params(self.params, atoms)
        try:
            for _ in iters:
                self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
                electrons, pmove = self.pm_sampler(
                    fermi_params,
                    electrons,
                    atoms,
                    subkeys,
                    self.width
                )
                if adapt_step_width:
                    self.width = self.width_scheduler(pmove.mean())
                if show_progress:
                    iters.set_postfix({
                        'pmove': pmove[0, 0],
                        'width': self.width[0]
                    })
        except KeyboardInterrupt:
            iters.close()
            return electrons
        if show_progress:
            iters.close()
        return electrons

    def update_step(self, electrons: jnp.ndarray, atoms: jnp.ndarray):
        """Does an parameter update step.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)

        Returns:
            (jnp.ndarray, (jnp.ndarray, jnp.ndarray, jnp.ndarray)): (new electrons, (energy, energy variance, pmove))
        """
        # Do update step
        self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
        (electrons, self.params, self.opt_state, E, E_var, pmove), train_state = self.train_step(
            self.epoch,
            self.params,
            electrons,
            atoms,
            self.opt_state,
            subkeys,
            self.width,
            **self.train_state.value
        )
        self.train_state.update(train_state, self.train_state_deacy)
        self.width = self.width_scheduler(np.mean(pmove))

        self.epoch += 1
        return electrons, (E, E_var, pmove)

    def pre_update_step(self, electrons: jnp.ndarray, atoms: jnp.ndarray, scfs: List[Scf]):
        """Performs an pretraining update step.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)
            scfs (List[Scf]): List of SCF solutions for the provided atom configurations.

        Returns:
            (jnp.ndarray, jnp.ndarray, jnp.ndarray): (loss, new electrons, move probability)
        """
        self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
        targets = eval_orbitals(scfs, electrons, self.spins)

        self.params, electrons, self.pre_opt_state, loss, pmove = self.pretrain_step(
            self.params,
            electrons,
            atoms,
            targets,
            self.pre_opt_state,
            subkeys,
            self.width
        )
        self.width = self.width_scheduler(pmove.mean())
        return loss, electrons, pmove

    def eval_energy(
            self,
            electrons: jnp.ndarray,
            atoms: jnp.ndarray,
            n_repeats: int,
            thermalize_steps: int,
            show_progress: bool = True) -> jnp.ndarray:
        """Evaluates the energy for the given molecular structure.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)
            n_repeats (int): How often we sample and compute the energy
            thermalize_steps (int): Thermalizing steps between energy computations
            show_progress (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            jnp.ndarray: evaluated energies
        """
        fermi_params = self.pm_get_fermi_params(self.params, atoms)
        n_configs = np.prod(atoms.shape[:2])
        if show_progress:
            means = [OnlineMean() for _ in range(n_configs)]

        iters = tqdm.trange(
            n_repeats, desc='Computing Energy') if show_progress else range(n_repeats)
        total_energies = []

        def iter_step(key, electrons):
            for j in range(thermalize_steps):
                key, subkey = jax_utils.p_split(key)
                electrons, pmove = self.pm_sampler(
                    fermi_params,
                    electrons,
                    atoms,
                    subkey,
                    self.width
                )
            energies = self.pm_local_energy(
                fermi_params, electrons, atoms
            )
            return key, energies, electrons

        for i in iters:
            self.shared_key, energies, electrons = iter_step(
                self.shared_key, electrons)
            total_energies.append(energies.reshape(n_configs, -1))
            if show_progress:
                for e, mean in zip(energies.reshape(n_configs, -1), means):
                    mean.update(e)
                logging.info('\t'.join(map(str, means)))
        total_energies = np.concatenate(total_energies, -1)
        return total_energies

    def checkpoint(self, folder: str, name: str, electrons: jnp.ndarray, atoms: jnp.ndarray):
        """Store checkpoint.

        Args:
            folder (str): Folder to store the checkpoint
            name (str): Checkpoint name
            electrons: electrons to be stored
            atoms: atoms to be stored
        """
        with open(os.path.join(folder, name), 'wb') as out:
            np.savez(out, {
                'params': self.params,
                'opt_state': self.opt_state,
                'width': self.width,
                'train_state': self.train_state.value,
                'epoch': self.epoch,
                'electrons': np.array(electrons),
                'atoms': np.array(atoms),
            })

    def load_checkpoint(self, file_path):
        """Load checkpoint

        Args:
            file_path (str): Path to checkpoint file
        """
        with open(file_path, 'rb') as inp:
            data = dict(np.load(inp, allow_pickle=True))['arr_0'][None][0]
        self.params = jax_utils.broadcast(data['params'])
        self.opt_state = jax_utils.broadcast(data['opt_state'])
        self.width = jax_utils.broadcast(data['width'])
        self.epoch = jax_utils.broadcast(data['epoch'])
        self.train_state = jax_utils.broadcast(data['train_state'])
