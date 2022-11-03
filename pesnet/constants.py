"""
Physical constants + some utility functions.
Some of these functions are taken from 
https://github.com/deepmind/ferminet/blob/jax/ferminet/constants.py
"""

import functools

import jax
from jax import core

BOHR_TO_ANGSTROM = 0.52917721067
ANGSTROM_TO_BOHR = 1./BOHR_TO_ANGSTROM

HARTREE_TO_EV = 27.211386024367243
EV_TO_KCAL = 23.060548012069493
KCAL_TO_EV = 1. / EV_TO_KCAL
HARTREE_TO_KCAL = HARTREE_TO_EV * EV_TO_KCAL
KCAL_TO_HARTREE = 1. / HARTREE_TO_KCAL

HARTREE_TO_INV_CM = 219474.63
INV_CM_TO_HARTREE = 1. / HARTREE_TO_INV_CM


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(jax.lax.pmean, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(jax.lax.psum, axis_name=PMAP_AXIS_NAME)
pmax = functools.partial(jax.lax.pmax, axis_name=PMAP_AXIS_NAME)


def wrap_if_pmap(p_func):
    def p_func_if_pmap(obj):
        try:
            core.axis_frame(PMAP_AXIS_NAME)
            return p_func(obj)
        except NameError:
            return obj
    return p_func_if_pmap


pmean_if_pmap = wrap_if_pmap(pmean)
psum_if_pmap = wrap_if_pmap(psum)
pmax_if_pmap = wrap_if_pmap(pmax)
