"""
This file contains two classes: Atom and Molecule.
An atom consists of an element and coordinates while a molecule
is composed by a set of atoms.

The classes contain simple logic functions to obtain spins, charges
and coordinates for molecules.
"""
import math
import numbers
from typing import Counter, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from pesnet.constants import ANGSTROM_TO_BOHR
from pesnet.systems import ELEMENT_BY_ATOMIC_NUM, ELEMENT_BY_SYMBOL
from pyscf import gto


class Atom:
    def __init__(self, name: str, coords: Optional[Sequence[float]] = None, units='bohr') -> None:
        assert units in ['bohr', 'angstrom']
        if isinstance(name, str):
            self.element = ELEMENT_BY_SYMBOL[name]
        elif isinstance(name, numbers.Number):
            self.element = ELEMENT_BY_ATOMIC_NUM[name]
        else:
            raise ValueError()
        self.coords = coords
        if self.coords is None:
            self.coords = (0, 0, 0)
        assert len(self.coords) == 3
        self.coords = np.array(coords)
        if units == 'angstrom':
            self.coords *= ANGSTROM_TO_BOHR

    @property
    def atomic_number(self):
        return self.element.atomic_number

    @property
    def symbol(self):
        return self.element.symbol

    def __str__(self) -> str:
        return self.element.symbol


class Molecule:
    def __init__(self, atoms: Sequence[Atom], spins: Optional[Tuple[int, int]] = None) -> None:
        self.atoms = atoms
        self._spins = spins

    def charges(self):
        return jnp.array([a.atomic_number for a in self.atoms])

    def coords(self):
        coords = jnp.array([a.coords for a in self.atoms])
        coords -= coords.mean(0, keepdims=True)
        return coords

    def spins(self):
        if self._spins is not None:
            return self._spins
        else:
            n_electrons = sum(self.charges())
            return (math.ceil(n_electrons/2), math.floor(n_electrons/2))

    def to_pyscf(self, basis='STO-3G', verbose: int = 3):
        center = np.array([a.coords for a in self.atoms]).mean(0)
        mol = gto.Mole(atom=[
            [a.symbol, a.coords - center]
            for a in self.atoms
        ], basis=basis, unit='bohr', verbose=verbose)
        spins = self.spins()
        mol.spin = spins[0] - spins[1]
        nuc_charge = sum(a.atomic_number for a in self.atoms)
        e_charge = sum(spins)
        mol.charge = nuc_charge - e_charge
        mol.build()
        return mol

    def __str__(self) -> str:
        result = ''
        if len(self.atoms) == 1:
            result = str(self.atoms[0])
        elif len(self.atoms) == 2:
            distance = np.linalg.norm(
                self.atoms[0].coords - self.atoms[1].coords)
            result = f'{str(self.atoms[0])}-{str(self.atoms[1])}_{distance:.2f}'
        else:
            vals = dict(Counter(str(a) for a in self.atoms))
            result = ''.join(f'{key}{val}' for key, val in vals.items())
        if sum(self.spins()) < sum(self.charges()):
            result += 'plus'
        elif sum(self.spins()) > sum(self.charges()):
            result += 'minus'
        return result
