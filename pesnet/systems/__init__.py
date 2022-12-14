"""
This file contains various system configurations, e.g., atomic, 
diatomic systems, cyclobutadiene, ...
"""
import os
import numpy as np

from .element import ELEMENT_BY_ATOMIC_NUM, ELEMENT_BY_SYMBOL
from .system import Atom, Molecule
from .xyz_parser import read_xyz


def atomic(symbol: str, *args, **kwargs):
    return Molecule([
        Atom(symbol, (0.0, 0.0, 0.0))
    ])


def diatomic(symbol1: str, symbol2: str, R: float, units='bohr', *args, **kwargs):
    return Molecule([
        Atom(symbol1, (-R/2, 0.0, 0.0), units=units),
        Atom(symbol2, (R/2, 0.0, 0.0), units=units),
    ])


def H2(R: float, *args, **kwargs):
    return diatomic('H', 'H', R, *args, **kwargs)


def LiH(R: float, *args, **kwargs):
    return diatomic('Li', 'H', R, *args, **kwargs)


def h4_plane(theta: float, R: float, *args, **kwargs):
    y = np.sin(np.radians(theta/2)) * R
    x = np.cos(np.radians(theta/2)) * R
    return Molecule([
        Atom('H', (x, y, 0.0)),
        Atom('H', (x, -y, 0.0)),
        Atom('H', (-x, y, 0.0)),
        Atom('H', (-x, -y, 0.0))
    ])


def h2o(theta: float, R: float, *args, **kwargs):
    assert 0 < theta < 180
    return Molecule([
        Atom('H', (R * np.sin(np.radians(theta / 2)), R * np.cos(np.radians(theta / 2)), 0.)),
        Atom('O', (0., 0., 0.)),
        Atom('H', (- R * np.sin(np.radians(theta / 2)), R * np.cos(np.radians(theta / 2)), 0.)),
    ])


def h_chain(n: int, R: float, *args, **kwargs):
    span = (n-1)*R
    center = span/2
    return Molecule([
        Atom('H', (i*R - center, 0.0, 0.0))
        for i in range(n)
    ])


def regular_hn_plane(n: int, R: float, *args, **kwargs):
    return Molecule([
        Atom('H', (
            np.sin(i/n * 2 * np.pi) * R,
            np.cos(i/n * 2 * np.pi) * R,
            0.0
        ))
        for i in range(n)
    ])


def by_positions(symbols, positions, units='bohr', spins=None, *args, **kwargs):
    assert len(symbols) == len(positions)
    positions = np.array(positions)
    return Molecule([
        Atom(sym, coords)
        for sym, coords in zip(symbols, positions)
    ], spins=spins)


def h4plus(positions: np.ndarray, *args, **kwargs):
    return by_positions(['H']*len(positions), positions, spins=(2, 1))


H4plus = h4plus


def cyclobutadiene(state: str):
    # https://github.com/deepmind/ferminet/blob/jax/ferminet/configs/organic.py
    if state == 'ground':
        return Molecule([
            Atom('C', (0.0000000e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.9555318e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.9555318e+00, 2.5586891e+00, 0.0000000e+00)),
            Atom('C', (0.0000000e+00, 2.5586891e+00, 0.0000000e+00)),
            Atom('H', (-1.4402903e+00, -1.4433100e+00, 1.7675451e-16)),
            Atom('H', (4.3958220e+00, -1.4433100e+00, -1.7675451e-16)),
            Atom('H', (4.3958220e+00, 4.0019994e+00, 1.7675451e-16)),
            Atom('H', (-1.4402903e+00, 4.0019994e+00, -1.7675451e-16)),
        ])
    elif state == 'transition':
        return Molecule([
            Atom('C', (0.0000000e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.7419927e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.7419927e+00, 2.7419927e+00, 0.0000000e+00)),
            Atom('C', (0.0000000e+00, 2.7419927e+00, 0.0000000e+00)),
            Atom('H', (-1.4404647e+00, -1.4404647e+00, 1.7640606e-16)),
            Atom('H', (4.1824574e+00, -1.4404647e+00, -1.7640606e-16)),
            Atom('H', (4.1824574e+00, 4.1824574e+00, 1.7640606e-16)),
            Atom('H', (-1.4404647e+00, 4.1824574e+00, -1.7640606e-16))
        ])


def bicyclobutane(state: str):
    # https://pubs.acs.org/doi/abs/10.1021/jp065721k
    # Supplementary
    # https://github.com/deepmind/ferminet/blob/jax/ferminet/configs/organic.py
    if state == 'bicbut':
        return Molecule([
            Atom('C', (1.0487346562, 0.5208579773, 0.2375867187), units='angstrom'),
            Atom('C', (0.2497284256, -0.7666691493, 0.0936474818), units='angstrom'),
            Atom('C', (-0.1817326465, 0.4922777820, -0.6579637266), units='angstrom'),
            Atom('C', (-1.1430708301, -0.1901383337, 0.3048494250), units='angstrom'),
            Atom('H', (2.0107137141, 0.5520589541, -0.2623459977), units='angstrom'),
            Atom('H', (1.0071921280, 1.0672669240, 1.1766131856), units='angstrom'),
            Atom('H', (0.5438033167, -1.7129829738, -0.3260782874), units='angstrom'),
            Atom('H', (-0.2580605320, 0.6268443026, -1.7229636111), units='angstrom'),
            Atom('H', (-1.3778676954, 0.2935640723, 1.2498189977), units='angstrom'),
            Atom('H', (-1.9664163102, -0.7380906148, -0.1402911727), units='angstrom')
        ])
    elif state == 'con_TS':
        return Molecule([
            Atom('C', (1.0422528085, 0.5189448459, 0.2893513723), units='angstrom'),
            Atom('C', (0.6334392052, -0.8563584473, -0.1382423606), units='angstrom'),
            Atom('C', (-0.2492035181, 0.3134656784, -0.5658962512), units='angstrom'),
            Atom('C', (-1.3903646889, 0.0535204487, 0.2987506023), units='angstrom'),
            Atom('H', (1.8587636947, 0.9382817031, -0.2871146890), units='angstrom'),
            Atom('H', (0.9494853889, 0.8960565051, 1.3038563129), units='angstrom'),
            Atom('H', (0.3506375894, -1.7147937260, 0.4585707483), units='angstrom'),
            Atom('H', (-0.3391417369, 0.6603641863, -1.5850373819), units='angstrom'),
            Atom('H', (-1.2605467656, 0.0656225945, 1.3701508857), units='angstrom'),
            Atom('H', (-2.3153892612, -0.3457478660, -0.0991685880), units='angstrom'),
        ])
    elif state == 'dis_TS':
        return Molecule([
            Atom('C', (1.5864390444, -0.1568990400, -0.1998155990), units='angstrom'),
            Atom('C', (-0.8207390911, 0.8031532550, -0.2771554962), units='angstrom'),
            Atom('C', (0.2514913592, 0.0515423448, 0.4758741643), units='angstrom'),
            Atom('C', (-1.0037104567, -0.6789877402, -0.0965401189), units='angstrom'),
            Atom('H', (2.4861305372, 0.1949133826, 0.2874101433), units='angstrom'),
            Atom('H', (1.6111805503, -0.2769458302, -1.2753251100), units='angstrom'),
            Atom('H', (-1.4350764228, 1.6366792379, 0.0289087336), units='angstrom'),
            Atom('H', (0.2833919284, 0.1769734467, 1.5525271253), units='angstrom'),
            Atom('H', (-1.7484283536, -1.0231589431, 0.6120702030), units='angstrom'),
            Atom('H', (-0.8524391649, -1.3241689195, -0.9544331346), units='angstrom')
        ])
    elif state == 'g-but':
        return Molecule([
            Atom('C', (1.4852019019, 0.4107781008, 0.5915178362), units='angstrom'),
            Atom('C', (0.7841417614, -0.4218449588, -0.2276848579), units='angstrom'),
            Atom('C', (-0.6577970182, -0.2577617373, -0.6080850660), units='angstrom'),
            Atom('C', (-1.6247236649, 0.2933006709, 0.1775352473), units='angstrom'),
            Atom('H', (1.0376813593, 1.2956518484, 1.0267024109), units='angstrom'),
            Atom('H', (2.5232360753, 0.2129135014, 0.8248568552), units='angstrom'),
            Atom('H', (1.2972328960, -1.2700686671, -0.6686116041), units='angstrom'),
            Atom('H', (-0.9356614935, -0.6338686329, -1.5871170536), units='angstrom'),
            Atom('H', (-1.4152018269, 0.6472889925, 1.1792563311), units='angstrom'),
            Atom('H', (-2.6423222755, 0.3847635835, -0.1791755263), units='angstrom')
        ])
    elif state == 'gt_TS':
        return Molecule([
            Atom('C', (1.7836595975, 0.4683155866, -0.4860478101), units='angstrom'),
            Atom('C', (0.7828892933, -0.4014025715, -0.1873880949), units='angstrom'),
            Atom('C', (-0.6557274850, -0.2156646805, -0.6243545354), units='angstrom'),
            Atom('C', (-1.6396999531, 0.2526943506, 0.1877948644), units='angstrom'),
            Atom('H', (1.6003117673, 1.3693309737, -1.0595471944), units='angstrom'),
            Atom('H', (2.7986234673, 0.2854595500, -0.1564989895), units='angstrom'),
            Atom('H', (1.0128486304, -1.2934621995, 0.3872559845), units='angstrom'),
            Atom('H', (-0.9003245968, -0.4891235826, - 1.6462438855), units='angstrom'),
            Atom('H', (-1.4414954784, 0.5345813494, 1.2152198579), units='angstrom'),
            Atom('H', (-2.6556262424, 0.3594422237, -0.1709361970), units='angstrom')
        ])
    elif state == 't-but':
        return Molecule([
            Atom('C', (0.6109149108, 1.7798412991, -0.0000000370), units='angstrom'),
            Atom('C', (0.6162339625, 0.4163908910, -0.0000000070), units='angstrom'),
            Atom('C', (-0.6162376752, -0.4163867945, -0.0000000601), units='angstrom'),
            Atom('C', (-0.6109129465, -1.7798435851, 0.0000000007), units='angstrom'),
            Atom('H', (1.5340442204, 2.3439205382, 0.0000000490), units='angstrom'),
            Atom('H', (-0.3156117962, 2.3419017314, 0.0000000338), units='angstrom'),
            Atom('H', (1.5642720455, -0.1114324578, -0.0000000088), units='angstrom'),
            Atom('H', (-1.5642719469, 0.1114307897, -0.0000000331), units='angstrom'),
            Atom('H', (-1.5340441021, -2.3439203971, 0.0000000714), units='angstrom'),
            Atom('H', (0.3156133277, -2.3419020150, -0.0000000088), units='angstrom')
        ])
    else:
        raise ValueError()
        
        
def nh3o3(state: str):
    if state == 'complex':
        return Molecule([
            Atom('N', (0.010935, 0.161494, 0.027196), units='angstrom'),
            Atom('O', (0.316501, -0.937927, 2.923490), units='angstrom'),
            Atom('O', (1.235012, -0.094654, 2.885792), units='angstrom'),
            Atom('O', (2.220860, -0.343633, 2.163064), units='angstrom'),
            Atom('H', (0.288632, -0.808619, -0.051241), units='angstrom'),
            Atom('H', (0.457140, 0.652245, -0.736668), units='angstrom'),
            Atom('H', (-0.984332, 0.192257, -0.151477 ), units='angstrom'),
        ])
    elif state == 'product':
        return Molecule([
            Atom('N', (1.316951, -0.489095, 0.170975), units='angstrom'),
            Atom('O', (0.584512, 0.592153, -0.357588), units='angstrom'),
            Atom('O', (-0.659216, 0.601455, 0.321757), units='angstrom'),
            Atom('O', (-1.410596, -0.485778, -0.166783), units='angstrom'),
            Atom('H', (-1.062691, -1.206172, 0.375654), units='angstrom'),
            Atom('H', (2.088067, -0.073829, 0.681183), units='angstrom'),
            Atom('H', (1.689773, -0.961244, -0.643837), units='angstrom'),
        ])
    elif state == 'separated':
        return Molecule([
            Atom('N', (-0.49545756, 0.32989928, -0.98139775), units='angstrom'),
            Atom('O', (7.9962378, -3.9339628, 19.886362), units='angstrom'),
            Atom('O', (8.914749, -3.09069, 19.848665), units='angstrom'),
            Atom('O', (9.900597, -3.3396688, 19.125937), units='angstrom'),
            Atom('H', (-0.21776058, -0.6402137, -1.0598347), units='angstrom'),
            Atom('H', (-0.04925258, 0.8206503, -1.7452617), units='angstrom'),
            Atom('H', (-1.4907246, 0.36066228, -1.1600707), units='angstrom'),
        ])
    else:
        raise NotImplementedError


def from_xyz(file_name, state):
    configs = read_xyz(file_name)
    return configs[state]
