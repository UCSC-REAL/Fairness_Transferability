#!/usr/bin/env python

'''
A namespace for population response functions

Signature for each function:

Args:
    setting: (Setting object) - current setting parameters
    state: (State object) - current state
    phi: (numpy array) - classifier's feature threshold for each group

Returns:
    (State object): velocity vector $(s_g[t+1] - s_g[t] : g \in \mathcal{G})$
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize as opt

from state import State
import util

epsilon = np.finfo(float).eps

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Nimbus Roman',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def replicator_equation(setting, state, phi):

    U, Q1, Q0, mu, sg = (
        setting.U, setting.Q1, setting.Q0,
        setting.mu, state.sg
    )

    # Qy is proportion of (y = qualified) agents that are rejected
    # It is used to interpolate from U[y,y_hat=1] to U[y,y_hat=0]
    W1 = U[1,1] + (U[1,0] - U[1,1]) * Q1(phi)
    W0 = U[0,1] + (U[0,0] - U[0,1]) * Q0(phi)

    # sg is proportion qualified in each group
    # Wg is average fitness of each group
    Wg = sg * W1 + (1 - sg) * W0

    # velocity vector in state space, where unit time separates each round
    return State(mu, (sg * W1 / Wg) - sg)
