#!/usr/bin/env python

import numpy as np
from scipy import optimize as opt

from matplotlib import pyplot as plt

from util import dot

epsilon = 0.005
max_float = np.finfo(float).max
min_float = np.finfo(float).min

'''
A namespace for classifier functions

Signature for each function:

Args:
    setting: (Setting object) - current setting parameters
    state: (State object) - current state
    rate_limit: float: [0, 1] - max allowed acceptance rate

Returns:
    phi: (numpy array) - classifier's feature threshold for each group
'''

def demographic_parity_bayes(setting, state, rate_limit=1):
    '''
    Bayes-optimal classifier contrained by demographic parity
    '''

    V, Q1, Q0, inv_q1_q0, sg, mu = (
        setting.V, setting.Q1, setting.Q0, setting.inv_q1_q0, state.sg, state.mu
    )

    x = np.linspace(-10, 10, 1000)

    def perturbed(state, gamma):
        '''get thresholds phi'''

        # Eq. 136a
        n = (V[0,0] - V[0,1] - gamma)
        d = (V[1,1] - V[1,0] + gamma)

        # n / d should range from 0 (phi = -inf) to inf (phi = inf)
        # we check boundary values at line 247
        with np.testing.suppress_warnings() as sup:

            sup.filter(RuntimeWarning, 'invalid value encountered in true_divide')

            if any(((n / d) * (1 - state.sg) / state.sg) < 0):
                return None

            return inv_q1_q0(
                (n / d) * (1 - state.sg) / state.sg
            )

    def diff(gamma_a):
        '''
        difference in acceptance rates
        '''

        gamma_b = - mu[0] / mu[1] * gamma_a # (Eq. 136b)

        gamma = np.array([gamma_a, gamma_b]).reshape((2,))

        phi = perturbed(state, gamma)

        if phi is None:
            return gamma_a

        beta = setting.beta(phi, state)

        return beta[0] - beta[1]

    def u(phi):
        return (
            dot(     Q0(phi),  (V[0,0] * (1 - sg))) + # utility term for true negative
            dot((1 - Q0(phi)), (V[0,1] * (1 - sg))) + # utility term for false positive
            dot(     Q1(phi),  (V[1,0] * sg)) + # utility term for false negative
            dot((1 - Q1(phi)), (V[1,1] * sg)) # utility term for true positive
        )

    def u_gamma(gamma_a):
        '''
        # Utility function to maximize subject to DP
        '''

        with np.testing.suppress_warnings() as sup:

            sup.filter(RuntimeWarning, 'overflow encountered in double_scalars')

            gamma_b = - mu[0] / mu[1] * gamma_a # (Eq. 136b)

        gamma = np.array([gamma_a, gamma_b])
        phi = perturbed(state, gamma)

        return u(phi)


    # root finding with scipy library code
    try:
        gamma_a = opt.bisect(diff, -1.5, 1.5)

        # x = np.linspace(-1.5, 1.5, 500)
        # plt.plot(x, [u(a) for a in x], label='u')
        # plt.plot(x, [diff(a) for a in x], label='v')
        # plt.plot([-1, 1], [0, 0])
        # plt.plot([gamma_a, gamma_a], [0, 0.5], label='gamma_a')
        # plt.legend()
        # plt.show()

    except RuntimeError:
        print('RUNTIME ERROR in classifiers.py')
        x = np.linspace(-1.5, 1.5, 500)
        plt.plot(x, [u(a) for a in x], label='u')
        plt.plot(x, [diff(a) for a in x], label='v')
        plt.legend()
        plt.show()

    # Handle boundaries

    gamma_b = - mu[0] / mu[1] * gamma_a # (Eq. 136b)
    phi = perturbed(state, np.array([gamma_a, gamma_b]))

    utility = u_gamma(gamma_a)
    if u(5) >= utility:
        phi = 5 * np.ones(2)
    if u(-5) > utility:
        phi = -5 * np.ones(2)

    # we naturally have stricter standards than limited resources require

    # print(dot(setting.beta(phi, state), setting.mu))
    if dot(setting.beta(phi, state), setting.mu) < rate_limit:
        return phi

    # we must admit 'rate_limit' from each group. Find the thresholds that do this.
    return setting.inv_beta(rate_limit, state)

