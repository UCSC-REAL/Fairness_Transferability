#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from state import State
import responses
import util

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Nimbus Roman',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

class Setting:
    def __init__(self, mu, U, V, q0_mean=-1, q1_mean=1, sigma=1):
        '''
        Class for bundling parameters of setting.

        Assumes
        * Two groups

        * Gaussian q0, q1 with the same variance (to satisfy Assumption 3).

          This choice of functions merely gives us concrete mappings to and from
          the (real-valued) space of features (e.g., X, phi); all results are
          preserved by monotonic transformations of this space. For example,
          Coate and Loury use scores in the interval [0, 1], but the situation is
          identical: an appropriate mapping of score values yields the isomorphism.
          As noted by Mouzannar et. al. (2019), all that really matters are the
          observable statistics in the outcome (Y, hat{Y}) space.

        * Matrix representations for V, U (Eqns. 5, 9)

        Args:
            mu: numpy array of len (2)
            U: 2x2 numpy array - Strategy fitness matrix by outcome: U[Y, \hat{Y}]
                overloaded as T for markov transitions (see main.py: s_markov)
                overloaded for utility matrix for coate&loury model (see s_best_response)

            V: 2x2 numpy array - Classifier utility matrix by outcome: V[Y, \hat{Y}]
            q0_mean: float - mean of q0
            q1_mean: float - mean of q1 (must be greater than q0_mean)
            sigma: float - stdev of q0, q1
        '''

        # proportions of groups
        self.mu = mu

        assert np.sum(mu) == 1

        # Agent Fitness Matrix (Eq. 10)
        self.U = U # U[y, y_hat]

        # Classifier Utility Matrix (Eq. 5)
        self.V = V # V[y, y_hat]

        self.xi = (V[0,0] - V[0,1]) / (V[1,1] - V[1,0])

        # self.theta = (V[0,0] - V[0,1]) / (V[1,1] - V[1,0] + V[0,0] - V[0,1])

        assert q1_mean > q0_mean
        self.q1_mean = q1_mean
        self.q0_mean = q0_mean

        self.sigma = sigma

    def q0(self, x):
        return util.gaussian(x, self.q0_mean, self.sigma)

    def q1(self, x):
        return util.gaussian(x, self.q1_mean, self.sigma)

    def Q1(self, x):
        '''
        cumuative distribution function of q1 (Eq. 11)
        '''
        return util.gaussian_cdf(x, self.q1_mean, self.sigma)

    def Q0(self, x):
        '''
        cumulative distribution function of q0 (Eq. 11)
        '''
        return util.gaussian_cdf(x, self.q0_mean, self.sigma)

    def invQ1(self, y):
        '''
        invert Q1
        '''
        return util.gaussian_cdf_inv(y, self.q1_mu, self.sigma)

    def invQ0(self, y):
        '''
        invert Q0
        '''
        return util.gaussian_cdf_inv(y, self.q0_mean, self.sigma)


    def inv_q1_q0(self, y):
        '''
        inverse of q1(x)/q0(x) (Eq. 6)
        '''

        with np.testing.suppress_warnings() as sup:

            sup.filter(RuntimeWarning, 'invalid value encountered in log')

            def inv(y):
                n = 2 * np.log(y) + self.q1_mean ** 2 - self.q0_mean ** 2
                d = 2 * (self.q1_mean - self.q0_mean)
                return n / d

            # limit to machine precision
            return np.where(
                y <= np.finfo(float).eps,
                inv(np.finfo(float).eps),
                inv(y)
            )

    def plot_w1w0(self, savename=None):
        U = self.U
        Q1 = self.Q1
        Q0 = self.Q0
        inv_q1_q0 = self.inv_q1_q0

        x = np.linspace(-5, 5, 1000)

        def W1W0(x):
            return (
                (U[1,1] + (U[1,0] - U[1,1]) * Q1(x)) -
                (U[0,1] + (U[0,0] - U[0,1]) * Q0(x))
            )

        plt.figure(figsize=(3,3))

        plt.plot(x, W1W0(x), label='$W_1 - W_0$')
        plt.plot(x, x * 0) #, color='#e7e7e7')
        plt.plot(x, x * 0 + U[1,0] - U[0,0], label='$U_{1\hat{0}} - U_{0\hat{0}}$')
        plt.plot(x, x * 0 + U[1,1] - U[0,1], label='$U_{1\hat{1}} - U_{0\hat{1}}$')
        plt.xlabel(r'classifier threshold $\phi$')
        plt.title(r'$W_1(\phi) - W_0(\phi)$ (strictly quasi-concave)')
        plt.legend()

        if not savename:
            plt.show()
        else:
            filename = f'images/{savename}.pdf'
            print('saving', filename)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    def sanity_check(self):

        U = self.U
        Q1 = self.Q1
        Q0 = self.Q0
        inv_q1_q0 = self.inv_q1_q0

        # Set up figure(s)
        num_figs = 4
        f, axs = plt.subplots(2, num_figs // 2)

        try:
            iter(axs)
        except TypeError:
            axs = [axs]

        x = np.linspace(-10, 10, 1000)

        #--------------------------------------------
        # Plot q0, q1, Q0, Q1

        def q0(x):
            return util.gaussian(x, self.q0_mean, self.sigma)

        def q1(x):
            return util.gaussian(x, self.q1_mean, self.sigma)

        axs[0,0].plot(x, q0(x), label='$q_0$')
        axs[0,0].plot(x, q1(x), label='$q_1$')
        axs[0,0].plot(x, Q0(x), label='$Q_0$')
        axs[0,0].plot(x, Q1(x), label='$Q_1$')
        axs[0,0].set_xlabel('x')
        axs[0,0].set_title('Q0, Q1 vs x (Gaussian CDF)')
        axs[0,0].legend()

        #--------------------------------------------
        # Plot q1/q0 inv_q1_q0

        axs[1,1].plot(x, inv_q1_q0(q1(x) / q0(x)))
        axs[1,1].set_title('Checking correct inverse for $\\frac{q_1}{q_0}$')
        axs[1,1].set_xlabel('x')

        #--------------------------------------------
        # Plot s vs phi

        xx = np.linspace(0, 1, 100)

        # threshold as function of qualification rate
        def phi_of_s(s):
            V = self.V

            return inv_q1_q0(
                (
                    (V[0,0] - V[0,1]) / (V[1,1] - V[1,0])
                ) * (1 - s) / s
            )

        axs[1,0].plot(xx, phi_of_s(xx), label='Classifier Perception')
        axs[1,0].set_title('Monotonically decreasing')
        axs[1,0].set_ylabel('Feature Threshold $\\phi(s)$')
        axs[1,0].set_xlim((0, 1))
        axs[1,0].set_xlabel('Qualification Rate $s$')
        axs[1,0].legend()

        #--------------------------------------------
        # Plot (W1 - W0) vs phi

        def W1W0(x):
            return (
                (U[1,1] + (U[1,0] - U[1,1]) * Q1(x)) -
                (U[0,1] + (U[0,0] - U[0,1]) * Q0(x))
            )

        axs[0,1].plot(x, W1W0(x), label='$W_1 - W_0$')
        axs[0,1].plot(x, x * 0) #, color='#e7e7e7')
        axs[0,1].plot(x, x * 0 + U[1,0] - U[0,0], label='$U_{10} - U_{00}$')
        axs[0,1].plot(x, x * 0 + U[1,1] - U[0,1], label='$U_{11} - U_{01}$')
        axs[0,1].set_xlabel('classifier threshold phi')
        axs[0,1].set_title('W1 - W0 vs phi (strictly quasi-concave function)')
        axs[0,1].legend()

        plt.show()

    def beta(self, phi, state):
        '''
        Acceptance Rate per group

        Assumes two groups

        Args:
            phi: classifier's feature threshold (float)
            state: State object with only two groups

        Returns:
            Acceptance rate of group 1, Acceptance rate of group 2, ...
        '''

        Q0 = self.Q0
        Q1 = self.Q1

        return 1 - (state.sg * Q1(phi) + (1 - state.sg) * Q0(phi))

    def inv_beta(self, target_rate, state):
        '''
        Find the threshold phi for the target acceptance rate in each group.
        '''

        Q0 = self.Q0
        Q1 = self.Q1

        phi = []
        for s in state.sg:

            def violation(phi_g):

                return ((1 - (s * Q1(phi_g) + (1 - s) * Q0(phi_g))) - target_rate) ** 2

            res = minimize(violation, 0)
            phi.append(res.x[0])

        return np.array(phi)

    def fpr(self, phi, state):
        '''
        False Positive Rate per group
        (predicted qualified given not qualified)

        Assumes only two groups

        Args:
            phi: classifier threshold (float)
            state: State object with only two groups

        Returns:
            False positive rate of group 1, False positive rate of group 2
        '''

        assert state.n == 2

        Q0 = self.Q0

        # Q0 is rejection rate for unqualified individuals
        # (1 - Q0) is acceptance for unqualified individuals (false positive)
        return (
            (1 - Q0(phi[0])),
            (1 - Q0(phi[1]))
        )

    def fnr(self, phi, state):
        '''
        False Negative Rate per group
        (predicted unqualified given qualified)

        Assumes only two groups

        Args:
            phi: classifier threshold (float)
            state: State object with only two groups

        Returns:
            False negative rate of group 1, False negative rate of group 2
        '''

        assert state.n == 2

        Q1 = self.Q1

        # Q1 is rejection rate for qualified individuals (false negative)
        return (
            Q1(phi[0]),
            Q1(phi[1])
        )

    def ppr(self, phi, state):
        '''
        (actually qualified given predicted qualified)
        '''
        Q1 = self.Q1

        return state.sg * (1 - Q1(phi)) / self.beta(phi, state)

    def inv_ppr(self, target_ppr, state):
        '''
        find phi that yields target ppr for each group
        '''

        if not 0 < target_ppr < 1:
            assert False

        Q0, Q1 = self.Q0, self.Q1
        q0_mean, q1_mean = self.q0_mean, self.q1_mean

        # vectorized binary search

        l = -5 * self.sigma - q0_mean * np.ones(2)
        r = 5 * self.sigma + q1_mean * np.ones(2)

        # x = np.linspace(-5, 5, 100)
        # plt.plot(x, [self.ppr(a, state) for a in x])
        # plt.show()

        phi = np.array([0.0, 0.0])
        ppr = self.ppr(phi, state)

        while sum(np.abs(ppr - target_ppr)) > 0.00001:

            for i in [0, 1]:
                if ppr[i] < target_ppr:
                    l[i] = phi[i]
                    phi[i] = (phi[i] + r[i]) / 2
                else:
                    r[i] = phi[i]
                    phi[i] = (phi[i] + l[i]) / 2

            ppr = self.ppr(phi, state)

        return phi

    def avg_fitness(self, phi, state):
        '''
        Average Fitness per group

        Assumes only two groups

        Args:
            phi: classifier threshold (float)
            state: State object with only two groups

        Returns:
            fitness of group 1, fitness of group 2
        '''

        assert state.n == 2

        Q0 = self.Q0
        Q1 = self.Q1
        U = self.U

        # group 1
        W1g1 = U[1,1] + (U[1,0] - U[1,1]) * Q1(phi[0]) # avg fitness of qualified
        W0g1 = U[0,1] + (U[0,0] - U[0,1]) * Q0(phi[0]) # avg fitness of unqualified

        # group 2
        W1g2 = U[1,1] + (U[1,0] - U[1,1]) * Q1(phi[1]) # avg fitness of qualified
        W0g2 = U[0,1] + (U[0,0] - U[0,1]) * Q0(phi[1]) # avg fitness of unqualified

        return (
            (state[0] * W1g1 + (1 - state[0]) * W0g1),
            (state[1] * W1g2 + (1 - state[1]) * W0g2)
        )

    def copy(self, **kwargs):
        default = {
            'mu': self.mu,
            'U': self.U,
            'V': self.V,
            'q0_mean': self.q0_mean,
            'q1_mean': self.q1_mean,
            'sigma': self.sigma
        }
        return Setting(**(default | kwargs))

if __name__ == "__main__":
    s1 = Setting(
        mu=np.array([0.5, 0.5]),
        U=np.array([
            [0.1, 5.5],
            [0.5, 1.0]
        ]),
        V=np.array([
            [0.5, -0.5],
            [-0.25, 1.0]
        ])
    )

    x = np.linspace(-5, 5, 100)
    # s1.q1
    # inv_q1_q0(self, x) =
