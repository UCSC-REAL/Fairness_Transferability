#!/usr/bin/env python3

import os
import pickle

import numpy as np

from setting import Setting
from system import System

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plot_foreground_color = 'white'
foreground_color = 'black'
background_color = 'white'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Nimbus Roman',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.facecolor': background_color,
    'axes.labelcolor': foreground_color,
    'text.color': foreground_color,
    'xtick.color': foreground_color,
    'ytick.color': foreground_color,
})

import responses
import classifiers

################################################################################

# W_1 - W_0 vs phi
#
#           _____<___
#          /
# 0 ------s---------
#        /
#       /
# -->--'
#
# U01 > U11 > U10 > U00

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
assert s1.U[0,1] > s1.U[1,1] > s1.U[1,0] > s1.U[0,0]

################################################################################

def display(setting, response_func, res=64, strength_parameter=0.1, rate_limit=1, savename=None):
    '''
    setting: Setting object comprising parameters
    response_func: function from `responses` to model population response
    res: number of points per axis to calculate vector field
    strength_parameter: to use with feedback control
    rate_limit: max acceptance rate any classifier may allow
    savename: name to save file under in `images/` directory
    kwargs: forwarded to `plot.py` function `plot`
    '''

    if os.path.exists('s.p'):
        with open('s.p', 'rb') as f:
            s = pickle.load(f)

    else:
        s = System(
            'Demographic Parity', setting,
            classifiers.demographic_parity_bayes,
            response_func,
            res=res,
            cls_kwargs=dict(rate_limit=rate_limit)
        )

        s.calculate()

        with open('s.p', 'wb') as f:
            pickle.dump(s, f)

    vmax = np.max(np.array([
        np.abs(s.next_A1 - s.next_A2),
        s.bound1 + s.bound2
        ])
    )

    fig = plt.figure(figsize=(9, 3))
    axs = []
    axs.append(fig.add_subplot(1, 3, 1, projection='3d'))
    axs.append(fig.add_subplot(1, 3, 2, projection='3d'))
    axs.append(fig.add_subplot(1, 3, 3, projection='3d'))

    for ax in axs:
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel('Group 1 qualification rate $s_1$')
        # ax.set_zlabel('Differential DP Violation')
        ax.view_init(40, -80)


    # fig.subplots_adjust(hspace=1, wspace=1)
    axs[-1].set_ylabel('Group 2 qualification rate $s_2$')

    diff = ((s.bound1 + s.bound2) -  np.abs(s.next_A1 - s.next_A2))
    diff = np.where(
        diff < 0,
        0,
        diff
    )

    X, Y = np.meshgrid(s.x, s.y)

    axs[0].set_title('Modelled Shift')
    cs = axs[0].plot_surface(
        X, Y, np.abs(s.next_A1 - s.next_A2),
        cmap='viridis',
        # levels=np.array(np.linspace(colormin, colormax, 9)),
        # alpha=0.7,
        antialiased=True
    )


    axs[1].set_title('Theoretical Bound')
    cs = axs[1].plot_surface(
        X, Y, (s.bound1 + s.bound2),
        cmap='viridis',
        # levels=np.array(np.linspace(colormin, colormax, 9)),
        # alpha=0.8,
        antialiased=True
    )

    axs[2].set_title('Bound - Modelled Shift')
    cs = axs[2].plot_surface(
        X, Y, ((s.bound1 + s.bound2) -  np.abs(s.next_A1 - s.next_A2)),
        cmap='viridis',
        # levels=np.array(np.linspace(colormin, colormax, 9)),
        # alpha=0.8,
        antialiased=True
    )

    ############################################################################

    # display or save plot
    plt.tight_layout()
    if savename is None:
        plt.show()
    else:
        filename = f'{savename}.pdf'
        print('saving', filename)
        plt.savefig(filename, bbox_inches='tight')

display(s1, responses.replicator_equation, res=64, savename='transfer')
