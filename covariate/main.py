import os
import pickle

import itertools

import numpy as np
import pandas as pd

from scipy import optimize as opt
from opt_einsum import contract

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Nimbus Roman',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

################################################################################

epsilon = 0.001

def dot(a, b):
    '''dot product'''
    return np.sum(a * b)


# G = Group (race, in this dataset)
# X = Credit score. 0-100 in source data. We divide by 100 for range 0-1
# Y = Whether loan was repaid. Y=0 indicates default on a loan

# We must represent (cumulative) probability (density) functions as datastructures.
# As a first approximation, we simply store them like a histogram, in which
# the data is binned by G (treated as a dictionary key), then by intervals in X.
# Distributions represented in this fashion are named with a leading underscore.

# _example[group_string][X_val]

# We distinguish cumulative distribution functions from density functions by
# naming the former with capitalized variables and the latter with lowercase

# Pr(G = g)
mu = pd.read_csv('totals.csv', index_col=0)
mu = (mu / mu.sum(axis=1)['SSA']).to_numpy()


################################################################################
#//  _        _           _       _   _
#// | |_ __ _| |__  _   _| | __ _| |_(_) ___  _ __  ___
#// | __/ _` | '_ \| | | | |/ _` | __| |/ _ \| '_ \/ __|
#// | || (_| | |_) | |_| | | (_| | |_| | (_) | | | \__ \
#//  \__\__,_|_.__/ \__,_|_|\__,_|\__|_|\___/|_| |_|___/
#//

def get_density_from_cdf(df):
    '''
    Treats each bin

    assuming that df represents Pr(X <= x | G =g)
    '''

    density = {}

    for g in df.columns:
        out = []
        prev = 0

        for xi, x in enumerate(df.index):

            if (xi == 0):
                out.append(df[g][x])

            else:
                out.append(df[g][x] - df[g][prev_x])

            prev_x = x

        density[g] = out

    out_df = pd.DataFrame(density, index=df.index)

    # normalize and return
    return out_df / out_df.sum()

# Pr(Y = 0 | G = g, X = x)
_y0_xg = pd.read_csv('default_by_race.csv', index_col=0) / 100 # convert from percentage

# Pr(X <= x | G = g)
_X_g = pd.read_csv('cdf_by_race.csv', index_col=0) / 100 # convert from percentage

# G values in the source data
groups = list(_X_g.columns)

# X values in source data
idx = _X_g.index
x_min = idx.min()
x_max = idx.max()

# ensure columns and indices agree for imported data
assert all(_X_g.columns == _y0_xg.columns)
assert all(idx == _y0_xg.index)

# p(x | G = g)
_x_g = get_density_from_cdf(_X_g)

# p(Y = 1, X = x | G = g) = p(x | G) * Pr(Y = 1 | G = g, X = x)
_xy1_g = _x_g * (1 - _y0_xg)

# = p(Y = 0, X = x | G = g) = p(x | G) * Pr(Y = 0 | G = g, X = x)
_xy0_g = _x_g * _y0_xg

# p(X <= x, Y = 0 | G = g)
_Xy0_g = _xy0_g.cumsum()

# p(X <= x, Y = 1 | G = g)
_Xy1_g = _xy1_g.cumsum()

################################################################################

def discretize(phi):
    '''
    Get first x value such that phi <= x
    '''
    return idx[np.digitize(phi, idx, right=True)]

def tp_phi(phi, _Xy1_g=_Xy1_g):
    '''
    True positive FRACTION given per-group X-threshold classifiers
    e.g., phi = np.array([0.5, 0.4, 0.6, 0.3])

    Pr(Y_hat = 1, Y = 1 | G = g)
    '''

    phi = discretize(phi)

    _tp = []
    for i, p in enumerate(phi):

        g = groups[i]

        _tp.append(
            # Pr( Y = 1, X > phi | g )
            _Xy1_g[g][x_max] - _Xy1_g[g][p]
        )

    return np.array(_tp)

def fn_phi(phi, _Xy1_g=_Xy1_g):
    '''
    False negative FRACTION given per-group X-threshold classifiers
    e.g., phi = np.array([0.5, 0.4, 0.6, 0.3])

    Pr(Y_hat = -1, Y = 1 | G = g)
    '''

    phi = discretize(phi)

    _fn = []
    for i, p in enumerate(phi):
        g = groups[i]

        _fn.append(
            # Pr( Y = 1, X < phi | g )
            _Xy1_g[g][p]
        )

    return np.array(_fn)

def ac_phi(phi, _X_g=_X_g):
    '''
    Acceptance rates given per-group X-threshold classifiers
    e.g., phi = np.array([0.5, 0.4, 0.6, 0.3])

    Pr( Y_hat = 1 | G = g )
    '''

    phi = discretize(phi)

    return np.array([
        (1 - _X_g[groups[i]][p]) for i, p in enumerate(phi)
    ])

def fnr_phi(phi, _Xy1_g=_Xy1_g):
    '''
    False negative RATE given per-group X-threshold classifiers
    e.g., phi = np.array([0.5, 0.4, 0.6, 0.3])

    Pr( Y_hat = 0 | Y = 1, G = g )
    '''

    phi = discretize(phi)

    _fnr = []
    for i, p in enumerate(phi):
        g = groups[i]

        _fnr.append(
            # Pr( Y_hat = 0, Y = 1 | g ) / Pr( Y = 1 | g )
            _Xy1_g[g][p] / _Xy1_g[g][x_max]
        )

    return np.array(_fnr)

def fpr_phi(phi, _Xy0_g=_Xy0_g):
    '''
    False positive RATE given per-group X-threshold classifiers
    e.g., phi = np.array([0.5, 0.4, 0.6, 0.3])

    Pr( Y_hat = 1 | Y = 0, G = g )
    '''

    phi = discretize(phi)

    _fpr = []
    for i, p in enumerate(phi):
        g = groups[i]

        _fpr.append(
            # 1 - [Pr( Y_hat = 0, Y = 0 | g ) / Pr( Y = 0 | g )]
            1 - ( _Xy0_g[g][p] / _Xy0_g[g][x_max] )
        )

    return np.array(_fpr)


################################################################################
#//       _               _  __ _
#//   ___| | __ _ ___ ___(_)/ _(_) ___ _ __
#//  / __| |/ _` / __/ __| | |_| |/ _ \ '__|
#// | (__| | (_| \__ \__ \ |  _| |  __/ |
#//  \___|_|\__,_|___/___/_|_| |_|\___|_|
#//

def phi_ac(target_ac, _X_g=_X_g):
    '''
    Get X-thresholds for given target acceptance rate in each group
    e.g., target_ac = np.array([0.5, 0.4, 0.6, 0.3]) such that the actual
    acceptance rate is >= the target acceptance rate.
    '''

    # vectorized binary search
    l = np.ones(len(groups)) * x_min
    r = np.ones(len(groups)) * x_max
    c = (l + r) / 2

    current = ac_phi(c, _X_g=_X_g)
    while any( np.abs(current - target_ac) > epsilon ):
        l = np.where(
            current > target_ac, # accepting too many; threshold too low
            c,
            l
        )
        r = np.where(
            current < target_ac, # must accept more; lower threshold
            c,
            r
        )
        c = (l + r) / 2
        update = ac_phi(c, _X_g=_X_g)
        if all(update == current):
            break
        else:
            current = update

    return c

def phi_fpr(target_fpr, _Xy0_g=_Xy0_g):
    '''
    Get X-thresholds for given target false positive rate in each group
    e.g., target_fpr = np.array([0.5, 0.4, 0.6, 0.3])
    '''

    # vectorized binary search
    l = np.ones(len(groups)) * x_min
    r = np.ones(len(groups)) * x_max
    c = (l + r) / 2

    current = fpr_phi(c, _Xy0_g=_Xy0_g)
    while any( (np.abs(current - target_fpr) > epsilon) & ((r - l) > epsilon) ):
        l = np.where(
            current > target_fpr, # too many false positives; raise threshold
            c,
            l
        )
        r = np.where(
            current < target_fpr, # not enough false positives; lower threshold
            c,
            r
        )
        c = (l + r) / 2
        update = fpr_phi(c, _Xy0_g=_Xy0_g)
        if all(update == current):
            break
        else:
            current = update

    return c

def phi_fnr(target_fnr, _Xy1_g=_Xy1_g):
    '''
    Get X-thresholds for given target false negative rate in each group
    e.g., target_fnr = np.array([0.5, 0.4, 0.6, 0.3])
    '''

    # vectorized binary search
    l = np.ones(len(groups)) * x_min
    r = np.ones(len(groups)) * x_max
    c = (l + r) / 2

    current = fnr_phi(c, _Xy1_g=_Xy1_g)
    while any( (np.abs(current - target_fnr) > epsilon) & ((r - l) > epsilon) ):
        l = np.where(
            current < target_fnr, # not enough false negatives; raise threshold
            c,
            l
        )
        r = np.where(
            current > target_fnr, # too many false negatives; lower threshold
            c,
            r
        )
        c = (l + r) / 2
        update = fnr_phi(c, _Xy1_g=_Xy1_g)
        if all(update == current):
            break
        else:
            current = update

    return c


################################################################################
#//                  _ _                _               _
#//  ___  __ _ _ __ (_) |_ _   _    ___| |__   ___  ___| | __
#// / __|/ _` | '_ \| | __| | | |  / __| '_ \ / _ \/ __| |/ /
#// \__ \ (_| | | | | | |_| |_| | | (__| | | |  __/ (__|   <
#// |___/\__,_|_| |_|_|\__|\__, |  \___|_| |_|\___|\___|_|\_\
#//                        |___/

# x = np.linspace(0, 1, 30)

# plt.title('checking monotonicity')
# plt.plot(x, [phi_ac(a) for a in x], label=groups)
# plt.plot([ac_phi(a * np.ones(len(groups))) for a in x * 100], x * 100, label=groups)
# plt.xlabel('Acceptance Rate')
# plt.ylabel('Threshold')
# plt.legend()
# plt.show()

# plt.title('checking inverse functions')
# plt.plot(x, [phi_fnr(a) for a in x], label=[f'fnr {g}' for g in groups])
# plt.plot(x, [phi_fpr(a) for a in x], label=[f'fpr {g}' for g in groups])
# plt.plot([fnr_phi(a * np.ones(len(groups))) for a in x * 100], x * 100,
#          label=[f'fnr {g}' for g in groups])
# plt.plot([fpr_phi(a * np.ones(len(groups))) for a in x * 100], x * 100,
#          label=[f'fpr {g}' for g in groups])
# plt.xlabel('FPR or FNR')
# plt.ylabel('Threshold')
# plt.legend()
# plt.show()

# plt.title('checking monotonicity')
# plt.plot(x, [fnr_phi(phi_ac(a)) for a in x], label=[f'fnr {g}' for g in groups])
# plt.plot(x, [fpr_phi(phi_ac(a)) for a in x], label=[f'fpr {g}' for g in groups])
# plt.xlabel('Acceptance Rate')
# plt.ylabel('FPR or FNR')
# plt.legend()
# plt.show()

################################################################################

# # Assume everyone applies for loans of equal amounts.
# # All defaults yeild no profit
# # All repaid loans yield the same profit (No partial repayments)
# # Utility is maximized proportional to
# # \sum_{y,y_hat} V_{y, y_hat} * Pr(Y_hat = y_hat | Y = y)

# # classifier utility per outcome
# V=np.array([
#     [0.5, -0.5], # tn, fp
#     [-0.25, 1.0] # fn, tp
# ])

# def u_phi(phi, V=V):
#     '''
#     Classifier utility for given group-specific thresholds
#     '''

#     _fpr = fpr_phi(phi)
#     _fnr = fnr_phi(phi)
#     u = 0
#     u += dot(mu, (1 - _fpr)) * V[0,0] # tnr
#     u += dot(mu, _fpr)       * V[0,1] # fpr
#     u += dot(mu, _fnr)       * V[1,0] # fnr
#     u += dot(mu, (1 - _fnr)) * V[1,1] # tpr
#     return u

# def dp_thresholds(_X_g=_X_g, V=V):
#     '''
#     Get Demographic Parity thresholds.
#     Maximize utility, sweep over group-independent acceptance rates
#     '''

#     def obj(ac):
#         '''negative utility as a function of acceptance rate'''
#         return -u_phi(phi_ac(ac, _X_g=_X_g), V=V)

#     sol = opt.minimize_scalar(
#         obj, # objective function
#         bounds=(0, 1),
#         method='bounded'
#     )

#     return phi_ac(sol.x, _X_g=_X_g)

# x = np.linspace(0, 1, 200)
# plt.plot(x, [-u_phi(phi_ac(a)) for a in x])
# plt.xlabel('Group-independent Acceptance Rate')
# plt.ylabel('Negative Utility (to minimize)')
# plt.show()

# def eo_thresholds(_Xy1_g=_Xy1_g, V=V):
#     '''
#     Get Equal Opportunity thresholds
#     Maximize utility, sweep over group-independent false negative rates
#     (true positive rates)
#     '''

#     def obj(fnr):
#         '''negative utility as a function of acceptance rate'''
#         return -u_phi(phi_fnr(fnr, _Xy1_g=_Xy1_g), V=V)

#     sol = opt.minimize_scalar(
#         obj, # objective function
#         bounds=(0, 1),
#         method='bounded'
#     )

#     return phi_fnr(sol.x, _Xy1_g=_Xy1_g)

# x = np.linspace(0, 1, 200)
# plt.plot(x, [-u_phi(phi_fnr(a)) for a in x])
# plt.xlabel('Group-independent False Negative Rate')
# plt.ylabel('Negative Utility (to minimize)')
# plt.show()

################################################################################
#//                           _       _
#//   ___ _____   ____ _ _ __(_) __ _| |_ ___
#//  / __/ _ \ \ / / _` | '__| |/ _` | __/ _ \
#// | (_| (_) \ V / (_| | |  | | (_| | ||  __/
#//  \___\___/ \_/ \__,_|_|  |_|\__,_|\__\___|
#//      _     _  __ _
#//  ___| |__ (_)/ _| |_
#// / __| '_ \| | |_| __|
#// \__ \ | | | |  _| |_
#// |___/_| |_|_|_|  \__|
#//

def p_manipulate(x, phi, theta):
    '''

    Probability of manipulation given feature x, threshold phi

    probability of manipulating ~= 1 - |x' - x| * theta to left of threshold

    theta = 100000 => nobody manipulates;
    theta = 0.0001 => everyone manipulates;

    Args:
        x: array of indices to calculate
    '''
    return np.where(
        x > phi,
        0,
        np.maximum(0, 1 - (phi - x) * theta)
    )

def manipulation_resamples_from(x, x_new, phi, theta):
    '''
    Given a manipulator with original feature x, its new feature
    will be sampled from the following density function:

    Pr(X_new=x_new | X=x, Manip=True, G=g)

    (uniform b/w threshold and max of domain)

    x is original feature
    x_new is new feature
    phi is decision threshold
    1/theta is manipulation budget

    Resulting distribution is renormalized after this function is called
    '''

    x = np.array(x).reshape(len(x), 1, 1)
    x_new = np.array(x_new).reshape(1, len(x_new), 1)
    phi = np.array(phi).reshape(1, 1, len(phi))

    # depends on original feature
    ub = np.maximum(phi, np.minimum(x_max, phi + (1 / theta)))

    zero = np.zeros((len(x), len(x_new), len(phi)))
    one = np.ones((len(x), len(x_new), len(phi)))

    # indexed by X, then X_new, then G
    dist = np.where(
        x_new < phi,
        zero,
        np.where(
            x_new > ub,
            zero,
            one
        )
    )
    s = contract('xng->xg', dist) # sum over x_new for normalization

    # Pr(X_new=x_new | X=x, Manip=True, G=g)
    return np.where(
        s == 0,
        zero,
        dist / s
    )

def eo_dot(a, b, _y0_xg=_y0_xg):
    '''
    dot product

    a, b are dataframes with matching columns and index to _y0_xg
    '''
    return ((1 - _y0_xg) * a * b).sum().to_numpy()

def covariate_shift(phi, _x_g=_x_g):
    '''
    Given classifier X-thresholds, what is realized version of _x_g,
    _x_g_R?

    Uses p_manipulate

    Returns function of theta, a shift parameter
    '''

    def of_theta(theta):

        # Pr(Y_hat = 0 -> Y_hat = 1 | X = x, G = g)
        # probability of manipulation
        # as numpy array indexed by X, then G
        manip = np.array([p_manipulate(x, phi, theta) for x in idx])

        array_x_g = _x_g.to_numpy() # X, G

        # Pr(X_new=x_new | X=x, Manip=True)
        # indexed by X, then X_new, then G
        resample = manipulation_resamples_from(idx, idx, phi, theta)

        # Pr(X_new=x_new, Manip=True | G=g)
        # = sum_x Pr(X_new=x_new | X=x, Manip=True, G=g) Pr(Manip=True | X=x, G) Pr(X=x | G=g)
        resample_dist = contract('xng,xg,xg->ng', resample, manip, _x_g)

        # indexed by X, G
        return _x_g * (1 - manip) + resample_dist

    return of_theta

################################################################################
#//  _____                  _
#// | ____|__ _ _   _  __ _| |
#// |  _| / _` | | | |/ _` | |
#// | |__| (_| | |_| | (_| | |
#// |_____\__, |\__,_|\__,_|_|
#//          |_|
#//   ___                         _               _ _
#//  / _ \ _ __  _ __   ___  _ __| |_ _   _ _ __ (_) |_ _   _
#// | | | | '_ \| '_ \ / _ \| '__| __| | | | '_ \| | __| | | |
#// | |_| | |_) | |_) | (_) | |  | |_| |_| | | | | | |_| |_| |
#//  \___/| .__/| .__/ \___/|_|   \__|\__,_|_| |_|_|\__|\__, |
#//       |_|   |_|                                     |___/



def div_eo(_x_g, _x_g_R, _y0_xg=_y0_xg):
    '''
    Get divergence b/w distributions _x_g and _x_g_R, as defined for Equal opportunity
    analysis and bounds
    '''

    # Divergence of distributions for EO case
    eo_div2 = np.maximum(
        0, # the negative values are caused by floating point round errors
        (
            eo_dot(_x_g, _x_g, _y0_xg=_y0_xg) +
            eo_dot(_x_g_R, _x_g_R, _y0_xg=_y0_xg) -
            2 * eo_dot(_x_g, _x_g_R, _y0_xg=_y0_xg)
        )
    )

    return np.sqrt(eo_div2)

def eo_violation(tpr):
    '''
    Sum of group-pairwise differences is tpr

    tpr is array (one entry per group)
    '''
    x = tpr.reshape((1, len(tpr)))
    return np.sum(np.abs(x - x.T)) / 2

def bound_eo(phi, _x_g=_x_g, _y0_xg=_y0_xg):
    '''
    (assumes that policy is fair on source distribution)

    First, get maximum possible change in TPR per group
    as a function of the group-divergence bound B_g
    between _r = _x_g and _r_R = _x_g_R.

    Project ball of radius B_g around r vector to 1-t plane,
    extremize ratio of angles,
    convert to extremized TPR

    Return a bound on maximum possible violation of equal opportunity
    accounting for changes in all groups AS A FUNCTION of Bg.
    '''

    _r = _x_g
    _s = _y0_xg

    _s_inv = 1/_y0_xg

    # pr(Yhat = 1 | x, g)
    # depends on phi
    _t = pd.DataFrame({
        g: np.array([(1 if x >= phi[gi] else 0) for x in idx])
        for gi, g in enumerate(groups)
    }, index=idx)

    # before shift
    _tp = eo_dot(_t, _r) # = tp_phi(phi)
    _fn = eo_dot(1 - _t, _r) # = fn_phi(phi)

    norm_t = np.sqrt(eo_dot(_t, _t))
    norm_1 = np.sqrt(eo_dot(1, 1))
    norm_r = np.sqrt(eo_dot(_r, _r))
    norm_s_inv = np.sqrt(eo_dot(_s_inv, _s_inv))

    # unit vectors
    unit_t = _t / norm_t
    unit_r = _r / norm_r
    unit_1 = 1 / norm_1
    unit_s_inv = _s_inv / norm_s_inv


    basis_t = unit_t

    # projection of 1 in (1-t) plane, orthogonal to t
    # a la Gram-Schmidt
    ortho = 1 - unit_t * eo_dot(1, unit_t)
    # as unit vector
    basis_1 = ortho / np.sqrt(eo_dot(ortho, ortho))

    # check is close to zero
    # print(eo_dot(basis_t, ortho))

    # check is close to one
    # print(eo_dot(basis_1, basis_1))

    # project r into 1-t plane
    r_proj = (
        basis_t * eo_dot(_r, basis_t) +
        basis_1 * eo_dot(_r, basis_1)
    )
    norm_r_proj = np.sqrt(eo_dot(r_proj, r_proj))
    unit_r_proj = r_proj / norm_r_proj

    # should be the same
    # print(eo_dot(r_proj, basis_t))
    # print(eo_dot(_r, basis_t))

    # should be the same
    # print(eo_dot(r_proj, basis_1))
    # print(eo_dot(_r, basis_1))

    # should be the same
    # print(eo_dot(unit_r, _t) / eo_dot(unit_r, unit_1) )
    # print(eo_dot(unit_r_proj, _t) / eo_dot(unit_r_proj, unit_1) )

    # should be the same
    # print(eo_dot(unit_r, unit_t) / eo_dot(unit_r, unit_1) )
    # print(eo_dot(unit_r_proj, unit_t) / eo_dot(unit_r_proj, unit_1) )

    # angle between t and r_proj
    a_rt_proj = np.arccos( eo_dot(unit_r_proj, unit_t) )

    # angle between r_proj and 1
    a_r1_proj = np.arccos( eo_dot(unit_r_proj, unit_1) )

    # print(
    #     np.cos(a_rt_proj) / np.cos(a_r1_proj) * norm_t / norm_1 - (_tp / (_tp + _fn))
    # )

    # all three should be the same
    # print(_tp / (_tp + _fn))
    # print(eo_dot(_r, _t) / eo_dot(_r, 1) )
    # print(np.cos(a_rt_proj) / np.cos(a_r1_proj) * norm_t / norm_1 )

    # This vector, dotted with _s_inv, is 1
    _s_inv_bias = _s_inv / norm_s_inv ** 2

    # project s_inv to t-1 plane
    s_inv_proj = (
        basis_t * eo_dot(_s_inv, basis_t) +
        basis_1 * eo_dot(_s_inv, basis_1)
    )
    norm_s_inv_proj = np.sqrt(eo_dot(s_inv_proj, s_inv_proj))
    unit_s_inv_proj = s_inv_proj / norm_s_inv_proj

    # direction of semi-major axis of projected ellipse
    # is rotated 90 deg relative to projection of s_inv
    unit_aa = (
        basis_t * eo_dot(unit_s_inv_proj, basis_1)
        - basis_1 * eo_dot(unit_s_inv_proj, basis_t)
    )

    # direction of semi-minor axis of projected ellipse
    unit_bb = (
        basis_t * eo_dot(unit_s_inv_proj, basis_t) +
        basis_1 * eo_dot(unit_s_inv_proj, basis_1)
    )

    # check is close to zero
    # print(eo_dot(unit_aa, unit_bb))

    # check is close to one
    # print(eo_dot(unit_aa, unit_aa))

    # check is close to one
    # print(eo_dot(unit_bb, unit_bb))

    unit_bb = unit_bb

    def of_bg(bg):
        '''
        Get extremal possible values of TPR as function of budget Bg

        We can extremize difference in ratios of angles in the 1-t plane, because
        the projection to the 1-t plane is all that matters.
        '''

        # semi major axis
        aa = unit_aa * bg

        # semi minor axis
        bb = unit_bb * eo_dot(unit_s_inv_proj, unit_s_inv) * bg

        def to_minimize(th):

            # point on ellipse as function of angle th in t-1 plane
            p = aa * np.cos(th) + bb * np.sin(th) + r_proj

            norm_p = np.sqrt(eo_dot(p, p))

            # maximize sum of angles away from center of ellipse as seen from origin for each group
            return np.sum(-np.arccos(eo_dot(p / norm_p, unit_r_proj)))

        # find angles domains over which to maximize separately
        # outputs from -pi to pi
        split_th = np.arctan2(eo_dot(unit_bb, unit_r_proj), eo_dot(unit_aa, unit_r_proj))

        bounds = np.vstack((split_th, split_th + np.pi))
        x0 = bounds.sum(axis=0) / 2

        sol1 = opt.minimize(
            to_minimize, # objective function
            x0=x0,
            bounds=np.transpose(bounds),
            method='Nelder-Mead'
        ).x

        bounds = np.vstack((split_th - np.pi, split_th))
        x0 = bounds.sum(axis=0) / 2

        sol2 = opt.minimize(
            to_minimize, # objective function
            x0=x0,
            bounds=np.transpose(bounds),
            method='Nelder-Mead'
        ).x

        p1 = aa * np.cos(sol1) + bb * np.sin(sol1) + r_proj
        p2 = aa * np.cos(sol2) + bb * np.sin(sol2) + r_proj

        # from origin
        unit_p1 = p1 / np.sqrt(eo_dot(p1, p1))
        unit_p2 = p2 / np.sqrt(eo_dot(p2, p2))

        # Minimum possible TPR, maximum possible TPR
        # recall all angles are bounded between 0 and pi/2

        cos_p1_t = eo_dot(unit_p1, unit_t)
        cos_p1_1 = eo_dot(unit_p1, unit_1)

        cos_p2_t = eo_dot(unit_p2, unit_t)
        cos_p2_1 = eo_dot(unit_p2, unit_1)

        b1 = cos_p1_t / cos_p1_1 * norm_t / norm_1
        b2 = cos_p2_t / cos_p2_1 * norm_t / norm_1

        extrema = (
            np.minimum(
                np.maximum(
                    np.minimum(
                        b1, b2
                    ),
                    0
                ),
                1
            ),
            np.maximum(
                np.minimum(
                    np.maximum(
                        b2, b2
                    ),
                    1
                ),
                0
            )
        )

        for gi, g in enumerate(groups):

            if cos_p1_t[gi] < 0:
                extrema[0][gi] = 0
                extrema[1][gi] = 1

            if cos_p1_1[gi] < 0:
                extrema[0][gi] = 0
                extrema[1][gi] = 1

            if cos_p2_t[gi] < 0:
                extrema[0][gi] = 0
                extrema[1][gi] = 1

            if cos_p2_1[gi] < 0:
                extrema[0][gi] = 0
                extrema[1][gi] = 1

        bound = 0

        # strategic response covariate shift implies TPR strictly
        # increases if Pr(Y=1 | X) is monotonic in X, but, in general
        # we will use the adversarial bound, which requires
        # computation of all possible combinations
        n = len(groups)
        for i in range(n):
            tpr = np.array([extrema[int(x)][j] for j, x in enumerate(bin(i)[2:].zfill(n))])
            bound = max(bound, eo_violation(tpr))

        return bound

    return of_bg

# Verifying that extremal angles extremize TPR
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# for angle_sum in tqdm(np.linspace(0, np.pi/2, 30)):
#     ratios=[]
#     thetas=[]
#     for theta in np.linspace(angle_sum - np.pi/2 + 0.1, np.pi/2, 30):
#         ratios.append(np.cos(theta) / np.cos(angle_sum - theta))
#         thetas.append(theta)
#     ax.set_xlabel('angle_sum')
#     ax.set_ylabel('numerator angle')
#     ax.set_zlabel('cos ratio')
#     ax.scatter([angle_sum], thetas, ratios)
# plt.show()

m = 10
k = m

tpr_range = np.linspace(0.05, 0.8, m)
th_range = np.linspace(0, 0.1, k)


if not os.path.exists('max_bgs'):

    source_tpr = [tpr_range]
    max_bgs = [np.zeros(len(tpr_range))]

    # change in fairness violation
    strategic_manipulation = [np.zeros(len(tpr_range))]
    upper_bound = [np.zeros(len(tpr_range))]

    # possible tpr values
    for tpr in tqdm(tpr_range):

        eo_phi = phi_fnr(1 - tpr)

        shift_of_theta = covariate_shift(eo_phi)
        bound_of_bg = bound_eo(eo_phi)

        violation = []
        _max_bgs = []
        bound = []

        # observed by strategic classification with
        # possible theta values
        for th in th_range:
            _x_g_R = shift_of_theta(th)

            # plt.plot(idx, _x_g[groups[0]], color='blue')
            # plt.plot(idx, _x_g_R[groups[0]], color='red')
            # plt.show()

            # covariate shift assumes _y0_xg does not change
            # _y0_xg = Pr(Y=0 | X=x, G=g)
            _Xy1_g_R = ((_x_g_R * (1 - _y0_xg)).cumsum())

            R_tp = tp_phi(eo_phi, _Xy1_g=_Xy1_g_R)
            R_fn = fn_phi(eo_phi, _Xy1_g=_Xy1_g_R)
            R_tpr = R_tp / (R_tp + R_fn)

            violation.append(eo_violation(R_tpr))
            bg = div_eo(_x_g, _x_g_R)
            max_bg = max(bg)

            _max_bgs.append(max_bg)

            bound.append(bound_of_bg(max_bg * np.ones(len(groups))))

        max_bgs.append(_max_bgs)
        upper_bound.append(bound)
        strategic_manipulation.append(violation)
        source_tpr.append(tpr * np.ones(k))

        # ax.plot(max_bgs, tpr * np.ones(k), violation, linestyle='dotted', color='blue')
        # ax.plot(max_bgs, tpr * np.ones(k), bound, linestyle='dotted', color='black')

    source_tpr = np.array(source_tpr).flatten()
    strategic_manipulation = np.array(strategic_manipulation).flatten()
    upper_bound = np.array(upper_bound).flatten()
    max_bgs = np.array(max_bgs).flatten()

    with open('source_tpr', 'wb') as f:
        pickle.dump(source_tpr, f)
    with open('strategic_manipulation', 'wb') as f:
        pickle.dump(strategic_manipulation, f)
    with open('upper_bound', 'wb') as f:
        pickle.dump(upper_bound, f)
    with open('max_bgs', 'wb') as f:
        pickle.dump(max_bgs, f)
else:
    with open('source_tpr', 'rb') as f:
        source_tpr = pickle.load(f)
    with open('strategic_manipulation', 'rb') as f:
        strategic_manipulation = pickle.load(f)
    with open('upper_bound', 'rb') as f:
        upper_bound = pickle.load(f)
    with open('max_bgs', 'rb') as f:
        max_bgs = pickle.load(f)

######
# Plotting

fig = plt.figure(figsize=(9,4))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(max_bgs, source_tpr, strategic_manipulation, alpha=0.7, antialiased=True)
ax.plot_trisurf(max_bgs, source_tpr, upper_bound, alpha=0.4, antialiased=True, cmap='viridis')
ax.view_init(20, -65)
ax.set_xlabel('Max observed group shift')
ax.set_ylabel('Source TPR')
ax.set_zlabel('Violation of Equal Opportunity')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_trisurf(max_bgs, source_tpr, strategic_manipulation, alpha=0.7, antialiased=True)
ax.plot_trisurf(max_bgs, source_tpr, upper_bound, alpha=0.4, antialiased=True, cmap='viridis')
ax.view_init(20, -65 - 4)
ax.set_xlabel('Max observed group shift')
ax.set_ylabel('Source TPR')
ax.set_zlabel('Violation of Equal Opportunity')

plt.tight_layout()
plt.savefig('EO_bound.pdf', bbox_inches='tight')

#####

#//  ____                                             _     _
#// |  _ \  ___ _ __ ___   ___   __ _ _ __ __ _ _ __ | |__ (_) ___
#// | | | |/ _ \ '_ ` _ \ / _ \ / _` | '__/ _` | '_ \| '_ \| |/ __|
#// | |_| |  __/ | | | | | (_) | (_| | | | (_| | |_) | | | | | (__
#// |____/ \___|_| |_| |_|\___/ \__, |_|  \__,_| .__/|_| |_|_|\___|
#//                             |___/          |_|
#//  ____            _ _
#// |  _ \ __ _ _ __(_) |_ _   _
#// | |_) / _` | '__| | __| | | |
#// |  __/ (_| | |  | | |_| |_| |
#// |_|   \__,_|_|  |_|\__|\__, |
#//                        |___/

def div_dp(_x_g, _x_g_R):

    w = _x_g_R / _x_g

    # average of w sampled from source distribution is
    # int_X realized_x_g dx
    mean_w = _x_g_R.sum() # = 1

    # variance is average value of (w - mean_w)^2
    return (((w - mean_w) ** 2) * _x_g).sum()

def dp_violation(ac):
    '''
    Sum of group-pairwise differences in acceptance rate

    ac is array (one entry per group)
    '''
    ac = ac.reshape((len(ac), 1))
    return np.sum(np.abs(ac - ac.T)) / 2

def bound_dp(phi, _X_g=_X_g):
    '''
    Return bound on maximum possible violation
    of DP as a function of Bg

    (assumes that policy is fair on source distribution)
    '''

    source_ac = ac_phi(phi, _X_g=_X_g)

    def of_bg(bg):
        '''
        bg is variance of w according to source distribution
        '''

        # Need pairwise sums, but this bound is not as tight as it could be
        # (6) vs 4...

        # Assumes 4 groups
        return np.minimum(
            4,
            (3 * bg * source_ac).sum()
        )

    return of_bg

m = 30
k = m

ac_range = np.linspace(0.1, 0.9, m)
th_range = np.linspace(2, 10, k)

if not os.path.exists('dp_max_bgs'):

    source_ac = [ac_range]
    max_bgs = [np.zeros(len(ac_range))]

    # change in fairness violation
    strategic_manipulation = [np.zeros(len(ac_range))]
    upper_bound = [np.zeros(len(ac_range))]

    # possible ac values
    for ac in tqdm(ac_range):

        dp_phi = phi_ac(ac)

        source_violation = dp_violation(ac_phi(dp_phi))

        shift_of_theta = covariate_shift(dp_phi)
        bound_of_bg = bound_dp(dp_phi)

        violation = []
        _max_bgs = []
        bound = []

        # observed by strategic classification with
        # possible theta values
        for th in th_range:
            _x_g_R = shift_of_theta(th)

            # plt.plot(idx, _x_g[groups[0]], color='blue')
            # plt.plot(idx, _x_g_R[groups[0]], color='red')
            # plt.show()

            _X_g_R = _x_g_R.cumsum()

            # Pr(X <= x | G = g)
            R_ac = ac_phi(dp_phi, _X_g=_X_g_R)

            violation.append(dp_violation(R_ac))
            bg = div_dp(_x_g, _x_g_R)
            max_bg = max(bg)

            _max_bgs.append(max_bg)

            # bound.append(bound_of_bg(bg) + source_violation)
            bound.append(bound_of_bg(max_bg * np.ones(len(groups))) + source_violation)

        max_bgs.append(_max_bgs)
        upper_bound.append(bound)
        strategic_manipulation.append(violation)
        source_ac.append(ac * np.ones(k))

        # ax.plot(max_bgs, ac * np.ones(k), violation, linestyle='dotted', color='blue')
        # ax.plot(max_bgs, ac * np.ones(k), bound, linestyle='dotted', color='black')

    source_ac = np.array(source_ac).flatten()
    strategic_manipulation = np.array(strategic_manipulation).flatten()
    upper_bound = np.array(upper_bound).flatten()
    max_bgs = np.array(max_bgs).flatten()

    with open('dp_source_ac', 'wb') as f:
        pickle.dump(source_ac, f)
    with open('dp_strategic_manipulation', 'wb') as f:
        pickle.dump(strategic_manipulation, f)
    with open('dp_upper_bound', 'wb') as f:
        pickle.dump(upper_bound, f)
    with open('dp_max_bgs', 'wb') as f:
        pickle.dump(max_bgs, f)
else:
    with open('dp_source_ac', 'rb') as f:
        source_ac = pickle.load(f)
    with open('dp_strategic_manipulation', 'rb') as f:
        strategic_manipulation = pickle.load(f)
    with open('dp_upper_bound', 'rb') as f:
        upper_bound = pickle.load(f)
    with open('dp_max_bgs', 'rb') as f:
        max_bgs = pickle.load(f)

fig = plt.figure(figsize=(9,4))

print(max_bgs.shape)
print(source_ac.shape)
print(strategic_manipulation.shape)
print(upper_bound.shape)

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(max_bgs, source_ac, strategic_manipulation, alpha=0.7, antialiased=True)
ax.plot_trisurf(max_bgs, source_ac, upper_bound, alpha=0.4, antialiased=True, cmap='viridis')
ax.view_init(15, -30)
ax.set_xlabel('Max observed group shift')
ax.set_ylabel('Source acceptance rate')
ax.set_zlabel('Violation of Demographic Parity')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_trisurf(max_bgs, source_ac, strategic_manipulation, alpha=0.7, antialiased=True)
ax.plot_trisurf(max_bgs, source_ac, upper_bound, alpha=0.4, antialiased=True, cmap='viridis')
ax.view_init(15, -30 - 4)
ax.set_xlabel('Max observed group shift')
ax.set_ylabel('Source acceptance rate')
ax.set_zlabel('Violation of Demographic Parity')

plt.tight_layout()
plt.savefig('DP_bound.pdf', bbox_inches='tight')
