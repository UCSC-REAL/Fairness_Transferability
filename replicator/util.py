#!/usr/bin/env python

import numpy as np
from scipy.special import erf, erfinv

def dot(a, b):
    '''dot product'''
    return np.sum(a * b)

def gaussian(x, mean, sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def gaussian_deriv(x, mean, sigma):
    return -2 * (x - mean) / (2 * sigma ** 2) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def gaussian_cdf(x, mean, sigma):
    return (1 + erf( (x - mean) / (sigma * np.sqrt(2)))) / 2

def gaussian_cdf_inv(y, mean, sigma):
    return erfinv(2 * y - 1) * sigma * np.sqrt(2) + mean
