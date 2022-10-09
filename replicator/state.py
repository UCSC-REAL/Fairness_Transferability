#!/usr/bin/env python

import numpy as np

from util import dot

class State:
    '''
    Allows quick change of coordinates between (s_g) vector and (deltas, avg_s)
    vector.

    Implemented before I got a handle on numpy's __array_ufunc__ and
    __array_function__, which would allow a superior implementation. As it is,
    it's akward to pass `State` objects around when you need to compute on
    groups independently.  Finally, the utility of Monads is clear.
    '''
    def __init__(self, mu, sg):
        '''
        Class representing state vector and enabling coordinate transformations
        on the vector

        Args:
            mu: (numpy array) $\mu_1, \mu_2, ..., \mu_n$ - relative proportions of groups
            sg: (numpy array) $s_1, s_2, ..., s_n$ - qualification rates of groups

        "primary" representation is in  (s_1, s_2, ... ) basis
        "sdelta" representation is in (s_bar, delta_12, delta_23, ...) basis

        '''
        self.sg = sg
        self.mu = mu
        self.avg, self.delta = self._to_sdelta(mu, sg)
        self.n = len(sg)

    def __getitem__(self, index):
        '''
        Allow indexing of State object directly to get s_g
        vector components.

        Eg,
        >>> state = State(np.array([0.1, 0.2, 0.3]))
        >>> state[0]
        0.1
        '''
        return self.sg[index]

    @classmethod
    def from_sdelta(cls, mu, avg, delta):
        '''
        Alternate constructor of State instance in terms of $\bar{s}$ and
        vector of delta values (Eq. 14)

        Args:
            avg: $\bar{s}$ (float)

            delta: $\delta(1, 2), \delta(2, 3), ..., \delta(n-1, n)$
                   (numpy array)
        '''
        return cls(cls._to_primary(mu, avg, delta))

    @staticmethod
    def _to_sdelta(mu, sg):
        '''
        Get $\bar{s}$ and vector of delta values from vector of $s_g$ values

        Args:
            sg: $s_1, s_2, ..., s_n$
                (numpy array)

        Returns:
            (avg, delta)

            avg: $\bar{s}$ (float)
            delta: $\delta(1, 2), \delta(2, 3), ..., \delta(n-1, n)$
                   (numpy array)
        '''
        avg = dot(mu, sg)
        delta = (np.roll(sg, 1) - sg)[1:]

        return (avg, delta)

    @staticmethod
    def _to_primary(mu, avg, delta):
        '''
        Get vector of $s_g$ values from \bar{s}$ and vector of delta values

        Args:
            avg: $\bar{s}$ (float)
            delta: $\delta(1, 2), \delta(2, 3), ..., \delta(n-1, n)$
                   (numpy array)

        Returns $s_1, s_2, ..., s_n$ (numpy array)
        '''

        n = len(delta) + 1
        sg = np.ones(n) * avg
        for g in range(1, n + 1):
            sg[g-1] += delta[g-1:].sum()
            print()
            for h in range(1, n):
                sg[g-1] -= mu[:h].sum() * delta[h-1]
                print(mu[:h].sum())
        return sg
