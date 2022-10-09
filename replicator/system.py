#!/usr/bin/env python

import numpy as np

from tqdm import tqdm

from state import State

class System:
    '''
    Combine classifier and response function in setting.

    Used to store computed values of simulation with specified resolution.
    '''
    def __init__(self, name, setting, classifier_func, response_func, res, resp_kwargs={}, cls_kwargs={}):

        self.name = name # for plotting
        self.setting = setting
        self.classifier_func = classifier_func
        self.response_func = response_func
        self.res = res
        self.resp_kwargs = resp_kwargs
        self.cls_kwargs = cls_kwargs

    def calculate(self):
        '''
        Assumes two groups
        '''
        res = self.res

        self.x = np.linspace(0.01, 0.99, res) # s_1
        self.y = np.linspace(0.01, 0.99, res) # s_2

        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # Velocity vectors
        self.Vx = np.zeros((res, res)) # x-component (group 1)
        self.Vy = np.zeros((res, res)) # y-component (group 2)

        # Acceptance rate for each (of two) groups
        self.A1 = np.zeros((res, res))
        self.A2 = np.zeros((res, res))

        self.next_A1 = np.zeros((res, res))
        self.next_A2 = np.zeros((res, res))

        # Outcomes (false negative rates, false positive rates)
        self.fpr1 = np.zeros((res, res))
        self.fpr2 = np.zeros((res, res))
        self.fnr1 = np.zeros((res, res))
        self.fnr2 = np.zeros((res, res))

        # self.tp1 = np.zeros((res, res))
        # self.fp1 = np.zeros((res, res))
        # self.tn1 = np.zeros((res, res))
        # self.fn1 = np.zeros((res, res))

        # self.tp2 = np.zeros((res, res))
        # self.fp2 = np.zeros((res, res))
        # self.tn2 = np.zeros((res, res))
        # self.fn2 = np.zeros((res, res))

        # self.next_fpr1 = np.zeros((res, res))
        # self.next_fpr2 = np.zeros((res, res))
        # self.next_fnr1 = np.zeros((res, res))
        # self.next_fnr2 = np.zeros((res, res))

        # Average fitness of each group
        self.f1 = np.zeros((res, res))
        self.f2 = np.zeros((res, res))

        # https://eli.thegreenplace.net/2014/meshgrids-and-disambiguating-rows-and-columns-from-cartesian-coordinates/
        # Arrays indexed by row = y, column = x.


        self.bound1 = np.zeros((res, res))
        self.bound2 = np.zeros((res, res))

        Utp = self.setting.U[1, 1]
        Ufn = self.setting.U[1, 0]
        Utn = self.setting.U[0, 0]
        Ufp = self.setting.U[0, 1]

        for ix in tqdm(range(res)):
            for iy in range(res):

                state = State(self.setting.mu, np.array([self.x[ix], self.y[iy]]))

                # classifier
                phi = self.classifier_func(self.setting, state, **self.cls_kwargs)

                # resulting variables
                self.A1[iy,ix], self.A2[iy,ix] = self.setting.beta(phi, state)
                fpr = np.array(self.setting.fpr(phi, state))
                fnr = np.array(self.setting.fnr(phi, state))
                # self.f1[iy,ix], self.f2[iy,ix] = self.setting.avg_fitness(phi, state)

                tp1, tp2 = (1 - fnr) * state.sg
                fp1, fp2 = fpr * (1 - state.sg)
                tn1, tn2 = (1 - fpr) * (1 - state.sg)
                fn1, fn2 = fnr * state.sg

                n1 = Utp * tp1 + Ufn * fn1
                d1 = Utn * tn1 + Ufp * fp1
                n2 = Utp * tp2 + Ufn * fn2
                d2 = Utn * tn2 + Ufp * fp2

                # response
                vel = self.response_func(self.setting, state, phi, **self.resp_kwargs)
                self.Vx[iy,ix], self.Vy[iy,ix] = vel.sg

                self.bound1[iy, ix] = np.abs(vel.sg[0]) * abs(fpr[0] + fnr[0] - 1)
                self.bound2[iy, ix] = np.abs(vel.sg[1]) * abs(fpr[1] + fnr[1] - 1)

                # frozen policy variables
                next_state = State(self.setting.mu, state.sg + vel.sg)
                self.next_A1[iy, ix], self.next_A2[iy, ix] = self.setting.beta(phi, next_state)

                # self.next_fpr1[iy,ix], self.next_fpr2[iy,ix] = self.setting.fpr(phi, next_state)
                # self.next_fnr1[iy,ix], self.next_fnr2[iy,ix] = self.setting.fnr(phi, next_state)


        # normalize fitness over entire state space
        max_f = np.max((np.max(self.f1), np.max(self.f2)))
        min_f = np.min((np.min(self.f1), np.min(self.f2)))
        self.f1 = (self.f1 - min_f) / (max_f - min_f)
        self.f2 = (self.f2 - min_f) / (max_f - min_f)

        # # normalize acceptance rates
        # self.A1 = self.A1 / (varphi * 2)
        # self.A2 = self.A2 / (varphi * 2)
