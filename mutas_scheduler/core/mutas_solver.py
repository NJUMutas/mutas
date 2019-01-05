# Algorithm 1: The Mutas Algorithm

from typing import Dict
import logging as log
import random

import numpy as np
import cvxpy as cp

from mutas_scheduler.data.global_data import GlobalData
from mutas_scheduler.core.mutas_annealer import MutasAnnealer
from mutas_scheduler.helper.func_wrapper import cache_wrapper


class MutasSolver:

    def __init__(self):
        self.gd: GlobalData = None
        self.a: np.matrix = None
        self.f: np.matrix = None

    def solve(self, gd, verbose=True):
        if verbose:
            log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Solver start")
        self.gd = gd
        self.a, self.f = self.mutas()
        state, energy = self.anneal()
        a: np.matrix = state['a']
        f: np.matrix = state['f']
        a = a.reshape((self.gd.SERVER_NUM, self.gd.TRANS_NUM, self.gd.USER_NUM))
        f = f.reshape((self.gd.SERVER_NUM, self.gd.TRANS_NUM, self.gd.USER_NUM))
        log.info("======================")
        log.info(np.sum(a, axis=0))
        log.info("======================")
        log.info(np.sum(f, axis=0))
        log.info("======================")
        log.info("Solver end")

    # @cache_wrapper("mutas.pkl")
    def mutas(self):
        log.info("Mutas start")
        initial_assignment = np.zeros(shape=[self.gd.USER_NUM * self.gd.TRANS_NUM, self.gd.SERVER_NUM])
        for i in range(self.gd.USER_NUM):
            for j in range(self.gd.TRANS_NUM):
                server_id = random.randint(0, self.gd.SERVER_NUM - 1)
                while self.gd.FUNCTION_MASK[server_id, j] != 1:
                    server_id = random.randint(0, self.gd.SERVER_NUM - 1)
                initial_assignment[j * self.gd.USER_NUM + i, server_id] = 1
        ini_a = initial_assignment.flatten(order='F')

        f = self._f_resolution(ini_a)
        value = 0
        new_value = self._evaluation(ini_a, f)
        while abs(value - new_value) > 0.001:
            value = new_value
            a, new_value = self._a_resolution(f)
            log.info("One iteration over")
            f = self._f_resolution(a)
        e_current = self._evaluation(a, f)
        log.info("Converge, value=%f, new_value=%f, e_current=%f" % (value, new_value, e_current))
        log.info("Mutas end")
        return a, f

    # @cache_wrapper("anneal.pkl")
    def anneal(self):
        log.info("Simulate anneal begin")
        annealer = MutasAnnealer(self)
        annealer.set_schedule({'tmax': 5, 'tmin': 4.8, 'steps': 10, 'updates': 1})
        state, energy = annealer.anneal()
        log.info("Simulate anneal end")
        return state, energy

    def fun(self, x: cp.Variable, cons: Dict[str, cp.Constant]):
        y = 0
        for i in range(self.gd.USER_NUM):
            for j in range(self.gd.TRANS_NUM - 1):
                for m in range(self.gd.SERVER_NUM):
                    if self.gd.FUNCTION_MASK[m, j] == 1:
                        position1 = m * self.gd.USER_NUM * self.gd.TRANS_NUM + j * self.gd.USER_NUM + i
                        for n in range(self.gd.SERVER_NUM):
                            if self.gd.FUNCTION_MASK[n, j + 1] == 1:
                                position2 = n * self.gd.USER_NUM * self.gd.TRANS_NUM + (j + 1) * self.gd.USER_NUM + i
                                y = y + cp.maximum(x[position1] + x[position2] - 1, 0) * self.gd.LINK_LATENCY[m, n]
        return y + cp.matmul(cons['xishu'], x)

    def _a_resolution(self, f: np.ndarray):
        xishu = self._getXishu(f)
        x = cp.Variable(shape=self.gd.SERVER_NUM * self.gd.TRANS_NUM * self.gd.USER_NUM, nonneg=True)
        obj = cp.Minimize(self.fun(x=x, cons={'xishu': xishu}))
        cons = [cp.matmul(self.gd.C, x) == self.gd.b]
        prob = cp.Problem(obj, cons)
        opt_val = prob.solve()
        log.info("opt_val=%f, eval=%f" % (opt_val, self._evaluation(x.value, f)))
        return np.maximum(x.value, 0), opt_val

    def _f_resolution(self, a: np.ndarray):
        aCopy = a.copy()
        f = np.zeros(shape=self.gd.SERVER_NUM * self.gd.TRANS_NUM * self.gd.USER_NUM)
        weighted_datasize = np.sqrt(self.gd.NEW_DATA_SIZE * self.gd.NEW_FUNCTION_MASK * aCopy)
        server_load = weighted_datasize.copy()
        for i in range(self.gd.SERVER_NUM):
            start_index = i * self.gd.TRANS_NUM * self.gd.USER_NUM
            end_index = (i + 1) * self.gd.TRANS_NUM * self.gd.USER_NUM
            server_load[start_index:end_index] = np.sum(weighted_datasize[start_index:end_index])
        f = weighted_datasize / (server_load + self.gd.EPS)
        return f

    def _evaluation(self, a, f):
        aCopy = a.copy()
        xishu = self._getXishu(f)
        y = 0
        for i in range(self.gd.USER_NUM):
            for j in range(self.gd.TRANS_NUM - 1):
                for m in range(self.gd.SERVER_NUM):
                    if self.gd.FUNCTION_MASK[m, j] == 1:
                        position1 = m * self.gd.USER_NUM * self.gd.TRANS_NUM + j * self.gd.USER_NUM + i
                        for n in range(self.gd.SERVER_NUM):
                            if self.gd.FUNCTION_MASK[n, j + 1] == 1:
                                position2 = n * self.gd.USER_NUM * self.gd.TRANS_NUM + (j + 1) * self.gd.USER_NUM + i
                                y += max(aCopy[position1] + aCopy[position2] - 1, 0) * self.gd.LINK_LATENCY[m, n]
        return y + np.dot(xishu, aCopy)

    def _getXishu(self, f):
        fCopy = f.copy()
        for i in range(self.gd.SERVER_NUM):
            start_index = i * self.gd.USER_NUM * self.gd.TRANS_NUM
            end_index = (i + 1) * self.gd.USER_NUM * self.gd.TRANS_NUM
            fCopy[start_index: end_index] = self.gd.SERVER_CAPACITY[i] * fCopy[start_index: end_index]
        processing_time = self.gd.NEW_DATA_SIZE / (fCopy + self.gd.EPS)
        processing_time *= self.gd.NEW_FUNCTION_MASK
        xishu = self.gd.NEW_LATENCY + processing_time
        return xishu
