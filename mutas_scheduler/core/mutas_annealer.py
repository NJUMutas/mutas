import random
from simanneal import Annealer
import logging as log

from mutas_scheduler.helper.func_wrapper import time_wrapper


class MutasAnnealer(Annealer):
    def __init__(self, mutas):
        self.mutas = mutas
        self.gd = self.mutas.gd
        self.state = {'a': self.mutas.a, 'f': self.mutas.f}
        super(MutasAnnealer, self).__init__(self.state)  # important!

    def move(self):
        a = self.state['a']
        f = self.state['f']
        user1 = random.randint(0, self.gd.USER_NUM - 1)
        trans1 = random.randint(0, self.gd.TRANS_NUM - 1)
        position = None
        pre_server = None
        max_prob_server = float("-inf")
        for i in range(self.gd.SERVER_NUM):
            if self.gd.FUNCTION_MASK[i, trans1] == 1:
                position = i * self.gd.USER_NUM * self.gd.TRANS_NUM + trans1 * self.gd.USER_NUM + user1
                if a[position] > max_prob_server:
                    pre_server = i
                    max_prob_server = a[position]
        new_server = random.randint(0, self.gd.SERVER_NUM - 1)
        if new_server == pre_server or self.gd.FUNCTION_MASK[new_server, trans1] == 0:
            new_server = random.randint(0, self.gd.SERVER_NUM - 1)
        new_position = new_server * self.gd.USER_NUM * self.gd.TRANS_NUM + trans1 * self.gd.USER_NUM + user1
        temp = a[new_position]
        a[new_position] = a[position]
        a[position] = temp
        self.state['a'] = a
        self.state['f'] = f

    # @time_wrapper
    def energy(self):
        a = self.state['a']
        f = self.mutas._f_resolution(a)
        new_value = self.mutas._evaluation(a, f)
        value = 0
        while abs(value - new_value) > 0.001:
            log.info("Delta=%f", abs(value - new_value))
            value = new_value
            a, _ = self.mutas._a_resolution(f)
            f = self.mutas._f_resolution(a)
            new_value = self.mutas._evaluation(a, f)
            log.info("new_value=%s, evaluation=%s" % ( _ , new_value))
        e_new = self.mutas._evaluation(a, f)
        self.state['a'] = a
        self.state['f'] = f
        return e_new
