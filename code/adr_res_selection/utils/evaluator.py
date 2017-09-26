import numpy as np

from ..utils import say


class Evaluator(object):

    def __init__(self, bins='document'):
        self.index = 0.
        self.ttl_cost = 0.
        self.ttl_g_norm = 0.

        self.crr_adr = 0
        self.crr_res = 0
        self.crr_both = 0
        self.total = 0.

        self.acc_adr = 0.
        self.acc_res = 0.
        self.acc_both = 0.

        # 1D: binned_n_agents, 2D: (crr_a, crr_r, crr_both, total)
        if bins == 'document':
            self.results = np.zeros((7, 4), dtype='float32')  # 2, 3, 4, 5, 6-10, 11-
        elif bins == 'distance' or bins == 'context':
            self.results = np.zeros((16, 4), dtype='float32')  # 0, 1-15

        self.crr_adr_vec = []
        self.crr_res_vec = []

    def update(self, n_agents, cost, g_norm, pred_a, pred_r, gold_a, gold_r):
        self.index += 1

        if pred_a is not None:
            crr_adr_vec = map(lambda x: 1 if x[0] == x[1] else 0, zip(pred_a, gold_a))
            self.crr_adr_vec += crr_adr_vec
        else:
            crr_adr_vec = []
            self.crr_adr_vec += [-1] * len(gold_a)

        crr_res_vec = map(lambda x: 1 if x[0] == x[1] else 0, zip(pred_r, gold_r))
        self.crr_res_vec += crr_res_vec

        crr_adr = np.sum(crr_adr_vec)
        crr_res = np.sum(crr_res_vec)

        for i, a in enumerate(crr_adr_vec):
            n = n_agents[i]
            self.results[n][0] += a

        for i, r in enumerate(crr_res_vec):
            n = n_agents[i]
            self.results[n][1] += r
            self.results[n][3] += 1

        crr_both = 0.
        if pred_a is not None and pred_r is not None:
            for a, r, n in zip(crr_adr_vec, crr_res_vec, n_agents):
                crr_both += a * r
                self.results[n][2] += a * r

        self.crr_adr += crr_adr
        self.crr_res += crr_res
        self.crr_both += crr_both
        self.total += len(pred_r)

        self.ttl_cost += cost
        self.ttl_g_norm += g_norm

    def show_results(self):
        ###########################
        # Training-related values #
        ###########################
        say('\n\tTotal Loss: %f\tTotal Grad Norm: %f' % (self.ttl_cost, self.ttl_g_norm))
        say('\n\tAvg.  Loss: %f\tAvg.  Grad Norm: %f' % (self.ttl_cost / self.index, self.ttl_g_norm / self.index))

        ######################
        # Prediction results #
        ######################
        crr_adr = self.crr_adr
        crr_res = self.crr_res
        crr_both = self.crr_both
        total = self.total

        acc_adr = self.acc_adr = crr_adr / total
        acc_res = self.acc_res = crr_res / total
        acc_both = self.acc_both = crr_both / total

        text = 'Both: {:>7.2%} ({:>7}/{:>7})  Adr: {:>7.2%} ({:>7}/{:>7})  Res: {:>7.2%} ({:>7}/{:>7})\n'

        total = int(total)
        say('\n\n\tAccuracy\n\tTOTAL  ' + text.format(
            acc_both, int(crr_both), total,
            acc_adr, int(crr_adr), total,
            acc_res, int(crr_res), total))

        total_list = []
        acc_both_list = []
        acc_adr_list = []
        acc_res_list = []
        for n_agents in xrange(self.results.shape[0]):
            text = 'Both: {:>7.2%} ({:>7}/{:>7})  Adr: {:>7.2%} ({:>7}/{:>7})  Res: {:>7.2%} ({:>7}/{:>7})'
            result = self.results[n_agents]
            crr_adr = result[0]
            crr_res = result[1]
            crr_both = result[2]
            total = result[3]

            if total > 0:
                acc_both = crr_both / total
                acc_adr = crr_adr / total
                acc_res = crr_res / total
            else:
                acc_both = 0.
                acc_adr = 0.
                acc_res = 0.

            total = int(total)
            say('\n\t{:>5}  '.format(n_agents) + text.format(
                acc_both, int(crr_both), total,
                acc_adr, int(crr_adr), total,
                acc_res, int(crr_res), total))

            if n_agents <= 6:
                total_list.append(total)
                acc_both_list.append(crr_both)
                acc_adr_list.append(crr_adr)
                acc_res_list.append(crr_res)
            else:
                total_list[-1] += total
                acc_both_list[-1] += crr_both
                acc_adr_list[-1] += crr_adr
                acc_res_list[-1] += crr_res
        say('\n')

        acc_both_list = [100 * acc_both / total if total > 0. else 0. for acc_both, total in zip(acc_both_list,total_list)]
        acc_adr_list = [100 * acc_adr / total if total > 0. else 0. for acc_adr, total in zip(acc_adr_list,total_list)]
        acc_res_list = [100 * acc_res / total if total > 0. else 0. for acc_res, total in zip(acc_res_list,total_list)]
        #print total_list
        #print acc_both_list
        #print acc_adr_list
        #print acc_res_list

