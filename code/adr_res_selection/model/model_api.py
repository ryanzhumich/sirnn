import math
import time

import sys

import numpy as np
import theano
import theano.tensor as T

from model import StaticModel, DynamicModel
from model2 import HierModel, InterleaveModel
from model3 import InterleaveModelCross
from model4 import InterleaveModelCollab
from ..utils import say, load_data, dump_data, create_path, move_data, check_identifier
from ..utils.evaluator import Evaluator


class ModelAPI(object):
    def __init__(self, argv, init_emb, vocab, n_prev_sents):
        self.argv = argv
        self.init_emb = init_emb
        self.vocab = vocab
        self.max_n_agents = n_prev_sents + 1

        self.model = None
        self.train_f = None
        self.pred_f = None
        self.pred_r_f = None

    def set_model(self):
        say('\n\nBUILD A MODEL\n')
        argv = self.argv

        #####################
        # Network variables #
        #####################
        c = T.itensor3('c')
        r = T.itensor3('r')
        a = T.ftensor3('a')
        y_r = T.ivector('y_r')
        y_a = T.imatrix('y_a')
        n_agents = T.iscalar('n_agents')

        max_n_agents = self.max_n_agents
        init_emb = self.init_emb
        n_vocab = self.vocab.size()

        #################
        # Build a model #
        #################
        say('MODEL: %s  Unit: %s  Opt: %s  Activation: %s  ' % (argv.model, argv.unit, argv.opt, argv.activation))

        if argv.model == 'static':
            model = StaticModel
        elif argv.model == 'dynamic':
            model = DynamicModel
        elif argv.model == 'hier':
            model = HierModel
        elif argv.model == 'interleave':
            model = InterleaveModel
        elif argv.model == 'interleavecross':
            model = InterleaveModelCross
        elif argv.model == 'interleavecollab':
            model = InterleaveModelCollab

        self.model = model(argv, max_n_agents, n_vocab, init_emb)

        if argv.model == 'interleave' or argv.model == 'hier' or argv.model == 'interleavecross' or argv.model == 'interleavecollab':
            b = T.ftensor3('b')
            self.model.compile(c=c, r=r, a=a, b=b, y_r=y_r, y_a=y_a, n_agents=n_agents)
        else:
            self.model.compile(c=c, r=r, a=a, y_r=y_r, y_a=y_a, n_agents=n_agents)


    def load_model(self):
        self.model = load_data(self.argv.load_model)

    def save_model(self, output_fn=None, output_dir='../data/model'):
        argv = self.argv

        if output_fn is None:
            output_fn = 'model-%s.unit-%s.batch-%d.reg-%f.sents-%d.words-%d' % \
                        (argv.model, argv.unit, argv.batch, argv.reg, argv.n_prev_sents, argv.max_n_words)
            output_fn = check_identifier(output_fn)
        dump_data(self.model, output_fn)
        create_path(output_dir)
        move_data(output_fn, output_dir)

    def set_train_f(self, train_samples):
        model = self.model
        index = T.iscalar('index')
        self.train_f = theano.function(inputs=[index],
                                       outputs=[model.nll, model.g_norm, model.a_hat, model.r_hat],
                                       updates=model.update,
                                       givens={
                                           model.inputs[0]: train_samples[0][index],
                                           model.inputs[1]: train_samples[1][index],
                                           model.inputs[2]: train_samples[2][index],
                                           model.inputs[3]: train_samples[3][index],
                                           model.inputs[4]: train_samples[4][index],
                                           model.inputs[5]: train_samples[5][index]
                                       }
                                       )

    def set_train_f_interleave(self, train_samples):
        model = self.model
        index = T.iscalar('index')
        self.train_f = theano.function(inputs=[index],
                                       outputs=[model.nll, model.g_norm, model.a_hat, model.r_hat],
                                       updates=model.update,
                                       givens={
                                           model.inputs[0]: train_samples[0][index],
                                           model.inputs[1]: train_samples[1][index],
                                           model.inputs[2]: train_samples[2][index],
                                           model.inputs[3]: train_samples[3][index],
                                           model.inputs[4]: train_samples[4][index],
                                           model.inputs[5]: train_samples[5][index],
                                           model.inputs[6]: train_samples[6][index]
                                       },
                                       on_unused_input='ignore'
                                       )


    def set_train_online_f(self):
        model = self.model
        self.train_f = theano.function(inputs=model.inputs,
                                       outputs=[model.nll, model.g_norm, model.a_hat, model.r_hat],
                                       updates=model.update,
                                       )

    def set_test_f(self):
        model = self.model
        self.pred_f = theano.function(inputs=model.inputs,
                                      outputs=[model.a_hat, model.r_hat],
                                      on_unused_input='ignore'
                                      )
        self.pred_r_f = theano.function(inputs=model.inputs,
                                        outputs=model.r_hat,
                                        on_unused_input='ignore'
                                        )

    def set_test_f2(self):
        model = self.model
        self.pred_f = theano.function(inputs=model.inputs,
                                      outputs=[model.p_a_pred, model.p_r_pred],
                                      on_unused_input='ignore'
                                      )
        self.pred_r_f = theano.function(inputs=model.inputs,
                                        outputs=model.p_r_pred,
                                        on_unused_input='ignore'
                                        )

        self.o_a_f = theano.function(inputs=model.inputs,
                                        outputs=model.o_a,
                                        on_unused_input='ignore'
                                        )

        self.o_r_f = theano.function(inputs=model.inputs,
                                        outputs=model.o_r,
                                        on_unused_input='ignore'
                                        )

        self.z_f = theano.function(inputs=model.inputs,
                                        outputs=model.z,
                                        on_unused_input='ignore'
                                        )

        self.A_adr_f = theano.function(inputs=model.inputs,
                                        outputs=model.A_adr,
                                        on_unused_input='ignore'
                                        )

        self.a_res_f = theano.function(inputs=model.inputs,
                                        outputs=model.a_res,
                                        on_unused_input='ignore'
                                        )

    def set_test_f3(self):
        model = self.model
        self.pred_f = theano.function(inputs=model.inputs,
                                      outputs=[model.p_a, model.p_r, model.p_a_pred, model.p_r_pred],
                                      on_unused_input='ignore'
                                      )
        self.pred_r_f = theano.function(inputs=model.inputs,
                                        outputs=model.p_r_pred,
                                        on_unused_input='ignore'
                                        )


    def train(self, c, r, a, res_vec, adr_vec, n_agents):
        nll, g_norm, pred_a, pred_r = self.train_f(c, r, a, res_vec, adr_vec, n_agents)
        return nll, g_norm, pred_a, pred_r

    def train_all(self, batch_indices, evalset):
        evaluator = Evaluator()
        np.random.shuffle(batch_indices)
        np.random.shuffle(batch_indices)
        start = time.time()

        for index, b_index in enumerate(batch_indices):
            if index != 0 and index % 100 == 0:
                say("  {}/{}".format(index, len(batch_indices)))

            cost, g_norm, pred_a, pred_r = self.train_f(b_index)
            binned_n_agents, labels_a, labels_r = evalset[b_index]

            if math.isnan(cost):
                say('\n\nLoss is NAN: Mini-Batch Index: %d,%d\n' % (index,b_index))
                say('\n\ng_norm is %f\n' % g_norm)
                exit()

            evaluator.update(binned_n_agents, cost, g_norm, pred_a, pred_r, labels_a, labels_r)

        end = time.time()
        say('\n\tTime: %f' % (end - start))
        evaluator.show_results()

    def predict(self, c, r, a, b, y_r, y_a, n_agents):
        pred_a = None
        '''
        print n_agents
        A_adr = self.A_adr_f(c, r, a, b, y_r, y_a, n_agents)
        print A_adr.shape
        a_res = self.a_res_f(c, r, a, b, y_r, y_a, n_agents)
        print a_res.shape
        z = self.z_f(c, r, a, b, y_r, y_a, n_agents)
        print z.shape
        o_a = self.o_a_f(c, r, a, b, y_r, y_a, n_agents)
        print o_a.shape
        o_r = self.o_r_f(c, r, a, b, y_r, y_a, n_agents)
        print o_r.shape
        '''
        if n_agents > 1:
            if self.argv.model == 'interleave' or self.argv.model == 'hier' or self.argv.model == 'interleavecross':
                pred_a, pred_r = self.pred_f(c, r, a, b, y_r, y_a, n_agents)
            else:
                pred_a, pred_r = self.pred_f(c, r, a, y_r, y_a, n_agents)
        else:
            if self.argv.model == 'interleave' or self.argv.model == 'hier' or self.argv.model == 'interleavecross':
                pred_r = self.pred_r_f(c, r, a, b, y_r, y_a, n_agents)
            else:
                pred_r = self.pred_r_f(c, r, a, y_r, y_a, n_agents)
        return pred_a, pred_r

    def predict_collab(self, c, r, a, b, y_r, y_a, n_agents):
        p_a = None
        p_r = None
        pred_a = None
        if n_agents > 1:
            p_a, p_r, pred_a, pred_r = self.pred_f(c, r, a, b, y_r, y_a, n_agents)
        else:
            pred_r = self.pred_r_f(c, r, a, b, y_r, y_a, n_agents)
        return p_a, p_r, pred_a, pred_r


    def predict_all(self, samples):
        evaluator = Evaluator(bins=self.argv.bins)
        start = time.time()

        sample_idx = 0
        for i, sample in enumerate(samples):
            if i != 0 and i % 100 == 0:
                say("  {}/{}".format(i, len(samples)))

            x = sample[0]
            binned_n_agents = sample[1]
            labels_a = sample[2]
            labels_r = sample[3]
            agent_index = sample[4]
            binned_n_agents_in_distance = sample[5]

            if self.argv.crosstest == 1:
                p_a_pred, p_r_pred = self.predict(c=x[0], r=x[1], a=x[2], b=x[3], y_r=x[4], y_a=x[5], n_agents=x[6])
                #TODO: how to make prediction from these two matrix
                if p_a_pred is not None:
                    pred_a = []
                    pred_r = []
                    #p_a_pred: batch x n_cands x n_agents-1
                    #p_r_pred: batch x n_agents-1 x n_cands
                    p_ar = p_a_pred + p_r_pred.transpose((0,2,1)) # directly add sigmoid scores
                    #p_ar = p_a_pred * p_r_pred.transpose((0,2,1)) # directly multiply sigmoid scores
                    for j in range(p_ar.shape[0]):
                        (r_idx,a_idx)  = np.unravel_index(p_ar[j].argmax(),p_ar[j].shape)
                        pred_a.append(a_idx)
                        pred_r.append(r_idx)
                    pred_a = np.array(pred_a)
                    pred_r = np.array(pred_r)
                else:
                    pred_a = None
                    #p_r_pred: batch x 2 x n_cands
                    #print p_r_pred.shape
                    pred_r = p_r_pred[:,0,:].argmax(axis=1)
                    #print pred_r.shape
                    #exit()
            elif self.argv.crosstest == 4:
                p_a, p_r, p_a_pred, p_r_pred = self.predict_collab(c=x[0], r=x[1], a=x[2], b=x[3], y_r=x[4], y_a=x[5], n_agents=x[6])
                if p_a_pred is not None:
                    pred_a = []
                    pred_r = []
                    #p_a: batch x n_agents-1
                    #p_r: batch x n_cands
                    #p_a_pred: batch x n_cands x n_agents-1
                    #p_r_pred: batch x n_agents-1 x n_cands

                    p_a_pred = (p_a_pred.transpose(2,0,1) * p_r).transpose(1,2,0)
                    p_r_pred = (p_r_pred.transpose(2,0,1) * p_a).transpose(1,2,0)
                    p_ar = p_a_pred + p_r_pred.transpose((0,2,1))

                    for j in range(p_ar.shape[0]):
                        (r_idx,a_idx)  = np.unravel_index(p_ar[j].argmax(),p_ar[j].shape)
                        pred_a.append(a_idx)
                        pred_r.append(r_idx)
                    pred_a = np.array(pred_a)
                    pred_r = np.array(pred_r)
                else:
                    pred_a = None
                    #p_r_pred: batch x 2 x n_cands
                    pred_r = p_r_pred[:,0,:].argmax(axis=1)
            elif self.argv.crosstest == 5:
                p_a, p_r, p_a_pred, p_r_pred = self.predict_collab(c=x[0], r=x[1], a=x[2], b=x[3], y_r=x[4], y_a=x[5], n_agents=x[6])
                if p_a_pred is not None:
                    pred_a = []
                    pred_r = []
                    #p_a: batch x n_agents-1
                    #p_r: batch x n_cands
                    #p_a_pred: batch x n_cands x n_agents-1
                    #p_r_pred: batch x n_agents-1 x n_cands

                    p_a_pred = (p_a_pred.transpose(2,0,1) * p_r).transpose(1,2,0)
                    p_r_pred = (p_r_pred.transpose(2,0,1) * p_a).transpose(1,2,0)
                    p_ar = p_a_pred * p_r_pred.transpose((0,2,1))

                    for j in range(p_ar.shape[0]):
                        (r_idx,a_idx)  = np.unravel_index(p_ar[j].argmax(),p_ar[j].shape)
                        pred_a.append(a_idx)
                        pred_r.append(r_idx)
                    pred_a = np.array(pred_a)
                    pred_r = np.array(pred_r)
                else:
                    pred_a = None
                    #p_r_pred: batch x 2 x n_cands
                    pred_r = p_r_pred[:,0,:].argmax(axis=1)
            elif self.argv.crosstest == 3:
                p_a_pred, p_r_pred = self.predict(c=x[0], r=x[1], a=x[2], b=x[3], y_r=x[4], y_a=x[5], n_agents=x[6])
                #TODO: how to make prediction from these two matrix
                if p_a_pred is not None:
                    pred_a = []
                    pred_r = []
                    #print p_a_pred.shape
                    #print p_r_pred.shape
                    #p_a_pred: batch x n_cands+1 x n_agents-1
                    #p_r_pred: batch x n_agents x n_cands

                    p_a = p_a_pred[:,-1,:] # batch x n_agents-1 P(a|C)
                    p_r = p_r_pred[:,-1,:] # batch x n_cands    P(r|C)

                    p_a_pred = p_a_pred[:,:-1,:]  # batch x n_cands x n_agents-1  P(a|C,r)
                    p_r_pred = p_r_pred[:,:-1,:]  # batch x n_agents-1 x n_cands  P(r|C,a)

                    p_a_pred = (p_a_pred.transpose(2,0,1) * p_r).transpose(1,2,0)
                    p_r_pred = (p_r_pred.transpose(2,0,1) * p_a).transpose(1,2,0)
                    p_ar = p_a_pred + p_r_pred.transpose((0,2,1))

                    for j in range(p_ar.shape[0]):
                        (r_idx,a_idx)  = np.unravel_index(p_ar[j].argmax(),p_ar[j].shape)
                        pred_a.append(a_idx)
                        pred_r.append(r_idx)
                    pred_a = np.array(pred_a)
                    pred_r = np.array(pred_r)
                else:
                    pred_a = None
                    #p_r_pred: batch x 1 x n_cands
                    pred_r = p_r_pred[:,0,:].argmax(axis=1)
            elif self.argv.crosstest == 2:
                p_a_pred, p_r_pred = self.predict(c=x[0], r=x[1], a=x[2], b=x[3], y_r=x[4], y_a=x[5], n_agents=x[6])
                #TODO: how to make prediction from these two matrix
                if p_a_pred is not None:
                    pred_a = []
                    pred_r = []
                    #p_a_pred: batch x n_cands x n_agents-1
                    #p_r_pred: batch x n_agents-1 x n_cands
                    p_a_pred = p_a_pred.max(axis=1)
                    p_r_pred = p_r_pred.max(axis=1)
                    for j in range(p_r_pred.shape[0]):
                        a_idx = p_a_pred[j].argmax()
                        r_idx = p_r_pred[j].argmax()
                        pred_a.append(a_idx)
                        pred_r.append(r_idx)
                    pred_a = np.array(pred_a)
                    pred_r = np.array(pred_r)
                else:
                    pred_a = None
                    #p_r_pred: batch x 2 x n_cands
                    pred_r = p_r_pred[:,0,:].argmax(axis=1)
            else:   # self.argv.crosstest == 0
                pred_a, pred_r = self.predict(c=x[0], r=x[1], a=x[2], b=x[3], y_r=x[4], y_a=x[5], n_agents=x[6])

            if self.argv.bins == 'document':
                evaluator.update(binned_n_agents, 0., 0., pred_a, pred_r, labels_a, labels_r)
            elif self.argv.bins == 'context':
                evaluator.update([int(x[6])] * len(labels_a), 0., 0., pred_a, pred_r, labels_a, labels_r)
            elif self.argv.bins == 'distance':
                evaluator.update(binned_n_agents_in_distance, 0., 0., pred_a, pred_r, labels_a, labels_r)

            if self.argv.output:
                batch_size = len(labels_a)
                for j in range(batch_size):
                    if pred_a is not None:
                        pred_aj = pred_a[j]
                    else:
                        pred_aj = None
                    self.output(sample_idx,x[0][j],x[1][j],x[2][j],x[3][j],labels_a[j],labels_r[j],pred_aj,pred_r[j],agent_index[j])
                    sample_idx += 1

        end = time.time()
        say('\n\tTime: %f' % (end - start))
        evaluator.show_results()

        return evaluator.acc_both, evaluator.acc_adr, evaluator.acc_res

    def get_pnorm_stat(self):
        lst_norms = []
        for p in self.model.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    def output(self,sample_idx,context,response,sender,addressee,label_adr,label_res,pred_adr,pred_res,agent_index):
        result_file = open(self.argv.output,'a')
 
        if pred_adr is not None:
            crr_adr = 1 if label_adr == pred_adr else 0
        else:
            crr_adr = -1
        crr_res = 1 if label_res == pred_res else 0
        crr_both = 1 if crr_res == 1 and crr_adr == 1 else 0
        print >> result_file, 'Sample', sample_idx, crr_both, crr_adr, crr_res, '\n'
         
        
        # Print Context
        agent_index_reverse = {}
        for k,v in agent_index.iteritems():
            agent_index_reverse[v] = k

        vocab = self.vocab
        def print_sent_idx(sent_idx):
            return ' '.join(vocab.get_word(w_id) for w_id in sent_idx if w_id != 0)

        assert len(context) == len(sender) == len(addressee)
        for (i,c) in enumerate(context):
            sender_idx = -1
            for (j,s) in enumerate(sender[i]):
                if s == 1:
                    sender_idx = j
                    break
            sender_id = agent_index_reverse[sender_idx]
            
            adr_idx = -1
            for (j,s) in enumerate(addressee[i]):
                if s == 1:
                    adr_idx = j
                    break
            if adr_idx == -1:
                adr_id = None
            else:
                adr_id = agent_index_reverse[adr_idx]

            print >> result_file, '%s\t%s\t%s' % (sender_id, adr_id, print_sent_idx(c))

        # Print Responding Agent
        print >> result_file, '%s\t?\t?\n' % agent_index_reverse[0]   # current speaker

        # Print Result
        print >> result_file, agent_index_reverse
        if pred_adr is not None:
            pred_adr += 1
        print >> result_file, 'label_adr', label_adr+1, 'pred_adr', pred_adr, '\n'
        for r in response:
            print >> result_file, print_sent_idx(r)
        print >> result_file, 'label_res', label_res, 'pred_res', pred_res, '\n\n'

