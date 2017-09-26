import theano
import theano.tensor as T

from ..nn import binary_cross_entropy, regularization, get_grad_norm, optimizer_select, adam_clip
from ..nn import initialize_weights, build_shared_zeros, sigmoid
from ..nn import rnn, attention
from ..utils import say


class Model(object):

    def __init__(self, argv, max_n_agents, n_vocab, init_emb):
        self.argv = argv

        ###################
        # Input variables #
        ###################
        self.inputs = []
        self.layers = []
        self.params = []

        ###################
        # Hyperparameters #
        ###################
        self.n_words = argv.max_n_words
        self.max_n_agents = max_n_agents
        self.n_vocab = n_vocab
        self.init_emb = init_emb
        self.opt = argv.opt
        self.lr = argv.lr
        self.dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        self.dim_hidden = argv.dim_hidden
        self.L2_reg = argv.reg
        self.unit = argv.unit
        self.attention = argv.attention

        ####################
        # Output variables #
        ####################
        self.a_hat = None
        self.r_hat = None
        self.nll = None
        self.cost = None
        self.g_norm = None
        self.update = None

    def set_params(self):
        for l in self.layers:
            self.params += l.params
        say("Parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def predict(self, p_a, p_r):
        self.a_hat = T.argmax(p_a, axis=1)
        self.r_hat = T.argmax(p_r, axis=1)

    def objective_f(self, p_a, p_r, y_a, y_r, n_agents):
        # y_a: 1D: batch; 0/1
        # p_a: 1D: batch, 2D: n_agents-1; elem=scalar

        # p_arranged: 1D: batch, 2D: n_agents-1; C0=True, C1-Cn=False
        mask = T.cast(T.neq(y_a, -1), theano.config.floatX)
        p_a_arranged = p_a.ravel()[y_a.ravel()]
        p_a_arranged = p_a_arranged.reshape((y_a.shape[0], y_a.shape[1]))
        p_a_arranged = p_a_arranged * mask

        true_p_a = p_a_arranged[:, 0]
        false_p_a = T.max(p_a_arranged[:, 1:], axis=1)

        true_p_r = p_r[T.arange(y_r.shape[0]), y_r]
        false_p_r = p_r[T.arange(y_r.shape[0]), 1 - y_r]

        ########
        # Loss #
        ########
        nll_a = binary_cross_entropy(true_p_a, false_p_a) * T.cast(T.gt(n_agents, 2), dtype=theano.config.floatX)
        nll_r = binary_cross_entropy(true_p_r, false_p_r)
        nll = 0.5 * nll_a + 0.5 * nll_r

        ########
        # Cost #
        ########
        L2_sqr = regularization(self.params)
        cost = nll + self.L2_reg * L2_sqr / 2.

        return nll, cost

    def optimization(self):
        grads = T.grad(self.cost, self.params)
        self.g_norm = get_grad_norm(grads)
        if self.opt == 'adam':
            self.update = adam_clip(self.cost, self.params)
        #self.update = optimizer_select(self.opt, self.params, grads, self.lr)


class InterleaveModel(Model):

    def __init__(self, argv, max_n_agents, n_vocab, init_emb):
        super(InterleaveModel, self).__init__(argv, max_n_agents, n_vocab, init_emb)

    def compile(self, c, r, a, b, y_r, y_a, n_agents):
        self.inputs = [c, r, a, b, y_r, y_a, n_agents]
        batch = T.cast(c.shape[0], 'int32')
        n_prev_sents = T.cast(c.shape[1], 'int32')
        n_cands = T.cast(r.shape[1], 'int32')

        ################
        # Architecture #
        ################
        x_c, x_r = self.lookup_table(c, r)
        h_c, h_r = self.intra_sentential_layer(x_c, x_r, batch, n_prev_sents, n_cands)
        a_res, A_adr, z = self.inter_sentential_layer(h_c, a, b, n_agents, batch)
        p_a, p_r = self.output_layer(a_res, A_adr, z, h_r)

        ######################
        # Training & Testing #
        ######################
        self.nll, self.cost = self.objective_f(p_a, p_r, y_a, y_r, n_agents)
        self.set_params()
        self.optimization()
        self.predict(p_a, p_r)

    def lookup_table(self, c, r):
        pad = build_shared_zeros((1, self.dim_emb))
        if self.init_emb is None:
            emb = theano.shared(initialize_weights(self.n_vocab - 1, self.dim_emb))
            self.params += [emb]
        else:
            emb = theano.shared(self.init_emb)

        E = T.concatenate([pad, emb], 0)

        x_c = E[c]
        x_r = E[r]

        x_c = x_c.reshape((-1, self.n_words, self.dim_emb)).dimshuffle(1, 0, 2)
        x_r = x_r.reshape((-1, self.n_words, self.dim_emb)).dimshuffle(1, 0, 2)

        return x_c, x_r

    def intra_sentential_layer(self, x_c, x_r, batch, n_prev_sents, n_cands):
        ############
        # Encoders #
        ############
        if self.unit == 'rnn':
            layer = rnn.RNN(n_i=self.dim_emb, n_h=self.dim_hidden)
        else:
            layer = rnn.GRU(n_i=self.dim_emb, n_h=self.dim_hidden)
        self.layers.append(layer)

        ##########
        # Encode #
        ##########
        h_c0 = T.zeros(shape=(batch * n_prev_sents, self.dim_hidden), dtype=theano.config.floatX)
        h_r0 = T.zeros(shape=(batch * n_cands, self.dim_hidden), dtype=theano.config.floatX)

        # 1D: n_words, 2D: batch * n_prev_sents, 3D: n_hidden
        H_c, _ = theano.scan(fn=layer.recurrence, sequences=x_c, outputs_info=h_c0)
        h_c = H_c[-1].reshape((batch, n_prev_sents, self.dim_hidden)).dimshuffle(1, 0, 2)

        # 1D: n_words, 2D: batch * n_cands, 3D: n_hidden
        H_r, _ = theano.scan(fn=layer.recurrence, sequences=x_r, outputs_info=h_r0)
        h_r = H_r[-1].reshape((batch, n_cands, self.dim_hidden))

        return h_c, h_r

    def inter_sentential_layer(self, h_c, a, b, n_agents, batch):
        ############
        # Encoders #
        ############
        if self.unit == 'rnn':
            #layer = rnn.InterleaveRNN2(n_i=self.dim_hidden, n_h=self.dim_hidden)
            layer = rnn.InterleaveRNN3(n_i=self.dim_hidden, n_h=self.dim_hidden)
        elif self.unit == 'gru':
            layer = rnn.InterleaveGRU(n_i=self.dim_hidden, n_h=self.dim_hidden)
        self.layers.append(layer)

        a = a[:, :, :n_agents].dimshuffle(1, 0, 2)
        b = b[:, :, :n_agents].dimshuffle(1, 0, 2)

        # 1D: n_prev_sents, 2D: batch, 3D: n_agents, 4D: dim_agent
        A0 = T.zeros(shape=(batch, n_agents, self.dim_hidden), dtype=theano.config.floatX)
        A, _ = theano.scan(fn=layer.recurrence_interleave, sequences=[h_c, a, b], outputs_info=A0)

        ###########################
        # Extract summary vectors #
        ###########################
        # Responding Agent: 1D: Batch, 2D: dim_agent
        a_res = A[-1][:, 0]

        # Addressee Agents: 1D: batch, 2D: n_agents - 1, 3D: dim_agent
        A_adr = A[-1][:, 1:]

        #############
        # Attention #
        #############
        # 1D: batch, 2D: dim_agent
        if self.attention not in [-1,0]:
            print 'Using Attention ', self.attention
            layer = attention.Attention(n_h=self.dim_hidden, attention=self.attention)
            self.layers.append(layer)
            z = layer.f(A[-1], a_res)
        elif self.attention == 0:
            z = T.max(A[-1], 1)
        elif self.attention == -1:
            print 'Using mean ... ...'
            z = T.mean(A[-1], 1)

        return a_res, A_adr, z

    def output_layer(self, a_res, A_adr, z, h_r):
        layer_a = rnn.Layer(self.dim_hidden * 2, self.dim_hidden)
        layer_r = rnn.Layer(self.dim_hidden * 2, self.dim_hidden)
        self.layers += [layer_a, layer_r]

        o = T.concatenate([a_res, z], axis=1)  # 1D: batch, 2D: dim_hidden + dim_hidden
        o_a = layer_a.forward(o)  # 1D: batch, 2D: dim_hidden
        o_r = layer_r.forward(o)  # 1D: batch, 2D: dim_hidden

        score_a = T.batched_dot(o_a, A_adr.dimshuffle(0, 2, 1))  # 1D: batch, 2D: n_agents-1; elem=scalar
        score_r = T.batched_dot(o_r, h_r.dimshuffle(0, 2, 1))  # 1D: batch, 2D: n_cands; elem=scalar

        p_a = sigmoid(score_a)  # 1D: batch, 2D: n_agents-1; elem=scalar
        p_r = sigmoid(score_r)  # 1D: batch, 2D: n_cands; elem=scalar

        return p_a, p_r


class HierModel(Model):

    def __init__(self, argv, max_n_agents, n_vocab, init_emb):
        super(HierModel, self).__init__(argv, max_n_agents, n_vocab, init_emb)

    def compile(self, c, r, a, b, y_r, y_a, n_agents):
        self.inputs = [c, r, a, b, y_r, y_a, n_agents]
        batch = T.cast(c.shape[0], 'int32')
        n_prev_sents = T.cast(c.shape[1], 'int32')
        n_cands = T.cast(r.shape[1], 'int32')

        ################
        # Architecture #
        ################
        x_c, x_r = self.lookup_table(c, r)
        h_c, h_r = self.intra_sentential_layer(a, n_agents, x_c, x_r, batch, n_prev_sents, n_cands)
        a_res, h_d, A_p = self.inter_sentential_layer(h_c, a, n_agents, batch, n_prev_sents)
        p_a, p_r = self.output_layer(a_res, h_d, h_r, A_p)

        ######################
        # Training & Testing #
        ######################
        self.nll, self.cost = self.objective_f(p_a, p_r, y_a, y_r, n_agents)
        self.set_params()
        self.optimization()
        self.predict(p_a, p_r)

    def lookup_table(self, c, r):
        pad = build_shared_zeros((1, self.dim_emb))
        if self.init_emb is None:
            emb = theano.shared(initialize_weights(self.n_vocab - 1, self.dim_emb))
            self.params += [emb]
        else:
            emb = theano.shared(self.init_emb)

        E = T.concatenate([pad, emb], 0)

        x_c = E[c]
        x_r = E[r]

        x_c = x_c.reshape((-1, self.n_words, self.dim_emb)).dimshuffle(1, 0, 2)
        x_r = x_r.reshape((-1, self.n_words, self.dim_emb)).dimshuffle(1, 0, 2)

        return x_c, x_r

    def intra_sentential_layer(self, a, n_agents, x_c, x_r, batch, n_prev_sents, n_cands):
        ############
        # Encoders #
        ############
        if self.unit == 'rnn':
            layer = rnn.RNN(n_i=self.dim_emb, n_h=self.dim_hidden)
        else:
            layer = rnn.GRU(n_i=self.dim_emb, n_h=self.dim_hidden)
        self.layers.append(layer)

        A = rnn.Layer(self.max_n_agents, self.dim_emb)
        self.A = A
        self.layers.append(A)

        ##########
        # Encode #
        ##########
        #x_c: max_n_words x (batch x n_prev_sents) x dim_emb
        x_a = A.forward(a).reshape((-1, 1, self.dim_emb)).dimshuffle(1,0,2) # 1 x (batch x n_prev_sents) x dim_emb
        x_c = T.concatenate([x_a, x_c], 0) # (max_n_words + 1) x (batch x n_prev_sents) x dim_emb

        h_c0 = T.zeros(shape=(batch * n_prev_sents, self.dim_hidden), dtype=theano.config.floatX)
        h_r0 = T.zeros(shape=(batch * n_cands, self.dim_hidden), dtype=theano.config.floatX)

        # 1D: n_words, 2D: batch * n_prev_sents, 3D: n_hidden
        H_c, _ = theano.scan(fn=layer.recurrence, sequences=x_c, outputs_info=h_c0)
        h_c = H_c[-1].reshape((batch, n_prev_sents, self.dim_hidden)).dimshuffle(1, 0, 2)

        # 1D: n_words, 2D: batch * n_cands, 3D: n_hidden
        H_r, _ = theano.scan(fn=layer.recurrence, sequences=x_r, outputs_info=h_r0)
        h_r = H_r[-1].reshape((batch, n_cands, self.dim_hidden))

        return h_c, h_r

    def inter_sentential_layer(self, h_c, a, n_agents, batch, n_prev_sents):
        ############
        # Encoders #
        ############
        if self.unit == 'rnn':
            layer = rnn.RNN(n_i=self.dim_emb+self.dim_hidden, n_h=self.dim_hidden)
        else:
            layer = rnn.GRU(n_i=self.dim_emb+self.dim_hidden, n_h=self.dim_hidden)
        self.layers.append(layer)

        A = self.A
        #h_c: n_prev_sents x batch_size x dim_hidden
        x_a = A.forward(a).reshape((batch, n_prev_sents, self.dim_emb)).dimshuffle(1,0,2)
        h_c = T.concatenate([x_a,h_c],2) # n_prev_sents x batch_size x (dim_emb + dim_hidden)

        h_d0 = T.zeros(shape=(batch, self.dim_hidden), dtype=theano.config.floatX)
        H_d, _ = theano.scan(fn=layer.recurrence, sequences=h_c, outputs_info=h_d0)
        h_d = H_d[-1].reshape((batch, self.dim_hidden)) # h_d: batch x dim_hidden

        '''
        a = a[:, :, :n_agents].dimshuffle(1, 0, 2)

        # 1D: n_prev_sents, 2D: batch, 3D: n_agents, 4D: dim_agent
        A0 = T.zeros(shape=(batch, n_agents, self.dim_hidden), dtype=theano.config.floatX)
        A, _ = theano.scan(fn=layer.recurrence_inter, sequences=[h_c, a], outputs_info=A0)

        ###########################
        # Extract summary vectors #
        ###########################
        # Responding Agent: 1D: Batch, 2D: dim_agent
        a_res = A[-1][:, 0]

        # Addressee Agents: 1D: batch, 2D: n_agents - 1, 3D: dim_agent
        A_adr = A[-1][:, 1:]

        #############
        # Attention #
        #############
        # 1D: batch, 2D: dim_agent
        if self.attention:
            layer = attention.Attention(n_h=self.dim_hidden, attention=self.attention)
            self.layers.append(layer)
            z = layer.f(A[-1], a_res)
        else:
            z = T.max(A[-1], 1)

        return a_res, A_adr, z
        '''

        a_res = T.repeat(A.W[0].dimshuffle('x', 0), batch, 0) #a_res: batch x dim_emb
        A_p = A.W[1: n_agents]  # (n_agents - 1) x dim_emb

        return a_res, h_d, A_p

    def output_layer(self, a_res, h_d, h_r, A_p):
        #h_r: batch x n_cands x dim_hidden
        layer_a = rnn.Layer(self.dim_emb + self.dim_hidden, self.dim_emb)
        layer_r = rnn.Layer(self.dim_emb + self.dim_hidden, self.dim_hidden)
        self.layers += [layer_a, layer_r]

        o = T.concatenate([a_res, h_d], axis=1)  # 1D: batch, 2D: dim_emb + dim_hidden
        o_a = layer_a.forward(o)  # batch x dim_emb
        o_r = layer_r.forward(o)  # batch x dim_hidden

        score_a = T.dot(o_a, A_p.T) # batch x (n_agents - 1) 
        score_r = T.batched_dot(o_r, h_r.dimshuffle(0, 2, 1))  # 1D: batch, 2D: n_cands; elem=scalar

        p_a = sigmoid(score_a)  # 1D: batch, 2D: n_agents-1; elem=scalar
        p_r = sigmoid(score_r)  # 1D: batch, 2D: n_cands; elem=scalar

        return p_a, p_r

