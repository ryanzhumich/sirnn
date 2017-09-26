import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, initialize_weights


class Layer(object):

    def __init__(self, n_i, n_h):
        self.W = theano.shared(initialize_weights(n_i, n_h))
        self.params = [self.W]

    def forward(self, h):
        return T.dot(h, self.W)


class InterleaveRNN1(object):
    def __init__(self, n_i, n_h, activation=tanh):
        self.W_in = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.W_h_a = theano.shared(initialize_weights(n_h, n_h))
        self.W_h_b = theano.shared(initialize_weights(n_h, n_h))
        self.W_h_other = theano.shared(initialize_weights(n_h, n_h))
        self.params = [self.W_in, self.W_h_a, self.W_h_b, self.W_h_other]
        self.activation = activation

    def recurrence_interleave(self, x_t, a_t, b_t, A_t):
        """
        :param  x_t: 1D: batch, 2D: dim_hidden
        :param  a_t: 1D: batch, 2D: n_agents; elem=one hot vector for speaker
        :param  b_t: 1D: batch, 2D: n_agents; elem=one hot vector for addressee
        :return A_t: 1D: batch, 2D: n_agents, 3D: dim_agent
        """

        h_a = A_t * a_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_b = A_t * b_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_other = A_t - h_a - h_b             # batch_size x n_agents x dim_agent

        h_a_sum = T.sum(h_a,1)                # batch_size x dim_agent
        h_x = T.dot(T.concatenate([x_t,h_a_sum],1), self.W_in)           # batch_size x dim_hidden

        h_a = T.dot(h_a, self.W_h_a)          # batch_size x n_agents x dim_hidden
        h_b = T.dot(h_b, self.W_h_b)          # batch_size x n_agents x dim_hidden
        h_other = T.dot(h_other, self.W_h_other) # batch_size x n_agents x dim_hidden

        return self.activation(h_x.dimshuffle(0,'x',1) + h_a + h_b + h_other)

class InterleaveRNN2(object):
    def __init__(self, n_i, n_h, activation=tanh):
        self.W_in = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.W_h = theano.shared(initialize_weights(n_h, n_h))
        self.V_h = theano.shared(initialize_weights(n_h, n_h))
        self.W_h_other = theano.shared(initialize_weights(n_h, n_h))
        self.params = [self.W_in, self.W_h, self.V_h, self.W_h_other]
        self.activation = activation


    def recurrence_interleave(self, x_t, a_t, b_t, A_t):
        """
        :param  x_t: 1D: batch, 2D: dim_hidden
        :param  a_t: 1D: batch, 2D: n_agents; elem=one hot vector for speaker
        :param  b_t: 1D: batch, 2D: n_agents; elem=one hot vector for addressee
        :return A_t: 1D: batch, 2D: n_agents, 3D: dim_agent
        """

        h_a = A_t * a_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_b = A_t * b_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_other = A_t - h_a - h_b             # batch_size x n_agents x dim_agent

        h_a = T.sum(h_a,1)                # batch_size x dim_agent
        h_b = T.sum(h_b,1)                # batch_size x dim_agent
        h_x = T.dot(T.concatenate([x_t,h_a],1), self.W_in)           # batch_size x dim_hidden

        h_other = T.dot(h_other, self.W_h_other) # batch_size x n_agents x dim_hidden

        # update for speaker
        A_t_a = self.activation(h_x + T.dot(h_a, self.W_h) + T.dot(h_b, self.V_h))
        A_t_a = A_t_a.dimshuffle(0,'x',1) * a_t.dimshuffle(0,1,'x')

        # update for addressee
        A_t_b = self.activation(h_x + T.dot(h_b, self.W_h) + T.dot(h_a, self.V_h))
        A_t_b = A_t_b.dimshuffle(0,'x',1) * b_t.dimshuffle(0,1,'x')

        # update for others
        A_t_other = self.activation(h_x.dimshuffle(0,'x',1) * (1 - (a_t.dimshuffle(0,1,'x') + b_t.dimshuffle(0,1,'x'))) + h_other)

        return A_t_a + A_t_b + A_t_other


class InterleaveRNN3(object):
    def __init__(self, n_i, n_h, activation=tanh):
        self.WA_in = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WA_h = theano.shared(initialize_weights(n_h, n_h))
        self.VA_h = theano.shared(initialize_weights(n_h, n_h))

        self.WB_in = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WB_h = theano.shared(initialize_weights(n_h, n_h))
        self.VB_h = theano.shared(initialize_weights(n_h, n_h))

        self.Wother_in = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.Wother_h = theano.shared(initialize_weights(n_h, n_h))

        self.params = [self.WA_in, self.WA_h, self.VA_h, 
                       self.WB_in, self.WB_h, self.VB_h,
                       self.Wother_in, self.Wother_h] 
        self.activation = activation


    def recurrence_interleave(self, x_t, a_t, b_t, A_t):
        """
        :param  x_t: 1D: batch, 2D: dim_hidden
        :param  a_t: 1D: batch, 2D: n_agents; elem=one hot vector for speaker
        :param  b_t: 1D: batch, 2D: n_agents; elem=one hot vector for addressee
        :return A_t: 1D: batch, 2D: n_agents, 3D: dim_agent
        """

        h_a = A_t * a_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_b = A_t * b_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_other = A_t - h_a - h_b             # batch_size x n_agents x dim_agent

        h_a = T.sum(h_a,1)                # batch_size x dim_agent
        h_b = T.sum(h_b,1)                # batch_size x dim_agent

        #h_x = T.dot(T.concatenate([x_t,h_a],1), self.W_in)           # batch_size x dim_hidden
        xt_ha = T.concatenate([x_t, h_a], 1)

        # update for speaker
        #A_t_a = self.activation(h_x + T.dot(h_a, self.W_h) + T.dot(h_b, self.V_h))
        A_t_a = self.activation(T.dot(xt_ha, self.WA_in) + T.dot(h_a, self.WA_h) + T.dot(h_b, self.VA_h))
        A_t_a = A_t_a.dimshuffle(0,'x',1) * a_t.dimshuffle(0,1,'x')

        # update for addressee
        #A_t_b = self.activation(h_x + T.dot(h_b, self.W_h) + T.dot(h_a, self.V_h))
        A_t_b = self.activation(T.dot(xt_ha, self.WB_in) + T.dot(h_b, self.WB_h) + T.dot(h_a, self.VB_h))
        A_t_b = A_t_b.dimshuffle(0,'x',1) * b_t.dimshuffle(0,1,'x')

        # update for others
        h_x = T.dot(xt_ha, self.Wother_in).dimshuffle(0,'x',1) * (1 - (a_t.dimshuffle(0,1,'x') + b_t.dimshuffle(0,1,'x')))
        h_other = T.dot(h_other, self.Wother_h) # batch_size x n_agents x dim_hidden
        A_t_other = self.activation(h_x + h_other)

        return A_t_a + A_t_b + A_t_other


class InterleaveGRU(object):

    def __init__(self, n_i, n_h, activation=tanh):
        # Parameters for speaker
        self.WA_xr = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WA_xp = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WA_xz = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WA_xh = theano.shared(initialize_weights(n_i + n_h, n_h))

        self.WA_hr = theano.shared(initialize_weights(n_h, n_h))
        self.WA_hp = theano.shared(initialize_weights(n_h, n_h))
        self.WA_hz = theano.shared(initialize_weights(n_h, n_h))
        self.WA_hh = theano.shared(initialize_weights(n_h, n_h))

        self.VA_hr = theano.shared(initialize_weights(n_h, n_h))
        self.VA_hp = theano.shared(initialize_weights(n_h, n_h))
        self.VA_hz = theano.shared(initialize_weights(n_h, n_h))
        self.VA_hh = theano.shared(initialize_weights(n_h, n_h))

        # Parameters for addressee
        self.WB_xr = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WB_xp = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WB_xz = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.WB_xh = theano.shared(initialize_weights(n_i + n_h, n_h))

        self.WB_hr = theano.shared(initialize_weights(n_h, n_h))
        self.WB_hp = theano.shared(initialize_weights(n_h, n_h))
        self.WB_hz = theano.shared(initialize_weights(n_h, n_h))
        self.WB_hh = theano.shared(initialize_weights(n_h, n_h))

        self.VB_hr = theano.shared(initialize_weights(n_h, n_h))
        self.VB_hp = theano.shared(initialize_weights(n_h, n_h))
        self.VB_hz = theano.shared(initialize_weights(n_h, n_h))
        self.VB_hh = theano.shared(initialize_weights(n_h, n_h))

        # Parameters for other 
        self.Wother_xr = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.Wother_xz = theano.shared(initialize_weights(n_i + n_h, n_h))
        self.Wother_xh = theano.shared(initialize_weights(n_i + n_h, n_h))

        self.Wother_hr = theano.shared(initialize_weights(n_h, n_h))
        self.Wother_hz = theano.shared(initialize_weights(n_h, n_h))
        self.Wother_hh = theano.shared(initialize_weights(n_h, n_h))


        self.params = [self.WA_xr, self.WA_xp, self.WA_xz, self.WA_xh, self.WA_hr, self.WA_hp, self.WA_hz, self.WA_hh, self.VA_hr, self.VA_hp, self.VA_hz, self.VA_hh,
                       self.WB_xr, self.WB_xp, self.WB_xz, self.WB_xh, self.WB_hr, self.WB_hp, self.WB_hz, self.WB_hh, self.VB_hr, self.VB_hp, self.VB_hz, self.VB_hh,
                       self.Wother_xr, self.Wother_xz, self.Wother_xh, self.Wother_hr, self.Wother_hz, self.Wother_hh
                      ]
        self.activation = activation

    def recurrence_interleave(self, x_t, a_t, b_t, A_t):
        """
        :param  x_t: 1D: batch_size, 2D: dim_hidden
        :param  a_t: 1D: batch, 2D: n_agents; elem=one hot vector for speaker
        :param  b_t: 1D: batch, 2D: n_agents; elem=one hot vector for addressee
        :return A_t: 1D: batch_size, 2D: n_agents, 3D: dim_agent
        """
        h_a = A_t * a_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_b = A_t * b_t.dimshuffle(0, 1, 'x') # batch_size x n_agents x dim_agent
        h_other = A_t - h_a - h_b             # batch_size x n_agents x dim_agent

        h_a = T.sum(h_a,1)                # batch_size x dim_agent
        h_b = T.sum(h_b,1)                # batch_size x dim_agent

        xt_ha = T.concatenate([h_a, x_t],1)         # batch_size x (dim_agent + dim_hidden)

        # update for speaker
        r_t = sigmoid(T.dot(xt_ha, self.WA_xr) + T.dot(h_a, self.WA_hr) + T.dot(h_b, self.VA_hr))
        p_t = sigmoid(T.dot(xt_ha, self.WA_xp) + T.dot(h_a, self.WA_hp) + T.dot(h_b, self.VA_hp))
        z_t = sigmoid(T.dot(xt_ha, self.WA_xz) + T.dot(h_a, self.WA_hz) + T.dot(h_b, self.VA_hz))
        h_hat_t = self.activation(T.dot(xt_ha, self.WA_xh) + T.dot((r_t * h_a), self.WA_hh) + T.dot((p_t * h_b), self.VA_hh))
        ha_t = (1. - z_t) * h_a + z_t * h_hat_t
        A_t_a = ha_t.dimshuffle(0,'x',1) * a_t.dimshuffle(0,1,'x')

        # update for addressee
        r_t = sigmoid(T.dot(xt_ha, self.WB_xr) + T.dot(h_b, self.WB_hr) + T.dot(h_a, self.VB_hr))
        p_t = sigmoid(T.dot(xt_ha, self.WB_xp) + T.dot(h_b, self.WB_hp) + T.dot(h_a, self.VB_hp))
        z_t = sigmoid(T.dot(xt_ha, self.WB_xz) + T.dot(h_b, self.WB_hz) + T.dot(h_a, self.VB_hz))
        h_hat_t = self.activation(T.dot(xt_ha, self.WB_xh) + T.dot((r_t * h_b), self.WB_hh) + T.dot((p_t * h_a), self.VB_hh))
        hb_t = (1. - z_t) * h_b + z_t * h_hat_t
        A_t_b = hb_t.dimshuffle(0,'x',1) * b_t.dimshuffle(0,1,'x')

        # update for others
        x_r = T.dot(xt_ha, self.Wother_xr).dimshuffle(0, 'x', 1) * (1 - (a_t.dimshuffle(0, 1, 'x') + b_t.dimshuffle(0, 1, 'x')))  # batch_size x n_agnets x dim_hidden
        x_z = T.dot(xt_ha, self.Wother_xz).dimshuffle(0, 'x', 1) * (1 - (a_t.dimshuffle(0, 1, 'x') + b_t.dimshuffle(0, 1, 'x')))  # batch_size x n_agnets x dim_hidden
        x_h = T.dot(xt_ha, self.Wother_xh).dimshuffle(0, 'x', 1) * (1 - (a_t.dimshuffle(0, 1, 'x') + b_t.dimshuffle(0, 1, 'x')))  # batch_size x n_agnets x dim_hidden

        r_t = sigmoid(x_r + T.dot(h_other, self.Wother_hr))
        z_t = sigmoid(x_z + T.dot(h_other, self.Wother_hz))
        h_hat_t = self.activation(x_h + T.dot((r_t * h_other), self.Wother_hh))
        h_t = (1. - z_t) * h_other + z_t * h_hat_t 
        A_t_other = h_t

        return A_t_a + A_t_b + A_t_other


class RNN(object):

    def __init__(self, n_i, n_h, activation=tanh):
        self.W_in = theano.shared(initialize_weights(n_i, n_h))
        self.W_h = theano.shared(initialize_weights(n_h, n_h))
        self.params = [self.W_in, self.W_h]
        self.activation = activation

    def recurrence(self, x_t, h_t):
        """
        :param x_t: 1D: batch * n_prev_sents * max_n_agents, 2D: dim_emb
        :return: h_t: 1D: batch * n_prev_sents * max_n_agents, 2D: dim_hidden
        """
        return self.activation(T.dot(x_t, self.W_in) + T.dot(h_t, self.W_h))

    def recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch, 2D: dim_hidden
        :param  i_t: 1D: batch, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch, 2D: n_agents, 3D: dim_agent
        """
        h_x = T.dot(x_t, self.W_in).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')
        h_a = T.dot(A_t, self.W_h)
        return self.activation(h_x + h_a)

    def skip_recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch_size, 2D: dim_hidden
        :param  i_t: 1D: batch_size, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch_size, 2D: n_agents, 3D: dim_agent
        """
        h_x = T.dot(x_t, self.W_in)
        A_sub = A_t[T.arange(A_t.shape[0]), i_t]
        h_a = T.dot(A_sub, self.W_h)
        return self.activation(T.set_subtensor(A_sub, h_x + h_a))


class GRU(object):

    def __init__(self, n_i, n_h, activation=tanh):
        self.W_xr = theano.shared(initialize_weights(n_i, n_h))
        self.W_hr = theano.shared(initialize_weights(n_h, n_h))
        self.W_xz = theano.shared(initialize_weights(n_i, n_h))
        self.W_hz = theano.shared(initialize_weights(n_h, n_h))
        self.W_xh = theano.shared(initialize_weights(n_i, n_h))
        self.W_hh = theano.shared(initialize_weights(n_h, n_h))
        self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]
        self.activation = activation

    def recurrence(self, x_t, h_tm1):
        r_t = sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tm1, self.W_hz))
        h_hat_t = self.activation(T.dot(x_t, self.W_xh) + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t

    def recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch_size, 2D: dim_hidden
        :param  i_t: 1D: batch_size, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch_size, 2D: n_agents, 3D: dim_agent
        """
        x_r = T.dot(x_t, self.W_xr).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')
        x_z = T.dot(x_t, self.W_xz).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')
        x_h = T.dot(x_t, self.W_xh).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')

        r_t = sigmoid(x_r + T.dot(A_t, self.W_hr))
        z_t = sigmoid(x_z + T.dot(A_t, self.W_hz))
        h_hat_t = self.activation(x_h + T.dot((r_t * A_t), self.W_hh))
        h_t = (1. - z_t) * A_t + z_t * h_hat_t

        return h_t

    def skip_recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch_size, 2D: dim_hidden
        :param  i_t: 1D: batch_size, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch_size, 2D: n_agents, 3D: dim_agent
        """
        x_r = T.dot(x_t, self.W_xr)
        x_z = T.dot(x_t, self.W_xz)
        x_h = T.dot(x_t, self.W_xh)

        A_sub = A_t[T.arange(A_t.shape[0]), i_t]
        r_t = sigmoid(x_r + T.dot(A_sub, self.W_hr))
        z_t = sigmoid(x_z + T.dot(A_sub, self.W_hz))
        h_hat_t = self.activation(x_h + T.dot((r_t * A_sub), self.W_hh))
        h_t = (1. - z_t) * A_sub + z_t * h_hat_t

        return T.set_subtensor(A_sub, h_t)
