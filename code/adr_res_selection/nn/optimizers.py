from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from nn_utils import build_shared_zeros


class GradClip(theano.compile.ViewOp):
    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        def pgrad(g_out):
            g_out = T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound)
            g_out = ifelse(T.any(T.isnan(g_out)), T.ones_like(g_out)*0.00001, g_out)
            return g_out
        return [pgrad(g_out) for g_out in g_outs]

#gradient_clipper = GradClip(-10.0, 10.0)
#T.opt.register_canonicalize(theano.gof.OpRemove(gradient_clipper), name='gradient_clipper')

def adam_clip(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8, grad_clip=1.0):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """

    gradient_clipper = GradClip(-grad_clip, grad_clip)

    updates = []
    all_grads = theano.grad(gradient_clipper(loss), all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

def grad_clipping(g, s):
    g_norm = T.abs_(g)
    return T.switch(g_norm > s, (s * g) / g_norm, g)


def optimizer_select(opt, params, grads, lr=0.001):
    if opt == 'adagrad':
        return ada_grad(grads=grads, params=params, lr=lr)
    elif opt == 'ada_delta':
        return ada_delta(grads=grads, params=params)
    elif opt == 'adam':
        return adam(grads=grads, params=params, lr=lr)
    return sgd(grads=grads, params=params, lr=lr)


def sgd(grads, params, lr=0.1):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        updates[p] = p - lr * g
    return updates


def ada_grad(grads, params, lr=0.1, eps=1.):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        r = build_shared_zeros(p.get_value(True).shape)
        r_t = r + T.sqr(g)
        p_t = p - (lr / (T.sqrt(r_t) + eps)) * g
        updates[r] = r_t
        updates[p] = p_t
    return updates


def ada_delta(grads, params, b=0.999, eps=1e-8):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        r = build_shared_zeros(p.get_value(True).shape)
        v = build_shared_zeros(p.get_value(True).shape)
        s = build_shared_zeros(p.get_value(True).shape)
        r_t = b * r + (1 - b) * T.sqr(g)
        v_t = (T.sqrt(s) + eps) / (T.sqrt(r) + eps) * g
        s_t = b * s + (1 - b) * T.sqr(v_t)
        p_t = p - v_t
        updates[r] = r_t
        updates[v] = v_t
        updates[s] = s_t
        updates[p] = p_t
    return updates


def adam(grads, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    updates = OrderedDict()
    i = theano.shared(np.float32(0.))
    i_t = i + 1.

    for p, g in zip(params, grads):
        v = build_shared_zeros(p.get_value(True).shape)
        r = build_shared_zeros(p.get_value(True).shape)

        v_t = (b1 * v) + (1. - b1) * g
        r_t = (b2 * r) + (1. - b2) * T.sqr(g)

        r_hat = lr / (T.sqrt(r_t / (1. - b2 ** i_t)) + e)
        v_hat = v / (1. - b1 ** i_t)

        p_t = p - r_hat * v_hat
        updates[v] = v_t
        updates[r] = r_t
        updates[p] = p_t

    updates[i] = i_t
    return updates
