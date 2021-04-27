# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:14:20 2015
@author: Balázs Hidasi
"""

import os
import os.path

orig_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ[
    "THEANORC"
] = ".theanorc_gru4rec"  # Only affects the actual settings if theano was not imported before this point (by any module)
import theano
from theano import function
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from . import custom_opt, datatools
from .gpu_ops import gpu_diag, gpu_searchsorted

os.chdir(orig_cwd)
import pickle
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

mrng = MRG_RandomStreams()


class GRU4Rec:
    """
    GRU4Rec(loss='bpr-max', final_act='elu-1', hidden_act='tanh', layers=[100],
                 n_epochs=10, batch_size=32, dropout_p_hidden=0.0, dropout_p_embed=0.0, learning_rate=0.1, momentum=0.0, lmbd=0.0, embedding=0, n_sample=2048, sample_alpha=0.75, smoothing=0.0, constrained_embedding=False,
                 adapt='adagrad', adapt_params=[], grad_cap=0.0, bpreg=1.0, logq=0.0,
                 sigma=0.0, init_as_normal=False, train_random_order=False, time_sort=True,
                 session_key='SessionId', item_key='ItemId', time_key='Time')
    Initializes the network.

    Parameters
    -----------
    loss : 'top1', 'bpr', 'cross-entropy', 'xe_logit', 'top1-max', 'bpr-max'
        selects the loss function (default : 'bpr-max')
    final_act : 'softmax', 'linear', 'relu', 'tanh', 'softmax_logit', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>'
        selects the activation function of the final layer, <X> and <Y> are the parameters of the activation function (default : 'elu-1')
    hidden_act : 'linear', 'relu', 'tanh', 'leaky-<X>', 'elu-<X>', 'selu-<X>-<Y>'
        selects the activation function on the hidden states, <X> and <Y> are the parameters of the activation function (default : 'tanh')
    layers : list of int values
        list of the number of GRU units in the layers (default : [100])
    n_epochs : int
        number of training epochs (default: 10)
    batch_size : int
        size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 32)
    dropout_p_hidden : float
        probability of dropout of hidden units (default: 0.0)
    dropout_p_embed : float
        probability of dropout of the input units, applicable only if embeddings are used (default: 0.0)
    learning_rate : float
        learning rate (default: 0.05)
    momentum : float
        if not zero, Nesterov momentum will be applied during training with the given strength (default: 0.0)
    lmbd : float
        coefficient of the L2 regularization (default: 0.0)
    embedding : int
        size of the embedding used, 0 means not to use embedding (default: 0)
    n_sample : int
        number of additional negative samples to be used (besides the other examples of the minibatch) (default: 2048)
    sample_alpha : float
        the probability of an item used as an additional negative sample is supp^sample_alpha (default: 0.75)
        (e.g.: sample_alpha=1 --> popularity based sampling; sample_alpha=0 --> uniform sampling)
    smoothing : float
        (only works with cross-entropy and xe_logit losses) if set to non-zero class labels are smoothed with this value, i.e. the expected utput is (e/N, ..., e/N, 1-e+e/N, e/N, ..., e/N) instead of (0, ..., 0, 1, 0, ..., 0), where N is the number of outputs and e is the smoothing value (default: 0.0)
    constrained_embedding : bool
        if True, the output weight matrix is also used as input embedding (default: False)
    adapt : None, 'adagrad', 'rmsprop', 'adam', 'adadelta'
        sets the appropriate learning rate adaptation strategy, use None for standard SGD (default: 'adagrad')
    adapt_params : list
        parameters for the adaptive learning methods (default: [])
    grad_cap : float
        clip gradients that exceede this value to this value, 0 means no clipping (default: 0.0)
    bpreg : float
        score regularization coefficient for the BPR-max loss function (default: 1.0)
    logq : float
        logq normalization of negative samples (set between 0.0 and 1.0), usually useful with cross-entropy loss (default: 0.0)
    sigma : float
        "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0.0)
    init_as_normal : boolean
        False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
    train_random_order : boolean
        whether to randomize the order of sessions in each epoch (default: False)
    time_sort : boolean
        whether to ensure the the order of sessions is chronological (default: True)
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')

    """

    def __init__(
        self,
        loss="bpr-max",
        final_act="linear",
        hidden_act="tanh",
        layers=[100],
        n_epochs=10,
        batch_size=32,
        dropout_p_hidden=0.0,
        dropout_p_embed=0.0,
        learning_rate=0.1,
        momentum=0.0,
        lmbd=0.0,
        embedding=0,
        n_sample=2048,
        sample_alpha=0.75,
        smoothing=0.0,
        constrained_embedding=False,
        adapt="adagrad",
        adapt_params=[],
        grad_cap=0.0,
        bpreg=1.0,
        logq=0.0,
        sigma=0.0,
        init_as_normal=False,
        train_random_order=False,
        time_sort=True,
        session_key="SessionId",
        item_key="ItemId",
        time_key="Time",
        seed=42
    ):
        self.layers = layers
        if not isinstance(self.layers, list):
            self.layers = [self.layers]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_p_hidden = dropout_p_hidden
        self.dropout_p_embed = dropout_p_embed
        self.learning_rate = learning_rate
        self.adapt_params = adapt_params
        self.momentum = momentum
        self.sigma = sigma
        self.init_as_normal = init_as_normal
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.grad_cap = grad_cap
        self.bpreg = bpreg
        self.logq = logq
        self.train_random_order = train_random_order
        self.lmbd = lmbd
        self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.time_sort = time_sort
        self.adapt = adapt
        self.loss = loss
        self.set_loss_function(self.loss)
        self.final_act = final_act
        self.set_final_activation(self.final_act)
        self.hidden_act = hidden_act
        self.set_hidden_activation(self.hidden_act)
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.smoothing = smoothing
        self.seed = seed

    def set_loss_function(self, loss):
        if loss == "cross-entropy":
            self.loss_function = self.cross_entropy
        elif loss == "bpr":
            self.loss_function = self.bpr
        elif loss == "bpr-max":
            self.loss_function = self.bpr_max
        elif loss == "top1":
            self.loss_function = self.top1
        elif loss == "top1-max":
            self.loss_function = self.top1_max
        elif loss == "xe_logit":
            self.loss_function = self.cross_entropy_logits
        else:
            raise NotImplementedError

    def set_final_activation(self, final_act):
        if final_act == "linear":
            self.final_activation = self.linear
        elif final_act == "relu":
            self.final_activation = self.relu
        elif final_act == "softmax":
            self.final_activation = self.softmax
        elif final_act == "tanh":
            self.final_activation = self.tanh
        elif final_act == "softmax_logit":
            self.final_activation = self.softmax_logit
        elif final_act.startswith("leaky-"):
            self.final_activation = self.LeakyReLU(
                float(final_act.split("-")[1])
            ).execute
        elif final_act.startswith("elu-"):
            self.final_activation = self.Elu(float(final_act.split("-")[1])).execute
        elif final_act.startswith("selu-"):
            self.final_activation = self.Selu(
                *[float(x) for x in final_act.split("-")[1:]]
            ).execute
        else:
            raise NotImplementedError

    def set_hidden_activation(self, hidden_act):
        if hidden_act == "relu":
            self.hidden_activation = self.relu
        elif hidden_act == "tanh":
            self.hidden_activation = self.tanh
        elif hidden_act == "linear":
            self.hidden_activation = self.linear
        elif hidden_act.startswith("leaky-"):
            self.hidden_activation = self.LeakyReLU(
                float(hidden_act.split("-")[1])
            ).execute
        elif hidden_act.startswith("elu-"):
            self.hidden_activation = self.Elu(float(hidden_act.split("-")[1])).execute
        elif hidden_act.startswith("selu-"):
            self.hidden_activation = self.Selu(
                *[float(x) for x in hidden_act.split("-")[1:]]
            ).execute
        else:
            raise NotImplementedError

    def set_params(self, **kvargs):
        maxk_len = np.max([len(str(x)) for x in kvargs.keys()])
        maxv_len = np.max([len(str(x)) for x in kvargs.values()])
        for k, v in kvargs.items():
            if not hasattr(self, k):
                print("Unkown attribute: {}".format(k))
                raise NotImplementedError
            else:
                if type(v) == str and k == "adapt_params":
                    v = [float(l) for l in v.split("/")]
                elif type(v) == str and type(getattr(self, k)) == list:
                    v = [int(l) for l in v.split("/")]
                if type(v) == str and type(getattr(self, k)) == bool:
                    if v == "True" or v == "1":
                        v = True
                    elif v == "False" or v == "0":
                        v = False
                    else:
                        print("Invalid value for boolean parameter: {}".format(v))
                        raise NotImplementedError
                setattr(self, k, type(getattr(self, k))(v))
                if k == "loss":
                    self.set_loss_function(self.loss)
                if k == "final_act":
                    self.set_final_activation(self.final_act)
                if k == "hidden_act":
                    self.set_hidden_activation(self.hidden_act)
                print(
                    "SET   {}{}TO   {}{}(type: {})".format(
                        k,
                        " " * (maxk_len - len(k) + 3),
                        getattr(self, k),
                        " " * (maxv_len - len(str(getattr(self, k))) + 3),
                        type(getattr(self, k)),
                    )
                )

    ######################ACTIVATION FUNCTIONS#####################
    def linear(self, X):
        return X

    def tanh(self, X):
        return T.tanh(X)

    def softmax(self, X):
        e_x = T.exp(X - X.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def softmax_logit(self, X):
        X = X - X.max(axis=1, keepdims=True)
        return T.log(T.exp(X).sum(axis=1, keepdims=True)) - X

    def softmax_neg(self, X):
        hm = 1.0 - T.eye(*X.shape)
        X = X * hm
        e_x = T.exp(X - X.max(axis=1, keepdims=True)) * hm
        return e_x / e_x.sum(axis=1, keepdims=True)

    def relu(self, X):
        return T.maximum(X, 0)

    def sigmoid(self, X):
        return T.nnet.sigmoid(X)

    class Selu:
        def __init__(self, lmbd, alpha):
            self.lmbd = lmbd
            self.alpha = alpha

        def execute(self, X):
            return self.lmbd * T.switch(T.ge(X, 0), X, self.alpha * (T.exp(X) - 1))

    class Elu:
        def __init__(self, alpha):
            self.alpha = alpha

        def execute(self, X):
            return T.switch(T.ge(X, 0), X, self.alpha * (T.exp(X) - 1))

    class LeakyReLU:
        def __init__(self, leak):
            self.leak = leak

        def execute(self, X):
            return T.switch(T.ge(X, 0), X, self.leak * X)

    #################################LOSS FUNCTIONS################################
    def cross_entropy(self, yhat, M):
        if self.smoothing:
            n_out = M + self.n_sample
            return T.cast(
                T.sum(
                    (1.0 - (n_out / (n_out - 1)) * self.smoothing)
                    * (-T.log(gpu_diag(yhat) + 1e-24))
                    + (self.smoothing / (n_out - 1))
                    * T.sum(-T.log(yhat + 1e-24), axis=1)
                ),
                theano.config.floatX,
            )
        else:
            return T.cast(T.sum(-T.log(gpu_diag(yhat) + 1e-24)), theano.config.floatX)

    def cross_entropy_logits(self, yhat, M):
        if self.smoothing:
            n_out = M + self.n_sample
            return T.cast(
                T.sum(
                    (1.0 - (n_out / (n_out - 1)) * self.smoothing) * gpu_diag(yhat)
                    + (self.smoothing / (n_out - 1)) * T.sum(yhat, axis=1)
                ),
                theano.config.floatX,
            )
        else:
            return T.cast(T.sum(gpu_diag(yhat)), theano.config.floatX)

    def bpr(self, yhat, M):
        return T.cast(
            T.sum(-T.log(T.nnet.sigmoid(gpu_diag(yhat, keepdims=True) - yhat))),
            theano.config.floatX,
        )

    def bpr_max(self, yhat, M):
        softmax_scores = self.softmax_neg(yhat)
        return T.cast(
            T.sum(
                -T.log(
                    T.sum(
                        T.nnet.sigmoid(gpu_diag(yhat, keepdims=True) - yhat)
                        * softmax_scores,
                        axis=1,
                    )
                    + 1e-24
                )
                + self.bpreg * T.sum((yhat ** 2) * softmax_scores, axis=1)
            ),
            theano.config.floatX,
        )

    def top1(self, yhat, M):
        ydiag = gpu_diag(yhat, keepdims=True)
        return T.cast(
            T.sum(
                T.mean(
                    T.nnet.sigmoid(-ydiag + yhat) + T.nnet.sigmoid(yhat ** 2), axis=1
                )
                - T.nnet.sigmoid(ydiag ** 2) / (M + self.n_sample)
            ),
            theano.config.floatX,
        )

    def top1_max(self, yhat, M):
        softmax_scores = self.softmax_neg(yhat)
        y = softmax_scores * (
            T.nnet.sigmoid(-gpu_diag(yhat, keepdims=True) + yhat)
            + T.nnet.sigmoid(yhat ** 2)
        )
        return T.cast(T.sum(T.sum(y, axis=1)), theano.config.floatX)

    ###############################################################################
    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)

    def init_weights(self, shape, name=None):
        return theano.shared(self.init_matrix(shape), borrow=True, name=name)

    def init_matrix(self, shape):
        if self.sigma != 0:
            sigma = self.sigma
        else:
            sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return self.floatX(np.random.randn(*shape) * sigma)
        else:
            return self.floatX(np.random.rand(*shape) * sigma * 2 - sigma)

    def extend_weights(self, W, n_new):
        matrix = W.get_value()
        sigma = (
            self.sigma
            if self.sigma != 0
            else np.sqrt(6.0 / (matrix.shape[0] + matrix.shape[1] + n_new))
        )
        if self.init_as_normal:
            new_rows = self.floatX(np.random.randn(n_new, matrix.shape[1]) * sigma)
        else:
            new_rows = self.floatX(
                np.random.rand(n_new, matrix.shape[1]) * sigma * 2 - sigma
            )
        W.set_value(np.vstack([matrix, new_rows]))

    def init(self, data):
        datatools.sort_if_needed(data, [self.session_key, self.time_key])
        offset_sessions = datatools.compute_offset(data, self.session_key)
        np.random.seed(self.seed)
        self.Wx, self.Wh, self.Wrz, self.Bh, self.H = [], [], [], [], []
        if self.constrained_embedding:
            n_features = self.layers[-1]
        elif self.embedding:
            self.E = self.init_weights((self.n_items, self.embedding), name="E")
            n_features = self.embedding
        else:
            n_features = self.n_items
        for i in range(len(self.layers)):
            m = []
            m.append(
                self.init_matrix(
                    (self.layers[i - 1] if i > 0 else n_features, self.layers[i])
                )
            )
            m.append(
                self.init_matrix(
                    (self.layers[i - 1] if i > 0 else n_features, self.layers[i])
                )
            )
            m.append(
                self.init_matrix(
                    (self.layers[i - 1] if i > 0 else n_features, self.layers[i])
                )
            )
            self.Wx.append(
                theano.shared(value=np.hstack(m), borrow=True, name="Wx{}".format(i))
            )  # For compatibility's sake
            self.Wh.append(
                self.init_weights(
                    (self.layers[i], self.layers[i]), name="Wh{}".format(i)
                )
            )
            m2 = []
            m2.append(self.init_matrix((self.layers[i], self.layers[i])))
            m2.append(self.init_matrix((self.layers[i], self.layers[i])))
            self.Wrz.append(
                theano.shared(value=np.hstack(m2), borrow=True, name="Wrz{}".format(i))
            )  # For compatibility's sake
            self.Bh.append(
                theano.shared(
                    value=np.zeros((self.layers[i] * 3,), dtype=theano.config.floatX),
                    borrow=True,
                    name="Bh{}".format(i),
                )
            )
            self.H.append(
                theano.shared(
                    value=np.zeros(
                        (self.batch_size, self.layers[i]), dtype=theano.config.floatX
                    ),
                    borrow=True,
                    name="H{}".format(i),
                )
            )
        self.Wy = self.init_weights((self.n_items, self.layers[-1]), name="Wy")
        self.By = theano.shared(
            value=np.zeros((self.n_items, 1), dtype=theano.config.floatX),
            borrow=True,
            name="By",
        )
        return offset_sessions

    def dropout(self, X, drop_p):
        if drop_p > 0:
            retain_prob = 1 - drop_p
            X *= (
                mrng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
                / retain_prob
            )
        return X

    def adam(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        v1 = self.adapt_params[0]
        v2 = 1.0 - self.adapt_params[0]
        v3 = self.adapt_params[1]
        v4 = 1.0 - self.adapt_params[1]
        acc = theano.shared(param.get_value(borrow=False) * 0.0, borrow=True)
        meang = theano.shared(param.get_value(borrow=False) * 0.0, borrow=True)
        countt = theano.shared(param.get_value(borrow=False) * 0.0, borrow=True)
        if sample_idx is None:
            acc_new = v3 * acc + v4 * (grad ** 2)
            meang_new = v1 * meang + v2 * grad
            countt_new = countt + 1
            updates[acc] = acc_new
            updates[meang] = meang_new
            updates[countt] = countt_new
        else:
            acc_s = acc[sample_idx]
            meang_s = meang[sample_idx]
            countt_s = countt[sample_idx]
            #            acc_new = v3 * acc_s + v4 * (grad**2) #Faster, but inaccurate when an index occurs multiple times
            #            updates[acc] = T.set_subtensor(acc_s, acc_new) #Faster, but inaccurate when an index occurs multiple times
            updates[acc] = T.inc_subtensor(
                T.set_subtensor(acc_s, acc_s * v3)[sample_idx], v4 * (grad ** 2)
            )  # Slower, but accurate when an index occurs multiple times
            acc_new = updates[acc][
                sample_idx
            ]  # Slower, but accurate when an index occurs multiple times
            #            meang_new = v1 * meang_s + v2 * grad
            #            updates[meang] = T.set_subtensor(meang_s, meang_new) #Faster, but inaccurate when an index occurs multiple times
            updates[meang] = T.inc_subtensor(
                T.set_subtensor(meang_s, meang_s * v1)[sample_idx], v2 * (grad ** 2)
            )  # Slower, but accurate when an index occurs multiple times
            meang_new = updates[meang][
                sample_idx
            ]  # Slower, but accurate when an index occurs multiple times
            countt_new = countt_s + 1.0
            updates[countt] = T.set_subtensor(countt_s, countt_new)
        return (meang_new / (1 - v1 ** countt_new)) / (
            T.sqrt(acc_new / (1 - v1 ** countt_new)) + epsilon
        )

    def adagrad(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        acc = theano.shared(param.get_value(borrow=False) * 0.0, borrow=True)
        if sample_idx is None:
            acc_new = acc + grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            acc_new = acc_s + grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling

    def adadelta(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        v1 = self.adapt_params[0]
        v2 = 1.0 - self.adapt_params[0]
        acc = theano.shared(param.get_value(borrow=False) * 0.0, borrow=True)
        upd = theano.shared(param.get_value(borrow=False) * 0.0, borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * (grad ** 2)
            updates[acc] = acc_new
            grad_scaling = (upd + epsilon) / (acc_new + epsilon)
            upd_new = v1 * upd + v2 * grad_scaling * (grad ** 2)
            updates[upd] = upd_new
        else:
            acc_s = acc[sample_idx]
            #            acc_new = v1 * acc_s + v2 * (grad**2) #Faster, but inaccurate when an index occurs multiple times
            #            updates[acc] = T.set_subtensor(acc_s, acc_new) #Faster, but inaccurate when an index occurs multiple times
            updates[acc] = T.inc_subtensor(
                T.set_subtensor(acc_s, acc_s * v1)[sample_idx], v2 * (grad ** 2)
            )  # Slower, but accurate when an index occurs multiple times
            acc_new = updates[acc][
                sample_idx
            ]  # Slower, but accurate when an index occurs multiple times
            upd_s = upd[sample_idx]
            grad_scaling = (upd_s + epsilon) / (acc_new + epsilon)
            #            updates[upd] = T.set_subtensor(upd_s, v1 * upd_s + v2 * grad_scaling * (grad**2)) #Faster, but inaccurate when an index occurs multiple times
            updates[upd] = T.inc_subtensor(
                T.set_subtensor(upd_s, upd_s * v1)[sample_idx],
                v2 * grad_scaling * (grad ** 2),
            )  # Slower, but accurate when an index occurs multiple times
        gradient_scaling = T.cast(T.sqrt(grad_scaling), theano.config.floatX)
        if self.learning_rate != 1.0:
            print(
                "Warn: learning_rate is not 1.0 while using adadelta. Setting learning_rate to 1.0"
            )
            self.learning_rate = 1.0
        return grad * gradient_scaling  # Ok, checked

    def rmsprop(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        v1 = self.adapt_params[0]
        v2 = 1.0 - self.adapt_params[0]
        acc = theano.shared(param.get_value(borrow=False) * 0.0, borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            #            acc_new = v1 * acc_s + v2 * grad ** 2 #Faster, but inaccurate when an index occurs multiple times
            #            updates[acc] = T.set_subtensor(acc_s, acc_new) #Faster, but inaccurate when an index occurs multiple times
            updates[acc] = T.inc_subtensor(
                T.set_subtensor(acc_s, acc_s * v1)[sample_idx], v2 * grad ** 2
            )  # Slower, but accurate when an index occurs multiple times
            acc_new = updates[acc][
                sample_idx
            ]  # Slower, but accurate when an index occurs multiple times
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling

    def RMSprop(self, cost, params, full_params, sampled_params, sidxs, epsilon=1e-6):
        grads = [T.grad(cost=cost, wrt=param) for param in params]
        sgrads = [T.grad(cost=cost, wrt=sparam) for sparam in sampled_params]
        updates = OrderedDict()
        if self.grad_cap > 0:
            norm = T.cast(
                T.sqrt(
                    T.sum([T.sum([T.sum(g ** 2) for g in g_list]) for g_list in grads])
                    + T.sum([T.sum(g ** 2) for g in sgrads])
                ),
                theano.config.floatX,
            )
            grads = [
                [
                    T.switch(T.ge(norm, self.grad_cap), g * self.grad_cap / norm, g)
                    for g in g_list
                ]
                for g_list in grads
            ]
            sgrads = [
                T.switch(T.ge(norm, self.grad_cap), g * self.grad_cap / norm, g)
                for g in sgrads
            ]
        for p_list, g_list in zip(params, grads):
            for p, g in zip(p_list, g_list):
                if self.adapt == "adagrad":
                    g = self.adagrad(p, g, updates)
                elif self.adapt == "rmsprop":
                    g = self.rmsprop(p, g, updates)
                elif self.adapt == "adadelta":
                    g = self.adadelta(p, g, updates)
                elif self.adapt == "adam":
                    g = self.adam(p, g, updates)
                if self.momentum > 0:
                    velocity = theano.shared(
                        p.get_value(borrow=False) * 0.0, borrow=True
                    )
                    velocity2 = self.momentum * velocity - self.learning_rate * (
                        g + self.lmbd * p
                    )
                    updates[velocity] = velocity2
                    updates[p] = p + velocity2
                else:
                    updates[p] = (
                        p * (1.0 - self.learning_rate * self.lmbd)
                        - self.learning_rate * g
                    )
        for i in range(len(sgrads)):
            g = sgrads[i]
            fullP = full_params[i]
            sample_idx = sidxs[i]
            sparam = sampled_params[i]
            if self.adapt == "adagrad":
                g = self.adagrad(fullP, g, updates, sample_idx)
            elif self.adapt == "rmsprop":
                g = self.rmsprop(fullP, g, updates, sample_idx)
            elif self.adapt == "adadelta":
                g = self.adadelta(fullP, g, updates, sample_idx)
            elif self.adapt == "adam":
                g = self.adam(fullP, g, updates, sample_idx)
            if self.lmbd > 0:
                delta = self.learning_rate * (g + self.lmbd * sparam)
            else:
                delta = self.learning_rate * g
            if self.momentum > 0:
                velocity = theano.shared(
                    fullP.get_value(borrow=False) * 0.0, borrow=True
                )
                vs = velocity[sample_idx]
                velocity2 = self.momentum * vs - delta
                updates[velocity] = T.set_subtensor(vs, velocity2)
                updates[fullP] = T.inc_subtensor(sparam, velocity2)
            else:
                updates[fullP] = T.inc_subtensor(sparam, -delta)
        return updates

    def model(
        self,
        X,
        H,
        M,
        R=None,
        Y=None,
        drop_p_hidden=0.0,
        drop_p_embed=0.0,
        predict=False,
    ):
        sparams, full_params, sidxs = [], [], []
        if (
            (hasattr(self, "ST"))
            and (Y is not None)
            and (not predict)
            and (self.n_sample > 0)
        ):
            A = self.ST[self.STI]
            Y = T.concatenate([Y, A], axis=0)
        if self.constrained_embedding:
            if Y is not None:
                X = T.concatenate([X, Y], axis=0)
            S = self.Wy[X]
            Sx = S[:M]
            Sy = S[M:]
            y = self.dropout(Sx, drop_p_embed)
            H_new = []
            start = 0
            sparams.append(S)
            full_params.append(self.Wy)
            sidxs.append(X)
        elif self.embedding:
            Sx = self.E[X]
            y = self.dropout(Sx, drop_p_embed)
            H_new = []
            start = 0
            sparams.append(Sx)
            full_params.append(self.E)
            sidxs.append(X)
        else:
            Sx = self.Wx[0][X]
            vec = Sx + self.Bh[0]
            rz = T.nnet.sigmoid(vec[:, self.layers[0] :] + T.dot(H[0], self.Wrz[0]))
            h = self.hidden_activation(
                T.dot(H[0] * rz[:, : self.layers[0]], self.Wh[0])
                + vec[:, : self.layers[0]]
            )
            z = rz[:, self.layers[0] :]
            h = (1.0 - z) * H[0] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            H_new = [T.switch(R, 0, h) if not predict else h]
            start = 1
            sparams.append(Sx)
            full_params.append(self.Wx[0])
            sidxs.append(X)
        for i in range(start, len(self.layers)):
            vec = T.dot(y, self.Wx[i]) + self.Bh[i]
            rz = T.nnet.sigmoid(vec[:, self.layers[i] :] + T.dot(H[i], self.Wrz[i]))
            h = self.hidden_activation(
                T.dot(H[i] * rz[:, : self.layers[i]], self.Wh[i])
                + vec[:, : self.layers[i]]
            )
            z = rz[:, self.layers[i] :]
            h = (1.0 - z) * H[i] + z * h
            h = self.dropout(h, drop_p_hidden)
            y = h
            H_new.append(T.switch(R, 0, h) if not predict else h)
        if Y is not None:
            if (not self.constrained_embedding) or predict:
                Sy = self.Wy[Y]
                sparams.append(Sy)
                full_params.append(self.Wy)
                sidxs.append(Y)
            SBy = self.By[Y]
            sparams.append(SBy)
            full_params.append(self.By)
            sidxs.append(Y)
            if predict and self.final_act == "softmax_logit":
                y = self.softmax(T.dot(y, Sy.T) + SBy.flatten())
            else:
                y = T.dot(y, Sy.T) + SBy.flatten()
                if not predict and self.logq:
                    y = y - self.logq * T.log(
                        T.concatenate(
                            [self.P0[Y[:M]], self.P0[Y[M:]] ** self.sample_alpha],
                            axis=0,
                        )
                    )
                y = self.final_activation(y)
            return H_new, y, sparams, full_params, sidxs
        else:
            if predict and self.final_act == "softmax_logit":
                y = self.softmax(T.dot(y, self.Wy.T) + self.By.flatten())
            else:
                y = T.dot(y, self.Wy.T) + self.By.flatten()
                if not predict and self.logq:
                    y = y - self.logq * T.log(self.P0)
                y = self.final_activation(y)
            return H_new, y, sparams, full_params, sidxs

    def generate_neg_samples(self, pop, length):
        if self.sample_alpha:
            sample = np.searchsorted(pop, np.random.rand(self.n_sample * length))
        else:
            sample = np.random.choice(self.n_items, size=self.n_sample * length)
        if length > 1:
            sample = sample.reshape((length, self.n_sample))
        return sample

    def fit(self, data, sample_store=10000000, store_type="gpu"):
        """
        Trains the network.

        Parameters
        --------
        data : pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        sample_store : int
            If additional negative samples are used (n_sample > 0), the efficiency of GPU utilization can be sped up, by precomputing a large batch of negative samples (and recomputing when necessary).
            This parameter regulizes the size of this precomputed ID set. Its value is the maximum number of int values (IDs) to be stored. Precomputed IDs are stored in the RAM.
            For the most efficient computation, a balance must be found between storing few examples and constantly interrupting GPU computations for a short time vs. computing many examples and interrupting GPU computations for a long time (but rarely).
        store_type : 'cpu', 'gpu'
            Where to store the negative sample buffer (sample store). The cpu mode is legacy and is no longer supported.

        """
        self.predict = None
        self.error_during_train = False
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(
            data=np.arange(self.n_items), index=itemids, name="ItemIdx"
        )
        data["ItemIdx"] = self.itemidmap[data[self.item_key].values].values
        offset_sessions = self.init(data)
        pop = data.groupby(self.item_key).size()
        if self.logq:
            self.P0 = theano.shared(
                pop[self.itemidmap.index.values].values.astype(theano.config.floatX),
                name="P0",
                borrow=False,
            )
        if self.n_sample:
            pop = pop[self.itemidmap.index.values].values ** self.sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            if sample_store:
                generate_length = sample_store // self.n_sample
                if generate_length <= 1:
                    sample_store = 0
                    print("No example store was used")
                elif store_type == "cpu":
                    neg_samples = self.generate_neg_samples(pop, generate_length)
                    sample_pointer = 0
                    print(
                        "Created sample store with {} batches of samples (type=CPU)".format(
                            generate_length
                        )
                    )
                elif store_type == "gpu":
                    P = theano.shared(pop.astype(theano.config.floatX), name="P")
                    self.ST = theano.shared(
                        np.zeros((generate_length, self.n_sample), dtype="int64")
                    )
                    self.STI = theano.shared(np.asarray(0, dtype="int64"))
                    X = mrng.uniform((generate_length * self.n_sample,))
                    updates_st = OrderedDict()
                    updates_st[self.ST] = gpu_searchsorted(
                        P, X, dtype_int64=True
                    ).reshape((generate_length, self.n_sample))
                    updates_st[self.STI] = np.asarray(0, dtype="int64")
                    generate_samples = theano.function([], updates=updates_st)
                    generate_samples()
                    sample_pointer = 0
                    print(
                        "Created sample store with {} batches of samples (type=GPU)".format(
                            generate_length
                        )
                    )
                else:
                    print("Invalid store type {}".format(store_type))
                    raise NotImplementedError
            else:
                print("No example store was used")
        X = T.ivector(name="X")
        Y = T.ivector(name="Y")
        M = T.iscalar(name="M")
        R = T.bcol(name="R")
        H_new, Y_pred, sparams, full_params, sidxs = self.model(
            X, self.H, M, R, Y, self.dropout_p_hidden, self.dropout_p_embed
        )
        cost = self.loss_function(Y_pred, M) / self.batch_size
        params = [
            self.Wx if self.embedding or self.constrained_embedding else self.Wx[1:],
            self.Wh,
            self.Wrz,
            self.Bh,
        ]
        updates = self.RMSprop(cost, params, full_params, sparams, sidxs)
        for i in range(len(self.H)):
            updates[self.H[i]] = H_new[i]
        if hasattr(self, "STI"):
            updates[self.STI] = self.STI + 1
        train_function = function(
            inputs=[X, Y, M, R],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True,
            on_unused_input="ignore",
        )
        base_order = (
            np.argsort(data.groupby(self.session_key)[self.time_key].min().values)
            if self.time_sort
            else np.arange(len(offset_sessions) - 1)
        )
        data_items = data.ItemIdx.values
        for epoch in range(self.n_epochs):
            t0 = time.time()
            for i in range(len(self.layers)):
                self.H[i].set_value(
                    np.zeros(
                        (self.batch_size, self.layers[i]), dtype=theano.config.floatX
                    ),
                    borrow=True,
                )
            c = []
            cc = []
            session_idx_arr = (
                np.random.permutation(len(offset_sessions) - 1)
                if self.train_random_order
                else base_order
            )
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]
            finished = False
            while not finished:
                minlen = (end - start).min()
                out_idx = data_items[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    out_idx = data_items[start + i + 1]
                    if self.n_sample and store_type == "cpu":
                        if sample_store:
                            if sample_pointer == generate_length:
                                neg_samples = self.generate_neg_samples(
                                    pop, generate_length
                                )
                                sample_pointer = 0
                            sample = neg_samples[sample_pointer]
                            sample_pointer += 1
                        else:
                            sample = self.generate_neg_samples(pop, 1)
                        y = np.hstack([out_idx, sample])
                    else:
                        y = out_idx
                        if self.n_sample:
                            if sample_pointer == generate_length:
                                generate_samples()
                                sample_pointer = 0
                            sample_pointer += 1
                    reset = start + i + 1 == end - 1
                    cost = train_function(
                        in_idx, y, len(iters), reset.reshape(len(reset), 1)
                    )
                    c.append(cost)
                    cc.append(len(iters))
                    if np.isnan(cost):
                        print(str(epoch) + ": NaN error!")
                        self.error_during_train = True
                        return
                start = start + minlen - 1
                finished_mask = end - start <= 1
                n_finished = finished_mask.sum()
                iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
                maxiter += n_finished
                valid_mask = iters < len(offset_sessions) - 1
                n_valid = valid_mask.sum()
                if (n_valid == 0) or (n_valid < 2 and self.n_sample == 0):
                    finished = True
                    break
                mask = finished_mask & valid_mask
                sessions = session_idx_arr[iters[mask]]
                start[mask] = offset_sessions[sessions]
                end[mask] = offset_sessions[sessions + 1]
                iters = iters[valid_mask]
                start = start[valid_mask]
                end = end[valid_mask]
                if n_valid < len(valid_mask):
                    for i in range(len(self.H)):
                        tmp = self.H[i].get_value(borrow=True)
                        tmp = tmp[valid_mask]
                        self.H[i].set_value(tmp, borrow=True)
            c = np.array(c)
            cc = np.array(cc)
            avgc = np.sum(c * cc) / np.sum(cc)
            if np.isnan(avgc):
                print("Epoch {}: NaN error!".format(str(epoch)))
                self.error_during_train = True
                return
            t1 = time.time()
            dt = t1 - t0
            print(
                "Epoch{} --> loss: {:.6f} \t({:.2f}s) \t[{:.2f} mb/s | {:.0f} e/s]".format(
                    epoch + 1, avgc, dt, len(c) / dt, np.sum(cc) / dt
                )
            )
        if hasattr(self, "ST"):
            del self.ST
            del self.STI

    def predict_next_batch(
        self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100
    ):
        """
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        """
        if self.error_during_train:
            raise Exception
        if self.predict is None or self.predict_batch != batch:
            self.predict_batch = batch
            X = T.ivector()
            Y = T.ivector()
            M = (
                T.iscalar()
                if self.constrained_embedding or (predict_for_item_ids is not None)
                else None
            )
            for i in range(len(self.layers)):
                self.H[i].set_value(
                    np.zeros((batch, self.layers[i]), dtype=theano.config.floatX),
                    borrow=True,
                )
            if predict_for_item_ids is not None:
                H_new, yhat, _, _, _ = self.model(X, self.H, M, Y=Y, predict=True)
            else:
                H_new, yhat, _, _, _ = self.model(X, self.H, M, predict=True)
            updatesH = OrderedDict()
            for i in range(len(self.H)):
                updatesH[self.H[i]] = H_new[i]
            if predict_for_item_ids is not None:
                if self.constrained_embedding:
                    self.predict = function(
                        inputs=[X, Y, M],
                        outputs=yhat,
                        updates=updatesH,
                        allow_input_downcast=True,
                    )
                else:
                    self.predict = function(
                        inputs=[X, Y],
                        outputs=yhat,
                        updates=updatesH,
                        allow_input_downcast=True,
                    )
            else:
                if self.constrained_embedding:
                    self.predict = function(
                        inputs=[X, M],
                        outputs=yhat,
                        updates=updatesH,
                        allow_input_downcast=True,
                    )
                else:
                    self.predict = function(
                        inputs=[X],
                        outputs=yhat,
                        updates=updatesH,
                        allow_input_downcast=True,
                    )
            self.current_session = np.ones(batch) * -1
        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:
            for i in range(len(self.H)):
                tmp = self.H[i].get_value(borrow=True)
                tmp[session_change] = 0
                self.H[i].set_value(tmp, borrow=True)
            self.current_session = session_ids.copy()
        in_idxs = self.itemidmap[input_item_ids]
        if predict_for_item_ids is not None:
            iIdxs = self.itemidmap[predict_for_item_ids]
            if self.constrained_embedding:
                preds = np.asarray(self.predict(in_idxs, iIdxs, batch)).T
            else:
                preds = np.asarray(self.predict(in_idxs, iIdxs)).T
            return pd.DataFrame(data=preds, index=predict_for_item_ids)
        else:
            if self.constrained_embedding:
                preds = np.asarray(self.predict(in_idxs, batch)).T
            else:
                preds = np.asarray(self.predict(in_idxs)).T
            return pd.DataFrame(data=preds, index=self.itemidmap.index)

    def symbolic_predict(self, X, Y, M, items, batch_size):
        if not self.constrained_embedding:
            M = None
        H = []
        for i in range(len(self.layers)):
            H.append(
                theano.shared(
                    np.zeros((batch_size, self.layers[i]), dtype=theano.config.floatX)
                )
            )
        if items is not None:
            H_new, yhat, _, _, _ = self.model(X, H, M, Y=Y, predict=True)
        else:
            H_new, yhat, _, _, _ = self.model(X, H, M, predict=True)
        updatesH = OrderedDict()
        for i in range(len(H)):
            updatesH[H[i]] = H_new[i]
        return yhat, H, updatesH

    def savemodel(self, fname):
        # Get model parameters for GPU-CPU compatibility
        if self.embedding:
            self.E = self.E.get_value()
        for i in range(len(self.layers)):
            self.Wx[i] = self.Wx[i].get_value()
            self.Wrz[i] = self.Wrz[i].get_value()
            self.Wh[i] = self.Wh[i].get_value()
            self.Bh[i] = self.Bh[i].get_value()
            self.H[i] = self.H[i].get_value()
        self.Wy = self.Wy.get_value()
        self.By = self.By.get_value()
        # Write the model
        with open(fname, "wb") as f:
            pickle.dump(self, f)
        # Reload the parameters
        if self.embedding:
            self.E = theano.shared(self.E, borrow=True, name="E")
        for i in range(len(self.layers)):
            self.Wx[i] = theano.shared(self.Wx[i], borrow=True, name="Wx{}".format(i))
            self.Wrz[i] = theano.shared(
                self.Wrz[i], borrow=True, name="Wrz{}".format(i)
            )
            self.Wh[i] = theano.shared(self.Wh[i], borrow=True, name="Wh{}".format(i))
            self.Bh[i] = theano.shared(self.Bh[i], borrow=True, name="Bh{}".format(i))
            self.H[i] = theano.shared(self.H[i], borrow=True, name="H{}".format(i))
        self.Wy = theano.shared(self.Wy, borrow=True, name="Wy")
        self.By = theano.shared(self.By, borrow=True, name="By")

    @classmethod
    def loadmodel(cls, fname):
        gru = pd.read_pickle(fname)
        if gru.embedding:
            gru.E = theano.shared(gru.E, borrow=True, name="E")
        for i in range(len(gru.layers)):
            gru.Wx[i] = theano.shared(gru.Wx[i], borrow=True, name="Wx{}".format(i))
            gru.Wrz[i] = theano.shared(gru.Wrz[i], borrow=True, name="Wrz{}".format(i))
            gru.Wh[i] = theano.shared(gru.Wh[i], borrow=True, name="Wh{}".format(i))
            gru.Bh[i] = theano.shared(gru.Bh[i], borrow=True, name="Bh{}".format(i))
            gru.H[i] = theano.shared(gru.H[i], borrow=True, name="H{}".format(i))
        gru.Wy = theano.shared(gru.Wy, borrow=True, name="Wy")
        gru.By = theano.shared(gru.By, borrow=True, name="By")
        return gru

    def predict_next(
        self,
        session_id,
        input_item_id,
        predict_for_item_ids=None,
        skip=False,
        type="view",
        timestamp=0,
    ):
        """
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.
        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.
        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.
        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        """

        if not input_item_id in self.itemidmap.index:
            return None

        return self.predict_next_batch(
            np.array([session_id]), np.array([input_item_id]), predict_for_item_ids, 1
        )[0]

