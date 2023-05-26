from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)
        
        #### MLP ####
        if not hasattr(self.net, "preprocess") and \
           not hasattr(self.net, "rnn") and \
           not hasattr(self.net, "postprocess"):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        for n, dv in layer.grads.items():
            #############################################################################
            # TODO: Implement the SGD with (optional) Weight Decay                      #
            #############################################################################
            if self.weight_decay > 0:
                dv += self.weight_decay * layer.params[n]
            diff = -self.lr*dv
            layer.params[n] += -self.lr*dv - self.weight_decay * layer.params[n]
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################



class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t
        self.weight_decay=weight_decay

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam with [optinal] Weight Decay                      #
        #############################################################################
        self.t = self.t+1
        for n, dv in layer.grads.items():
            if n not in self.mt:
                self.mt[n] = np.zeros(layer.params[n].shape)
            if n not in self.vt:
                self.vt[n] = np.zeros(layer.params[n].shape)
            self.mt[n] = self.beta1*self.mt[n] + ((1 - self.beta1)*layer.grads[n])
            self.vt[n] = self.beta2*self.vt[n] + ((1 - self.beta2)*np.square(layer.grads[n]))
            M = self.mt[n]/(1 - self.beta1**self.t)
            V = self.vt[n]/(1 - self.beta2**self.t)
            layer.params[n] -= (M*self.lr)/(np.power(V,0.5) + self.eps) 
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
