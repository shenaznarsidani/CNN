from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels =3, kernel_size=3, number_filters = 3, 
                 name="conv2D"),
            MaxPoolingLayer(name="maxpool", pool_size = 2, stride = 2),
            flatten(name="flatten"),
            fc(input_dim=27, output_dim=5, init_scale = 0.02, name="fc")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            



            ConvLayer2D(input_channels = 3, kernel_size = 3, padding = 1,number_filters = 16, name = 'conv2D1'),
            
            gelu(name="geluc1"),
            
            ConvLayer2D(input_channels = 16, kernel_size = 3, stride=2, number_filters = 32, name = 'conv2D2'),
            
            gelu(name="geluc2"),
            MaxPoolingLayer(2,2,'maxpool1'),
            flatten("flatten"),
            
            fc(1568, 100, 0.1, name="fc1"),
            gelu(name="gelu1"),
            fc(100, 100, 0.1, name="fc2"),
            gelu(name="gelu2"),
            fc(100, 100, 0.1, name="fc3"),
            gelu(name="gelu3"),
            fc(100, 100, 0.1, name="fc4"),
            gelu(name="gelu4"),
            fc(100, 100, 0.1, name="fc5"),
            gelu(name="gelu5"),
            fc(100, 20, 0.1, name="fc6")  
            
            ########### END ###########
        )