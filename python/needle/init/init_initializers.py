import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(2) * math.sqrt(3.0 / (fan_in)) 
    if shape != None: 
        # fan_in = shape[0] * shape[1] * shape[2] # K * K * Cin 
        # fan_out = shape[0] * shape[1] * shape[3] # K * K * Cout 
        return rand(*shape, low = -bound, high = bound, **kwargs) 
    else: 
        return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs) 
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2) / math.sqrt(fan_in) 
    return randn(fan_in, fan_out, mean = 0, std = std, **kwargs) 
    ### END YOUR SOLUTION