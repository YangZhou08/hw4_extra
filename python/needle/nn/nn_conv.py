"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module 
import math 
from .nn_basic import BatchNorm2d 
from .nn_basic import ReLU 

class ConvBN(Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, bias = True, device = None, dtype = "float32"): 
        super().__init__() 
        self.conv = Conv(in_channels, out_channels, kernel_size, stride = stride, bias = bias, device = device, dtype = dtype) 
        self.bn = BatchNorm2d(out_channels, device = device, dtype = dtype) 
        self.relu = ReLU() 
        
    def forward(self, x): 
        return self.relu(self.bn(self.conv(x))) 
class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in = in_channels * kernel_size * kernel_size, fan_out = out_channels * kernel_size * kernel_size, shape = (kernel_size, kernel_size, in_channels, out_channels), nonlinearity = "relu", device = device, dtype = dtype)) # kkio
        
        inchanelkernelsize = in_channels * kernel_size * kernel_size
        if bias: 
            self.bias = Parameter(init.rand(out_channels, low = -1.0 / math.sqrt(inchanelkernelsize), high = 1.0 / math.sqrt(inchanelkernelsize)), device = device, dtype = dtype) # o 
        else: 
            self.bias = None 
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_prod = x.transpose((1, 2)).transpose((2, 3)) 
        
        padding = (self.kernel_size - 1) // 2 
        # output = ops.conv(x_prod, self.kernel, stride = self.stride, padding = padding) 
        output = ops.conv(x_prod, self.weight, stride = self.stride, padding = padding) 
        if self.bias != None: 
            output = output + ops.reshape(self.bias, (1, 1, 1, self.out_channels)).broadcast_to(output.shape) 
        return output.transpose((2, 3)).transpose((1, 2)) # (N, H, W, C) -> (N, C, H, W) 
        ### END YOUR SOLUTION