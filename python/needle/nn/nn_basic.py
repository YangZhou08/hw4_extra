"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device = device, dtype = dtype)) 
        # self.bias = Parameter(init.kaiming_uniform(fan_in = 1, fan_out = out_features, device = device, dtype = dtype)) 
        if bias: 
            self.bias = Parameter(init.kaiming_uniform(fan_in = out_features, fan_out = 1, device = device, dtype = dtype).transpose()) 
        else: 
            self.bias = None 
        # self.bias.data = self.bias.data.transpose() 
        # self.bias = ops.broadcast_to(self.bias, (1, out_features)) 
        # self.bias = ops.transpose(self.bias) 
        # self.bias = Parameter(init.kaiming_uniform(fan_in = 1, fan_out = out_features, device = device, dtype = dtype)) 
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        intermediates = ops.matmul(X, self.weight) 
        # print(self.bias.shape) 
        if self.bias != None: 
            biasbroadcast = ops.broadcast_to(self.bias, intermediates.shape) 
            # print(biasbroadcast.shape) 
            # print("linear output {}".format((intermediates + biasbroadcast).dtype)) 
            return intermediates + biasbroadcast 
        else: 
            return intermediates 
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x) 
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device = device, dtype = dtype)) 
        self.bias = Parameter(init.zeros(dim, device = device, dtype = dtype)) 
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes = (1,)) / x.shape[1] 
        print("shapeone {} oneshape {}".format((x.shape[0], 1), x.shape)) 
        mean = ops.broadcast_to(mean.reshape((x.shape[0], 1)), x.shape) 
        # print(mean.shape) 
        # power = init.ones_like(x, device = x.device) * 2 
        variance = ops.summation(ops.power_scalar(x - mean, 2), axes = (1,)) / (x.shape[1]) 
        # power = init.ones_like(variance, device = variance.device) * 0.5 
        std = ops.power_scalar(variance + self.eps, 0.5) 
        std = ops.broadcast_to(std.reshape((x.shape[0], 1)), x.shape) 
        intermediates = (x - mean) / std 
        # print(variance.shape) 
        # intermediates = (x - mean) / ops.power(variance + self.eps, 0.5) 
        # print(intermediates.shape) 
        intermediates = intermediates * ops.broadcast_to(self.weight.reshape((1, x.shape[1])), intermediates.shape) + ops.broadcast_to(self.bias.reshape((1, x.shape[1])), intermediates.shape) 
        return intermediates 
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training: 
            # self.p = np.float32(self.p) 
            # mask = init.randb(*x.shape, p = (1 - self.p), device = x.device, dtype = "float32") 
            mask = init.randb(*x.shape, p = (1 - self.p), device = x.device, dtype = x.dtype) 
            mask = mask / (1 - self.p) 
            output = x * mask 
            # print("dropout output {}".format(output.dtype)) 
            return output 
        else: 
            return x 
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
