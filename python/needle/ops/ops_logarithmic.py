from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis = 1, keepdims = True) 
        intermediate = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis = 1, keepdims = True)) 
        
        # output = intermediate + array_api.squeeze(max_Z, axis = -1) 
        output = intermediate 
        # print("logsoftmax output {}".format(((Z - max_Z) - output).dtype)) 
        return (Z - max_Z) - output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0] 
        
        max_Z = array_api.max(Z, axis = 1, keepdims = True) 
        # intermediates = exp(add(Z, -max_Z)) 
        # return intermediates/summation(intermediates, axis = self.axes) * out_grad 
        intermediates = array_api.exp(Z - max_Z.broadcast_to(Z.shape)) 
        # if isinstance(out_grad, Value): 
            # out_grad = out_grad.numpy().astype("float32") 
        
        # output = intermediates / array_api.sum(intermediates, axis = self.axes, keepdims = True) * out_grad 
        # output = out_grad / array_api.sum(intermediates, axis = self.axes) 
        output = array_api.summation(intermediates, axis = 1, keepdims = True) 
        axes = tuple(range(len(Z.shape))) 
        # for axis in axes: 
            # out_grad = out_grad.expand_dims(axis) 
            # out_grad = array_api.expand_dims(out_grad, axis) 
        # output = array_api.broadcast_to(array_api.reshape(output, newshape), Z.shape) 
        # out_grad = array_api.broadcast_to(array_api.reshape(out_grad, newshape), Z.shape) 
        # out_grad = array_api.broadcast_to(out_grad, Z.shape) 
        output = intermediates / output 
        
        # output = array_api.ones((Z.shape[1], Z.shape[1])) - output 
        # output = array_api.eye(Z.shape[1]) - output 
        # output = output * out_grad 
        # output = array_api.matmul(out_grad, output) 
        output = out_grad - (array_api.broadcast_to(array_api.summation(out_grad, axis = -1, keepdims = True), out_grad.shape) * array_api.exp(node)) 
        
        # print("logsoftmax gradient {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis = self.axes, keepdims = True)
        print("z shape {}, max_z shape {}".format(Z.shape, max_Z.shape)) 
        if self.axes == None: 
            max_Z = max_Z.reshape([1 for i in range(len(Z.shape))]) 
        # intermediate = log(summation(exp(add(Z, -max_Z)), axis = self.axes)) 
        intermediate = array_api.log(array_api.summation(array_api.exp(Z - max_Z.broadcast_to(Z.shape)), axis = self.axes)) 
        
        output = intermediate + Z.max(axis = self.axes) 
        # print("logsumexp output {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0] 
        
        max_Z = array_api.max(Z.realize_cached_data(), axis = self.axes, keepdims = True) 
        max_Z = Tensor(max_Z, device = Z.device) 
        # intermediates = exp(add(Z, -max_Z)) 
        # return intermediates/summation(intermediates, axis = self.axes) * out_grad 
        intermediates = exp(Z - max_Z.broadcast_to(Z.shape)) 
        # if isinstance(out_grad, Value): 
            # out_grad = out_grad.numpy().astype("float32") 
        
        # output = intermediates / array_api.sum(intermediates, axis = self.axes, keepdims = True) * out_grad 
        # output = out_grad / array_api.sum(intermediates, axis = self.axes) 
        output = summation(intermediates, axes = self.axes) 
        newshape = list(Z.shape) 
        output = out_grad / output 
        if self.axes == None: 
            self.axes = tuple(range(len(Z.shape))) 
        elif isinstance(self.axes, int): 
            self.axes = (self.axes,) 
        for axis in self.axes: 
            # out_grad = out_grad.expand_dims(axis) 
            newshape[axis] = 1 
        output = output.reshape(newshape).broadcast_to(Z.shape) 
        # output = array_api.broadcast_to(array_api.reshape(output, newshape), Z.shape) 
        # out_grad = array_api.broadcast_to(array_api.reshape(out_grad, newshape), Z.shape) 
        # out_grad = array_api.broadcast_to(out_grad, Z.shape) 
        # output = intermediates / output 
        output = intermediates * output 
        
        '''
        if isinstance(out_grad, Value): 
            out_grad = out_grad.numpy().astype("float32") 
        print(self.axes) 
        if self.axes is None: 
            self.axes = tuple(range(len(Z.shape))) 
        max_Z = array_api.max(Z, axis = self.axes, keepdims = True) 
        intermediates = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis = self.axes)) 
        
        # output = array_api.exp((Z - max_Z) - intermediates) 
        output = array_api.exp((Z - max_Z) - intermediates) 
        output = output * out_grad 
        output = Tensor(output) 
        '''
        # print("logsumexp gradient {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

