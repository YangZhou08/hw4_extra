"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input1, input2 = node.inputs 
        output1 = out_grad * input2 * array_api.power(input1, (input2 - 1)) 
        output2 = out_grad * array_api.power(input1, input2) * array_api.log(input1) 
        return output1, output2 
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a.power(self.scalar) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input1 = node.inputs[0] 
        # output = out_grad * self.scalar * array_api.power(input1, (self.scalar - 1)) 
        # output = out_grad * self.scalar * input1.power(self.scalar - 1) 
        output = out_grad * self.scalar * input1**(self.scalar - 1) 
        return output 
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input1, input2 = node.inputs 
        output1 = out_grad / input2 
        output2 = out_grad * (-input1 / power_scalar(input2, 2)) 
        return output1, output2 
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input1 = node.inputs[0] 
        output1 = out_grad / self.scalar 
        return output1 
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes != None: 
            if len(self.axes) != len(a.shape): 
                newaxes = [] 
                for i in range(len(a.shape)): 
                    if i not in self.axes: 
                        newaxes.append(i) 
                    else: 
                        newaxes.append(self.axes[(self.axes.index(i) + 1) % len(self.axes)]) 
                self.axes = tuple(newaxes) 
        else: 
            # newaxes = list(array_api.arange(len(a.shape))) 
            newaxes = list(range(len(a.shape))) 
            # newaxes2 = list(array_api.arange(len(a.shape))) 
            newaxes2 = list(range(len(a.shape))) 
            newaxes[-1] = newaxes2[-2] 
            newaxes[-2] = newaxes2[-1] 
            self.axes = tuple(newaxes) 
        return array_api.transpose(a, self.axes) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes != None: 
            if len(self.axes) != len(node.inputs[0].shape): 
                newaxes = [] 
                for i in range(len(node.inputs[0].shape)): 
                    if i not in self.axes: 
                        newaxes.append(i) 
                    else: 
                        newaxes.append(self.axes[(self.axes.index(i) + 1) % len(self.axes)]) 
                self.axes = tuple(newaxes) 
        else: 
            # newaxes = list(array_api.arange(len(node.inputs[0].shape))) 
            newaxes = list(range(len(node.inputs[0].shape))) 
            # newaxes2 = list(array_api.arange(len(node.inputs[0].shape))) 
            newaxes2 = list(range(len(node.inputs[0].shape)))
            newaxes[-1] = newaxes2[-2] 
            newaxes[-2] = newaxes2[-1] 
            self.axes = tuple(newaxes) 
        if isinstance(out_grad, Tensor): 
            return Tensor(array_api.transpose(out_grad.realize_cached_data(), self.axes), device = out_grad.device, dtype = out_grad.dtype, requires_grad = out_grad.requires_grad) 
        else: 
            return array_api.transpose(out_grad, self.axes) 
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape 
        # print(input_shape) 
        # print(out_grad.shape) 
        # return array_api.reshape(out_grad.numpy(), input_shape) 
        return out_grad.reshape(input_shape) 
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        original_shape = node.inputs[0].shape 
        # pad with ones 
        orig_shape = [1,] * (len(self.shape) - len(original_shape)) + list(original_shape) 
        
        # identify the broadcasted axes 
        broadcastedaxes = [i for i in range(len(self.shape)) if self.shape[i] != orig_shape[i] or orig_shape[i] == 1] 
        
        # sum out the broadcasted axes 
        # out_grad = out_grad.numpy() 
        for axes in sorted(broadcastedaxes, reverse = True): 
            # out_grad = array_api.summation(out_grad, axis = axes) 
            out_grad = out_grad.sum(axes = axes) 
        
        # remove the added ones 
        # out_grad = array_api.squeeze(out_grad, axis = tuple(range(len(self.shape) - len(original_shape)))) 
        out_grad = out_grad.reshape(original_shape) 
        
        # out_grad = out_grad.numpy() 
        # out_grad = array_api.sum(out_grad, axis=tuple(range(len(out_grad.shape) - len(node.inputs[0].shape)))) 
        
        return Tensor(out_grad) 
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        print("sum a shape {}".format(a.shape)) 
        if not isinstance(self.axes, int) and self.axes != None: # support for multiple axes 
            newaxes = list(self.axes) 
            newaxes = sorted(newaxes)[::-1] 
            for i in newaxes: 
                a = array_api.sum(a, axis = i) 
            return a 
        else: 
            return array_api.sum(a, axis=self.axes) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape) 
        if self.axes == None: 
            self.axes = tuple(range(len(node.inputs[0].shape))) 
        elif isinstance(self.axes, int): 
            self.axes = (self.axes,) 
        for i in list(self.axes): 
            # out_grad = array_api.expand_dims(out_grad, axis = i) we cannot use expand_dims unsaqweeze anymore 
            new_shape[i] = 1 
        out_grad = array_api.reshape(out_grad, new_shape) 
        out_grad = array_api.broadcast_to(out_grad, node.inputs[0].shape) 
        # out_grad = out_grad.broadcast_to(node.inputs[0].shape) 
        return out_grad 
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # return array_api.matmul(a, b) # (n, m) * (m, p) = (n, p) 
        print("matmul a {}".format(a.shape)) 
        print("matmul b {}".format(b.shape)) 
        output = a @ b # (n, m) * (m, p) = (n, p) 
        # print("matmul {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # output1 = matmul(out_grad, transpose(node.inputs[1])) 
        # output2 = matmul(transpose(node.inputs[0]), out_grad) 
        a = node.inputs[0] 
        b = node.inputs[1] 
        
        # A = a.numpy() 
        # B = b.numpy() 
        
        A_shape = a.shape 
        B_shape = b.shape 
        
        # # Compute output shape
        # out_shape = array_api.broadcast_shapes(A_shape[:-2], B_shape[:-2]) + (A_shape[-2], B_shape[-1]) 
        
        # # Reshape inputs to 2D or 3D
        # A_reshaped = array_api.broadcast_to(A, out_shape[:-1] + (A_shape[-1],)).reshape(-1, A_shape[-2], A_shape[-1]) 
        # B_reshaped = array_api.broadcast_to(B, out_shape[:-2] + (B_shape[-2], B_shape[-1])).reshape(-1, B_shape[-2], B_shape[-1]) 
        # # grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-2], grad_output.shape[-1]) 
        # out_grad = out_grad.numpy() 
        # out_grad = out_grad.reshape(-1, out_grad.shape[-2], out_grad.shape[-1]) 
        
        # Compute gradients
        # grad_A = np.matmul(grad_output_reshaped, B_reshaped.swapaxes(-1, -2)) 
        grad_A = out_grad @ b.transpose() 
        # grad_B = np.matmul(A_reshaped.swapaxes(-1, -2), grad_output_reshaped) 
        grad_B = a.transpose() @ out_grad 
        
        # Reshape gradients to match input shapes
        # grad_A = grad_A.reshape(out_shape[:-1] + (A_shape[-1],)) 
        # grad_B = grad_B.reshape(out_shape[:-2] + (B_shape[-2], B_shape[-1])) 
        
        # Sum over broadcasted dimensions 
        if len(grad_A.shape) > len(a.shape): 
            grad_A = grad_A.sum(axes=tuple(range(len(grad_A.shape) - len(a.shape)))) 
        if len(grad_B.shape) > len(b.shape): 
            grad_B = grad_B.sum(axes=tuple(range(len(grad_B.shape) - len(b.shape)))) 
        
        # output1 = array_api.squeeze(output1, axis = tuple(range(len(output1.shape) - len(a.shape)))).astype("float32") 
        # output2 = array_api.squeeze(output2, axis = tuple(range(len(output2.shape) - len(b.shape)))).astype("float32") 
        
        # output1 = Tensor(output1) 
        # output2 = Tensor(output2) 
        # print(type(output1), type(output2)) 
        # print("matmul gradient {} and {}".format(output1.dtype, output2.dtype)) 
        
        # return output1, output2 
        return grad_A, grad_B 
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.negative(a) 
        # output = array_api.negative(a) 
        # output = mul_scalar(a, -1) 
        output = a.negative() 
        # print("negate {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return array_api.negative(out_grad) 
        # output = array_api.negative(out_grad) 
        # output = mul_scalar(out_grad, -1) 
        if isinstance(out_grad, Tensor): 
            output = mul_scalar(out_grad, -1) 
        else: 
            output = out_grad.negative() 
        # print("negate gradient {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        output = array_api.log(a) 
        # print("log {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return out_grad * (1 / node.inputs[0].numpy().astype(float)) 
        output = out_grad * (1 / node.inputs[0]) 
        # print("log gradient {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.exp(a) 
        output = array_api.exp(a) 
        # print("exp {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # output = out_grad * array_api.exp(node.inputs[0].numpy().astype(float)) 
        output = out_grad * exp(node.inputs[0]) 
        # print("exp gradient {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.maximum(a, 0) 
        # output = array_api.maximum(a, 0) 
        output = a.maximum(0) 
        # print("relu {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return out_grad * array_api.where(node.inputs[0].numpy() > 0, 1, 0) 
        # print("out_grad {}".format(out_grad.dtype)) 
        # output = out_grad * array_api.where(node.inputs[0].numpy() > 0, 1, 0).astype("float32") 
        output = out_grad * Tensor(node.inputs[0].realize_cached_data() > 0, device = node.inputs[0].device, dtype = node.inputs[0].dtype) 
        # print("relu gradient {}".format(output.dtype)) 
        return output 
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        output = array_api.tanh(a) 
        
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        output = (-tanh(node.inputs[0]) ** 2 + 1) * out_grad 
        return output 
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        newshape = list(args[0].shape) 
        newshape.insert(self.axis, len(args)) 
        output = array_api.empty(newshape, dtype = args[0].dtype, device = args[0].device) 
        for i, tensor in enumerate(args): 
            index = [slice(None) for _ in range(len(newshape))] 
            index[self.axis] = i 
            output[tuple(index)] = tensor 
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis) 
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis] 
        each_shape = list(A.shape) 
        each_shapetwo = [] 
        for i in range(len(each_shape)): 
            if i != self.axis: 
                each_shapetwo.append(each_shape[i]) 
        each_shape = tuple(each_shapetwo) 
        splits = [] 
        for i in range(n): 
            index = [slice(None) for _ in range(len(A.shape))] 
            index[self.axis] = i 
            # newtensor = array_api.empty(each_shapetwo, dtype = A.dtype, device = A.device) 
            newtensor = A[tuple(index)].compact() 
            newtensor = newtensor.reshape(each_shapetwo) 
            splits.append(newtensor) 
        return tuple(splits) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis) 
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes) 
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        newshape = list(a.shape) 
        if isinstance(self.axes, int): 
            self.axes = (self.axes,) 
        for axis in self.axes: 
            if axis >= len(a.shape): 
                continue 
            newshape[axis] = newshape[axis] + newshape[axis] * self.dilation 
        
        # output = array_api.empty(newshape, dtype = a.dtype, device = a.device) 
        output = array_api.full(newshape, 0, dtype = a.dtype, device = a.device) 
        
        index = [slice(None) for _ in range(len(newshape))] 
        for axis in self.axes: 
            if axis >= len(a.shape): 
                continue 
            index[axis] = slice(None, newshape[axis], self.dilation + 1) 
        output[tuple(index)] = a 
        
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation) 
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        newshape = list(a.shape) 
        if isinstance(self.axes, int): 
            self.axes = (self.axes,) 
        
        index = [slice(None) for _ in range(len(newshape))] 
        for axis in self.axes: 
            if axis >= len(a.shape): 
                continue 
            index[axis] = slice(None, a.shape[axis], self.dilation + 1) 
        
        return a[tuple(index)].compact() 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation) 
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))) 
        
        N, H, W, Cin = A.shape 
        K, _, _, Cout = B.shape 
        Ns, Hs, Ws, Cs = A.strides 
        inner_dim = K * K * Cin 
        
        # out_h = (H - K + 2 * self.padding) // self.stride + 1 
        out_h = (H - K) // self.stride + 1 
        # out_w = (W - K + 2 * self.padding) // self.stride + 1 
        out_w = (W - K) // self.stride + 1 
        withstridedhs = Hs * self.stride 
        withstridedws = Ws * self.stride 
        
        output = A.as_strided(
            shape = (N, out_h, out_w, K, K, Cin), 
            strides = (Ns, withstridedhs, withstridedws, Hs, Ws, Cs) 
        ).compact().reshape((N * out_h * out_w, inner_dim)) 
        B = B.compact() 
        output = output @ B.reshape((inner_dim, Cout)) 
        output = output.compact() 
        output = output.reshape((N, out_h, out_w, Cout)) 
        
        return output 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X = node.inputs[0] 
        W = node.inputs[1] 
        
        # compute Xgrad 
        K, K, Cin, Cout = W.shape 
        '''
        grad_prod = out_grad.pad(((0, 0), (K - 1, K - 1), (K - 1, K - 1), (0, 0))) 
        ''' 
        if self.stride > 1: 
            grad_prod = dilate(out_grad, (1, 2), self.stride - 1) 
        else: 
            grad_prod = out_grad 
        W_prod = flip(W, (0, 1)) 
        W_prod = transpose(W_prod, (0, 1, 3, 2)) # (K, K, Cin, Cout) -> (K, K, Cout, Cin) 
        Xgrad = conv(grad_prod, W_prod, padding = K - 1 - self.padding) 
        
        # compute Wgrad 
        if self.stride > 1: 
            grad_prod = dilate(out_grad, (1, 2), self.stride - 1) 
        else: 
            grad_prod = out_grad 
        # (N, H - K + 1, W - K + 1, Cout) -> (H - K + 1, W - K + 1, N, Cout) 
        grad_prod = transpose(grad_prod, (1, 2, 0, 3)) # (H - K + 1, N, W - K + 1, Cout) -> (N, H - K + 1, W - K + 1, Cout) 
        X_prod = transpose(X, (3, 1, 2, 0)) # (N, H, W, Cin) -> (Cin, W, H, N) 
        interme = conv(X_prod, grad_prod, padding = self.padding) 
        Wgrad = transpose(interme, (1, 2, 0, 3)) # (K, Cin, K, Cout) -> (K, K, Cin, Cout) 
        
        return Xgrad, Wgrad 
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


