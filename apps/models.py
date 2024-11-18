import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
         ###
        # raise NotImplementedError() ### 
        self.conv1 = nn.ConvBN(3, 16, 7, 4, device = device, dtype = dtype) 
        self.conv2 = nn.ConvBN(16, 32, 3, 2, device = device, dtype = dtype) 
        self.conv3 = nn.ConvBN(32, 32, 3, 1, device = device, dtype = dtype) 
        self.conv4 = nn.ConvBN(32, 32, 3, 1, device = device, dtype = dtype) 
        # self.conv6 = ConvBN(32, 64, 3, 2) 
        self.conv5 = nn.ConvBN(32, 64, 3, 2, device = device, dtype = dtype) 
        # self.conv7 = ConvBN(64, 128, 3, 2) 
        self.conv6 = nn.ConvBN(64, 128, 3, 2, device = device, dtype = dtype) 
        # self.conv8 = ConvBN(128, 128, 3, 1) 
        self.conv7 = nn.ConvBN(128, 128, 3, 1, device = device, dtype = dtype) 
        # self.conv9 = ConvBN(128, 128, 3, 1) 
        self.conv8 = nn.ConvBN(128, 128, 3, 1, device = device, dtype = dtype) 
        self.fc = nn.Linear(128, 128, device = device, dtype = dtype) 
        self.relu = nn.ReLU() 
        self.en = nn.Linear(128, 10, device = device, dtype = dtype) 
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv1(x) 
        x = self.conv2(x) 
        output = self.conv3(x) 
        output = self.conv4(output) 
        output += x 
        x = output 
        x = self.conv5(x) 
        x = self.conv6(x) 
        output = self.conv7(x) 
        output = self.conv8(output) 
        output += x 
        x = output 
        x = x.reshape((x.shape[0], 128)) 
        x = self.fc(x) 
        x = self.relu(x) 
        x = self.en(x) 
        
        return x 
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device = device, dtype = dtype) 
        if seq_model == 'rnn': 
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device = device, dtype = dtype) 
        else: 
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device = device, dtype = dtype) 
        self.fc = nn.Linear(hidden_size, output_size, device = device, dtype = dtype) 
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x) 
        out, h = self.seq_model(x, h) 
        out = out.reshape((out.shape[0] * out.shape[1], out.shape[2])) 
        out = self.fc(out) 
        return out, h 
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
