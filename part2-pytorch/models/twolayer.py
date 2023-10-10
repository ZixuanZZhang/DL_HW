import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.linear1=nn.Linear(input_dim,hidden_size)
        self.linear2=nn.Linear(hidden_size,num_classes)
        self.sigmoid=nn.Sigmoid()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x= x.view(x.size(0), -1)
        h_sigmoid=self.sigmoid(self.linear1(x))
        out=self.linear2(h_sigmoid)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out