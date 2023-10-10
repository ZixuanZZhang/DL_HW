import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.conv=nn.Conv2d(3, 32, 7, 1,0)
        self.pool=nn.MaxPool2d(2, 2)
        self.linear=nn.Linear(5408,10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        out1=self.conv(x)
        out2=self.pool(out1)
        out3=out2.view(out2.size(0), -1)
        outs=self.linear(out3)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs