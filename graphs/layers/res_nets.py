import torch
from torch import nn
from torch.nn import functional as F
from graphs.layers.masked_conv2d import MaskedConv2d


class ResBlock(nn.Module):
    """
    One ResNet block that will be used repeatedly to build the GMM net.
    """
    def __init__(self, features, bias=True):
        """
        (parameters follow nn.Linear())
        :param features: size of each input and output sample
        :param bias: If set to False, the layer will not learn an additive bias. Default: True
        """
        super(ResBlock, self).__init__()
        # some layers
        self.linear = nn.Linear(features, features, bias=bias)
        # self.bnorm = nn.BatchNorm1d(features)
        self.nonlinearity = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        y = self.linear(x)
        # y = self.bnorm(y)
        z = self.nonlinearity(y)
        # z = self.dropout(z)
        return x + z


class ResBlock2d(nn.Module):
    """
    One ResNet2d block that will be used repeatedly to build the GMM net in gmm with masked convolutions.
    """
    def __init__(self, mask_type, *args, **kwargs):
        """
        (parameters follow )
        """
        super(ResBlock2d, self).__init__()
        # some layers
        self.mconv2d_1 = MaskedConv2d(mask_type, *args, **kwargs)
        self.nonlTanh = nn.Tanh()
        self.activation = nn.LeakyReLU()
        self.mconv2d_2 = MaskedConv2d(mask_type, *args, **kwargs)

    def forward_old(self, x):
        # see for resnet variants :
        # https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        xout = self.mconv2d_1(x)
        y = self.nonlTanh(xout)
        yout = self.mconv2d_2(y)
        z = self.nonlTanh(yout)
        return x + z

    def forward(self, x):
        # see for resnet variants :
        # https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        xout = self.mconv2d_1(x)
        y = self.activation(xout)
        yout = self.mconv2d_2(y)
        z = yout + x
        return z
