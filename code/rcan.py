import torch
import torch.nn as nn
from networks import *


class Net(nn.Module):
    def __init__(self, scale_factor, base_filter=256, K=5, L=3, M=3, N=3, num_channel=3, num_filters=64, kernel_size=3):
        super(Net, self).__init__()
        self.kernel_size = kernel_size
        self.CONV = RegularConv(in_channels=num_channel, out_channels=base_filter, kernel_size=kernel_size, K=K)
        self.RCAM = ResidualAttentionModule(in_channels=num_channel, out_channels=num_filters, kernel_size=kernel_size,
                                            L=L, M=M)
        self.RRDB = RRDB(in_channels=num_channel, out_channels=num_filters, kernel_size=kernel_size, N=N)
        self.UPSACLE = Upsampling(scale_factor=scale_factor)

    def forward(self, x, r):
        residual = r
        conv = self.CONV(x)
        rcam = self.RCAM(conv)
        rrdb = self.RRDB(rcam)
        up = self.UPSACLE(rrdb)
        layer = torch.add(up, residual)
        out = RegularConv(3, 1, kernel_size=self.kernel_size)(layer)

        return out
