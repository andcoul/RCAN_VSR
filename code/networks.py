import torch
import torch.nn as nn


class RegularConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation='relu', bn=True,
                 bias=False, convolution="3d", K=0):
        super(RegularConv, self).__init__()
        self.in_channels = in_channels
        self.convolution = convolution
        if self.convolution == '3d':
            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias)
            self.bn = nn.BatchNorm3d(in_channels) if bn else None
        if self.convolution == '2d':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias)
            self.bn = nn.BatchNorm2d(in_channels) if bn else None
        self.K = K

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        K = self.K
        if K == 0:
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.activation is not None:
                x = self.act(x)
        else:
            for i in range(K):
                x = self.conv(x)
                if self.bn is not None:
                    x = self.bn(x)
                if self.activation is not None:
                    x = self.act(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, stride=1, padding=1, dilation=2, activation='relu',
                 bn=True, bias=False, convolution='3d'):
        super(DilatedConv, self).__init__()
        self.in_channel = in_channel
        if self.convolution == '3d':
            self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias)
            self.bn = nn.BatchNorm3d(in_channel) if bn else None
        if self.convolution == '2d':
            self.conv = nn.Conv2d(in_channels=in_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=bias)
            self.bn = nn.BatchNorm2d(in_channel) if bn else None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.act(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseBlock, self).__init__()
        self.conv1 = RegularConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = RegularConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv3 = RegularConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv4 = RegularConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv5 = RegularConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 activation='None')

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return out + x


'''Residual In Residual Dense Block'''


class RRDB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N):
        super(RRDB, self).__init__()
        self.DB1 = DenseBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.DB2 = DenseBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.DB3 = DenseBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.N = N

    def forward(self, x):
        for i in range(self.N):
            res = self.DB1(x)
            res = self.DB2(res)
            out = self.DB3(res)
            x = out + x

        return x


'''Attention branchs(gates) '''


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = DilatedConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, activation='sigmoid')

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        return x * x_out


'''Cross-domain Attention'''


class CrossAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(CrossAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


''' Residual Cross-domain Attention Block'''


class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = RegularConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = RegularConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 activation='None')
        self.CA = CrossAttention()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        at = self.CA(x2)
        return at + x


'''Residual Cross-domain Attention Group'''


class ResidualAttentionGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, M):
        super(ResidualAttentionGroup, self).__init__()
        self.M = M
        self.RCAB = ResidualAttentionBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.M):
            x = self.RCAB(x)
        return x


'''Residual Cross-domain Attention Module'''


class ResidualAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, L, M):
        super(ResidualAttentionModule, self).__init__()
        self.L = L
        self.M = M
        self.RCAG = ResidualAttentionGroup(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           M=M)

    def forward(self, x):
        out = self.RCAG(x)
        for i in range(self.L - 1):
            out = self.RCAG(out)
        return out


class Upsampling(nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling, self).__init__()

        return "..."
