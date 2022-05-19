# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChebConv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, DEVICE, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(ChebConv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = DEVICE
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (N, V, C, T)
        :return: (N, V, C_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (N, V, C_in)
            graph_signal = graph_signal.to(self.DEVICE)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (N, V, C_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (V,V)
                T_k = T_k.to(self.DEVICE)
                T_k_with_at = T_k.mul(spatial_attention.to(self.DEVICE))   # (V,V)(V,V)=(V,V)
                T_k_with_at = T_k_with_at.to(self.DEVICE)

                theta_k = self.Theta[k].to(self.DEVICE)  # (in_channel, out_channel)
                T_k_with_at = T_k_with_at.permute(0, 2, 1)
                T_k_with_at = T_k_with_at.to(self.DEVICE)
                rhs = T_k_with_at.matmul(graph_signal)  # (V,V)(N, V, C_in) = (N,V,C_in)

                output = output + rhs.matmul(theta_k)  # (N,V,C_in)(C_in, C_out) = (N,V,C_out)

            outputs.append(output.unsqueeze(-1))  # (N,V,C_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (N,V,C_out, T)
        
class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        print('x size', x.shape)
        print('A size', A.shape)
        print('Kernel size', self.kernel_size)
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
