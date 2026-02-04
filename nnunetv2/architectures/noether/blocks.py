import torch
import torch.nn as nn
import torch.nn.functional as F

class NoetherBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 expansion: int = 3, learn_residual: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_residual = learn_residual
        self.kernel_size = kernel_size

        kernel_size = [kernel_size, kernel_size, kernel_size]

        # Parameters for the first convolution
        self.conv1_weights = nn.Parameter(torch.randn(in_channels, in_channels, *kernel_size))
        self.conv1_bias = nn.Parameter(torch.zeros(in_channels))

        self.act1 = nn.ReLU()

        # Parameters for the second convolution
        self.conv2_weights = nn.Parameter(torch.randn(in_channels * expansion, in_channels, 1, 1, 1))
        self.conv2_bias = nn.Parameter(torch.zeros(in_channels * expansion))

        # Parameters for the third convolution
        self.conv3_weights = nn.Parameter(torch.randn(out_channels, in_channels * expansion, 1, 1, 1))
        self.conv3_bias = nn.Parameter(torch.zeros(out_channels))

        self.norm1 = nn.GroupNorm(in_channels, in_channels)

        # Convolution info sharing parameters
        self.c2c1_scaler = nn.Parameter(torch.ones(in_channels))
        self.c2c3_scaler = nn.Parameter(torch.ones(out_channels))

        if self.learn_residual and self.in_channels != self.out_channels:
            self.res_conv = nn.Conv3d(self.in_channels, self.out_channels, 
                                      kernel_size = 1)

    def forward(self, x, dummy_tensor=None):
        sum_weights = self.conv2_weights.flatten().sum()
        self.conv1_weights *= self.c2c1_scaler[:, None, None, None, None] * sum_weights
        self.conv3_weights *= self.c2c3_scaler[:, None, None, None, None] * sum_weights

        x_ = x
        x = F.conv3d(x, self.conv1_weights, self.conv1_bias, padding = self.kernel_size // 2)
        x = self.norm1(x)
        x = F.conv3d(x, self.conv2_weights, self.conv2_bias)
        x = self.act1(x)
        x = F.conv3d(x, self.conv3_weights, self.conv3_bias)

        if self.learn_residual:
            if self.in_channels != self.out_channels:
                x_ = self.res_conv(x_)

            x += x_

        return x
    
class NoetherDownBlock(NoetherBlock):

    def __init__(self, in_channels, out_channels, kernel_size = 3, expansion = 3, learn_residual = False):
        super().__init__(in_channels, out_channels, kernel_size, expansion, learn_residual)

        if learn_residual:
            self.res_conv = nn.Conv3d(self.in_channels, self.out_channels, 
                                      kernel_size = 1, stride = 2)

    def forward(self, x, dummy_tensor=None):
        sum_weights = self.conv2_weights.flatten().sum()
        self.conv1_weights *= self.c2c1_scaler[:, None, None, None, None] * sum_weights
        self.conv3_weights *= self.c2c3_scaler[:, None, None, None, None] * sum_weights

        x_ = x
        x = F.conv3d(x, self.conv1_weights, self.conv1_bias, stride = 2, 
                     padding = self.kernel_size // 2)
        x = self.norm1(x)
        x = F.conv3d(x, self.conv2_weights, self.conv2_bias)
        x = self.act1(x)
        x = F.conv3d(x, self.conv3_weights, self.conv3_bias)

        if self.learn_residual:
            x += self.res_conv(x_)

        return x
    
class NoetherUpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, expansion = 3, learn_residual = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_residual = learn_residual
        self.kernel_size = kernel_size

        kernel_size = [kernel_size, kernel_size, kernel_size]

        # Parameters for the first convolution
        self.conv1_weights = nn.Parameter(torch.randn(in_channels, in_channels, *kernel_size))
        self.conv1_bias = nn.Parameter(torch.zeros(in_channels))

        self.act1 = nn.ReLU()

        # Parameters for the second convolution
        self.conv2_weights = nn.Parameter(torch.randn(in_channels * expansion, in_channels, 1, 1, 1))
        self.conv2_bias = nn.Parameter(torch.zeros(in_channels * expansion))

        # Parameters for the third convolution
        self.conv3_weights = nn.Parameter(torch.randn(out_channels, in_channels * expansion, 1, 1, 1))
        self.conv3_bias = nn.Parameter(torch.zeros(out_channels))

        self.norm1 = nn.GroupNorm(in_channels, in_channels)

        # Convolution info sharing parameters
        self.c2c1_scaler = nn.Parameter(torch.ones(in_channels))
        self.c2c3_scaler = nn.Parameter(torch.ones(out_channels))

        if learn_residual:
            self.res_conv = nn.ConvTranspose3d(self.in_channels, self.out_channels,
                                               kernel_size = 1, stride = 2)

    def forward(self, x, dummy_tensor=None):
        sum_weights = self.conv2_weights.flatten().sum()
        w1 = self.conv1_weights * (self.c2c1_scaler[:, None, None, None, None] * sum_weights)
        w3 = self.conv3_weights * (self.c2c3_scaler[:, None, None, None, None] * sum_weights)

        x_ = x
        x = F.conv_transpose3d(x, w1, self.conv1_bias, stride = 2, 
                               padding = self.kernel_size // 2, output_padding = 1)
        x = self.norm1(x)
        x = F.conv3d(x, self.conv2_weights, self.conv2_bias)
        x = self.act1(x)
        x = F.conv3d(x, w3, self.conv3_bias)

        if self.learn_residual:
            x_ = self.res_conv(x_)
            x_ = F.pad(x_, (1,0,1,0,1,0))
            x += x_

        return x

class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.conv_out = nn.ConvTranspose3d(in_channels, n_classes, kernel_size=1)
    
    def forward(self, x, dummy_tensor=None): 
        return self.conv_out(x)   
    
if __name__ == "__main__":
    
    network = NoetherUpBlock(in_channels = 4, out_channels = 4, learn_residual = True)
    network.cuda()

    with torch.no_grad():
        x = torch.randn((1, 4, 8, 8, 8)).cuda()
        print(x)
        print(network(x))