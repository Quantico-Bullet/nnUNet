import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from nnunetv2.architectures.noether.blocks import *

class NoetherNet(nn.Module):

    def __init__(self, in_channels: int, n_channels: int, n_classes: int, 
                 block_counts: list = [2,2,2,2,2,2,2,2,2], 
                 deep_supervision: bool = True,
                 do_checkpointing: bool = True):
        super().__init__()

        self.do_checkpointing = do_checkpointing
        self.deep_supervision = deep_supervision

        num_channels =  [n_channels, n_channels*2, n_channels*4, n_channels*8, n_channels*16]
        self.stem = nn.Conv3d(in_channels, num_channels[0], kernel_size=1)
        
        self.enc_block_0 = nn.Sequential(*[
            NoetherBlock(num_channels[0], num_channels[0])
            for i in range(block_counts[0])]
        )

        self.down_block_0 = NoetherDownBlock(num_channels[0], num_channels[1])

        self.enc_block_1 = nn.Sequential(*[
            NoetherBlock(num_channels[1], num_channels[1])
            for i in range(block_counts[1])]
        )

        self.down_block_1 = NoetherDownBlock(num_channels[1], num_channels[2])

        self.enc_block_2 = nn.Sequential(*[
            NoetherBlock(num_channels[2], num_channels[2])
            for i in range(block_counts[2])]
        )

        self.down_block_2 = NoetherDownBlock(num_channels[2], num_channels[3])

        self.enc_block_3 = nn.Sequential(*[
            NoetherBlock(num_channels[3], num_channels[3])
            for i in range(block_counts[3])]
        )

        self.down_block_3 = NoetherDownBlock(num_channels[3], num_channels[4])

        self.bottleneck = nn.Sequential(*[
            NoetherBlock(num_channels[4], num_channels[4])
            for i in range(block_counts[4])]
        )

        self.up_block_3 = NoetherBlock(num_channels[4], num_channels[3])

        self.dec_block_3 = nn.Sequential(*[
            NoetherBlock(num_channels[3], num_channels[3])
            for i in range(block_counts[5])]
        )

        self.up_block_2 = NoetherBlock(num_channels[3], num_channels[2])

        self.dec_block_2 = nn.Sequential(*[
            NoetherBlock(num_channels[2], num_channels[2])
            for i in range(block_counts[6])]
        )

        self.up_block_1 = NoetherBlock(num_channels[2], num_channels[1])

        self.dec_block_1 = nn.Sequential(*[
            NoetherBlock(num_channels[1], num_channels[1])
            for i in range(block_counts[7])]
        )

        self.up_block_0 = NoetherBlock(num_channels[1], num_channels[0])

        self.dec_block_0 = nn.Sequential(*[
            NoetherBlock(num_channels[0], num_channels[0])
            for i in range(block_counts[8])]
        )

        self.out0 = OutBlock(num_channels[0], n_classes)

        if deep_supervision:
            self.out1 = OutBlock(num_channels[1], n_classes)
            self.out2 = OutBlock(num_channels[2], n_classes)
            self.out3 = OutBlock(num_channels[3], n_classes)
            self.out4 = OutBlock(num_channels[4], n_classes)

        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    def iterative_checkpoint(self, block_sequence: nn.Sequential, x):
        for block in block_sequence:
            x = checkpoint.checkpoint(block, x, self.dummy_tensor)

        return x

    def forward(self, x):

        if self.do_checkpointing:
            x = self.stem(x)
            x = self.iterative_checkpoint(self.enc_block_0, x) 
            x = checkpoint.checkpoint(self.down_block_0, x)
            x = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_block_1, x)
            x = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_block_2, x)
            x = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_block_3, x)
            x = self.iterative_checkpoint(self.bottleneck, x)

            if self.deep_supervision:
                x_ds_4 = checkpoint.checkpoint(self.out4, x)

            x = checkpoint.checkpoint(self.up_block_3, x)
            x = self.iterative_checkpoint(self.dec_block_3, x)

            if self.deep_supervision:
                x_ds_3 = checkpoint.checkpoint(self.out3, x)

            x = checkpoint.checkpoint(self.up_block_2, x)
            x = self.iterative_checkpoint(self.dec_block_2, x)

            if self.deep_supervision:
                x_ds_2 = checkpoint.checkpoint(self.out2, x)

            x = checkpoint.checkpoint(self.up_block_1, x)
            x = self.iterative_checkpoint(self.dec_block_1, x)

            if self.deep_supervision:
                x_ds_1 = checkpoint.checkpoint(self.out1, x)

            x = checkpoint.checkpoint(self.up_block_0, x)
            x = self.iterative_checkpoint(self.dec_block_0, x)
            x = checkpoint.checkpoint(self.out0, x)

        else:
            x = self.stem(x)
            x = self.enc_block_0(x)
            x = self.down_block_0(x)
            x = self.enc_block_1(x)
            x = self.down_block_1(x)
            x = self.enc_block_2(x)
            x = self.down_block_2(x)
            x = self.enc_block_3(x)
            x = self.down_block_3(x)
            x = self.bottleneck(x)
            x = self.up_block_3(x)
            x = self.dec_block_3(x)
            x = self.up_block_2(x)
            x = self.dec_block_2(x)
            x = self.up_block_1(x)
            x = self.dec_block_1(x)
            x = self.up_block_0(x)
            x = self.dec_block_0(x)
            x = self.out0(x)            

        if self.deep_supervision:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        
        else: 
            return x