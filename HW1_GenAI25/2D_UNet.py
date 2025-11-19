'''
Code sources:
https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=1841s
https://huggingface.co/blog/annotated-diffusion
'''
import torch
from torch import nn, Tensor, einsum
from einops.layers.torch import Rearrange
from einops import rearrange

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super().__init__()

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.rearrange(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.upsample(x))

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.conv(x))
        return self.act(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x: Tensor):
        h = self.block1(x)
        return self.block2(h) + self.skip(x)

class RMSNorm(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        ## pytorch normalize takes the signal and computes x / sum(x ** 2, dim = 1).sqrt()
        ## we want x / (sum(x ** 2, dim = 1) / sqrt(num_channels))
        return nn.functional.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

## https://huggingface.co/blog/annotated-diffusion
class LinearAttention(nn.Module):
    '''
    Computes attention that scales linearly in memory O(n) rather than polynomial O(n^2).

    https://arxiv.org/pdf/2006.16236
    '''
    def __init__(self, in_channels: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head

        self.norm = RMSNorm(in_channels)
        self.qkv = nn.Conv2d(in_channels, 3*hidden_dim, kernel_size = 1, bias = False)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size = 1),
            RMSNorm(in_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, height, width = x.shape
        ## in the einsums i corresponds to height and j corresponds to width

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = 1)

        ## takes the channel dimention and reshapes into heads, channels
        ## also flattens the feature maps into vectors
        ## the shape is now b h c d where d is dim_head
        q = rearrange(q, 'b (h c) i j -> b h c (i j)', h = self.heads)
        k = rearrange(k, 'b (h c) i j -> b h c (i j)', h = self.heads)
        v = rearrange(v, 'b (h c) i j -> b h c (i j)', h = self.heads)
        ## softmax along dim_head dim
        q = q.softmax(dim = 2) * self.scale
        ## softmax along flattened image dim
        k = k.softmax(dim = 3)
        ## compute comparison betweeen keys and values to produce context.
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (i j) -> b (h c) i j', i = height, j = width)
        return self.output(out)

## https://huggingface.co/blog/annotated-diffusion
class Attention(nn.Module):
    '''
    Computes full pixelwise attention.
    '''
    def __init__(self, in_channels: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head

        self.norm = RMSNorm(in_channels)
        self.qkv = nn.Conv2d(in_channels, hidden_dim * 3, kernel_size = 1, bias = False)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size = 1),
            RMSNorm(in_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, height, width = x.shape

        x = self.norm(x)

        ## compute the queries, keys, and values of the incoming feature maps
        q, k, v = torch.chunk(self.qkv(x), 3, dim = 1)
        ## takes the channel dimention and reshapes into heads, channels

        q = rearrange(q, 'b (h c) i j -> b h c (i j)', h = self.heads)
        k = rearrange(k, 'b (h c) i j -> b h c (i j)', h = self.heads)
        v = rearrange(v, 'b (h c) i j -> b h c (i j)', h = self.heads)
        q = q * self.scale
        ## multiplication of the query and key matrixes for each head
        sim = einsum('b h c i, b h c j -> b h i j', q, k)
        ## subtract the maximum value of each row
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        ## make each row into a probability distribution
        attn = sim.softmax(dim = -1)
        ## weight the values according to the rows of the attention matrix
        y = einsum('b h i j, b h d j -> b h i d', attn, v)
        # reshape the weighted values back into feature maps
        y = rearrange(y, 'b h (i j) d -> b (h d) i j', i = height, j = width)
        return self.output(y)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.input_layer = nn.Conv2d(in_channels, 32, kernel_size = 1)

        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResBlock(32, 32),
                ResBlock(32, 32),
                LinearAttention(32, heads, dim_head),
                DownBlock(32, 32)
            ]),

            nn.ModuleList([
                ResBlock(32, 32),
                ResBlock(32, 32),
                LinearAttention(32, heads, dim_head),
                DownBlock(32, 64)
            ]),

            nn.ModuleList([
                ResBlock(64, 64),
                ResBlock(64, 64),
                LinearAttention(64, heads, dim_head),
                nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
            ])
        ])

        self.mid_block1 = ResBlock(128, 128)
        self.mid_attention = Attention(128, heads, dim_head)
        self.mid_block2 = ResBlock(128, 128)

        self.ups = nn.ModuleList([
            nn.ModuleList([
                ResBlock(128 + 64, 128),
                ResBlock(128 + 64, 128),
                LinearAttention(128, heads, dim_head),
                UpBlock(128, 64)
            ]),

            nn.ModuleList([
                ResBlock(64 + 32, 64),
                ResBlock(64 + 32, 64),
                LinearAttention(64, heads, dim_head),
                UpBlock(64, 32)
            ]),

            nn.ModuleList([
                ResBlock(32 + 32, 32),
                ResBlock(32 + 32, 32),
                LinearAttention(32, heads, dim_head),
                nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
            ])
        ])

        self.output_res = ResBlock(32 + 32, 32)
        self.output_layer = nn.Conv2d(32, in_channels, kernel_size = 1)
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, height, width = x.shape

        y = self.input_layer(x) ## (b x 32 x height x width)
        r = y.clone()
        
        residuals = []
        for res1, res2, attention, downsample in self.downs:
            y = res1(y)
            residuals.append(y)
            y = res2(y)
            y = attention(y) + y
            residuals.append(y)
            y = downsample(y)

        y = self.mid_block1(y)
        y = self.mid_attention(y) + y
        y = self.mid_block2(y)

        for res1, res2, attention, upsample in self.ups:
            y = res1(torch.cat((y, residuals.pop()), dim = 1))
            y = res2(torch.cat((y, residuals.pop()), dim = 1))
            y = attention(y) + y
            y = upsample(y)

        ## final skip connection to residual layer
        y = self.output_res(torch.cat((y, r), dim = 1))
        y = self.output_layer(y)
        return y
    
if __name__ == '__main__':

    batch_size = 32
    in_channels = 3
    img_hight = 128
    img_width = 128

    model = UNet(in_channels)

    x = torch.randn(batch_size, in_channels, img_hight, img_width)
    print(x.shape)

    y = model(x)
    print(x.shape, y.shape)