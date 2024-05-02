import torch
import torch.nn as nn
import numpy as np
from math import log

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#convolution class that switches between 2D and 3D convolutions
class ConvolutionalLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size, stride, padding, dim, bias = True):
        super().__init__()
        if dim == 2:
            self.conv = nn.Conv2d(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=kernel_size, stride=stride, padding=padding, bias = bias)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=kernel_size, stride=stride, padding=padding, bias = bias)
        else:
            raise ValueError("dim must be 2 or 3")

    def forward(self, x):
        #print("dtype in conv layer: ", x.dtype)
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, n_channels, dim=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvolutionalLayer(n_channels_in=n_channels, n_channels_out=n_channels, kernel_size=3, stride=1, padding=1, dim=dim)#nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, n_channels, dim):
        super().__init__()
        self.conv = ConvolutionalLayer(n_channels_in=n_channels, n_channels_out=n_channels, kernel_size=3, stride=2, padding=1, dim=dim) #nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding
    
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, dim: int):
        """
        * `dim` is the number of dimensions in the embedding
        """
        super().__init__()
        self.dim = dim
        self.encoding = PositionalEncoding(dim)
        # First linear layer
        self.lin1 = nn.Linear(self.dim, self.dim*4)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.dim*4, self.dim)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        #PositionalEncoding
        #half_dim = self.dim // 8
        #emb = log(10_000) / (half_dim - 1)
        #emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        #emb = t[:, None] * emb[None, :]
        #emb = torch.cat((emb.sin(), emb.cos()), dim=1) #do I need a 3D time embedding?
        emb = self.encoding(t)
        #emb = emb.to(torch.float32)  # convert emb to Float

        # Transform with the MLP (noise_level_mlp)
        emb = self.lin1(emb)
        emb = self.act(emb)
        emb = self.lin2(emb)

        #
        return emb

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32, dim=3):
        super().__init__()

        self.n_head = n_head
        
        self.indices1 = "bnc{0}, bnc{1} -> bn{2}".format(
            "hw" if dim == 2 else "hwd",
            "yx" if dim ==2 else "yxz", 
            "hwyx" if dim == 2 else "hwdyxz")
        self.indices2 = "bn{0}, bnc{1} -> bnc{2}".format(
            "hwyx" if dim == 2 else "hwdyxz",
            "yx" if dim ==2 else "yxz", 
            "hw" if dim == 2 else "hwd")

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = ConvolutionalLayer(n_channels_in=in_channel, n_channels_out=in_channel*3, kernel_size=1, stride=1, padding=0, dim=dim, bias = False)
        self.out = ConvolutionalLayer(n_channels_in=in_channel, n_channels_out=in_channel, kernel_size=1, stride=1, padding=0, dim=dim, bias = True)

    def forward(self, input):
        #print("Executing attention block")
        batch, channel,(*d) = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, *d)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx
        
        attn = torch.einsum(
            self.indices1, query, key
        ).contiguous() / np.sqrt(channel)
        attn = attn.view(batch, n_head, *d, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, *d, *d)
        
        out = torch.einsum(self.indices2, attn, value).contiguous()
        out = self.out(out.view(batch, channel, *d))

        return out + input

class ResnetBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, time_channels, norm_groups = 32, dropout = 0, with_attn=False, dim=2):#, use_affine_level=False
        """
        n_channels_in: int
            Number of input channels
        n_channels_out: int
            Number of output channels
        norm_groups: int
            Number of groups for the group normalization (must divide n_channels)
        noise_level_emb_dim: int
            Number of dimensions for the time embedding (match the input dimensions?)
        """
        super().__init__()
        #self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        
        #block1
        self.block1 = nn.Sequential(
            nn.GroupNorm(norm_groups, n_channels_in),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            ConvolutionalLayer(n_channels_in=n_channels_in, n_channels_out=n_channels_out, kernel_size=3, stride=1, padding=1, dim=dim),#nn.Conv2d(n_channels_in, n_channels_out, kernel_size=(3, 3), padding=(1, 1))
        )

        #noise_func/FeatureWiseAffine = act and linear for timeembedding........
        self.time_emb = nn.Linear(time_channels, n_channels_out)
        self.time_act = Swish()
        if dim == 2:
            self.temb_reshape = (-1, 1, 1)
        elif dim == 3:
            self.temb_reshape = (-1, 1, 1, 1)
        


        #block2
        self.block2 = nn.Sequential(
            nn.GroupNorm(norm_groups, n_channels_out),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            ConvolutionalLayer(n_channels_in=n_channels_out, n_channels_out=n_channels_out, kernel_size=3, stride=1, padding=1, dim=dim),
            )

        #If the number of input channels is not equal to the number of 
        #output channels we have to project the shortcut connection
        if n_channels_in != n_channels_out:
            self.shortcut = ConvolutionalLayer(n_channels_in=n_channels_in, n_channels_out=n_channels_out, kernel_size=1, stride=1, padding=0, dim=dim) #nn.Conv2d(n_channels_in, n_channels_out, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if with_attn:
            self.attn = SelfAttention(n_channels_out, norm_groups=norm_groups)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x, temb):
        batch = x.shape[0]
        temb = self.time_emb(self.time_act(temb)).view(batch, *self.temb_reshape)
        
        h = self.block1(x)
        h = h + temb
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
        

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=2,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8,),
        res_blocks=3,
        dropout = 0, 
        with_attn=False,
        image_size=128, 
        dim=2
        ):
        super().__init__()

        self.temb = TimeEmbedding(inner_channel)
        

        #ResnetBlock(n_channels_in, n_channels_out,  time_channels, norm_groups = 32, dropout = 0, with_attn=False, dim=2)

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size if with_attn else False
        downs = [ConvolutionalLayer(n_channels_in=in_channel, n_channels_out=inner_channel, kernel_size=3, stride=1, padding=1, dim=dim)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlock(
                    n_channels_in=pre_channel, n_channels_out=channel_mult, time_channels=inner_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn, dim=dim))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel, dim=dim))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)


        self.mid = nn.ModuleList([
            ResnetBlock(n_channels_in=pre_channel, n_channels_out=pre_channel, time_channels=inner_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn, dim=dim),
            ResnetBlock(n_channels_in=pre_channel, n_channels_out=pre_channel, time_channels=inner_channel, norm_groups=norm_groups, 
                        dropout=dropout, with_attn=use_attn, dim=dim)
            #ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
            #                   dropout=dropout, with_attn=True),
            #ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
            #                   dropout=dropout, with_attn=False)
            ])
        
        ups = []

        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(
                    ResnetBlock(n_channels_in=pre_channel+feat_channels.pop(), n_channels_out=channel_mult, time_channels=inner_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn, dim=dim)
                    #ResnetBlocWithAttn(pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,dropout=dropout, with_attn=use_attn)
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel, dim=dim))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)
        
        self.final_block = nn.Sequential(
            nn.GroupNorm(norm_groups, pre_channel),
            Swish(),
            ConvolutionalLayer(n_channels_in=pre_channel, n_channels_out=out_channel, kernel_size=3, stride=1, padding=1, dim=dim),#nn.Conv2d(n_channels_in, n_channels_out, kernel_size=(3, 3), padding=(1, 1))
        )
    def forward(self, x, time):
        t = self.temb(time)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlock):
                #print("Downs shape: ", x.shape)
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlock):
                #print("Mids shape: ", x.shape)
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
                #print("Ups shape: ", x.shape)
                f = feats.pop()
                x = layer(torch.cat((x, f), dim=1), t)
            else:
                x = layer(x)

        return self.final_block(x)
