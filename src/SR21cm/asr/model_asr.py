import torch
import torch.nn as nn
from timm.models.layers import DropPath
from typing import Callable
from functools import partial
import math
from einops import repeat
from SR21cm.asr.mambairunet_arch import MambaIREncoder
from SR21cm.asr.arch_utils import SS3D, PixelShuffle3d, PixelUnshuffle3d
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception as e:
    print(e)
    print("Selective scan not available, using reference implementation and enabling test mode")
    test_mode = True
    selective_scan_fn = None

class VSSBlock3D(nn.Module):
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1,
      **kwargs,
      ):
    super().__init__()
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=expansion_factor, **kwargs)
    self.drop_path = DropPath(drop_path)

    self.skip_scale1= nn.Parameter(torch.ones(hidden_dim)) #added
    self.skip_scale2= nn.Parameter(torch.ones(hidden_dim))
    self.conv_blk = CAB(hidden_dim,False)
    self.ln_2 = nn.LayerNorm(hidden_dim)

  def forward(self, input: torch.Tensor, x_size):
    #x = self.ln_1(input)
    #x = input*self.skip_scale1+ self.drop_path(self.self_attention(x))

    #x_ln_conv = self.conv_blk(self.ln_2(x)) #check shape
    #x = x*self.skip_scale2 + x_ln_conv


    # x [B,HWD,C]
    B, L, C = input.shape
    input = input.view(B, *x_size, C).contiguous()  # [B,H,W,D,C]
    x = self.ln_1(input)
    x = input*self.skip_scale1 + self.drop_path(self.self_attention(x))
    x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 4, 1, 2, 3).contiguous()).permute(0, 2, 3, 4, 1).contiguous()
    x = x.view(B, -1, C).contiguous()
    return x

class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock3D(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expansion_factor=self.mlp_ratio,
                input_resolution=input_resolution,
                is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for i,blk in enumerate(self.blocks):
            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 img_size=None,
                 patch_size=None,
                 is_light_sr = False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            is_light_sr = is_light_sr
            )

        # build the last conv layer in each residual state space group

        self.conv = nn.Conv3d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x



class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
    
class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=2):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)
    
class MambaIR(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 **kwargs):
        super(MambaIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim


        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1], patches_resolution[2]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                img_size=img_size,
                patch_size=patch_size,
                is_light_sr = self.is_light_sr
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        # -------------------------3. high-quality image reconstruction ------------------------ #

        # for lightweight SR (to save parameters)
        self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3], x.shape[4])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)

        for i,layer in enumerate(self.layers):
            print(f"before layer: x size: {x_size}, x shape: {x.shape}")
            x = layer(x, x_size)
            print(f"after layer: x size: {x_size}, x shape: {x.shape}")

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        print(f"after unembed: x size: {x_size}, x shape: {x.shape}")

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        # for lightweight SR
        print(f"before conv_first: x size: {x.shape}")
        x = self.conv_first(x)
        print(f"after conv_first: x size: {x.shape}")
        x_after_body = self.forward_features(x)
        x_after_body = self.conv_after_body(x_after_body)
        x = x_after_body + x
        print(f"after conv_after_body: x size: {x.shape}")
        x = self.upsample(x)
        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv3d(num_feat, (scale**3) * num_out_ch, 3, 1, 1))
        m.append(PixelShuffle3d(scale))
        super(UpsampleOneStep, self).__init__(*m)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

import warnings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple([x] * n)
    return parse
to_3tuple = _ntuple(3)

class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1], x_size[2])  # b Ph*Pw*Pd c
        return x

    def flops(self):
        flops = 0
        return flops





class VSSLayer3D(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        mlp_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=64,
        version = 'v5', #None, v5, v6
        expansion_factor = 1,
        scan_type = 'scan',
        orientation_order = None,
        size = 12,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        if version is None:
            print('Vanilla VSS')
            self.blocks = nn.ModuleList([
                VSSBlock3D(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    expansion_factor=expansion_factor,
                )
                for i in range(depth)])
        else:
            raise Exception('define a valid VSS version') 
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class ResConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, id = True, preact = True): #id was False
        super().__init__()
        if preact:
            self.conv = nn.Sequential(
                nn.GroupNorm(8, ch_in),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                )
        id = (ch_in == ch_out) and id
        self.identity = (lambda x: x) if id else nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        x = self.conv(inp)
        residual = self.identity(inp)
        return residual + x

class down_block(nn.Module):
    def __init__(self, ch_in, ch_out, maxpool = False, id=True, preact = True, kernel_size = 3):
        super().__init__()
        if maxpool:
            downsample = nn.MaxPool3d(kernel_size=2,stride=2)
            resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=preact)
            self.down = nn.Sequential(downsample, resblock)
        else:
            downconv = nn.Conv3d(
                                in_channels=ch_in,
                                out_channels=ch_out,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=kernel_size//2 - (1 - kernel_size % 2),
                                )
            if preact:
                act = nn.Sequential(nn.GroupNorm(8, ch_in),nn.ReLU(inplace = True))# if act else (lambda x: x)
                resblock = ResConvBlock3D(ch_in=ch_out, ch_out=ch_out, id=id, preact=True)
                self.down = nn.Sequential(act, downconv, resblock)
            else:
                act = nn.Sequential(nn.GroupNorm(8, ch_out),nn.ReLU(inplace = True))# if act else (lambda x: x)
                resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=True)
                self.down = nn.Sequential(downconv, act, resblock)

    def forward(self,x):
        x = self.down(x)
        return x

class modified_net2(nn.Module):
    def __init__(
        self,
        channel_sizes = 32, 
        mamba_d_state = 32, # for vss block
        num_layers = 8,
        vss_version = None, # None for vanilla
        mlp_dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        drop_path_rate=0.3,
        ssm_expansion_factor=1,
        scan_type = 'scan',
        downconv = True,
        ):
        super().__init__()

        self.expand = nn.Conv3d(1, channel_sizes, kernel_size=3, stride=1, padding=1)

        if downconv:
            self.downconv = down_block(ch_in=channel_sizes, ch_out=channel_sizes, maxpool=False, id=False, preact=False)

        self.VSS_stages = []
        for i in range(4):
            self.VSS_stages.append(
                VSSLayer3D(dim = channel_sizes,
                           depth = num_layers,
                           drop_path = drop_path_rate,
                           attn_drop = attn_dropout_rate,
                           mlp_drop=mlp_dropout_rate,
                           d_state = mamba_d_state,
                           version=vss_version,
                           expansion_factor=ssm_expansion_factor,
                           scan_type=scan_type)
                           )
        self.VSS_stages = nn.ModuleList(self.VSS_stages)

    def forward(self,x):
        x = self.expand(x)
        if hasattr(self, 'downconv'):
            x = self.downconv(x)
        
        #save skip connections and concatenate in the end
        skips = [x, ]
        x = x.permute(0,2,3,4,1) # B H W D C
        for stage in self.VSS_stages:
            x = stage(x)
            skips.append(x.permute(0,4,1,2,3)) # B C H W D
        x = torch.cat(skips, dim=1)
        return x
    
class MLP_decoder(nn.Module):
    def __init__(self, in_dim=128 + 3, out_dim=1, depth=4, width=256):
        super(MLP_decoder, self).__init__()
        stage_one = []
        stage_two = []
        for i in range(depth):
            if i == 0:
                stage_one.append(nn.Linear(in_dim, width))
                stage_two.append(nn.Linear(in_dim, width))
            elif i == depth - 1:
                stage_one.append(nn.Linear(width, in_dim))
                stage_two.append(nn.Linear(width, out_dim))
            else:
                stage_one.append(nn.Linear(width, width))
                stage_two.append(nn.Linear(width, width))
            stage_one.append(nn.ReLU())
            stage_two.append(nn.ReLU())
        self.stage_one = nn.Sequential(*stage_one)
        self.stage_two = nn.Sequential(*stage_two)

    def forward(self, x):
        h = self.stage_one(x)
        return self.stage_two(x + h)


class ArSSR(nn.Module):
    def __init__(self, encoder, feature_dim, decoder_depth, decoder_width, **kwargs):
        super(ArSSR, self).__init__()
        self.multi_gpu = kwargs.get('multi_gpu', False)
        self.device = kwargs.get('device', torch.device('cpu'))

        self.encoder = encoder
        self.decoder = MLP_decoder(in_dim=feature_dim + 3, out_dim=1, depth=decoder_depth, width=decoder_width)

        self.loss = []
        self.loss_validation = []

    def forward(self, img_lr, xyz_hr):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image
        """
        # extract feature map from LR image
        feature_map = self.encoder(img_lr)  # 

        # generate feature vector for coordinate through trilinear interpolation (Equ. 4 & Fig. 3).
        coords = xyz_hr.flip(-1)
        coords = coords.unsqueeze(1).unsqueeze(1)
        feature_vector = nn.functional.grid_sample(feature_map, coords,
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        # concatenate coordinate with feature vector
        feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr], dim=-1)  # N×K×(3+feature_dim)
        # estimate the voxel intensity at the coordinate by using decoder.
        N, K = xyz_hr.shape[:2]
        img_sr = self.decoder(feature_vector_and_xyz_hr.view(N * K, -1)).view(N, K, -1)

        h = w = d = int(round(K**(1/3)))
        img_sr = img_sr.view(N, 1, h, w, d)
        return img_sr
    
    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    model = self.state_dict(), 
                    optimizer = self.optG.state_dict(),
                    loss = self.loss,
                    loss_validation = self.loss_validation,
                    ),
                    f = path
                    )
        else:
            if self.device.index == 0:
                print("Saving model!", flush=True)
                torch.save(
                    obj = dict(
                        model = self.module.state_dict(), 
                        optimizer = self.optG.state_dict(), 
                        loss = self.loss,
                        loss_validation = self.loss_validation,
                        ),
                        f = path
                        )

    def load_network(self, path, device = "cpu"):
        print("Loading model!", flush=True)
        loaded_state = torch.load(path, map_location = device)
        self.load_state_dict(loaded_state['model'])
        if self.multi_gpu:
            self.model.to(device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[device.index])
        self.optG.load_state_dict(loaded_state['optimizer'])
        self.loss = loaded_state['loss']
        self.loss_validation = loaded_state['loss_validation']

    
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    Shape: [D, H, W]
    ranges: [[z0, z1], [y0, y1], [x0, x1]]
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
    
if __name__=='__main__':
    d = 64 #96
    test_input = torch.randn(1,1,d,d,d)#.cuda()
    b,c,h,w,d = test_input.shape
    
    #encoder = modified_net2(downconv=True)#.cuda() #
    #out = encoder(test_input) #
    #print("encoder our shape test: ", out.shape) #

    #encoder = MambaIR(img_size=32., patch_size=1, in_chans=1, embed_dim=16, depths=(6, 6, 6, 6), drop_rate=0., d_state = 16, mlp_ratio=1., drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.,)
    encoder = MambaIREncoder(inp_channels=1, out_channels=1, dim=8, num_blocks=[4, 4, 4, 4], num_refinement_blocks=4, mlp_ratio=1., bias=False, )
    #test_input = torch.rand(2,1,64,64,64)
    #out = encoder(test_input)
    #print("encoder out shape test: ", out.shape)
    scale = 2.734135
    xyz_hr = make_coord([h*scale, w*scale, d*scale], ranges=None, flatten=False)
    xyz_hr = xyz_hr.view(1, -1, 3)
    xyz_hr = xyz_hr.repeat(b, 1, 1)
    network = ArSSR(feature_dim=128, encoder=encoder, decoder_depth=4, decoder_width=32)
    #output = network(img_lr=test_input, xyz_hr=xyz_hr)
    #print(output.shape)
    # print model
    for key in network.state_dict().keys():
        print(key)

