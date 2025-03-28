import torch
import torch.nn as nn
import math
from einops import repeat
from SR21cm.utils import get_subcubes, normalize, augment_dataset
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception as e:
    print(e)
    print("Selective scan not available, using reference implementation and enabling test mode")
    test_mode = True
    selective_scan_fn = None



class SS3D(nn.Module): #for the original Vanilla VSS block, worse as described in VMamba paper
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1, #2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cpu',
            dtype=None,
            **kwargs,
            ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model #channel dim, 512 or 1024, gets expanded
        self.d_state = d_state
        
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
        del self.x_proj

        self.dt_projs = (
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
        ).contiguous()
        #('A', A.shape)
        A_log = torch.log(A)    # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)    # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        #0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = 8

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L), torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L), torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)], dim=1).view(B, 4, -1, L)
        # hwd, whd, dwh, hdw; reversed
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, c, l)
        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        
        xs = xs.float().view(B, -1, L) # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)    # (k * d, d_state)
        
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        
        #print("shapes in: ", xs.shape, dts.shape, As.shape, Bs.shape, Cs.shape, Ds.shape, dt_projs_bias.shape)
        #shapes in:  torch.Size([1, 8192, 1728]) torch.Size([1, 8192, 1728]) torch.Size([8192, 64]) torch.Size([1, 8, 64, 1728]) torch.Size([1, 8, 64, 1728]) torch.Size([8192]) torch.Size([8192])
        out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
        ).view(B, K, -1, L) if not test_mode else torch.randn(B,K,self.d_model,L, device=x.device) #B, K, channelsize, L
        #print("shapes out: ", out_y.shape)
        #shapes out:  torch.Size([1, 8, 1024, 1728])
        assert out_y.dtype == torch.float

        # hwd b, 1, c, l >
        # whd b, 1, c, l >
        # dwh b, 1, c, l >
        # hdw b, 1, c, l >
        # hwd reversed l
        # whd reversed l
        # dwh reversed l
        # hdw reversed l

        #revert back to all hwd forward l

        #out1 = out_y[:,0,:,:]
        out2 = torch.transpose(out_y[:, 1].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous().view(B, -1, L)
        out3 = torch.transpose(out_y[:, 2].view(B, -1, W, H, D), dim0=2, dim1=4).contiguous().view(B, -1, L)
        out4 = torch.transpose(out_y[:, 3].view(B, -1, W, H, D), dim0=3, dim1=4).contiguous().view(B, -1, L)

        out5 = torch.flip(out_y[:, 0], dims=[-1]).view(B, -1, L)
        out6 = torch.flip(out2, dims=[-1]).view(B, -1, L)
        out7 = torch.flip(out3, dims=[-1]).view(B, -1, L)
        out8 = torch.flip(out4, dims=[-1]).view(B, -1, L)

        return out_y[:, 0], out2, out3, out4, out5, out6, out7, out8

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape #!!!
        #d_model = C
        
        xz = self.in_proj(x) # (b, h, w, d, d_model) -> (b, h, w, d, d_inner * 2)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d, d_inner), z for the multiplicative path
        
        x = x.permute(0, 4, 1, 2, 3).contiguous()    
        x = self.act(self.conv3d(x)) # (b, d, h, w)
        
        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x) # 1 1024 1728
        
        assert y1.dtype == torch.float32
        
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1) #bcl > blc > bhwdc
        y = self.out_norm(y)
        y = y * nn.functional.silu(z) #multiplicative path, ignored in v2 because ssm is inherently selective, described in VMamba
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out

class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f'pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)'
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f'pixel_shuffle expects its input\'s \'channel\' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}'
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)

class PixelUnshuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f'pixel_unshuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)'
            )
        elif x.shape[-3] % self.upscale_factor != 0:
            raise RuntimeError(
                f'pixel_unshuffle expects depth to be divisible by downscale_factor, but input.size(-3)={x.shape[-3]} is not divisible by {self.upscale_factor}'
            )
        elif x.shape[-2] % self.upscale_factor != 0:
            raise RuntimeError(
                f'pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)={x.shape[-2]} is not divisible by {self.upscale_factor}'
            )
        elif x.shape[-1] % self.upscale_factor != 0:
            raise RuntimeError(
                f'pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)={x.shape[-1]} is not divisible by {self.upscale_factor}'
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]

        out_depth = in_depth // self.upscale_factor
        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor
        nOut = channels * self.upscale_factor**3

        input_view = x.contiguous().view(
            *x.shape[:-4],
            channels,
            out_depth,
            self.upscale_factor,
            out_height,
            self.upscale_factor,
            out_width,
            self.upscale_factor
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-5, -3, -1, -6, -4, -2]
        output = input_view.permute(axes).contiguous()
        
        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)
