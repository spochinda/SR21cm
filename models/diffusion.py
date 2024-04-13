import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] #check if master address exists
    except:
        os.environ["MASTER_ADDR"] = "localhost"
    
    os.environ["MASTER_PORT"] = "12356"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #backend gloo for cpus?
    

    
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from math import log
from torch_ema import ExponentialMovingAverage
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS

from utils import DataManager

#device = (
#    "cuda"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
#)
#device="cpu" #because 3D convolutions are not supported on MPS
#print(f"Using {device} device", flush=True)

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


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    Variance schedule. cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    beta_schedule = 1 - (alphas[1:] / alphas[: -1])
    result = torch.clip(input = beta_schedule, min = 0.0001, max = 0.9999)

    return result


def linear_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
    beta_schedule = torch.linspace(beta_start, beta_end, timesteps)
    return beta_schedule



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        model_opt,
        loss_type='l1',
        noise_schedule=None,
        noise_schedule_opt=None,
        learning_rate=1e-4,
        rank = 0,
    ):
        super().__init__()
        self.multi_gpu = torch.cuda.device_count() > 1 
        if self.multi_gpu:
            self.device = torch.device(f'cuda:{rank}')
        else:
            self.device = "cpu"
        self.model = model(**model_opt) if not self.multi_gpu else DDP(model(**model_opt).to(self.device), device_ids=[rank])
        #self.model_opt = model_opt
        init_weights(self.model, init_type='orthogonal')
        self.optG = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.loss = []
        self.losses_validation_history = []
        self.loss_type = loss_type
        self.noise_schedule = noise_schedule
        self.noise_schedule_opt = noise_schedule_opt
        if noise_schedule is not None:
            self.set_new_noise_schedule(noise_schedule_opt)

    def set_new_noise_schedule(self, noise_schedule_opt):

        self.betas = self.noise_schedule(**self.noise_schedule_opt).to(self.device)
        self.timesteps, = self.betas.shape
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.], dtype=torch.float32, device=self.device), self.alphas_cumprod[:-1]))
        #self.alphas_cumprod_prev = torch.tensor(np.append(1., self.alphas_cumprod[:-1]), dtype=torch.float32) #need to specify dtype because np.append with 1. is float64
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def predict_start_from_noise(self, x_t, t, noise):
        b,(*d) = x_t.shape
        alpha_cumprod_t = self.alphas_cumprod[t].view(b,*[1]*len(d))
        x0 = x_t/torch.sqrt(alpha_cumprod_t) - torch.sqrt(1-alpha_cumprod_t) * noise/torch.sqrt(alpha_cumprod_t)
        return x0

    def q_sample(self, x0, t, noise=None): #forward diffusion
        #"t and batch number dim should be the same"
        b,(*d) = x0.shape
        t=torch.tensor(t).view(b,*[1]*len(d))
        alphas_cumprod_t = self.alphas_cumprod[t]
        noise = torch.randn_like(x0, device=self.device) if noise==None else noise
        x_t = torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, x_t, t, conditionals=None, clip_denoised=True, mean_approach = "DDPM SR3", ema=False):
        b,(*d) = x_t.shape
        time = t
        t=torch.tensor(b*[t]).view(b,*[1]*len(d))
        
        alpha_t, alpha_t_cumprod, alpha_t_cumprod_prev, beta_t = self.alphas[t], self.alphas_cumprod[t], self.alphas_cumprod_prev[t], self.betas[t]
        noise_level = alpha_t_cumprod
        
        if mean_approach=="DDIM": #
            ddim_alpha_t = self.ddim_alpha[t]
            ddim_alpha_t_sqrt = self.ddim_alpha_sqrt[t]
            ddim_alpha_t_prev = self.ddim_alpha_prev[t]
            ddim_sigma_t = self.ddim_sigma[t]
            ddim_sqrt_one_minus_alpha_t = self.ddim_sqrt_one_minus_alpha[t]
            noise_level = ddim_alpha_t

            #print("ddim t={0}, alpha_t={1:.4f}, alpha_t-1={2:.4f}: ".format( t.item(), ddim_alpha_t.item(), ddim_alpha_t_prev.item()) )


        posterior_variance_t = self.posterior_variance[t] #torch.sqrt(beta_t)
        noise = torch.randn_like(x_t) #if t > 0 else torch.zeros_like(x_t)

        if ema:
            with self.ema.average_parameters():
                pred_noise = self.model(x=torch.cat([x_t, *conditionals], dim=1), time=noise_level)
        else:
            pred_noise = self.model(x=torch.cat([x_t, *conditionals], dim=1), time=noise_level)

        if mean_approach=="DDPM Classic":
            x0 = None
            posterior_mean_t = (torch.sqrt(1/alpha_t)) * (x_t - beta_t/torch.sqrt(1 - alpha_t_cumprod) * pred_noise) #approach used in most papers   
            x_t = posterior_mean_t + noise * posterior_variance_t if t.item() > 0 else posterior_mean_t
            x_t = torch.clamp(x_t, -1.0, 1.0) if clip_denoised else x_t

        elif mean_approach=="DDPM SR3":
            x0 = self.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise) #eq in text above eq 9 rewritten for x0
            x0 = torch.clamp(x0, -1.0, 1.0) if clip_denoised else x0
            #beta_t_tilde = beta_t*(1-alpha_t_cumprod_prev)/(1-alpha_t_cumprod)
            posterior_mean_t = (torch.sqrt(alpha_t_cumprod_prev)*beta_t/(1-alpha_t_cumprod)) * x0 + \
                (torch.sqrt(alpha_t)*(1-alpha_t_cumprod_prev)/(1-alpha_t_cumprod)) * x_t #mu_tilde_t in the ddpm paper. q_posterior on github. SR3 approach
            x_t = posterior_mean_t + noise * posterior_variance_t if time > 0 else posterior_mean_t

        elif mean_approach=="DDIM":
            x0 = (x_t - ddim_sqrt_one_minus_alpha_t * pred_noise) / (ddim_alpha_t ** 0.5)
            #x0 = x0.clamp(-1., 1.) if clip_denoised else x0
            dir_xt = (1. - ddim_alpha_t_prev - ddim_sigma_t ** 2).sqrt() * pred_noise
            
            noise = noise*self.temperature
            x_t = (ddim_alpha_t_prev ** 0.5) * x0 + dir_xt + ddim_sigma_t * noise #x_t-1
        
        for i,(x,t_) in enumerate(zip(x_t,t)):
            if t_==0:
                noise[i] = torch.zeros_like(x)
            else:
                noise[i] = torch.randn_like(x)

        #if (t.item() == self.timesteps-1) or (t.item() == 0):
        #    print("posterior variance: {0:.4f}".format(posterior_variance_t.item()))
        #    print("noise mean, std: {0:.4f}, {1:.4f}".format(torch.mean(noise).item(), torch.std(noise).item()))
        #    print("x_t mean, std: {0:.4f}, {1:.4f}".format(torch.mean(x_t).item(), torch.std(x_t).item()))
        #    #print alpha_t, alpha_t_cumprod, alpha_t_cumprod_prev, beta_t
        #    print("alpha_t: {0:.4f}, alpha_t_cumprod: {1:.4f}, alpha_t_cumprod_prev: {2:.4f}, beta_t: {3:.4f}".format(alpha_t.item(), alpha_t_cumprod.item(), alpha_t_cumprod_prev.item(), beta_t.item()))
        
        return x_t, noise, pred_noise, x0 #torch.sqrt(beta_t)*noise

    @torch.no_grad()
    def p_sample_loop(self, conditionals=None, n_save=10, clip_denoised=True, mean_approach = True, save_slices=False, ema=False, ddim_n_steps = None, verbose = True, device="cpu"):
        assert mean_approach in ["DDPM Classic", "DDPM SR3", "DDIM"], "mean_approach must be one of ['DDPM Classic', 'DDPM SR3', 'DDIM']"
        
        #print("last alpha: ", self.alphas[1000])
        t_steps = self.timesteps
        
        if mean_approach=="DDIM":
            assert ddim_n_steps is not None, "ddim_n_steps must be specified (int) for DDIM mean approach"

            #ddim_timesteps = np.asarray(list(range(0, self.timesteps, self.timesteps // ddim_n_steps))) + 1
            if True:
                #print("method 1")
                ddim_timesteps = torch.linspace(0, 1+self.timesteps-(self.timesteps//(ddim_n_steps)), ddim_n_steps, dtype=torch.int) #improved denoising diffusion probabilistic models 
                self.ddim_alpha = self.alphas_cumprod[ddim_timesteps]
                self.ddim_alpha_prev = torch.cat([self.alphas_cumprod[0:1], self.alphas_cumprod[ddim_timesteps[:-1]]])
            else:
                #print("method 2")
                ddim_timesteps = torch.linspace(1, 1+self.timesteps-(self.timesteps//(ddim_n_steps+1))  , ddim_n_steps+1,dtype=int)
                ddim_timesteps_prev = ddim_timesteps[:-1]
                ddim_timesteps = ddim_timesteps[1:]
                self.ddim_alpha = self.alphas_cumprod[ddim_timesteps]
                self.ddim_alpha_prev = self.alphas_cumprod[ddim_timesteps_prev]

            
            t_steps = len(ddim_timesteps)
            
            ddim_eta = 0.
            
            
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            

            self.ddim_sigma = (ddim_eta *
                        ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                        (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5 
            self.temperature = 1.

        
        #print("tsteps nsave: ", t_steps, n_save)
        self.model.eval()
        sample_inter = t_steps//n_save if n_save <= t_steps else 1
        #print("sample inter: {0}, t_steps {1}, n_save {2}".format(sample_inter, t_steps, n_save))
        b,(*d)  = conditionals[-1].shape #select the last conditional to get the shape (should be T21_lr because order is delta,vbv,T21_lr)
        
        x_t = torch.randn((b,*d), device=device)
        x_sequence = [x_t] #use channel dimension as time axis
        x_slices = [x_t[:,:,:,:,d[-1]//2]] if save_slices else []
        
        noises = []
        pred_noises = []
        x0_preds = []
        
        interval = reversed(range(0, t_steps)) #if (mean_approach=="DDPM Classic") or (mean_approach=="DDPM SR3") else reversed(range(ddim_n_steps))
        sampling_timesteps = t_steps #if (mean_approach=="DDPM Classic") or (mean_approach=="DDPM SR3") else ddim_n_steps

        assert n_save <= sampling_timesteps, "n_save must be smaller or equal to time steps"

        for t in tqdm(interval, desc='sampling loop time step', total=sampling_timesteps, disable = not verbose):
            x_t, noise, pred_noise, x0 = self.p_sample(x_t, t, conditionals=conditionals, clip_denoised=clip_denoised, mean_approach=mean_approach, ema=ema)
            if t % sample_inter == 0:
                noises.append(noise)
                pred_noises.append(pred_noise)
                x0_preds.append(x0)
                x_sequence.append(x_t)
                
            if save_slices:
                x_slices.append(x_t[:,:,:,:,d[-1]//2])

        noises = torch.cat(noises, dim=1)
        pred_noises = torch.cat(pred_noises, dim=1)
        x0_preds = torch.cat(x0_preds, dim=1)
        x_sequence = torch.cat(x_sequence, dim=1)
        
        if clip_denoised:
            x_sequence[:,-1] = x_sequence[:,-1].clamp_(-1,1)

        if save_slices:
            x_slices = torch.cat(x_slices, dim=1)
        #if continous:
        return x_sequence, x_slices, noises, pred_noises, x0_preds
        #else:
        #    return x_sequence[:,-1]
    
    def p_sample_loop_2d_slices(self, conditionals=None, continous=False):
        #sample_inter = (1 | (self.timesteps//100))
        b,(*d)  = conditionals[-1].shape #select the last conditional to get the shape (should be T21_lr because order is delta,vbv,T21_lr)
        x_t = torch.randn((b,*d))
        
        x_sequence = x_t[:,:,:,:,d[-1]//2] #use channel dimension as time axis and just save 2d slices

        for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            x_t = self.p_sample(x_t, b*[t], conditionals=conditionals)
            #if t % sample_inter == 0:
            x_sequence = torch.cat([x_sequence, x_t[:,:,:,:,d[-1]//2]], dim=1)
        if continous:
            return x_sequence
        else:
            return x_sequence[:,-1]
        
    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    model = self.model.state_dict(), 
                    optimizer = self.optG.state_dict(),# epoch = e ),
                    ema = self.ema.state_dict(),
                    loss = self.loss,
                    losses_validation_history = self.losses_validation_history,
                    noise_schedule_opt = self.noise_schedule_opt),
                    f = path
                    )
        else:
            if str(self.device) == "cuda:0":
                print("Saving model", flush=True)
                torch.save(
                    obj = dict(
                        model = self.model.module.state_dict(), 
                        optimizer = self.optG.state_dict(),# epoch = e ),
                        #ema = self.ema.state_dict(),
                        loss = self.loss,
                        losses_validation_history = self.losses_validation_history,
                        noise_schedule_opt = self.noise_schedule_opt),
                        f = path
                        )

    def load_network(self, path):
        loaded_state = torch.load(path)
        self.model.load_state_dict(loaded_state['model'])
        self.optG.load_state_dict(loaded_state['optimizer'])
        self.ema.load_state_dict(loaded_state['ema'])
        self.loss = loaded_state['loss']
        try:
            self.noise_schedule_opt = loaded_state['noise_schedule_opt']
            self.set_new_noise_schedule(self.noise_schedule_opt)
            self.losses_validation_history = loaded_state['losses_validation_history']
        except:
            print("Noise schedule or losses_validation_history couldn't be loaded")
            self.losses_validation_history = []






def rot_onto_sides_torch(x):
    x1 = torch.rot90(x, k=1, dims=(3,4)).unsqueeze(0)
    x2 = torch.rot90(x, k=1, dims=(2,4)).unsqueeze(0)
    x3 = torch.rot90(x, k=1, dims=(2,3)).unsqueeze(0)
    x = torch.cat([x1,x2,x3], dim=0)
    return x

def rot_onto_sides(x):
    x1 = torch.rot90(x, k=1, dims=(3,4))
    x2 = torch.rot90(x, k=1, dims=(2,4))
    x3 = torch.rot90(x, k=1, dims=(2,3))
    return [x1,x2,x3]

def rot_on_base(cubes):
    x = []
    for cube in cubes:
        for i in range(4):
            x.append(torch.rot90(cube, k=i, dims=(2,3)))
    return x

def rot_to_opposite_corner(x):
    x = torch.rot90(x, k=2, dims=(2,4))
    x = torch.rot90(x, k=1, dims=(2,3))
    return x

def all_rotations(x):
    corner1 = rot_on_base(rot_onto_sides(x))
    corner2 = rot_on_base(rot_onto_sides(rot_to_opposite_corner(x)))
    #print("corner1[0] shape", corner1[0].shape, "corner2[0] shape", corner2[0].shape, flush=True)
    result = torch.cat([*corner1, *corner2], dim=0) #np.array(corner1 + corner2)
    #print("rotations shape: ", result.shape, flush=True)
    return result

def random_rotations(x, n=1):

    N = np.random.choice(np.arange(0,24), size=n,replace=False)
    N = torch.tensor(N, device=x.device)


    rotations = all_rotations(x)
    
    result = rotations[N]

    return result

def augment_dataset(T21, delta, vbv, T21_lr, n=8):
    dataset = []
    for i,(x1,x2,x3,x4) in enumerate(zip(T21, delta, vbv, T21_lr)):
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)
        x4 = x4.unsqueeze(0)
        data = torch.cat([x1,x2,x3,x4],dim=1) #cat data along channels so T21 and corresponding delta, vbv, T21_lr are rotated the same way
        data = random_rotations(data[:1], n=n)#.tolist()
        #data = torch.cat(data, dim=0)
        dataset.append(data)
        
    dataset = torch.cat(dataset,dim=0)
    
    T21 = dataset[:,:1]
    delta = dataset[:,1:2]
    vbv = dataset[:,2:3]
    T21_lr = dataset[:,3:]
    return T21, delta, vbv, T21_lr

def calculate_power_spectrum(data_x, Lpix=3, kbins=100, dsq = False, method="torch"):
    #Simulation box variables
    batch, channel,(*d) = data_x.shape
    assert channel == 1, "Channel must be 1"
    Npix = d[-1]
    Vpix = Lpix**3
    Lbox = Npix * Lpix
    Vbox = Lbox**3

    if method == "numpy":
        kspace = np.fft.fftfreq(Npix, d=Lpix/(2*np.pi)) #Calculating wavevectors k for the simulation grid
        kx, ky, kz = np.meshgrid(kspace,kspace,kspace)
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        data_k = np.fft.fftn(data_x[0,0]) #Dont need to scipy.fft.fftshift since kspace isn't fftshift'ed
        k_bin_edges = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), endpoint=True, num=kbins+1) #Bin k values and calculate power spectrum
        k_vals = np.zeros(kbins)
        P_k = np.zeros(kbins)
        for i in range(kbins):
            cond = ((k >= k_bin_edges[i]) & (k < k_bin_edges[i+1]))
            k_vals[i] = (k_bin_edges[i+1] + k_bin_edges[i])/2
            P_k[i] = (Vpix/Vbox) * Vpix * np.average(np.absolute(data_k[cond]))**2
        P_k = P_k*k_vals**3/(2*np.pi**2) if dsq else P_k
        return k_vals, P_k
    elif method == "torch":
        kspace_torch = torch.fft.fftfreq(Npix, d=Lpix/(2*np.pi))
        kx_torch, ky_torch, kz_torch = torch.meshgrid(kspace_torch,kspace_torch,kspace_torch)#.view(batch,channel,*d)
        k_torch = torch.sqrt(kx_torch**2 + ky_torch**2 + kz_torch**2)#.unsqueeze(0).unsqueeze(0).repeat(batch,1,*(len(d)*[1]))
        data_k_torch = torch.fft.fftn(input=data_x, dim=(2,3,4))
        kmin_mask_torch = k_torch > 0
        kmin_torch = torch.min(k_torch[kmin_mask_torch])
        kmax_torch = torch.max(k_torch)
        k_bin_edges_torch = torch.logspace(start=torch.log10(kmin_torch), end=torch.log10(kmax_torch), steps=kbins+1)
        k_vals_torch = torch.zeros(kbins)
        P_k_torch = torch.zeros(batch,channel,kbins)
        
        conditions = [(k_torch >= k_bin_edges_torch[i]) & (k_torch < k_bin_edges_torch[i+1]) for i in range(kbins)]

        for i in range(kbins):
            cond_torch = conditions[i]
            k_vals_torch[i] = (k_bin_edges_torch[i+1] + k_bin_edges_torch[i])/2
            means = torch.mean(torch.abs(data_k_torch[...,cond_torch]),dim=2, keepdim=False)
            P_k_torch[:,:,i] = (Vpix/Vbox) * Vpix * means**2
        P_k_torch = P_k_torch*k_vals_torch**3/(2*np.pi**2) if dsq else P_k_torch
        return k_vals_torch, P_k_torch
    else:
        raise ValueError("Method must be numpy or torch")

def normalize(x):
    x_min = torch.amin(x, dim=(1,2,3,4), keepdim=True)
    x_max = torch.amax(x, dim=(1,2,3,4), keepdim=True)
    x = (x - x_min) / (x_max - x_min)
    x = 2 * x - 1
    return x

def get_subcubes(cubes, cut_factor):
    batch, channel,(*d) = cubes.shape
    image_size = d[0]//cut_factor
    sub_cubes = []
    for cube in cubes:
        for i in range(cut_factor):
            for j in range(cut_factor):
                for k in range(cut_factor):
                    sub_cube = cube[:,i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size,k*image_size:(k+1)*image_size]
                    sub_cubes.append(sub_cube)
    sub_cubes = torch.cat(sub_cubes, dim=0).unsqueeze(1)
    return sub_cubes

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('ConvolutionalLayer') != -1:
        #print("    Above")
        init.orthogonal_(m.conv.weight.data, gain=1)
        if m.conv.bias is not None:
            m.conv.bias.data.zero_()
    elif classname.find('Conv') != -1:
        #print("    Above")
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        #print("    Above")
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('GroupNorm') != -1:
        #print("    Above")
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
    #Downsample and Upsample

def init_weights(net, init_type='orthogonal'):
    #print('Initialization method [{:s}]'.format(init_type))
    if init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)

def prepare_data(path, upscale=4, cut_factor=4, redshift=10, IC_seeds=list(range(1000,1002)), device="cpu" ):
    # Load training data
    Data = DataManager(path, redshifts=[redshift,], IC_seeds=IC_seeds)
    T21, delta, vbv = Data.load()

    # Convert to pytorch
    T21 = torch.from_numpy(T21).to(device)
    delta = torch.from_numpy(delta).to(device)
    vbv = torch.from_numpy(vbv).to(device)
    
    T21 = T21.permute(0,4,1,2,3) # Convert from 8,128,128,128,1 to 8,1,128,128,128
    delta = delta.unsqueeze(1) # Expand delta and vbv dims from 8,128,128,128 to 8,1,128,128,128
    vbv = vbv.unsqueeze(1)
    T21_lr = torch.nn.functional.interpolate( # Define low resolution input that has been downsampled and upsampled again
        torch.nn.functional.interpolate(T21, scale_factor=1/upscale, mode='trilinear'),
        scale_factor=upscale, mode='trilinear')

    T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
    delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
    vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
    T21_lr = get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
    
    T21 = normalize(T21)
    delta = normalize(delta)
    vbv = normalize(vbv)
    T21_lr = normalize(T21_lr)
    
    return T21, delta, vbv, T21_lr

def prepare_dataloader(path, batch_size=2*4, upscale=4, cut_factor=4, redshift=10, IC_seeds=list(range(1000,1002)), device="cpu", multi_gpu=False):
    ###START load_train_objs() and prepare_dataloader() pytorch multi-gpu tutorial###
    
    #model_i = "20"
    #model_path = path + "/trained_models/diffusion_model_test_{0}.pth".format(model_i)
    
    T21, delta, vbv, T21_lr = prepare_data(path, upscale=upscale, cut_factor=cut_factor, redshift=redshift, IC_seeds=IC_seeds, device=device)
    dataset = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)
    print("Prepare dataloader dataset shapes: ", T21.shape, delta.shape, vbv.shape, T21_lr.shape)
    data = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = DistributedSampler(dataset) if multi_gpu else None) #4
    ###END load_train_objs() and prepare_dataloader() pytorch multi-gpu tutorial###
    return data

def train_step(netG, epoch, train_data, device="cpu", multi_gpu = False,
          ):
    """
    Train the model
    """
    netG.model.train()
    
    #multi_gpu = torch.cuda.device_count() > 1
    #if multi_gpu:
    #    device = torch.device(f'cuda:{rank}')
    #else:
    #    device = "cpu"
    #print("Multi GPU: ", multi_gpu)

    ###START load_train_objs() and prepare_dataloader() pytorch multi-gpu tutorial###
    #path = os.getcwd().split("/21cmGAN")[0] + "/21cmGAN"
    #model_path = path + "/trained_models/diffusion_model_test_{0}.pth".format(model_i)
    
    #T21, delta, vbv, T21_lr = prepare_data(path, upscale=upscale, cut_factor=cut_factor, redshift=redshift, IC_seeds=IC_seeds, rank=rank)
    #dataset = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)
    #train_data = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = DistributedSampler(dataset) if multi_gpu else None) #4
    ###END load_train_objs() and prepare_dataloader() pytorch multi-gpu tutorial###
    
    
    #stime = time.time()
    
    avg_batch_loss = torch.tensor(0.0, device=device)



    ###START _run_epoch pytorch multi-gpu tutorial###
    #b_sz = len(train_data[0])
    
    if multi_gpu:
        #print(f"[GPU{rank}] Epoch {e} ")
        print(f"[{device}] Epoch {epoch} | (Mini)Batchsize: {train_data.batch_size} | Steps (batches): {len(train_data)}")
        train_data.sampler.set_epoch(epoch)
    for i,(T21_,delta_,vbv_, T21_lr_) in enumerate(train_data):
        #if (str(device)=="cpu") or (str(device)=="cuda:0"):

            
        
        T21_,delta_,vbv_, T21_lr_ = augment_dataset(T21_,delta_,vbv_, T21_lr_, n=1) #support device
        ###START _run_batch pytorch multi-gpu tutorial###

        netG.optG.zero_grad()
        
        ts = torch.randint(low = 0, high = netG.timesteps, size = (train_data.batch_size // 2 + 1, ), device=device)
        ts = torch.cat([ts, netG.timesteps - ts - 1], dim=0)[:train_data.batch_size] # antithetic sampling
        alphas_cumprod = netG.alphas_cumprod[ts]     
        xt, target_noise = netG.q_sample(T21_, ts)
        X = torch.cat([xt, delta_, vbv_, T21_lr_], dim = 1)
        predicted_noise = netG.model(X, alphas_cumprod)
        loss = nn.MSELoss(reduction='mean')(target_noise, predicted_noise) # torch.nn.L1Loss(reduction='mean')(target_noise, predicted_noise) 
        
        avg_batch_loss += loss / len(train_data)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
        netG.optG.step()
        ###END _run_batch pytorch multi-gpu tutorial###
        ###END _run_epoch pytorch multi-gpu tutorial###
        

        #losses.append(loss.item())
        if (device=="cuda:0") or (device=="cpu"):
            if True: #i%(len(train_data)//16) == 0:
                print(f"Batch {i} of {len(train_data)} batches")

        #netG.ema.update() #Update netG.model with exponential moving average
    
    if multi_gpu:
        torch.distributed.all_reduce(tensor=avg_batch_loss, op=torch.distributed.ReduceOp.AVG)
        #print("Multigpu avg batch loss: ", avg_batch_loss)
        #print("{0} loss: {1:.4f}".format(device, avg_batch_loss.item()))
    #print("{0} loss: {1:.4f}".format(device, avg_batch_loss.item()))

    netG.loss.append(avg_batch_loss.item())
    
    return avg_batch_loss.item()


def plot_checkpoint(x_true, x_pred, x_true_lr, delta, vbv, path = None, device="cpu"):

    fig = plt.figure(figsize=(15,15))
    gs = GS(3, 3, figure=fig,) #height_ratios=[1,1,1.5])

    ax_delta = fig.add_subplot(gs[0,0])#, wspace = 0.2)
    ax_vbv = fig.add_subplot(gs[0,1])
    ax_x_true_lr = fig.add_subplot(gs[0,2])

    ax_delta.imshow(delta[0,0,:,:,delta.shape[-1]//2], vmin=-1, vmax=1)
    ax_delta.set_title("Delta (input)")
    ax_vbv.imshow(vbv[0,0,:,:,vbv.shape[-1]//2], vmin=-1, vmax=1)
    ax_vbv.set_title("Vbv (input)")
    ax_x_true_lr.imshow(x_true_lr[0,0,:,:,x_true_lr.shape[-1]//2], vmin=-1, vmax=1)
    ax_x_true_lr.set_title("T21 LR (input)")



    ax_x_true = fig.add_subplot(gs[1,0])
    ax_x_pred = fig.add_subplot(gs[1,1])




    ax_x_true.imshow(x_true[0,0,:,:,x_true.shape[-1]//2], vmin=-1, vmax=1)
    ax_x_true.set_title("T21 HR (Real)")
    
    ax_x_pred.imshow(x_pred[0,-1,:,:,x_pred.shape[-1]//2], vmin=-1, vmax=1)
    ax_x_pred.set_title("T21 SR (Generated)")


    #power spectrum
    k_vals_true, dsq_true = calculate_power_spectrum(x_true,Lpix=3, kbins=100, dsq = True, method="torch")
    
    k_vals_pred, dsq_pred = calculate_power_spectrum(x_pred[:,-1:,:,:,:], Lpix=3, kbins=100, dsq = True, method="torch")

    #k_vals_true_lr, dsq_true_lr = calculate_power_spectrum(x_true_lr, Lpix=3, kbins=100, dsq = True, method="torch")

    sgs = SGS(1,2, gs[2,:])
    ax_dsq = fig.add_subplot(sgs[0])

    ax_dsq.plot(k_vals_true, dsq_true[0,0], label="T21 HR", ls='solid')

    ax_dsq.plot(k_vals_pred, dsq_pred[0,0], label="T21 SR", ls='dotted')
    
    ax_dsq.set_ylabel('$\Delta^2(k)_\\mathrm{{norm}}$')
    ax_dsq.set_xlabel('$k$')
    ax_dsq.set_yscale('log')
    ax_dsq.grid()
    ax_dsq.legend()
    ax_dsq.set_title("Power Spectrum (output)")


    ax_hist = fig.add_subplot(sgs[1])
    ax_hist.hist(x_pred[0,-1,:,:,:].flatten(), bins=100, alpha=0.5, label="T21 SR", density=True)
    ax_hist.hist(x_true[0,0,:,:,:].flatten(), bins=100, alpha=0.5, label="T21 HR", density=True)
    
    ax_hist.set_xlabel("Norm. $T_{{21}}$")
    ax_hist.set_ylabel("PDF")
    ax_hist.legend()
    ax_hist.grid()
    ax_hist.set_title("Pixel Histogram (output)")

    plt.savefig(path)
    plt.close()




def validation_step(netG, validation_data, validation_type="DDIM", validation_loss_type="dsq", device="cpu", multi_gpu=False):
    #validation_type = "DDIM" # or "DDPM SR3" or "DDPM Classic" or "None" (to save at every minimum training loss)
    #validation_loss_type = "dsq" # or voxel

    #if len(netG.loss)>=validation_epoch: #only start checking voxel loss after n epochs #change this when it works 
    losses_validation = torch.tensor(0.0, device=device) #0 #avg_batch_loss = torch.tensor(0.0, device=device)
    #stime_ckpt = time.time()

    if validation_type == "DDIM":
        #print(validation_type + " validation")
        for i,(T21_validation_, delta_validation_, vbv_validation_, T21_lr_validation_) in enumerate(validation_data):
        #for i, (T21_validation_, delta_validation_, vbv_validation_, T21_lr_validation_) in tqdm(enumerate(validation_data), total=len(validation_data)):
            x_sequence, x_slices, noises, pred_noises, x0_preds = netG.p_sample_loop(conditionals=[delta_validation_, vbv_validation_, T21_lr_validation_], n_save=2, clip_denoised=True, mean_approach = "DDIM", save_slices=True, ema=False, ddim_n_steps = 10, verbose=False, device=device)
            
            if validation_loss_type == "dsq":
                k_vals_true, dsq_true  = calculate_power_spectrum(T21_validation_, Lpix=3, kbins=100, dsq = True, method="torch")
                k_vals_pred, dsq_pred  = calculate_power_spectrum(x_sequence[:,-1:], Lpix=3, kbins=100, dsq = True, method="torch")
                losses_validation += torch.nanmean(torch.square(dsq_pred - dsq_true)) / len(validation_data)
            elif validation_loss_type == "voxel":
                losses_validation += torch.nanmean(torch.square(x_sequence[:,-1:] - T21_validation_)) / len(validation_data) #nn.MSELoss(reduction='mean')(x_sequence[:,-1:], T21_validation_).item()
            else:
                assert False, "Validation loss type not recognized"


        #print("{0} loss: {1:.4f}".format(device, losses_validation.item()))
        if multi_gpu:
            torch.distributed.all_reduce(tensor=losses_validation, op=torch.distributed.ReduceOp.AVG)
            #print("{0} loss: {1:.4f}".format(device, losses_validation.item()))


        #losses_validation /= len(validation_data)
        netG.losses_validation_history.append(losses_validation.item())
        #print("{0} loss {1:.4f} and time {2:.2f}".format(validation_type, losses_validation, time.time()-stime_ckpt))
        #save_bool = losses_validation == np.min(netG.losses_validation_history)
    return losses_validation.item(), [T21_validation_, x_sequence, T21_lr_validation_, delta_validation_, vbv_validation_]
        #elif (validation_type == "DDPM SR3") or (validation_type=="DDPM Classic"):
        #    print(validation_type + " validation")
        #    T21_validation_, delta_validation_, vbv_validation_, T21_lr_validation_ = loader_validation.dataset.tensors
        #    #pick random int from batch shape 0 of validation data
        #    i = torch.randint(low=0, high=T21_validation_.shape[0], size=(1,)).item()
        #    x_sequence, x_slices, noises, pred_noises, x0_preds = netG.p_sample_loop(conditionals=[delta_validation_[i:i+1], vbv_validation_[i:i+1], T21_lr_validation_[i:i+1]], n_save=100, clip_denoised=True, mean_approach = validation_type, save_slices=False, ema=True, ddim_n_steps = 10, verbose=True)
        #    
        #    if validation_loss_type == "dsq":
        #        k_vals_true, dsq_true  = calculate_power_spectrum(T21_validation_[i:i+1], Lpix=3, kbins=100, dsq = True, method="torch")
        #        k_vals_pred, dsq_pred  = calculate_power_spectrum(x_sequence[:,-1:], Lpix=3, kbins=100, dsq = True, method="torch")
        #        losses_validation += torch.nanmean(torch.square(dsq_pred - dsq_true)).item() #nn.MSELoss(reduction='mean')(dsq_pred, dsq_true).item()
        #    elif validation_loss_type == "voxel":
        #        losses_validation += torch.nanmean(torch.square(x_sequence[:,-1:] - T21_validation_[i:i+1])).item()
        #    else:
        #        assert False, "Validation loss type not recognized"
        #
        #
        #    netG.losses_validation_history.append(losses_validation)
        #    print("{0} voxel loss {1:.4f} and time {2:.2f}".format(validation_type, losses_validation, time.time()-stime_ckpt))
        #    save_bool = losses_validation == np.min(netG.losses_validation_history)
        #
        #elif validation_type == "None":
        #    save_bool = True
        #
        #else:
        #    assert False, "Validation type not recognized"
    #else:
        #save_bool = True
    

    #if save_bool:
        #print("Saving model now. Loss history is: ", netG.losses_validation_history)
        ##netG.save_network(model_path)


###START main pytorch multi-gpu tutorial###
def main(rank, world_size=0, total_epochs = 1, batch_size = 2*4):
    
    multi_gpu = world_size > 1

    if multi_gpu:
        device = torch.device(f'cuda:{rank}')
        print("Multi GPU: {0}, device: {1}".format(multi_gpu,device))
        ddp_setup(rank, world_size=world_size)
    else:
        device = "cpu"
        print("Multi GPU: {0}, device: {1}".format(multi_gpu,device))
    


    #optimizer and model
    path = os.getcwd().split("/21cmGAN")[0] + "/21cmGAN"
    model = UNet
    model_opt = dict(in_channel=4, out_channel=1, inner_channel=8, norm_groups=8, channel_mults=(1, 2, 2, 4, 4), attn_res=(8,), res_blocks=2, dropout = 0, with_attn=True, image_size=16, dim=3)#T21.shape[-1], dim=3)
    noise_schedule_opt = dict(timesteps = 1000, s = 0.008) #dict(timesteps = 1000, beta_start = 1e-6, beta_end = 1e-2) #21cm ddpm ###

    netG = GaussianDiffusion(
            model=model,
            model_opt=model_opt,
            loss_type='l1',
            noise_schedule=cosine_beta_schedule,#linear_beta_schedule,#
            noise_schedule_opt=noise_schedule_opt,
            learning_rate=1e-4,
            rank=rank,
        )

    train_data = prepare_dataloader(path=path, batch_size=batch_size, upscale=4, cut_factor=4, redshift=10, IC_seeds=list(range(1000,1002)), device=device, multi_gpu=multi_gpu)
    validation_data = prepare_dataloader(path=path, batch_size=batch_size, upscale=4, cut_factor=4, redshift=10, IC_seeds=list(range(1010,1011)), device=device, multi_gpu=multi_gpu)

    for e in range(total_epochs):
        avg_batch_loss = train_step(netG=netG, epoch=e, train_data=train_data, device=device, multi_gpu=multi_gpu)

        if avg_batch_loss == torch.min(torch.tensor(netG.loss)).item():
            
            if len(netG.loss)>=100: #only start checking voxel loss after n epochs #change this when it works
                losses_validation, input_output = validation_step(netG=netG, validation_data=validation_data, validation_type="DDIM", validation_loss_type="dsq", device=device, multi_gpu=multi_gpu)
                input_output = [x.cpu() for x in input_output] if multi_gpu else input_output

                if (str(device)=="cuda:0") or (str(device)=="cpu"):
                    print("losses_validation: {0:.4f}, losses_validation_history minimum: {1:.4f}".format(losses_validation, torch.min(torch.tensor(netG.losses_validation_history)).item()))
                
                    if losses_validation == torch.min(torch.tensor(netG.losses_validation_history)).item():
                            
                        #print("Saving model", flush=True)
                        #[x.cpu() for x in input_output]
                        plot_checkpoint(*input_output, path = path + f"/trained_models/diffusion_epoch_{e}_model_20.png", device="cpu")
                        netG.save_network( path + "/trained_models/diffusion_model_test_20.pth"  )
                    else:
                        print("Not saving model. Validaiton did not improve", flush=True)

    
    if multi_gpu:#world_size > 1:
        destroy_process_group()
###END main pytorch multi-gpu tutorial###



if __name__ == "__main__":
   
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    if multi_gpu:
        print("Using multi_gpu", flush=True)
        for i in range(torch.cuda.device_count()):
            print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)
        mp.spawn(main, args=(world_size, 1000, 16), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
    else:
        print("Not using multi_gpu",flush=True)
        main(rank=0, world_size=0, total_epochs=1, batch_size=8)#2*4)
    
        


        
            

    

    """




    #ema = ExponentialMovingAverage(netG.model.parameters(), decay=0.995)
    #ema_model = copy.deepcopy(nn_model).eval().requires_grad_(False)
    model_i = "20"
    model_path = path + "/trained_models/diffusion_model_test_{0}.pth".format(model_i)
    if os.path.isfile(model_path):
        print("Loading checkpoint", flush=True)
        netG.load_network(model_path)
    else:
        print(f"No checkpoint found at {model_path}. Starting from scratch.", flush=True)

    
    print("Starting training", flush=True)

    train(netG=netG, epochs=1, T21=T21, delta=delta, vbv=vbv, T21_lr=T21_lr, 
          batch_size=2*cut_factor, gpu_id=0, multi_gpu=False)

    
    for e in range(1):
        #stime = time.time()
        train(netG, T21, delta, vbv, T21_lr, batch_size=2*cut_factor, shuffle=True)
        
        
        #augment and create dataset for torch DataLoader
        T21_aug, delta_aug, vbv_aug, T21_lr_aug = augment_dataset(T21, delta, vbv, T21_lr, n=1)
        dataset = torch.utils.data.TensorDataset(T21_aug, delta_aug, vbv_aug, T21_lr_aug)
        #dataset = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)

        

        loader = torch.utils.data.DataLoader( dataset, batch_size=2*cut_factor, shuffle=True) #4
        netG.model.train()
        
        losses = []
        
        for i,(T21_,delta_,vbv_, T21_lr_) in enumerate(loader):
            # antithetic sampling
            ts = torch.randint(low = 0, high = netG.timesteps, size = (loader.batch_size // 2 + 1, ))
            ts = torch.cat([ts, netG.timesteps - ts - 1], dim=0)[:loader.batch_size]
            alphas_cumprod = netG.alphas_cumprod[ts]
            
            xt, target_noise = netG.q_sample(T21_, ts)
            X = torch.cat([xt, delta_, vbv_, T21_lr_], dim = 1)
            
            predicted_noise = netG.model(X, alphas_cumprod)
            
            loss = nn.MSELoss(reduction='mean')(target_noise, predicted_noise) # torch.nn.L1Loss(reduction='mean')(target_noise, predicted_noise) 
            
            netG.optG.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
            netG.optG.step()
    
            if i%(len(loader)//16) == 0:
                print(f"Batch {i} of {len(loader)} batches")
    
    
    

            netG.ema.update() #Update netG.model with exponential moving average

            losses.append(loss.item())

            

        losses_epoch = np.mean(losses)
        netG.loss.append(losses_epoch)





        validation_type = "DDIM" # or "DDPM SR3" or "DDPM Classic" or "None" (to save at every minimum training loss)
        validation_loss_type = "dsq" # or voxel

        if True:#losses_epoch == np.min(netG.loss): #save_bool:
            if e>=0: #only start checking voxel loss after n epochs #change this when it works 
                losses_validation = 0
                stime_ckpt = time.time()

                if validation_type == "DDIM":
                    print(validation_type + " validation")
                    #for i,(T21_validation_, delta_validation_, vbv_validation_, T21_lr_validation_) in enumerate(loader_validation):
                    for i, (T21_validation_, delta_validation_, vbv_validation_, T21_lr_validation_) in tqdm(enumerate(loader_validation), total=len(loader_validation)):
                        x_sequence, x_slices, noises, pred_noises, x0_preds = netG.p_sample_loop(conditionals=[delta_validation_, vbv_validation_, T21_lr_validation_], n_save=2, clip_denoised=True, mean_approach = "DDIM", save_slices=True, ema=True, ddim_n_steps = 10, verbose=False)
                        
                        if validation_loss_type == "dsq":
                            k_vals_true, dsq_true  = calculate_power_spectrum(T21_validation_, Lpix=3, kbins=100, dsq = True, method="torch")
                            k_vals_pred, dsq_pred  = calculate_power_spectrum(x_sequence[:,-1:], Lpix=3, kbins=100, dsq = True, method="torch")
                            losses_validation += torch.nanmean(torch.square(dsq_pred - dsq_true)).item() #nn.MSELoss(reduction='mean')(x_sequence[:,-1:], T21_validation_).item()
                        elif validation_loss_type == "voxel":
                            losses_validation += torch.nanmean(torch.square(x_sequence[:,-1:] - T21_validation_)).item()
                        else:
                            assert False, "Validation loss type not recognized"

                    losses_validation /= len(loader)
                    netG.losses_validation_history.append(losses_validation)
                    print("{0} voxel loss {1:.4f} and time {2:.2f}".format(validation_type, losses_validation, time.time()-stime_ckpt))
                    save_bool = losses_validation == np.min(netG.losses_validation_history)

                elif (validation_type == "DDPM SR3") or (validation_type=="DDPM Classic"):
                    print(validation_type + " validation")
                    T21_validation_, delta_validation_, vbv_validation_, T21_lr_validation_ = loader_validation.dataset.tensors
                    #pick random int from batch shape 0 of validation data
                    i = torch.randint(low=0, high=T21_validation_.shape[0], size=(1,)).item()
                    x_sequence, x_slices, noises, pred_noises, x0_preds = netG.p_sample_loop(conditionals=[delta_validation_[i:i+1], vbv_validation_[i:i+1], T21_lr_validation_[i:i+1]], n_save=100, clip_denoised=True, mean_approach = validation_type, save_slices=False, ema=True, ddim_n_steps = 10, verbose=True)
                    
                    if validation_loss_type == "dsq":
                        k_vals_true, dsq_true  = calculate_power_spectrum(T21_validation_[i:i+1], Lpix=3, kbins=100, dsq = True, method="torch")
                        k_vals_pred, dsq_pred  = calculate_power_spectrum(x_sequence[:,-1:], Lpix=3, kbins=100, dsq = True, method="torch")
                        losses_validation += torch.nanmean(torch.square(dsq_pred - dsq_true)).item() #nn.MSELoss(reduction='mean')(dsq_pred, dsq_true).item()
                    elif validation_loss_type == "voxel":
                        losses_validation += torch.nanmean(torch.square(x_sequence[:,-1:] - T21_validation_[i:i+1])).item()
                    else:
                        assert False, "Validation loss type not recognized"
                

                    netG.losses_validation_history.append(losses_validation)
                    print("{0} voxel loss {1:.4f} and time {2:.2f}".format(validation_type, losses_validation, time.time()-stime_ckpt))
                    save_bool = losses_validation == np.min(netG.losses_validation_history)
                
                elif validation_type == "None":
                    save_bool = True
                
                else:
                    assert False, "Validation type not recognized"
            else:
                save_bool = True
            

            if save_bool:
                print("Saving model now. Loss history is: ", netG.losses_validation_history)
                #netG.save_network(model_path)
        else:
            save_bool = False
            
            


    
            
        ftime = time.time()
        print("Epoch {0} trained in {1:.2f}s. Average loss {2:.4f} over {3} batches. Saved: {4}".format(e, ftime - stime, losses_epoch, len(loader), save_bool),flush=True)
    
    
    #print("Losses:\n", np.round(netG.loss,4), "\n", flush=True)
    
    ####################Load validation data for testing
    Data = DataManager(path, redshifts=[10,], IC_seeds=list(range(1010,1011)))
    T21, delta, vbv = Data.load()
    T21 = torch.from_numpy(T21)
    delta = torch.from_numpy(delta)
    vbv = torch.from_numpy(vbv)
    T21 = T21.permute(0,4,1,2,3)
    delta = delta.unsqueeze(1)
    vbv = vbv.unsqueeze(1)
    upscale = 2
    T21_lr = torch.nn.functional.interpolate( #define low resolution input that has been downsampled and upsampled again
        torch.nn.functional.interpolate(T21, scale_factor=1/upscale, mode='trilinear'),
        scale_factor=upscale, mode='trilinear')
    T21 = T21[:1,:,60:92,74:106,48:80]#test cutout of ionized region
    delta = delta[:1,:,60:92,74:106,48:80]#test cutout of ionized region
    vbv = vbv[:1,:,60:92,74:106,48:80]#test cutout of ionized region
    T21_lr = T21_lr[:1,:,60:92,74:106,48:80]#test cutout of ionized region
    T21 = normalize(T21)#[:,:,:T21.shape[2]//reduce_dim,:T21.shape[3]//reduce_dim,:T21.shape[4]//reduce_dim] #train on reduce_dim dimensions
    delta = normalize(delta)#[:,:,:delta.shape[2]//reduce_dim,:delta.shape[3]//reduce_dim,:delta.shape[4]//reduce_dim]
    vbv = normalize(vbv)#[:,:,:vbv.shape[2]//reduce_dim,:vbv.shape[3]//reduce_dim,:vbv.shape[4]//reduce_dim]
    T21_lr = normalize(T21_lr)#[:,:,:T21_lr.shape[2]//reduce_dim,:T21_lr.shape[3]//reduce_dim,:T21_lr.shape[4]//reduce_dim]
    ####################End load validation data for testing
    
    
    netG.load_network(model_path)
    print("print model after training loop load: ", netG.model.state_dict()['final_block.2.conv.weight'][0,0,0,:,:])
    x_sequence, x_slices, noises, pred_noises, x0_preds = netG.p_sample_loop(conditionals=[delta,vbv,T21_lr], n_save=10, clip_denoised=True, mean_approach = "DDPM SR3", save_slices=False, ema=True, ddim_n_steps = 10, verbose=True)

    
    nrows = 3
    ncols = 5


    rng = np.linspace(0, x_sequence.shape[1]-1, nrows*ncols, dtype=int)

    fig,axes = plt.subplots(nrows, ncols, figsize=(nrows*4, ncols))
    for i,ax in zip(rng,axes.flatten()):
        ax.imshow(x_sequence[0,i,:,:,x_sequence.shape[4]//2], vmin=-1, vmax=1)
        ax.set_title(f"t={i}")
        ax.axis('off')

    plt.savefig(path + "/trained_models/diffusion_model_{0}.png".format(model_i))

    fig,axes = plt.subplots(2, 3, figsize=(15,10))

    axes[0,0].imshow(T21_lr[0,0,:,:,T21_lr.shape[4]//2], vmin=-1, vmax=1)
    axes[0,0].set_title("T21 LR (input)")
    axes[0,1].imshow(delta[0,0,:,:,delta.shape[4]//2], vmin=-1, vmax=1)
    axes[0,1].set_title("Delta (input)")
    axes[0,2].imshow(vbv[0,0,:,:,vbv.shape[4]//2], vmin=-1, vmax=1)
    axes[0,2].set_title("Vbv (input)")

    axes[1,0].imshow(x_sequence[0,-1,:,:,x_sequence.shape[4]//2], vmin=-1, vmax=1)
    axes[1,0].set_title("T21 SR (Generated)")
    axes[1,1].imshow(T21[0,0,:,:,T21.shape[4]//2], vmin=-1, vmax=1)
    axes[1,1].set_title("T21 HR (Real)")


    #k_vals_real, P_k_real = calculate_power_spectrum(T21[0,0,:,:,:].numpy())
    k_vals_real, dsq_real  = calculate_power_spectrum(T21, Lpix=3, kbins=100, dsq = True, method="torch")
    #dsq_real = P_k_real*k_vals_real**3/(2*np.pi**2)
    axes[1,2].plot(k_vals_real, dsq_real[0,0], label="T21 HR", ls='solid')

    #k_vals_gen, P_k_gen = calculate_power_spectrum(x_sequence[0,-1,:,:,:].numpy())
    k_vals_gen, dsq_gen  = calculate_power_spectrum(x_sequence[:,-1:], Lpix=3, kbins=100, dsq = True, method="torch")
    #dsq_gen = P_k_gen*k_vals_gen**3/(2*np.pi**2)
    axes[1,2].plot(k_vals_gen, dsq_gen[0,0], label="T21 SR", ls='dashed')

    axes[1,2].set_yscale('log')
    axes[1,2].grid()
    axes[1,2].legend()

    plt.savefig(path + "/trained_models/diffusion_model_sample_{0}.png".format(model_i))

    #save x_sequence and x_slices
    #torch.save(x_sequence, path + "/trained_models/diffusion_model_sample_{0}.pt".format(model_i))

#plt.show()

#torchrun
#    --standalone
#    --nnodes=1
#    --nproc-per-node=$NUM_TRAINERS
#    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
    """