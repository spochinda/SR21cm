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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device="cpu" #because 3D convolutions are not supported on MPS
print(f"Using {device} device", flush=True)

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
    ):
        super().__init__()
        self.model = model(**model_opt)
        #self.model_opt = model_opt
        init_weights(self.model, init_type='orthogonal')
        self.optG = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.loss = []
        self.loss_type = loss_type
        self.noise_schedule = noise_schedule
        self.noise_schedule_opt = noise_schedule_opt
        if noise_schedule is not None:
            self.set_new_noise_schedule(noise_schedule_opt)

    def set_new_noise_schedule(self, noise_schedule_opt):

        self.betas = self.noise_schedule(**self.noise_schedule_opt)
        self.timesteps, = self.betas.shape
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.tensor(np.append(1., self.alphas_cumprod[:-1]), dtype=torch.float32) #need to specify dtype because np.append with 1. is float64
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def predict_start_from_noise(self, x_t, t, noise):
        b,(*d) = x_t.shape
        alpha_cumprod = self.alphas_cumprod[t].view(b,*[1]*len(d))
        x0 = x_t/torch.sqrt(alpha_cumprod) - torch.sqrt(1-alpha_cumprod) * noise/torch.sqrt(alpha_cumprod)
        return x0

    def q_sample(self, x0, t, noise=None): #forward diffusion
        #"t and batch number dim should be the same"
        b,(*d) = x0.shape
        t=torch.tensor(t).view(b,*[1]*len(d))
        alphas_cumprod_t = self.alphas_cumprod[t]
        noise = torch.randn_like(x0) if noise==None else noise
        x_t = torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, x_t, t, conditionals=None):
        b,(*d) = x_t.shape
        t=torch.tensor(t).view(b,*[1]*len(d))
        alpha_t, alpha_t_cumprod, alpha_t_cumprod_prev, beta_t = self.alphas[t], self.alphas_cumprod[t], self.alphas_cumprod_prev[t], self.betas[t]
        

        pred_noise = self.model(x=torch.cat([x_t, *conditionals], dim=1), time=self.alphas_cumprod[t])
        #x0 = self.predict_start_from_noise(x_t=x_t, t=t, noise=noise)

        #posterior_mean = (torch.sqrt(self.alphas_cumprod_prev[t])*self.betas[t]/(1-self.alphas_cumprod[t])) * x0 + \
        #    (torch.sqrt(self.alphas_cumprod[t])*(1-self.alphas_cumprod_prev[t])/(1-self.alphas_cumprod[t])) * x_t #mu_tilde_t in the ddpm paper. q_posterior on github
        #posterior_mean = (torch.sqrt(alpha_t_cumprod_prev) * beta_t/(1-alpha_t_cumprod)) * x0 + \
        #    (torch.sqrt(alpha_t_cumprod)*(1-alpha_t_cumprod_prev)/(1-alpha_t_cumprod)) * x_t #mu_tilde_t in the ddpm paper. q_posterior on github
        posterior_mean_t = (torch.sqrt(1/alpha_t)) * (x_t - beta_t/torch.sqrt(1 - alpha_t_cumprod) * pred_noise)
        posterior_variance_t = self.posterior_variance[t] #torch.sqrt(beta_t)
        noise = torch.randn_like(x_t) #if t > 0 else torch.zeros_like(x_t)

        x_t = posterior_mean_t + noise * posterior_variance_t if t.item() > 0 else posterior_mean_t
        
        
        if t.item()%50 == 0:
            print("x_t mean: {0:.2f}, noise mean: {1:.2f}, \
                  posterior_variance: {2:.2f}, sqrt(1/a): {3:.4f}, (1-a)/sqrt(1-abar): {4:.4f}\
                  ".format(torch.mean(x_t).item(),torch.mean(noise).item(), 
                           posterior_variance_t.item(), torch.sqrt(1/alpha_t).item() ,((1-alpha_t)/torch.sqrt(1 - alpha_t_cumprod)).item() 
                           ))
            #print("posterior mean terms: term1: {0:.2f}, term2: {1:.4f}".format( ((torch.sqrt(1/alpha_t)) * x_t).mean().item(), ((torch.sqrt(1/alpha_t)) * (1-alpha_t)/torch.sqrt(1 - alpha_t_cumprod) * pred_noise).mean().item() ))
            #print("posterior variance * noise: {0:.4f}".format((posterior_variance_t * noise).mean().item()))
        
        for i,(x,t_) in enumerate(zip(x_t,t)):
            if t_==0:
                noise[i] = torch.zeros_like(x)
            else:
                noise[i] = torch.randn_like(x)
        
        
        return x_t, noise, pred_noise #torch.sqrt(beta_t)*noise

    @torch.no_grad()
    def p_sample_loop(self, conditionals=None, continous=False):
        self.model.eval()
        sample_inter = (1 | (self.timesteps//10))
        b,(*d)  = conditionals[-1].shape #select the last conditional to get the shape (should be T21_lr because order is delta,vbv,T21_lr)
        x_t = torch.randn((b,*d))
        x_sequence = x_t #use channel dimension as time axis
        noises = []
        pred_noises = []
        for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            x_t, noise, pred_noise = self.p_sample(x_t, b*[t], conditionals=conditionals)
            if t % sample_inter == 0:
                noises.append(noise)
                pred_noises.append(pred_noise)
                x_sequence = torch.cat([x_sequence, x_t], dim=1)
            if (t == 999) or (t == 700) or (t == 400) or (t == 100) or (t == 0):
                #print("self.alphas_cumprod[t] in p_sample_loop: ", self.alphas_cumprod[t])
                print("print model sampling: ", self.model.state_dict()['final_block.2.conv.weight'][0,0,0,:,:])
        noises = torch.cat(noises, dim=1)
        pred_noises = torch.cat(pred_noises, dim=1)
        if continous:
            return x_sequence, noises, pred_noises
        else:
            return x_sequence[:,-1]
    
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
        torch.save(
            obj = dict(
                model = self.model.state_dict(), 
                optimizer = self.optG.state_dict(),# epoch = e ),
                ema = self.ema.state_dict(),
                loss = self.loss),
                f = path
                )

    def load_network(self, path):
        loaded_state = torch.load(path)
        self.model.load_state_dict(loaded_state['model'])
        self.optG.load_state_dict(loaded_state['optimizer'])
        self.ema.load_state_dict(loaded_state['ema'])
        self.loss = loaded_state['loss']




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
    return np.array(corner1 + corner2)

def random_rotations(x, n=1):
    N = np.random.choice(np.arange(0,24), size=n,replace=False)
    rotations = all_rotations(x)
    return rotations[N]

def augment_dataset(T21, delta, vbv, T21_lr, n=8):
    dataset = []
    for i,(x1,x2,x3,x4) in enumerate(zip(T21, delta, vbv, T21_lr)):
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)
        x4 = x4.unsqueeze(0)
        data = torch.cat([x1,x2,x3,x4],dim=1) #cat data along channels so T21 and corresponding delta, vbv, T21_lr are rotated the same way
        data = random_rotations(data[:1], n=n).tolist()
        data = torch.cat(data, dim=0)
        dataset.append(data)
        
    dataset = torch.cat(dataset,dim=0)
    
    T21 = dataset[:,:1]
    delta = dataset[:,1:2]
    vbv = dataset[:,2:3]
    T21_lr = dataset[:,3:]
    return T21, delta, vbv, T21_lr

def calculate_power_spectrum(data_x, Lpix=3, kbins=100):
    #Simulation box variables
    Npix = data_x.shape[0]
    Vpix = Lpix**3
    Lbox = Npix * Lpix
    Vbox = Lbox**3

    #Calculating wavevectors k for the simulation grid
    kspace = np.fft.fftfreq(Npix, d=Lpix/(2*np.pi))
    kx, ky, kz = np.meshgrid(kspace,kspace,kspace)
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    #Dont need to scipy.fft.fftshift since kspace isn't fftshift'ed
    data_k = np.fft.fftn(data_x)

    #Bin k values and calculate power spectrum
    k_bin_edges = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), endpoint=True, num=kbins+1)
    k_vals = np.zeros(kbins)
    P_k = np.zeros(kbins)
    for i in range(kbins):
        cond = ((k >= k_bin_edges[i]) & (k < k_bin_edges[i+1]))
        k_vals[i] = (k_bin_edges[i+1] + k_bin_edges[i])/2
        P_k[i] = (Vpix/Vbox) * Vpix * np.average(np.absolute(data_k[cond]))**2
        
    return k_vals, P_k

def normalize(x):
    x_min = torch.amin(x, dim=(1,2,3,4), keepdim=True)
    x_max = torch.amax(x, dim=(1,2,3,4), keepdim=True)
    x = (x - x_min) / (x_max - x_min)
    x = 2 * x - 1
    return x

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

if __name__ == "__main__":
    from utils import DataManager

    path = os.getcwd().split("/21cmGAN")[0] + "/21cmGAN"

    Data = DataManager(path, redshifts=[10,], IC_seeds=list(range(1000,1009)))
    T21, delta, vbv = Data.load()

    #convert to pytorch
    T21 = torch.from_numpy(T21)
    delta = torch.from_numpy(delta)
    vbv = torch.from_numpy(vbv)

    #convert from 8,128,128,128,1 to 8,1,128,128,128
    T21 = T21.permute(0,4,1,2,3)
    #Expand delta and vbv dims from 8,128,128,128 to 8,1,128,128,128
    delta = delta.unsqueeze(1)
    vbv = vbv.unsqueeze(1)


    reduce_dim = 4#4
    T21 = normalize(T21)[:,:,:T21.shape[2]//reduce_dim,:T21.shape[3]//reduce_dim,:T21.shape[4]//reduce_dim] #train on reduce_dim dimensions
    delta = normalize(delta)[:,:,:delta.shape[2]//reduce_dim,:delta.shape[3]//reduce_dim,:delta.shape[4]//reduce_dim]
    vbv = normalize(vbv)[:,:,:vbv.shape[2]//reduce_dim,:vbv.shape[3]//reduce_dim,:vbv.shape[4]//reduce_dim]
    upscale = 4
    T21_lr = torch.nn.functional.interpolate( #define low resolution input that has been downsampled and upsampled again
        torch.nn.functional.interpolate(T21, scale_factor=1/upscale, mode='trilinear'),
        scale_factor=upscale, mode='trilinear')
    T21, delta, vbv, T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=12)

    #create dataset for torch DataLoader
    dataset = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)

    #optimizer and model
    model = UNet
    model_opt = dict(in_channel=4, out_channel=1, inner_channel=8, norm_groups=8, channel_mults=(1, 2, 2, 4, 4), attn_res=(8,), res_blocks=2, dropout = 0, with_attn=True, image_size=32, dim=3)#T21.shape[-1], dim=3)
    noise_schedule_opt = dict(timesteps = 2000, beta_start = 1e-6, beta_end = 1e-2) #21cm ddpm ###dict(timesteps = 1000, s = 0.008)

    netG = GaussianDiffusion(
            model=model,
            model_opt=model_opt,
            loss_type='l1',
            noise_schedule=linear_beta_schedule,#cosine_beta_schedule,
            noise_schedule_opt=noise_schedule_opt,
            learning_rate=1e-4
        )

    ###3D test
    #ts = torch.randint(low = 1, high = netG.timesteps, size = (4, ))
    #alphas_cumprod = netG.alphas_cumprod[ts]
    #xt, target_noise = netG.q_sample(T21[:4], ts)
    #X = torch.cat([xt, delta[:4], vbv[:4], T21_lr[:4]], dim = 1)
    #predicted_noise = netG.model(X, alphas_cumprod)

    #ema = ExponentialMovingAverage(netG.model.parameters(), decay=0.995)
    #ema_model = copy.deepcopy(nn_model).eval().requires_grad_(False)
    model_i = "9"
    model_path = path + "/trained_models/diffusion_model_test_{0}.pth".format(model_i)
    if os.path.isfile(model_path):
        print("Loading checkpoint", flush=True)
        netG.load_network(model_path)
    else:
        print(f"No checkpoint found at {model_path}. Starting from scratch.", flush=True)

    
    
    loss = nn.MSELoss(reduction='mean')

    print("Starting training", flush=True)
    for e in range(400):
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        netG.model.train()
        
        losses = []
        stime = time.time()
        for i,(T21,delta,vbv, T21_lr) in enumerate(loader):
            ts = torch.randint(low = 0, high = netG.timesteps, size = (loader.batch_size, ))
            alphas_cumprod = netG.alphas_cumprod[ts]
            
            #print("alphas_cumprod in train: ", alphas_cumprod)

            xt, target_noise = netG.q_sample(T21, ts)
            X = torch.cat([xt, delta, vbv, T21_lr], dim = 1)
            
            predicted_noise = netG.model(X, alphas_cumprod)
            
            loss = nn.MSELoss(reduction='mean')(target_noise, predicted_noise)
            
            netG.optG.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
            netG.optG.step()
            #Update netG.model with exponential moving average
            netG.ema.update()
            
            losses.append(loss.item())
            if False: #not i % (len(loader)//2):
                print(f"Bacth {i} of {len(loader)} batches")
        if e==0:
            #print("print model train: ", netG.model.state_dict()['final_block.2.conv.weight'][0,0,0,:,:])
            j = ts.argmin().item()
            fig,ax = plt.subplots(1,3, figsize=(15,5))
            ax[0].imshow(predicted_noise[j,0,:,:,predicted_noise.shape[4]//2].detach().numpy(), vmin=-1, vmax=1)
            ax[0].set_title("Predicted noise, time={0}, \nnoise min/max/mean/std={1:.2f}/{2:.2f}/{3:.2f}/{4:.2f})".format(ts[j], predicted_noise.min().item(), predicted_noise.max().item(), predicted_noise.mean().item(), predicted_noise.std().item()))
            ax[1].imshow(target_noise[j,0,:,:,target_noise.shape[4]//2].detach().numpy(), vmin=-1, vmax=1)
            ax[1].set_title("Target noise, time={0}, \nnoise min/max/mean/std={1:.2f}/{2:.2f}/{3:.2f}/{4:.2f})".format(ts[j], target_noise.min().item(), target_noise.max().item(), target_noise.mean().item(), target_noise.std().item()))
            #residual
            residual = predicted_noise - target_noise
            ax[2].imshow(residual[j,0,:,:,residual.shape[4]//2].detach().numpy(), vmin=-1, vmax=1)
            ax[2].set_title("Residual, time={0}, \nresidual min/max/mean/std={1:.2f}/{2:.2f}/{3:.2f}/{4:.2f})".format(ts[j], residual.min().item(), residual.max().item(), residual.mean().item(), residual.std().item()))
            plt.savefig(path + "/trained_models/diffusion_model_intraining_{0}.png".format(model_i))

        
        losses_epoch = np.mean(losses)
        netG.loss.append(losses_epoch)
        save_bool = losses_epoch == np.min(netG.loss)
        if save_bool:
            
            netG.save_network(model_path)
        ftime = time.time()
        print("Epoch {0} trained in {1:.2f}s. Average loss {2:.4f} over {3} batches. Saved: {4}".format(e, ftime - stime, losses_epoch, len(loader), save_bool),flush=True)

    
    print("Losses:\n", np.round(netG.loss,4), "\n", flush=True)

    T21 = T21[:1]#[:1,:,:16,:16,:16]#[:1]
    delta = delta[:1]#[:1,:,:16,:16,:16]#[:1]
    vbv = vbv[:1]#[:1,:,:16,:16,:16]#[:1]
    T21_lr = T21_lr[:1]#[:1,:,:16,:16,:16]#[:1]
    
    print("print model just outiside training loop: ", netG.model.state_dict()['final_block.2.conv.weight'][0,0,0,:,:])
    #netG.load_network(model_path)
    x_sequence,noise, pred_noises = netG.p_sample_loop(conditionals=[delta,vbv,T21_lr], continous=True)

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


    k_vals_real, P_k_real = calculate_power_spectrum(T21[0,0,:,:,:].numpy())
    dsq_real = P_k_real*k_vals_real**3/(2*np.pi**2)
    axes[1,2].plot(k_vals_real, dsq_real, label="T21 HR", ls='solid')

    k_vals_gen, P_k_gen = calculate_power_spectrum(x_sequence[0,-1,:,:,:].numpy())
    dsq_gen = P_k_gen*k_vals_gen**3/(2*np.pi**2)
    axes[1,2].plot(k_vals_gen, dsq_gen, label="T21 SR", ls='dashed')

    axes[1,2].set_yscale('log')
    axes[1,2].grid()
    axes[1,2].legend()

    plt.savefig(path + "/trained_models/diffusion_model_sample_{0}.png".format(model_i))

    t = np.arange(999)[np.arange(999)%(1000//10)==0]
    tss = torch.tensor(t)
    xt, target_noise = netG.q_sample(T21[:1].repeat(len(t),1,1,1,1), tss)

    fig,axes = plt.subplots(2,10, figsize=(50,10))

    for i,(ax0,ax1) in enumerate(zip(axes[0,:], axes[1,:])):
        #ax0.imshow(target[0,:,:,target.shape[3]//2], vmin=-1, vmax=1)
        ax0.imshow(noise[0,i,:,:,noise.shape[4]//2], vmin=-1, vmax=1)
        ax0.set_title(f"t={t[-i-1]} " + "target" if i==0 else f"t={t[-i-1]}")
        ax1.imshow(pred_noises[0,i,:,:,pred_noises.shape[4]//2], vmin=-1, vmax=1)
    plt.savefig(path + "/trained_models/diffusion_model_noise_{0}.png".format(model_i))

#plt.show()
    