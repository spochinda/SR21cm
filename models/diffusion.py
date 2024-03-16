import os
from utils import DataManager
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from math import log

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
    def __init__(self, n_channels_in, n_channels_out, kernel_size, stride, padding, dim):
        super().__init__()
        if dim == 2:
            self.conv = nn.Conv2d(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
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
            self.attn = nn.Identity() #SelfAttention
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
        #attn_res=(8),
        res_blocks=3,
        dropout = 0, 
        with_attn=False, 
        dim=2
        ):
        super().__init__()

        self.temb = TimeEmbedding(inner_channel)
        

        #ResnetBlock(n_channels_in, n_channels_out,  time_channels, norm_groups = 32, dropout = 0, with_attn=False, dim=2)

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        #now_res = image_size
        downs = [ConvolutionalLayer(n_channels_in=in_channel, n_channels_out=inner_channel, kernel_size=3, stride=1, padding=1, dim=dim)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            #use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlock(
                    n_channels_in=pre_channel, n_channels_out=channel_mult, time_channels=inner_channel, norm_groups=norm_groups, dropout=dropout, with_attn=with_attn, dim=dim))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel, dim=dim))
                feat_channels.append(pre_channel)
                #now_res = now_res//2
        self.downs = nn.ModuleList(downs)


        self.mid = nn.ModuleList([
            ResnetBlock(n_channels_in=pre_channel, n_channels_out=pre_channel, time_channels=inner_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=with_attn, dim=dim),
            ResnetBlock(n_channels_in=pre_channel, n_channels_out=pre_channel, time_channels=inner_channel, norm_groups=norm_groups, 
                        dropout=dropout, with_attn=with_attn, dim=dim)
            #ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
            #                   dropout=dropout, with_attn=True),
            #ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
            #                   dropout=dropout, with_attn=False)
            ])
        
        ups = []

        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            #use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(
                    ResnetBlock(n_channels_in=pre_channel+feat_channels.pop(), n_channels_out=channel_mult, time_channels=inner_channel, norm_groups=norm_groups, dropout=dropout, with_attn=with_attn, dim=dim)
                    #ResnetBlocWithAttn(pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,dropout=dropout, with_attn=use_attn)
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel, dim=dim))
                #now_res = now_res*2

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
                #print("HERE: ", layer)
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlock):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
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
        noise_schedule_opt=None
    ):
        super().__init__()
        self.model = model(**model_opt)
        #self.model_opt = model_opt
        self.optG = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
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
        b,(*d) = T21.shape
        t=torch.tensor(t).view(b,*[1]*len(d))
        noise = torch.randn_like(x0) if noise==None else noise
        x_t = torch.sqrt(self.alphas_cumprod[t]) * x0 + torch.sqrt(1 - self.alphas_cumprod[t]) * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, x_t, t, conditionals=None):
        b,(*d) = x_t.shape
        t=torch.tensor(t).view(b,*[1]*len(d))
        noise = self.model(x=torch.cat([x_t, *conditionals], dim=1), time=self.alphas_cumprod[t])
        x0 = self.predict_start_from_noise(x_t=x_t, t=t, noise=noise)

        
        posterior_mean = (torch.sqrt(self.alphas_cumprod_prev[t])*self.betas[t]/(1-self.alphas_cumprod[t])) * x0 + \
            (torch.sqrt(self.alphas_cumprod[t])*(1-self.alphas_cumprod_prev[t])/(1-self.alphas_cumprod[t])) * x_t #mu_tilde_t in the ddpm paper. q_posterior on github
        
        noise = torch.randn_like(x_t) #if t > 0 else torch.zeros_like(x_t)
        for i,(x,t_) in enumerate(zip(x_t,t)):
            if t_==0:
                noise[i] = torch.zeros_like(x)
            else:
                noise[i] = torch.randn_like(x)
        return posterior_mean + noise * self.posterior_variance[t]

    @torch.no_grad()
    def p_sample_loop(self, conditionals=None, continous=False):
        sample_inter = (1 | (self.timesteps//100))
        b,(*d)  = conditionals[-1].shape #select the last conditional to get the shape (should be T21_lr because order is delta,vbv,T21_lr)
        x_t = torch.randn((b,*d))
        x_sequence = x_t #use channel dimension as time axis

        for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            x_t = self.p_sample(x_t, b*[t], conditionals=conditionals)
            if t % sample_inter == 0:
                x_sequence = torch.cat([x_sequence, x_t], dim=1)
        if continous:
            return x_sequence
        else:
            return x_sequence[:,-1]
        
    def save_network(self, path):
        torch.save(
            obj = dict(model = self.model.state_dict(), optimizer = self.optG.state_dict()),# epoch = e ),
            f = path
        )

    def load_network(self, path):
        self.model.load_state_dict(torch.load(path)['model'])
        self.optG.load_state_dict(torch.load(path)['optimizer'])



import time 

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

path = os.getcwd()
Data = DataManager(path.split('/models')[0], redshifts=[10,], IC_seeds=list(range(1000,1008)))
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


reduce_dim = 4
T21 = normalize(T21)[:,:,:T21.shape[2]//reduce_dim,:T21.shape[3]//reduce_dim,:T21.shape[4]//reduce_dim] #train on reduce_dim dimensions
delta = normalize(delta)[:,:,:delta.shape[2]//reduce_dim,:delta.shape[3]//reduce_dim,:delta.shape[4]//reduce_dim]
vbv = normalize(vbv)[:,:,:vbv.shape[2]//reduce_dim,:vbv.shape[3]//reduce_dim,:vbv.shape[4]//reduce_dim]
T21_lr = torch.nn.functional.interpolate( #define low resolution input that has been downsampled and upsampled again
    torch.nn.functional.interpolate(T21, scale_factor=0.5, mode='trilinear'),
    scale_factor=2, mode='trilinear')
T21, delta, vbv, T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=12)

#create dataset for torch DataLoader
dataset = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)

#optimizer and model
model = UNet
model_opt = dict(in_channel=4, out_channel=1, inner_channel=8, norm_groups=8, channel_mults=(1, 2, 2, 4, 4), res_blocks=2, dropout = 0, with_attn=False, dim=3)
noise_schedule_opt = dict(timesteps = 1000, s = 0.008)

netG = GaussianDiffusion(
        model=UNet,
        model_opt=model_opt,
        loss_type='l1',
        noise_schedule=cosine_beta_schedule,
        noise_schedule_opt=noise_schedule_opt
    )

opt = torch.optim.Adam(netG.model.parameters(), lr = 1e-3)
loss = nn.MSELoss(reduction='mean')

model_path = "../trained_models/diffusion_model_1/model_1.pth"
if os.path.isfile(model_path):
    print("Loading checkpoint", flush=True)
    netG.load_network(model_path)
else:
    print(f"No checkpoint found at {model_path}. Starting from scratch.", flush=True)

print("Starting training", flush=True)
for e in range(650):
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    netG.model.train()
    
    losses = []
    stime = time.time()
    for i,(T21,delta,vbv, T21_lr) in enumerate(loader):
        ts = torch.randint(low = 1, high = netG.timesteps, size = (loader.batch_size, ))
        alphas_cumprod = netG.alphas_cumprod[ts]

        xt, target_noise = netG.q_sample(T21, ts)
        X = torch.cat([xt, delta, vbv, T21_lr], dim = 1)
        
        predicted_noise = netG.model(X, alphas_cumprod)
        
        loss = nn.MSELoss(reduction='mean')(target_noise, predicted_noise)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        if False: #not i % (len(loader)//2):
            print(f"Bacth {i} of {len(loader)} batches")

    netG.save_network("diffusion_model_test.pth")
    ftime = time.time()
    print("Epoch {0} trained in {1:.2f}s. Average loss {2:.2f} over {3} batches".format(e, ftime - stime, np.mean(losses), len(loader)),flush=True)

    

x_sequence = netG.p_sample_loop(conditionals=[delta,vbv,T21_lr], continous=True)

nrows = 3
ncols = 5
fig,axes = plt.subplots(nrows, ncols, figsize=(nrows*4, ncols))

rng = np.linspace(0, x_sequence.shape[1]-1, nrows*ncols, dtype=int)

for i,ax in zip(reversed(rng),axes.flatten()):
    ax.imshow(x_sequence[0,i,:,:,x_sequence.shape[4]//2], vmin=-1, vmax=1)
    ax.set_title(f"t={i}")
    ax.axis('off')

plt.savefig("plot.png")
