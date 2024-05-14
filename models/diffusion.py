
import torch.utils

from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import init

from torch_ema import ExponentialMovingAverage

from .utils import *
from .sde_lib import *
from .samplers import Sampler




class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        network,
        network_opt,
        beta_schedule_opt = {'schedule_type': "linear", 'schedule_opt': {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}},
        learning_rate=1e-4,
        rank = 0,
    ):
        super().__init__()
        
        self.rank = rank
        self.multi_gpu = torch.cuda.device_count() > 1 

        if self.multi_gpu:
            self.device = torch.device(f'cuda:{self.rank}')
        else:
            self.device = "cpu"
        
        
        self.network = network
        self.network_opt = network_opt
        self.model = self.network(**self.network_opt).to(self.device)
        init_weights(self.model, init_type='orthogonal')
        if self.multi_gpu:
            self.model = DDP(self.model, device_ids=[rank])

        
        self.optG = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.loss = []
        self.losses_validation_history = []
        self.beta_schedule_opt = beta_schedule_opt
        self.sample = Sampler()
        #self.noise_schedule = noise_schedule
        #self.noise_schedule_opt = noise_schedule_opt
        #if noise_schedule is not None:
        #    self.set_new_noise_schedule(noise_schedule_opt)
        self.set_new_noise_schedule()

    def set_new_noise_schedule(self):
        self.betas = beta_schedule(**self.beta_schedule_opt).to(self.device)

        self.timesteps = self.beta_schedule_opt["schedule_opt"]["timesteps"]
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.], dtype=torch.float32, device=self.device), self.alphas_cumprod[:-1]))
        #self.alphas_cumprod_prev = torch.tensor(np.append(1., self.alphas_cumprod[:-1]), dtype=torch.float32) #need to specify dtype because np.append with 1. is float64
        #self.betas = beta_schedule(**self.beta_schedule_opt).to(self.device) #self.noise_schedule(**self.noise_schedule_opt).to(self.device)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        if self.beta_schedule_opt["schedule_type"] == "VPSDE":
            self.SDE = VPSDE(beta_min=self.beta_schedule_opt["schedule_opt"]["beta_min"], 
                             beta_max=self.beta_schedule_opt["schedule_opt"]["beta_max"], 
                             timesteps=self.beta_schedule_opt["schedule_opt"]["timesteps"]
                             )
            
        
    def predict_start_from_noise(self, x_t, t, noise):
        b,(*d) = x_t.shape
        alpha_cumprod_t = self.alphas_cumprod[t].view(b,*[1]*len(d))
        x0 = x_t/torch.sqrt(alpha_cumprod_t) - torch.sqrt(1-alpha_cumprod_t) * noise/torch.sqrt(alpha_cumprod_t)
        return x0

    def q_sample(self, x0, t, noise=None): #forward diffusion
        #"t and batch number dim should be the same"
        b,(*d) = x0.shape
        t=torch.tensor(t).view(b,*[1]*len(d))
        noise = torch.randn_like(x0, device=self.device) if noise==None else noise
        
        if self.beta_schedule_opt["schedule_type"] != "VPSDE":
            alphas_cumprod_t = self.alphas_cumprod[t]
            x_t = torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise
            return x_t, noise
        else:
            #print("VPSDE q_sampling", flush=True)
            mean, std = self.SDE.marginal_prob(x=x0, t=t)
            x_t = mean + std * noise #x0 + noise * std[:, None, None, None]
            return x_t, noise, std

    @torch.no_grad()
    def p_sample(self, x_t, t, conditionals=None, clip_denoised=True, sampler = "DDPM SR3", ema=False):
        b,(*d) = x_t.shape
        time = t
        t=torch.tensor(b*[t]).view(b,*[1]*len(d))
        
        alpha_t, alpha_t_cumprod, alpha_t_cumprod_prev, beta_t = self.alphas[t], self.alphas_cumprod[t], self.alphas_cumprod_prev[t], self.betas[t]
        noise_level = alpha_t_cumprod
        
        if sampler=="DDIM": #
            ddim_alpha_t = self.ddim_alpha[t]
            ddim_alpha_t_sqrt = self.ddim_alpha_sqrt[t]
            ddim_alpha_t_prev = self.ddim_alpha_prev[t]
            ddim_sigma_t = self.ddim_sigma[t]
            ddim_sqrt_one_minus_alpha_t = self.ddim_sqrt_one_minus_alpha[t]
            noise_level = ddim_alpha_t



        posterior_variance_t = self.posterior_variance[t] #torch.sqrt(beta_t)
        noise = torch.randn_like(x_t) #if t > 0 else torch.zeros_like(x_t)

        if ema:
            with self.ema.average_parameters():
                pred_noise = self.model(x=torch.cat([x_t, *conditionals], dim=1), time=noise_level)
        else:
            pred_noise = self.model(x=torch.cat([x_t, *conditionals], dim=1), time=noise_level)

        if sampler=="DDPM Classic":
            x0 = None
            posterior_mean_t = (torch.sqrt(1/alpha_t)) * (x_t - beta_t/torch.sqrt(1 - alpha_t_cumprod) * pred_noise) #approach used in most papers   
            x_t = posterior_mean_t + noise * posterior_variance_t if t.item() > 0 else posterior_mean_t
            x_t = torch.clamp(x_t, -1.0, 1.0) if clip_denoised else x_t

        elif sampler=="DDPM SR3":
            x0 = self.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise) #eq in text above eq 9 rewritten for x0
            x0 = torch.clamp(x0, -1.0, 1.0) if clip_denoised else x0
            #beta_t_tilde = beta_t*(1-alpha_t_cumprod_prev)/(1-alpha_t_cumprod)
            posterior_mean_t = (torch.sqrt(alpha_t_cumprod_prev)*beta_t/(1-alpha_t_cumprod)) * x0 + \
                (torch.sqrt(alpha_t)*(1-alpha_t_cumprod_prev)/(1-alpha_t_cumprod)) * x_t #mu_tilde_t in the ddpm paper. q_posterior on github. SR3 approach
            x_t = posterior_mean_t + noise * posterior_variance_t if time > 0 else posterior_mean_t

        elif sampler=="DDIM":
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

        return x_t, noise, pred_noise, x0 #torch.sqrt(beta_t)*noise

    @torch.no_grad()
    def p_sample_loop(self, conditionals=None, n_save=10, clip_denoised=True, sampler = True, save_slices=False, ema=False, ddim_n_steps = None, verbose = True, device="cpu"):
        assert sampler in ["DDPM Classic", "DDPM SR3", "DDIM"], "sampler must be one of ['DDPM Classic', 'DDPM SR3', 'DDIM']"
        
        #print("last alpha: ", self.alphas[1000])
        t_steps = self.timesteps
        
        if sampler=="DDIM":
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
        
        interval = reversed(range(0, t_steps)) #if (sampler=="DDPM Classic") or (sampler=="DDPM SR3") else reversed(range(ddim_n_steps))
        sampling_timesteps = t_steps #if (sampler=="DDPM Classic") or (sampler=="DDPM SR3") else ddim_n_steps

        assert n_save <= sampling_timesteps, "n_save must be smaller or equal to time steps"

        for t in tqdm(interval, desc='sampling loop time step', total=sampling_timesteps, disable = not verbose):
            x_t, noise, pred_noise, x0 = self.p_sample(x_t, t, conditionals=conditionals, clip_denoised=clip_denoised, sampler=sampler, ema=ema)
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
    
        
    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    network_opt = self.network_opt,
                    model = self.model.state_dict(), 
                    optimizer = self.optG.state_dict(),# epoch = e ),
                    #ema = self.ema.state_dict(),
                    loss = self.loss,
                    losses_validation_history = self.losses_validation_history,
                    #noise_schedule_opt = self.noise_schedule_opt),
                    beta_schedule_opt = self.beta_schedule_opt),
                    f = path
                    )
        else:
            if str(self.device) == "cuda:0":
                print("Saving model!", flush=True)
                torch.save(
                    obj = dict(
                        network_opt = self.network_opt,
                        model = self.model.module.state_dict(), 
                        optimizer = self.optG.state_dict(),# epoch = e ),
                        #ema = self.ema.state_dict(),
                        loss = self.loss,
                        losses_validation_history = self.losses_validation_history,
                        #noise_schedule_opt = self.noise_schedule_opt),
                        beta_schedule_opt = self.beta_schedule_opt),
                        f = path
                        )

    def load_network(self, path):
        loaded_state = torch.load(path, map_location=self.device)
        self.network_opt = loaded_state['network_opt']
        self.model = self.network(**self.network_opt)
        self.model.load_state_dict(loaded_state['model'])
        if self.multi_gpu:
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank])
        self.optG.load_state_dict(loaded_state['optimizer'])
        self.loss = loaded_state['loss']
        self.losses_validation_history = loaded_state['losses_validation_history']
        self.beta_schedule_opt = loaded_state['beta_schedule_opt']
        self.set_new_noise_schedule()



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

#def prepare_data(path, upscale=4, cut_factor=2, redshift=10, IC_seeds=list(range(1000,1002)), device="cpu" ):
#    # Load training data
#    Data = DataManager(path, redshifts=[redshift,], IC_seeds=IC_seeds)
#    T21, delta, vbv = Data.load()
#
#    # Convert to pytorch
#    T21 = torch.from_numpy(T21).to(device)
#    delta = torch.from_numpy(delta).to(device)
#    vbv = torch.from_numpy(vbv).to(device)
#    
#    T21 = T21.permute(0,4,1,2,3) # Convert from 8,128,128,128,1 to 8,1,128,128,128
#    delta = delta.unsqueeze(1) # Expand delta and vbv dims from 8,128,128,128 to 8,1,128,128,128
#    vbv = vbv.unsqueeze(1)
#    T21_lr = torch.nn.functional.interpolate( # Define low resolution input that has been downsampled and upsampled again
#        torch.nn.functional.interpolate(T21, scale_factor=1/upscale, mode='trilinear'),
#        scale_factor=upscale, mode='trilinear')
#
#    T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
#    delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
#    vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
#    T21_lr = get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
#    
#    T21, min_max_T21 = normalize(T21)
#    delta, min_max_delta = normalize(delta)
#    vbv, min_max_vbv = normalize(vbv)
#    T21_lr, min_max_T21_lr = normalize(T21_lr)
#    
#    return T21, delta, vbv, T21_lr

#def prepare_dataloader(path, batch_size=2*4, upscale=4, cut_factor=2, redshift=10, IC_seeds=list(range(1000,1002)), device="cpu", multi_gpu=False):
#    ###START load_train_objs() and prepare_dataloader() pytorch multi-gpu tutorial###
#    
#    #model_i = "20"
#    #model_path = path + "/trained_models/diffusion_model_test_{0}.pth".format(model_i)
#    
#    T21, delta, vbv, T21_lr = prepare_data(path, upscale=upscale, cut_factor=cut_factor, redshift=redshift, IC_seeds=IC_seeds, device=device)
#    dataset = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)
#    #print("Prepare dataloader dataset shapes: ", T21.shape, delta.shape, vbv.shape, T21_lr.shape)
#    data = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = DistributedSampler(dataset) if multi_gpu else None) #4
#    ###END load_train_objs() and prepare_dataloader() pytorch multi-gpu tutorial###
#    return data


        
