#from .diffusion import *

import torch
import numpy as np
from tqdm import tqdm

class Sampler():
    def __init__(self):
        super().__init__()

    def Euler_Maruyama_sampler(self,
                               netG,
                               conditionals,  
                               num_steps=1000, 
                               #device='cuda', 
                               eps=1e-3,
                               clip_denoised=True,
                               verbose=False):

        assert netG.beta_schedule_opt["schedule_type"] == "VPSDE", "Sampler only supports VPSDE schedule type."
        
        netG.model.eval()
        b,(*d)  = conditionals[-1].shape #select the last conditional to get the shape (should be T21_lr because order is delta,vbv,T21_lr)

        #batch_size = conditionals[0].shape[0]
        #t = torch.ones(b, )
        t = torch.tensor(b*[1.], device=conditionals[-1].device).view(b,*[1]*len(d))
        #shape = conditionals[0].shape[2:]
        noise = torch.randn(b, *d, device=conditionals[-1].device)
        mean, std = netG.SDE.marginal_prob(noise, t)
        #print(mean.shape, std.shape, noise.shape)
        init_x = noise * std
        time_steps = torch.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        x_sequence = [x]
        with torch.no_grad():
            for time_step in tqdm(time_steps, desc='sampling loop time step', disable = not verbose):      
                batch_time_step = torch.tensor(b*[time_step]).view(b,*[1]*len(d)) #torch.ones(batch_size) * time_step
                drift, g = netG.SDE.sde(x=x,t=batch_time_step)
                #print(drift.shape, g.shape, x.shape)
                X = torch.cat([x, *conditionals], dim=1)
                mean_x = x + (g**2) * netG.model(X, batch_time_step) * step_size
                x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x) #+ 0.5 * g**2 * x
                x_sequence.append(x)
            x_sequence.append(mean_x)
            x_sequence = torch.cat(x_sequence, dim=1)
        # Do not include any noise in the last sampling step.
        if clip_denoised:
            x_sequence[:,-1] = x_sequence[:,-1].clamp_(-1,1)
        return x_sequence
    
    def ddim(self, 
             netG,
             conditionals, 
             num_steps=100, 
             clip_denoised=True, 
             verbose=False):
        
        #only supports "linear" or "cosine" schedule
        assert netG.beta_schedule_opt["schedule_type"] in ["linear", "cosine"], "Sampler only supports linear or cosine schedule type."

        self.ddim_timesteps = torch.linspace(0, 1+netG.timesteps-(netG.timesteps//(num_steps)), num_steps, dtype=torch.int) #improved denoising diffusion probabilistic models 
        self.ddim_alpha = netG.alphas_cumprod[self.ddim_timesteps]
        self.ddim_alpha_prev = torch.cat([netG.alphas_cumprod[0:1], netG.alphas_cumprod[self.ddim_timesteps[:-1]]])
        
        t_steps = len(self.ddim_timesteps)    
        ddim_eta = 0.
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_sigma = (ddim_eta *
                    ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                    (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
        self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5 
        self.temperature = 1.

        
        netG.model.eval()

        b,(*d)  = conditionals[-1].shape #select the last conditional to get the shape (should be T21_lr because order is delta,vbv,T21_lr)
        x_t = torch.randn((b,*d), device=conditionals[-1].device)
        x_sequence = [x_t] #use channel dimension as time axis
        sampling_timesteps = len(self.ddim_timesteps)
        with torch.no_grad():
            for t in tqdm(reversed(range(0, sampling_timesteps)), desc='sampling loop time step', total=sampling_timesteps, disable = not verbose):
                #x_t, noise, pred_noise, x0 = self.p_sample
                t=torch.tensor(b*[t]).view(b,*[1]*len(d))

                ddim_alpha_t = self.ddim_alpha[t]
                ddim_alpha_t_sqrt = self.ddim_alpha_sqrt[t]
                ddim_alpha_t_prev = self.ddim_alpha_prev[t]
                ddim_sigma_t = self.ddim_sigma[t]
                ddim_sqrt_one_minus_alpha_t = self.ddim_sqrt_one_minus_alpha[t]
                noise_level = ddim_alpha_t
                posterior_variance_t = netG.posterior_variance[t] #torch.sqrt(beta_t)
                noise = torch.randn_like(x_t)
                pred_noise = netG.model(x=torch.cat([x_t, *conditionals], dim=1), time=noise_level)
                x0 = (x_t - ddim_sqrt_one_minus_alpha_t * pred_noise) / (ddim_alpha_t ** 0.5)
                #x0 = x0.clamp(-1., 1.) if clip_denoised else x0
                dir_xt = (1. - ddim_alpha_t_prev - ddim_sigma_t ** 2).sqrt() * pred_noise
                    
                for i,(x,t_) in enumerate(zip(x_t,t)):
                    if t_==0:
                        noise[i] = torch.zeros_like(x)
                    else:
                        noise[i] = torch.randn_like(x)
                
                noise = noise*self.temperature
                x_t = (ddim_alpha_t_prev ** 0.5) * x0 + dir_xt + ddim_sigma_t * noise #x_t-1

                
                x_sequence.append(x_t)

        x_sequence = torch.cat(x_sequence, dim=1)
        
        if clip_denoised:
            x_sequence[:,-1] = x_sequence[:,-1].clamp_(-1,1)

        return x_sequence
    
    @torch.no_grad()
    def ddim_v2(self, 
                netG,
                conditionals, 
                num_steps=100, 
                clip_denoised=True, 
                verbose=False):
        netG.model.eval()

        a = netG.timesteps // num_steps
        time_steps = np.asarray(list(range(0, netG.timesteps, a)))
        time_steps = time_steps + 1
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        


        b,(*d)  = conditionals[-1].shape

        x_t = torch.randn((b,*d), device=conditionals[-1].device)
        x = [x_t]

        for i in tqdm(reversed(range(0, num_steps)), desc='sampling loop time step', total=num_steps, disable = not verbose):
            #x_t = self.sample_one_step(x_t, time_steps[t], time_steps_prev[i], eta)
            
            t = time_steps[i] * torch.ones(b, *len(d)*[1], device=conditionals[-1].device, dtype=torch.int)
            prev_t = time_steps_prev[i] * torch.ones(b, *len(d)*[1], device=conditionals[-1].device, dtype=torch.int)
            #print("time: ", t, prev_t, flush=True)

            # get current and previous alpha_cumprod

            alpha_t = netG.alphas_cumprod[t]
            alpha_t_prev = netG.alphas_cumprod[prev_t]
            #print("alpha_t: ", alpha_t, alpha_t_prev, flush=True)

            # predict noise using model
            epsilon_theta_t = netG.model(x=torch.cat([x_t, *conditionals], dim=1), time=alpha_t)

            # calculate x_{t-1}
            eta = 0.0
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            epsilon_t = torch.randn_like(x_t, device=conditionals[-1].device)
            x_t = (
                    torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                    (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                        (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                    sigma_t * epsilon_t
            )
            x.append(x_t)
        
        x = torch.cat(x, dim=1)
        
        if clip_denoised:
            x[:,-1] = x[:,-1].clamp_(-1,1)
        
        return x

        
    
    def ddpm_classic(self,
                     netG,
                     conditionals, 
                     n_save = 10, 
                     clip_denoised=True, 
                     verbose=False):
        #only supports "linear" or "cosine" schedule
        assert netG.beta_schedule_opt["schedule_type"] in ["linear", "cosine"], "Sampler only supports linear or cosine schedule type."

        netG.model.eval()
        sample_inter = netG.timesteps//n_save if n_save <= netG.timesteps else 1

        b,(*d)  = conditionals[-1].shape
        
        x_t = torch.randn((b,*d))
        x_sequence = [x_t] #use channel dimension as time axis

        assert n_save <= netG.timesteps, "n_save must be smaller or equal to time steps"
        with torch.no_grad():
            for t in tqdm(reversed(range(0, netG.timesteps)), desc='sampling loop time step', total=netG.timesteps, disable = not verbose):
                t=torch.tensor(b*[t]).view(b,*[1]*len(d))
                alpha_t, alpha_t_cumprod, alpha_t_cumprod_prev, beta_t = netG.alphas[t], netG.alphas_cumprod[t], netG.alphas_cumprod_prev[t], netG.betas[t]
                noise_level = alpha_t_cumprod
                posterior_variance_t = netG.posterior_variance[t] #torch.sqrt(beta_t)
                noise = torch.randn_like(x_t) #if t > 0 else torch.zeros_like(x_t)
                pred_noise = netG.model(x=torch.cat([x_t, *conditionals], dim=1), time=noise_level)
                posterior_mean_t = (torch.sqrt(1/alpha_t)) * (x_t - beta_t/torch.sqrt(1 - alpha_t_cumprod) * pred_noise) #approach used in most papers   
                
                for i,(x,t_) in enumerate(zip(x_t,t)):
                    if t_==0:
                        noise[i] = torch.zeros_like(x)
                    else:
                        noise[i] = torch.randn_like(x)
                
                x_t = posterior_mean_t + noise * posterior_variance_t if t.item() > 0 else posterior_mean_t
                x_t = torch.clamp(x_t, -1.0, 1.0) if clip_denoised else x_t
                
                if t % sample_inter == 0:
                    x_sequence.append(x_t)
                    
        x_sequence = torch.cat(x_sequence, dim=1)
        
        if clip_denoised:
            x_sequence[:,-1] = x_sequence[:,-1].clamp_(-1,1)

        return x_sequence
    
    def ddpm_sr3(self, 
                 netG,
                 conditionals, 
                 n_save = 10, 
                 clip_denoised=True, 
                 verbose=False):
        #only supports "linear" or "cosine" schedule
        assert netG.beta_schedule_opt["schedule_type"] in ["linear", "cosine"], "Sampler only supports linear or cosine schedule type."

        netG.model.eval()
        sample_inter = netG.timesteps//n_save if n_save <= netG.timesteps else 1

        b,(*d)  = conditionals[-1].shape
        
        x_t = torch.randn((b,*d))
        x_sequence = [x_t] #use channel dimension as time axis

        assert n_save <= netG.timesteps, "n_save must be smaller or equal to time steps"
        with torch.no_grad():
            for t in tqdm(reversed(range(0, netG.timesteps)), desc='sampling loop time step', total=netG.timesteps, disable = not verbose):
                t=torch.tensor(b*[t]).view(b,*[1]*len(d))
                alpha_t, alpha_t_cumprod, alpha_t_cumprod_prev, beta_t = netG.alphas[t], netG.alphas_cumprod[t], netG.alphas_cumprod_prev[t], netG.betas[t]
                noise_level = alpha_t_cumprod
                posterior_variance_t = netG.posterior_variance[t] #torch.sqrt(beta_t)
                noise = torch.randn_like(x_t) #if t > 0 else torch.zeros_like(x_t)
                pred_noise = netG.model(x=torch.cat([x_t, *conditionals], dim=1), time=noise_level)
                
                x0 = netG.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise) #eq in text above eq 9 rewritten for x0
                x0 = torch.clamp(x0, -1.0, 1.0) if clip_denoised else x0
                #beta_t_tilde = beta_t*(1-alpha_t_cumprod_prev)/(1-alpha_t_cumprod)
                posterior_mean_t = (torch.sqrt(alpha_t_cumprod_prev)*beta_t/(1-alpha_t_cumprod)) * x0 + \
                    (torch.sqrt(alpha_t)*(1-alpha_t_cumprod_prev)/(1-alpha_t_cumprod)) * x_t #mu_tilde_t in the ddpm paper. q_posterior on github. SR3 approach
                
                for i,(x,t_) in enumerate(zip(x_t,t)):
                    if t_==0:
                        noise[i] = torch.zeros_like(x)
                    else:
                        noise[i] = torch.randn_like(x)

                x_t = posterior_mean_t + noise * posterior_variance_t #if time > 0 else posterior_mean_t

                if t % sample_inter == 0:
                    x_sequence.append(x_t)
            
        x_sequence = torch.cat(x_sequence, dim=1)

        if clip_denoised:
            x_sequence[:,-1] = x_sequence[:,-1].clamp_(-1,1)

        return x_sequence