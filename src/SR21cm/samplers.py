#from .diffusion import *

import torch
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

from scipy import integrate #for ode solver

class Sampler():
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def Euler_Maruyama_sampler(self,
                               netG,
                               x_lr = None,
                               conditionals=None,
                               class_labels=None,  
                               num_steps=1000, 
                               #device='cuda', 
                               eps=1e-3,
                               use_amp=False,
                               clip_denoised=True,
                               verbose=False):

        assert netG.noise_schedule_opt["schedule_type"] == "VPSDE", "Sampler only supports VPSDE schedule type."
        
        netG.model.eval()
        b,*d  = conditionals[-1].shape #select the last conditional to get the shape (order is delta,vbv)

        #batch_size = conditionals[0].shape[0]
        #t = torch.ones(b, )
        t = torch.tensor(b*[1.], device=x_lr.device).view(b,*[1]*len(d))
        #shape = conditionals[0].shape[2:]
        noise = torch.randn(b, *d, device=x_lr.device)
        mean, std = netG.SDE.marginal_prob(noise, t)

        time_steps = torch.linspace(1., eps, num_steps, device=x_lr.device)
        step_size = time_steps[0] - time_steps[1]

        x = noise #* std
        #x_sequence on CPU
        x_sequence = torch.empty_like(x, device='cpu')
        #[x.detach().cpu()]
        
        
        #x_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(x_lr) #only for SongUNet
                
        for i, time_step in enumerate(tqdm(time_steps, desc='sampling loop time step', disable=not verbose)):
            
            batch_time_step = torch.tensor(b*[time_step], device=x_lr.device).view(b,*[1]*len(d)) #torch.ones(batch_size) * time_step
            #f, g = netG.SDE.sde(x=x,t=batch_time_step) #drift=f diffusion=g
            
            #mean, noise_labels = netG.SDE.marginal_prob(x=x, t=batch_time_step)
            #noise_labels = batch_time_step.flatten() #noise_labels.flatten()
            X = torch.cat([x, *conditionals, x_lr], dim = 1)

            #try:
            #if str(x.device) == "cuda:0":
            #    print(torch.cuda.memory_summary())
            #with torch.cuda.amp.autocast():
            #if use_amp
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                score = netG.model(x=X, noise_labels=batch_time_step.flatten(), class_labels=class_labels, augment_labels=None) #999 from wrapper get_score_fn
            std = netG.SDE.marginal_prob(x=torch.zeros_like(x), t=batch_time_step)[1] #from wrapper get_score_fn
            score = -score / std #from wrapper get_score_fn
            #except Exception as e:
            #    print(e)
            #    score = netG.model(x=x, time=batch_time_step, x_lr=x_lr, conditionals=conditionals) #old UNet model (maybe time is sigma from marginalprob?)    
            
            
            f, g = netG.SDE.rsde(x=x, t=batch_time_step, score=score, probability_flow=False)
            
            x_mean = x - f * step_size #double check signs
            x = x_mean + g * torch.sqrt(step_size) * torch.randn_like(x) #double check signs
            
            #x = x - f * step_size + g * torch.sqrt(step_size) * torch.randn_like(x) #double check signs

            x_sequence = torch.cat([x_sequence, x.detach().cpu()], dim=1)
            #x_sequence.append(x.detach().cpu())
            if False:#torch.cuda.current_device() == 0:
                #memory summary
                #print(torch.cuda.memory_summary())
                #peak memory in MiB
                print(f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB", flush=True)
                print(f"Iteration {i} out of {num_steps}. {100*i/num_steps:.1f}% completed.", flush=True)

            del batch_time_step, X, score, std, f, g, x_mean
            torch.cuda.empty_cache()
        
        #x_sequence.append(x_mean) # Do not include any noise in the last sampling step.
        #x_sequence = torch.cat(x_sequence, dim=1)
        
        #if clip_denoised:
            #x_sequence[:,-1] = x_sequence[:,-1].clamp_(-1,1)
        
        return x_sequence
    
    @torch.no_grad()
    def ode_sampler(self, 
                    netG,
                    x_lr,
                    conditionals,
                    class_labels=None,
                    atol=1e-5,
                    rtol=1e-5,
                    #device='cuda', 
                    eps=1e-3):
        """Generate samples from score-based models with black-box ODE solvers.

        Args:
            score_model: A PyTorch model that represents the time-dependent score-based model.
            marginal_prob_std: A function that returns the standard deviation 
            of the perturbation kernel.
            diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
            batch_size: The number of samplers to generate by calling this function once.
            atol: Tolerance of absolute errors.
            rtol: Tolerance of relative errors.
            device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
            z: The latent code that governs the final sample. If None, we start from p_1;
            otherwise, we start from the given z.
            eps: The smallest time step for numerical stability.
        """
        assert netG.noise_schedule_opt["schedule_type"] == "VPSDE", "Sampler only supports VPSDE schedule type."
        netG.model.eval()
        #x_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(x_lr)

        b,*d  = x_lr.shape #select the last conditional to get the shape (order is delta,vbv)
        batch_time_step = torch.ones(size=(b,*[1]*len(d)),dtype=torch.float32, device=x_lr.device)
        #torch.tensor(b*[1.], device=conditionals[-1].device).view(b,*[1]*len(d))

        noise = torch.randn(b, *d, device=conditionals[-1].device)
        #mean, std = netG.SDE.marginal_prob(noise, batch_time_step)
        # Create the latent code
        init_x = noise # * std # no std because VPSDE std at t=1 is 1
        shape = init_x.shape
        
        def ode_func(batch_time_step, x):        
            """The ODE function for use by the ODE solver."""
            x = torch.tensor(x.reshape(shape), dtype=torch.float32, device=x_lr.device)
            
            batch_time_step = torch.ones(size=(b,*[1]*len(d)),dtype=torch.float32, device=x_lr.device) * batch_time_step

            #mean, noise_labels = netG.SDE.marginal_prob(x=x, t=batch_time_step)
            #x, noise_labels = netG.SDE.marginal_prob(x=x, t=batch_time_step) ########???
            noise_labels = batch_time_step.flatten() #noise_labels.flatten()
            X = torch.cat([x, *conditionals, x_lr], dim = 1)
            
            try:
                score = netG.model(x=X, noise_labels=noise_labels, class_labels=class_labels, augment_labels=None)
            except:
                score = netG.model(x=x, time=batch_time_step, x_lr=x_lr, conditionals=conditionals) #old UNet model (maybe time is sigma from marginalprob?)
            std = netG.SDE.marginal_prob(torch.zeros_like(x), noise_labels)[1] #from wrapper get_score_fn
            score = -score / std #from wrapper get_score_fn
            
            drift = netG.SDE.rsde(x=x, t=batch_time_step, score=score, probability_flow=True)[0]
            #drift, g = netG.SDE.sde(x=x,t=batch_time_step)
            #g = g.cpu().numpy()
            drift = drift.detach().cpu().numpy()
            #score = score.cpu().numpy()
            return  drift.reshape((-1,)).astype(np.float32) #(drift - 0.5 * (g**2) * score).reshape((-1,)).astype(np.float32) #score
        
        # Run the black-box ODE solver.
        res = integrate.solve_ivp(fun=ode_func, t_span=(1., eps), y0=init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
        print(f"Number of function evaluations: {res.nfev}")
        
        x = torch.tensor(res.y[:, -1].reshape(shape), dtype=torch.float32)
        return x
