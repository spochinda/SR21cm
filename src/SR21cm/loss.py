import torch
from contextlib import nullcontext

class VPLoss:
    def __init__(self, beta_max=20., beta_min=0.1, epsilon_t=1e-5, use_amp=False):
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.use_amp = use_amp

    def __call__(self, net, images, conditionals, labels, augment_pipe=None):
        b,c,*d = images.shape
        #t = torch.rand(size=(images.shape[0],1,1,1,1), device=images.device) * (1. - self.epsilon_t) + self.epsilon_t
        #antithetic sampling
        t = torch.rand(size=(b // 2 + b%2, 1, *len(d)*[1,]), device=images.device) * (1. - self.epsilon_t) + self.epsilon_t
        t = torch.cat([t, 1 - t + self.epsilon_t ], dim=0)[:b]
        z = torch.randn_like(images, device=images.device)
        mean, std = net.SDE.marginal_prob(x=images, t=t)
        perturbed_data = mean + std * z
        x = torch.cat([perturbed_data, *conditionals], dim = 1)
        with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
            score = net.model(x=x, noise_labels=t.flatten(), class_labels=labels, augment_labels=None) # 999 from wrapper get_score_fn
        score = -score / std #from wrapper get_score_fn
        
        loss = torch.square(score * std + z)
        loss = torch.sum(loss, dim=(1,2,3,4))
        loss = torch.mean(loss) * 0.5
        return loss
    

    def sigma(self, t):  #Why is sigma sign opposite of sdelib?
        t = torch.as_tensor(t)
        return ((0.5 * (self.beta_max-self.beta_min) * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditionals, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        y = y + n
        #D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        D_yn = net.edmrecond_forward(x=y, conditionals=conditionals, sigma=sigma, class_labels=labels) #vprecond
        loss = weight * ((D_yn - y) ** 2)
        loss = torch.mean(loss)
        return loss
    
def loss_fn(netG, batch_size, x_true, x_lr = None, conditionals = None):
    #Old loss function 

    if netG.noise_schedule_opt["schedule_type"] != "VPSDE":
        ts = torch.randint(low = 0, high = netG.timesteps, size = (batch_size // 2 + 1, ), device=x_true.device)
        ts = torch.cat([ts, netG.timesteps - ts - 1], dim=0)[:batch_size] # antithetic sampling
        
        alphas_cumprod = netG.alphas_cumprod[ts]     
        
        xt, target_noise = netG.q_sample(x_true, ts)

        #X = torch.cat([xt, *conditionals], dim = 1)
        model_pred = netG.model(x=xt, time=alphas_cumprod, x_lr=x_lr, conditionals=conditionals)
        loss = torch.nn.MSELoss(reduction='mean')(target_noise, model_pred) # loss per x_true
    
    else:
        b, *d = x_true.shape
        eps = 1e-5
        ts = torch.rand(b, device=x_true.device) * (1. - eps) + eps  
        xt, target_noise, std = netG.q_sample(x0=x_true, t=ts, noise=None)
        
        #drop_cond = 0#.28 #Xiaosheng Zhao
        #if drop_cond > 0:
        #    print("drop_cond: ", drop_cond, flush=True)
        #    rng = torch.rand((b,*[1]*len(d)), device=x_true.device) > drop_cond
        #    conditionals = [c * rng for c in conditionals]
        #    x_lr = x_lr * rng

        if False:
            score = netG.model(x=xt, time=ts, x_lr=x_lr, conditionals=conditionals)
        else:
            #mean, noise_labels = netG.SDE.marginal_prob(xt, ts)
            noise_labels = std.flatten()
            class_labels_size = torch.tensor([conditionals[-1].shape[-1]],dtype=torch.float32)
            x_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(x_lr)
            xt = torch.cat([xt, *conditionals, x_lr], dim = 1)
            #score = netG.model(x=xt, noise_labels=noise_labels, class_labels=None, augment_labels=None)
            #score = netG.model(x=xt, noise_labels=ts, class_labels=None, augment_labels=None)
            score = netG.model(x=xt, noise_labels=ts, class_labels=class_labels_size, augment_labels=None)

        loss = torch.mean(torch.square(score  * std + target_noise)) # loss per x_true (weighting=lambda_t=sigma_t**2)
    
    return loss #