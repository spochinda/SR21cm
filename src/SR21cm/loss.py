import torch
from contextlib import nullcontext
import numpy as np

class VPLoss:
    def __init__(self, beta_max=20., beta_min=0.1, epsilon_t=1e-5, use_amp=False):
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.use_amp = use_amp

    def __call__(self, net, images, conditionals, labels, augment_pipe=None):
        b,c,*d = images.shape
        volume = torch.prod(torch.tensor(d))
        #t = torch.rand(size=(images.shape[0],1,1,1,1), device=images.device) * (1. - self.epsilon_t) + self.epsilon_t
        #antithetic sampling
        t = torch.rand(size=(b // 2 + b%2, 1, *len(d)*[1,]), device=images.device) * (1. - self.epsilon_t) + self.epsilon_t
        t = torch.cat([t, 1 - t + self.epsilon_t ], dim=0)[:b]
        z = torch.randn_like(images, device=images.device)
        mean, std = net.SDE.marginal_prob(x=images, t=t)
        perturbed_data = mean + std * z
        x = torch.cat([perturbed_data, *conditionals], dim = 1)
        #with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
        with torch.cuda.amp.autocast(enabled=self.use_amp) if self.use_amp else nullcontext():
            score = net.model(x=x, noise_labels=t.flatten(), class_labels=labels, augment_labels=None) # torch.randn_like(conditionals[0])
            score = -score / std #from wrapper get_score_fn
            loss = torch.square(score * std + z)
            loss = torch.sum(loss, dim=(1,2,3,4)) / volume #average loss per pixel
            loss = torch.mean(loss) * 0.5

        #check if all params are None
        



        #if torch.cuda.current_device()==0:  print(score[0,0,:5,0,0], "score\n", t.flatten(), "time\n", labels,"labels\n", std[0,0,:5,0,0],"std\n", z[0,0,:5,0,0],"z\n",loss, "loss\n")
        return loss
    

    def sigma(self, t):  #Why is sigma sign opposite of sdelib?
        t = torch.as_tensor(t)
        return ((0.5 * (self.beta_max-self.beta_min) * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
