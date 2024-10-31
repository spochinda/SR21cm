
import torch.utils

from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn as nn
from torch.nn import init

from torch_ema import ExponentialMovingAverage

from .sde_lib import VPSDE
from .samplers import Sampler




class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        network,
        network_opt,
        noise_schedule_opt = {'schedule_type': "linear", 'schedule_opt': {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}},
        loss_fn = None,
        learning_rate=1e-4,
        scheduler=False,
        mp = False,
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
        #init_weights(self.model, init_type='orthogonal')
        if mp and self.multi_gpu:
            self.model = DDP(self.model, device_ids=[rank])

        self.optG = torch.optim.Adam(self.model.parameters(), lr = learning_rate,) #weight_decay=1e-5)
        self.scheduler = scheduler
        if self.scheduler:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optG, gamma=0.998466, last_epoch=-1) #gamma=0.99954 (5000), gamma=0.999079 (2500), gamma=0.998466 (1500), gamma=0.992354 (300)
        
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.9999)
        self.loss = []
        self.loss_validation = {"loss": [1e20,], "loss_validation": [1e20,]}
        self.noise_schedule_opt = noise_schedule_opt
        self.loss_fn = loss_fn
        if self.loss_fn.use_amp:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = False
        self.sample = Sampler()
        #self.noise_schedule = noise_schedule
        #self.noise_schedule_opt = noise_schedule_opt
        #if noise_schedule is not None:
        #    self.set_new_noise_schedule(noise_schedule_opt)
        #self.set_new_noise_schedule()
        self.SDE = VPSDE(beta_min=self.noise_schedule_opt["schedule_opt"]["beta_min"],
                         beta_max=self.noise_schedule_opt["schedule_opt"]["beta_max"], 
                         timesteps=self.noise_schedule_opt["schedule_opt"]["timesteps"]
                         )

    
    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    network_opt = self.network_opt,
                    model = self.model.state_dict(), 
                    optimizer = self.optG.state_dict(),
                    scheduler = self.scheduler.state_dict() if self.scheduler is not False else False,
                    ema = self.ema.state_dict(),
                    loss = self.loss,
                    loss_validation = self.loss_validation,
                    #noise_schedule_opt = self.noise_schedule_opt),
                    noise_schedule_opt = self.noise_schedule_opt,
                    scaler=self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler!=False else False,
                    ),
                    f = path
                    )
        else:
            if True:#self.rank == 0:
                print("Saving model!", flush=True)
                torch.save(
                    obj = dict(
                        network_opt = self.network_opt,
                        model = self.model.module.state_dict(), 
                        optimizer = self.optG.state_dict(), 
                        scheduler = self.scheduler.state_dict() if self.scheduler is not False else False,
                        ema = self.ema.state_dict(),
                        loss = self.loss,
                        loss_validation = self.loss_validation,
                        #noise_schedule_opt = self.noise_schedule_opt),
                        noise_schedule_opt = self.noise_schedule_opt,
                        scaler=self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler!=False else False,
                        ),
                        f = path
                        )

    def load_network(self, path):
        if self.rank==0:
            print("Loading model!", flush=True)
        loaded_state = torch.load(path, map_location = self.device)
        self.network_opt = loaded_state['network_opt']
        self.model = self.network(**self.network_opt)
        self.model.load_state_dict(loaded_state['model'])
        if self.multi_gpu:
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank])
        self.optG.load_state_dict(loaded_state['optimizer'])
        try:
            self.scheduler.load_state_dict(loaded_state['scheduler'])
        except Exception as e:
            if self.rank==0:
                print(e, flush=True)
                print("Failed to load scheduler.", flush=True)
        try:
            self.ema.load_state_dict(loaded_state['ema'])
        except Exception as e:
            if self.rank==0:
                print(e, flush=True)
                print("Failed to load EMA.", flush=True)
        self.loss = loaded_state['loss']
        try:
            self.loss_validation = loaded_state['loss_validation']
        except:
            self.loss_validation = loaded_state['losses_validation_history']
        try:
            self.noise_schedule_opt = loaded_state['beta_schedule_opt']
        except:
            self.noise_schedule_opt = loaded_state['noise_schedule_opt'] #changed name to noise_schedule_opt
        try:
            self.scaler.load_state_dict(loaded_state['scaler'])
        except:
            self.scaler = False
            print("Failed to load scaler.", flush=True)

    
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
