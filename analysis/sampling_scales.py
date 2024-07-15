import os
import matplotlib.pyplot as plt
import torch
import torch.distributed
from models.diffusion import GaussianDiffusion
from models.utils import CustomDataset
from models.model_edm import SongUNet
from models.train import validation_step_v2

import torch.multiprocessing as mp
import torch.utils
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import time

def ddp_setup(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] #check if master address exists
        print("Found master address: ", os.environ["MASTER_ADDR"])
    except:
        print("Did not find master address variable. Setting manually...")
        os.environ["MASTER_ADDR"] = "localhost"

    
    os.environ["MASTER_PORT"] = "2596"#"12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #backend gloo for cpus?

def sample_scales(rank, world_size, cut_factor=2, sub_batch=1, num_steps=100, model_path=""):
    multi_gpu = world_size > 1

    if multi_gpu:
        device = torch.device(f'cuda:{rank}')
        print("Multi GPU: {0}, device: {1}".format(multi_gpu,device))
        ddp_setup(rank, world_size=world_size)
    else:
        device = "cpu"
        print("Multi GPU: {0}, device: {1}".format(multi_gpu,device))
    
    path = os.getcwd().split("/21cmGen")[0] + "/21cmGen"


    channels = int(model_path.split("channels_")[1].split("_")[0])

    
    network_opt = dict(img_resolution=64, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
                model_channels=channels, channel_mult=[2,2,2], attn_resolutions=[], #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                )

    #network = UNet
    network = SongUNet
    
    #noise_schedule_opt = {'schedule_type': "linear", 'schedule_opt': {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}} 
    #noise_schedule_opt = {'schedule_type': "cosine", 'schedule_opt': {"timesteps": 1000, "s" : 0.008}} 
    #noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    
    loss_fn = None #VPLoss(beta_max=20., beta_min=0.1, epsilon_t=1e-5)
    
    netG = GaussianDiffusion(
            network=network,
            network_opt=network_opt,
            noise_schedule_opt=noise_schedule_opt,
            loss_fn = loss_fn,
            learning_rate=1e-4,
            scheduler=False,
            rank=rank,
        )
    
    netG.model_name = model_path.split("/")[-1].split(".pth")[0]
    netG.load_network(model_path)




    test_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/IC_cubes/", 
    #test_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
                                    redshifts=[10,], IC_seeds=list(range(72,80)), upscale=4, cut_factor=0, transform=False, norm_lr=True, device=device)
    test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=1, shuffle=False if multi_gpu else True,
                                                    sampler = DistributedSampler(test_data_module) if multi_gpu else None)
    
    loss_validation, tensor_dict = validation_step_v2(netG=netG, validation_dataloader=test_dataloader, cut_factor=cut_factor, norm_factor=1., augment=False, split_batch = True, sub_batch = sub_batch, one_box_validation=False, num_steps=num_steps, device=device, multi_gpu=multi_gpu)
    
    if rank==0:
        print(f"RMSE={loss_validation**0.5:.3f}, cut_factor={cut_factor}, num_steps={num_steps}", flush=True)
        torch.save(obj=tensor_dict,
                    f=path + f"/analysis/model_3/save_data_tensors_{netG.model_name}_scale_{cut_factor}.pth")
        # #tensors_cut_{cut_factor}_model_{netG.model_name}.pth
        
    if multi_gpu:
        torch.distributed.barrier()
        destroy_process_group()


if __name__ == "__main__":
    
    
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
   
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    path = os.getcwd().split("/21cmGen")[0] + "/21cmGen"
    model = "DDPMpp_standard_channels_32_tts_70_VPSDE_3_normfactor1.pth"

    cut_factors = [2,1,0]
    sub_batches = [4,1,1]

    if multi_gpu:
        print("Using multi_gpu", flush=True)
        for i in range(torch.cuda.device_count()):
            print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)

        for cut_factor,sub_batch in zip(cut_factors,sub_batches):
            sample_time = time.time()
            mp.spawn(
                sample_scales, 
                args=(world_size,
                    cut_factor, #1
                    sub_batch, #1
                    100, #steps
                    path+f"/trained_models/model_3/{model}",
                    ), 
                nprocs=world_size
                )
            print(f"Sample time: {time.time()-sample_time:.3f}s", flush=True)
        
    else:
        print("Not using multi_gpu",flush=True)

        for cut_factor,sub_batch in zip(cut_factors,sub_batches):
            sample_scales(rank=0, world_size=0, cut_factor=cut_factor, sub_batch=sub_batch, model_path=path+f"/trained_models/model_3/{model}",)
