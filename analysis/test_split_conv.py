import os 
import torch
from models.model_edm import SongUNet
from models.diffusion import GaussianDiffusion

import torch.distributed
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

def ddp_setup(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] #check if master address exists
        print("Found master address: ", os.environ["MASTER_ADDR"])
    except:
        print("Did not find master address variable. Setting manually...")
        os.environ["MASTER_ADDR"] = "localhost"

    
    os.environ["MASTER_PORT"] = "2595"#"12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #backend gloo for cpus?


def main(rank, world_size=0,mp=True):
    multi_gpu = world_size > 1

    print("Multi-GPU: ", multi_gpu)
    #test 512 boxes forward pass through model to see if we get OOM
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    #for i in range(torch.cuda.device_count()):
    #    print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)

    if mp:
        ddp_setup(rank, world_size=world_size)
    #torch.distributed.barrier()

    
    #path = os.getcwd().split("/21cmGen")[0] + "/21cmGen"
    
    network = SongUNet

    network_opt = dict(img_resolution=128, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
                model_channels=8, channel_mult=[1,2,4,8,16], attn_resolutions=[8,], mid_attn=True, #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                )
    noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    #8 channels: mult: 1,2,8,32,128 attn_res: 8
    #8 channels: mult: 1,2,4,8,16 attn_res: 8
    #16 channels: mult: 1,2,4,32,128 attn_res: [] failed even without midattn
    #16 channels: mult: 1,2,4,8,16 attn_res: [] failed even without midattn
    loss_fn = None #VPLoss(beta_max=20., beta_min=0.1, epsilon_t=1e-5)

    netG = GaussianDiffusion(
            network=network,
            network_opt=network_opt,
            noise_schedule_opt=noise_schedule_opt,
            loss_fn = loss_fn,
            learning_rate=1e-4,
            scheduler=False,
            mp=mp,
            rank=rank,
        )

    #model_path = path + "/trained_models/model_4/DDPMpp_standard_channels_32_tts_70_VPSDE_1_normfactor1.pth"
    #netG.model_name = model_path.split("/")[-1].split(".pth")[0]
    #netG.load_network(model_path)

    pytorch_total_params = sum(p.numel() for p in netG.model.parameters() if p.requires_grad)
    print(f"{pytorch_total_params:,} trainable parameters", flush=True)
    netG.model.eval()
    with torch.no_grad():
        x = torch.randn(1,4,512,512,512, device=device).detach()
        b,c,h,w,d = x.shape
        noise_labels = torch.rand(size=(b,1,1,1,1), device=device)
    
        score = netG.model(x=x, noise_labels=noise_labels.flatten(), class_labels=None, augment_labels=None)
    
    if torch.cuda.current_device()==0:
        print(f"[{str(device)}] Completed forward pass through model...", flush=True)
    
    if mp:
        torch.distributed.barrier()
        destroy_process_group()

if __name__ == "__main__":
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
    if torch.cuda.current_device()==0:
        for i in range(torch.cuda.device_count()):
            print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)    

    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1
    
    mp_=False#True
    if mp_:
        mp.spawn(main, args=(world_size,mp_), nprocs=world_size)
    else:
        main(torch.cuda.current_device(),world_size,mp_)
