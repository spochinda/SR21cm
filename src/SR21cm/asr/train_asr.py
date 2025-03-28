import contextlib
import torch
import torch.distributed
import torch.utils
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

#from .utils import CustomDataset, get_subcubes, normalize, invert_normalization, augment_dataset, calculate_power_spectrum, sample_model_v3, initialize_model_directory, data_preprocess, get_paths
#from .plotting import plot_sigmas, plot_input, plot_hist
#from .diffusion import GaussianDiffusion
from SR21cm.utils import get_subcubes
from SR21cm.asr import model_asr
from SR21cm.asr.utils_asr import CustomDataset, data_preprocess

import os
import time
import psutil  # Import psutil
#from contextlib import nullcontext
from tqdm import tqdm
from datetime import datetime


def ddp_setup(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] #check if master address exists
        print("Found master address: ", os.environ["MASTER_ADDR"])
    except:
        print("Did not find master address variable. Setting manually...")
        os.environ["MASTER_ADDR"] = "localhost"

    os.environ["MASTER_PORT"] = "2594"#"12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="gloo", rank=rank, world_size=world_size) #backend gloo for cpus? nccl for gpus



def train_step(model, train_dataloader, config, device="cpu", multi_gpu = False,):
    """
    Train the model
    """
    
    
    

    for i,(T21_, delta_, vbv_, labels) in enumerate(train_dataloader):
        model.train()
        
        T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std, scale = data_preprocess(T21=T21_, delta=delta_, vbv=vbv_, **config["data_preprocess"])
        b,c,*d = T21.shape
        xyz_hr = model_asr.make_coord(d, ranges=None, flatten=False)
        xyz_hr = xyz_hr.view(1, -1, 3)
        xyz_hr = xyz_hr.repeat(b, 1, 1)
        
        model.optG.zero_grad()
        T21_pred = model(img_lr=T21_lr, xyz_hr=xyz_hr) * T21_lr_std + T21_lr_mean
        T21 = T21 * T21_lr_std + T21_lr_mean
        loss = torch.mean((T21_pred-T21)**2)
        loss.backward()
        model.optG.step()
            
        
    
    if multi_gpu:
        torch.distributed.all_reduce(tensor=loss, op=torch.distributed.ReduceOp.AVG) #total loss=sum(average total batch loss per gpu)
    
    loss = loss.detach().cpu().item()**0.5

    model.loss.append(loss)

    return loss, labels, scale



def train(rank, 
          world_size=0, 
          config = None,
          **kwargs):
    
    

    torch.backends.cudnn.benchmark = True #find fastest algorithms

    multi_gpu = world_size > 1

    if multi_gpu:
        device = torch.device(f'cuda:{rank}')
        ddp_setup(rank, world_size=world_size)
    elif world_size==1 and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    #optimizer and model
    encoder = getattr(model_asr, config["encoder"])(**config["encoder_opt"])
    model = getattr(model_asr, config["network"])(**config["network_opt"], encoder=encoder)
    model.optG = getattr(torch.optim, config["optimizer"])(model.parameters(), **config["optimizer_opt"])
    model.multi_gpu = multi_gpu
    #generate seed to share across GPUs
    if multi_gpu:
        seed = torch.randint(0, 1420, (1,)) if config["seed"] is None else torch.tensor(config["seed"])
        torch.distributed.barrier()
        torch.distributed.broadcast(tensor=seed, src=0)
        seed = seed.item()

    
    train_data_module = CustomDataset(**config["datasets"]["train"], device=device)
    train_sampler = DistributedSampler(dataset=train_data_module, shuffle=True, seed=seed) if multi_gpu else None
    train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=train_data_module.batch_size, shuffle=(train_sampler is None), sampler = train_sampler)

    if rank==0:
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print(f"[{str(device)}] Starting training at {current_time}...", flush=True)
    start_time_training = time.time()

    
    for e in range(config["total_epochs"]): #if False:#       
        if device.index==0 or device.type=="cpu":
            start_time = time.time()

        loss, labels, scale = train_step(model=model, train_dataloader=train_dataloader, config=config, device=device, multi_gpu = multi_gpu)
        if device.index==0 or device.type=="cpu":
            print("[{0}]: Iteration in {1:.2f}s | ".format(device.type, time.time()-start_time) +
                "loss: {0:.4f}, ".format(loss) +
                "redshift: {0}".format(labels[0].item() if labels is not None else "None"),
                "scale: {0:.2f}".format(scale),
                flush=True)

        #every 100 epochs after epoch 2000 save network
        if len(model.loss) >= 1000 and len(model.loss)%100==0:
            if len(model.loss) == 1000:
                for g in model.optG.param_groups:
                            g['lr'] = 1e-4
            model_dir = os.path.join(config["path"], config["name"])
            model.save_network(path = model_dir)
            if rank==0:
                print(f"[{device}] Saved network at {model_dir}", flush=True)
        if (time.time()-start_time_training > 3*60*60):
            if rank==0:
                print("3 hours passed. Aborting training...", flush=True)
            break

    if multi_gpu:#world_size > 1:
        torch.distributed.barrier()
        destroy_process_group()
###END main pytorch multi-gpu tutorial###
        
