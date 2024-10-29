import torch
import torch.distributed
import torch.utils
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import os
import sys
import time
from datetime import datetime, timedelta
import argparse

#sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from SR21cm.train import sample, train
from SR21cm.utils import sample_scales
from SR21cm.plotting import plot_scales

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


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the optional arguments
    parser.add_argument("--channels", type=int, default=8, help="channels")
    parser.add_argument("--nmodels", type=int, default=1, help="nmodels")
    parser.add_argument("--id", type=int, default=8, help="id") #4

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the optional arguments
    channels = args.channels
    nmodels = args.nmodels
    id = args.id

    
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
   
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    

    if multi_gpu:
        print("Using multi_gpu", flush=True)
        for i in range(torch.cuda.device_count()):
            print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)

        for channel,n_models,id in zip([channels,], [nmodels,], [id,]): #[8,8,4], [56,28,28], [3,2,3,3]
            print(f"Training with {channel} channels and {n_models} models and {id} id", flush=True)
            
            
            training_time = time.time()
            cut_factor = 0 #train on 128 or 256
            mp.spawn(train, args=(world_size, 10000, 1, n_models, channel, [1,2,4,8,16], cut_factor, 1., False, id), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
            print(f"Training with {n_models} models and {channel} channels took {(time.time()-training_time)/3600:.2f}hrs", flush=True)
            
            print("Sampling scales...", flush=True)
            sampling_time = time.time()
            mp.spawn(sample_scales, 
                     args=(
                         world_size,
                         "/home/sp2053/venvs/SR21cmtest/lib/python3.8/site-packages/SR21cm/trained_models/model_6/DDPMpp_standard_channels_8_mult_1-2-4-8-16_tts_1_VPSDE_8_normfactor1.pth",
                         ),
                     nprocs=world_size)
            mp.spawn(plot_scales, args=(world_size, "T21_scales_",), nprocs=world_size)
            print(f"Sampling scales and plotting took {(time.time()-sampling_time)/3600:.2f}hrs", flush=True)

            #cut_factor = 0 #sample 256
            #print(f"Sampling all {256//2**cut_factor}...", flush=True)
            #sampling_time = time.time()
            #mp.spawn(sample, args=(world_size, n_models, channel, [1,2,4,8,16], cut_factor, 1., id), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
            #print(f"Sampling took {(time.time()-sampling_time)/3600:.2f}hrs", flush=True)
    else:
        print("Not using multi_gpu",flush=True)
        try:
            train(rank=0, world_size=0, total_epochs=1, batch_size=1, train_models=56, model_channels=4, channel_mult=[1,2,4,8,16], cut_factor=1, norm_factor=1., memory_profiling=False, model_id=4)#2*4)
        except KeyboardInterrupt:
            print('Interrupted', flush=True)
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)
    
        
 
