import torch
import torch.distributed
import torch.utils
import torch.multiprocessing as mp

import os
import sys
import time
from datetime import datetime, timedelta
import argparse
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from SR21cm.train import sample, train
from SR21cm.utils import sample_scales, get_paths
from SR21cm.plotting import plot_scales

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the optional arguments
    parser.add_argument("--channels", type=int, default=8, help="channels")
    parser.add_argument("--nmodels", type=int, default=1, help="nmodels")
    parser.add_argument("--id", type=int, default=8, help="id") #4
    parser.add_argument("--config", type=str, default="configs/default.yml", help="config file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the optional arguments
    channels = args.channels
    nmodels = args.nmodels
    id = args.id
    config = args.config
    with open(config, "r") as file:
        config = yaml.safe_load(file)
    

    

    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    
    print("PyTorch version: ", torch.__version__, flush=True)

    if config["profiling"]["nsys"]:
        print("nsys is enabled;, have you ran the program with nsys: nsys profile --trace=cuda,osrt,nvtx,cudnn --stats=true --capture-range=cudaProfilerApi --capture-range-end=stop -s none --cuda-memory-usage=true python train.py ", flush=True)

    if multi_gpu:
        print("CUDA version: ", torch.version.cuda)
        print("Using multi_gpu", flush=True)
        for i in range(torch.cuda.device_count()):
            print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)
        
        name = config["name"]
        
        for i in range(id+1,id+2):
            print(f"Training {i}...", flush=True)
        
            config["name"] = name + f"_{i}"
            
            training_time = time.time()
            cut_factor = 0 #train on 128 or 256
            if i != 0 and i != 5 and i != 10:
                mp.spawn(train, 
                            args=(
                                world_size, 
                                config, 
                                ), 
                            nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
                #mp.spawn(train, args=(world_size, 10000, 1, n_models, channel, [1,2,4,8,16], cut_factor, 1., False, id), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
                print(f"Training took {(time.time()-training_time)/3600:.2f}hrs", flush=True)
            
                print("Sampling scales...", flush=True)
                sampling_time = time.time()
                #mp.spawn(sample_scales, 
                #            args=(
                #                world_size,
                #                config, #"/home/sp2053/venvs/SR21cmtest/lib/python3.8/site-packages/SR21cm/trained_models/model_6/DDPMpp_standard_channels_8_mult_1-2-4-8-16_tts_1_VPSDE_8_normfactor1.pth",
                #                ),
                #            nprocs=world_size)
                print(f"Sampling scales and plotting took {(time.time()-sampling_time)/3600:.2f}hrs", flush=True)
                
            #model_path, model_dir, plot_dir, data_dir = get_paths(config)
            #mp.spawn(plot_scales, 
            #            args=(
            #                world_size, 
            #                data_dir,
            #                plot_dir,
            #                ), 
            #            nprocs=world_size)
            

            #cut_factor = 0 #sample 256
            #print(f"Sampling all {256//2**cut_factor}...", flush=True)
            #sampling_time = time.time()
            #mp.spawn(sample, args=(world_size, n_models, channel, [1,2,4,8,16], cut_factor, 1., id), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
            #print(f"Sampling took {(time.time()-sampling_time)/3600:.2f}hrs", flush=True)
    else:
        print("Not using multi_gpu",flush=True)
        #train(rank=0, world_size=world_size, total_epochs=1, batch_size=1, train_models=56, model_channels=4, channel_mult=[1,2,4,8,16], cut_factor=1, norm_factor=1., memory_profiling=False, model_id=4)#2*4)    
        #train(rank=0, world_size=world_size, config=config, memory_profiling=False, model_id=4)#2*4)    
        train(rank=0, world_size=world_size, config=config)
        #sample_scales(rank=0, world_size=world_size, model_path="/Users/simonpochinda/Documents/PhD/SR21cm/trained_models/model_6/DDPMpp_standard_channels_8_mult_1-2-4-8-16_tts_1_VPSDE_8_normfactor1.pth")
        #plot_scales(rank=0, world_size=world_size, load_path="/Users/simonpochinda/Documents/PhD/SR21cm/analysis/model_6/T21_scales_0.pth")
        
 
