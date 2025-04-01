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
from SR21cm.asr.train_asr import train

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the optional arguments
    parser.add_argument("--config", type=str, default="/home/sp2053/rds/hpc-work/SR21cm/configs/default_asr.yml", help="config file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the optional arguments
    config = args.config
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    
    print(f"PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}, Multi-gpu: {multi_gpu}", flush=True)

    if config["profiling"]["nsys"]:
        print("nsys is enabled;, have you ran the program with nsys: nsys profile --trace=cuda,osrt,nvtx,cudnn --stats=true --capture-range=cudaProfilerApi --capture-range-end=stop -s none --cuda-memory-usage=true python train.py ", flush=True)

    if multi_gpu:
        start = time.time()
        mp.spawn(train, 
                    args=(
                        world_size, 
                        config, 
                        ), 
                    nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
        print(f"Training took {(time.time()-start)/3600:.2f}hrs", flush=True)
    else:
        train(rank=0, world_size=world_size, config=config)
 
