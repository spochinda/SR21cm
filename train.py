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

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from SR21cm.train import train_step, sample
from SR21cm.loss import VPLoss
from SR21cm.utils import CustomDataset, get_subcubes, normalize, invert_normalization, augment_dataset, calculate_power_spectrum, plot_sigmas, sample_model_v3
from SR21cm.diffusion import GaussianDiffusion
from SR21cm.model_edm import SongUNet
from SR21cm.loss import VPLoss

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

def main(rank, world_size=0, total_epochs = 1, batch_size = 1, train_models = 56, model_channels = 32, channel_mult = [1,2,4,8,16], cut_factor=1, norm_factor=1., memory_profiling=False, model_id=1):
    #train_models = 56
    #model_channels = 8
    #model_id = 3
    torch.backends.cudnn.benchmark = True #find fastest algorithms

    multi_gpu = world_size > 1

    if multi_gpu:
        device = torch.device(f'cuda:{rank}')
        ddp_setup(rank, world_size=world_size)
    else:
        device = "cpu"
    


    #optimizer and model
    path = os.getcwd().split("/SR21cm")[0] + "/SR21cm"


    #network_opt = dict(in_channel=4, out_channel=1, inner_channel=32, norm_groups=8, channel_mults=(1, 2, 4, 8, 8), attn_res=(16,8,), res_blocks=2, dropout = 0, with_attn=True, image_size=64, dim=3)
    #network_opt = dict(in_channel=4, out_channel=1, inner_channel=32, norm_groups=8, channel_mults=(1, 2, 4, 8, 8), attn_res=(8,), res_blocks=2, dropout = 0, with_attn=True, image_size=32, dim=3)
    #network = UNet
    network_opt = dict(img_resolution=128, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
                    model_channels=model_channels, channel_mult=channel_mult, num_blocks = 4, attn_resolutions=[8,], mid_attn=True, #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                    embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                    )
    #img_resolution=64, channel_mult=[1,2,4,8,16], num_blocks = 4, attn_resolutions=[4,], mid_attn=True, largest model to pass through 512 on 80GB and 40GB
    #512 compatible:
    #channels 4, 1,2,8,8,16
    #channels 4, 1,2,8,16,
    #channels 4, 1,2,16,64
    #channels 4, 1-2-32-64-128
    #channels 4, 1,2,8,32,128, attn_res [8,] best
    #channels 4, 1,2,4,8,16, attn_res [8,] best works on 40 GB
    
    #256 compatible:
    #channels 32, 1,2,4,8,16 attn_res [8,] works on 40GB
    #channels 32, 1,2,8,32,128 attn_res [8,] OOM even on 80GB
    #channels 16, 1,2,8,32,128 attn_res [8,] on 80GB passed
    #channels 16, 1,2,4,8,16 attn_res [8,] on 80GB passed

    #network = UNet
    network = SongUNet
    
    #noise_schedule_opt = {'schedule_type': "linear", 'schedule_opt': {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}} 
    #noise_schedule_opt = {'schedule_type': "cosine", 'schedule_opt': {"timesteps": 1000, "s" : 0.008}} 
    #noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    
    #loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
    loss_fn = VPLoss(beta_max=20., beta_min=0.1, epsilon_t=1e-5, use_amp=False)
    
    netG = GaussianDiffusion(
            network=network,
            network_opt=network_opt,
            noise_schedule_opt=noise_schedule_opt,
            loss_fn = loss_fn,
            learning_rate=1e-3,
            scheduler=False,
            mp=True,
            rank=rank,
        )
    
    #generate seed to share across GPUs
    if multi_gpu:
        #if torch.cuda.current_device()==0:
        seed = torch.randint(0, 1420, (1,))
        torch.distributed.barrier()
        torch.distributed.broadcast(tensor=seed, src=0)
        if (model_channels==8) and (train_models==56) and (model_id==9):
            seed = 687
        elif (model_channels==8) and (train_models==56) and (model_id==10):
            seed = 264
        else:
            seed = seed.item()# 61#687
        print(f"Rank {rank} seed: {seed}", flush=True)


    train_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/", 
    #train_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/", 
                                    redshifts=[10,], IC_seeds=list(range(0,train_models)), Npix=256, device=device)
    #train_data_module.getFullDataset()
    train_sampler = DistributedSampler(dataset=train_data_module, shuffle=True, seed=seed) if multi_gpu else None
    train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=batch_size, shuffle=(train_sampler is None), sampler = train_sampler)#, num_workers=world_size*4)#, pin_memory=True) #rule of thumb 4*num_gpus

    validation_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/", 
    #validation_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
                                    redshifts=[10,], IC_seeds=list(range(train_models,72)), Npix=256, device=device)
    #validation_data_module.getFullDataset()
    validation_sampler = DistributedSampler(validation_data_module, shuffle=True, seed=seed) if multi_gpu else None
    validation_dataloader = torch.utils.data.DataLoader(validation_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = validation_sampler)#, num_workers=world_size*4)#, pin_memory=True)
    
    test_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/", 
    #test_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
                                    redshifts=[10,], IC_seeds=list(range(72,80)), Npix=256, device=device)
    test_sampler = DistributedSampler(test_data_module) if multi_gpu else None
    test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = test_sampler)#, num_workers=world_size*4)#, pin_memory=True)
    

    try:
        fn = path + "/trained_models/model_6/DDPMpp_standard_channels_{0}_mult_{1}_tts_{2}_{3}_{4}_normfactor{5}".format(
            netG.network_opt["model_channels"],
            "".join(f"{m}-" for i,m in enumerate(network_opt["channel_mult"]))[:-1], 
            len(train_data_module.IC_seeds) * 100 // 80,
            netG.noise_schedule_opt["schedule_type"], 
            model_id, int(norm_factor))
        
        netG.model_name = fn.split("/")[-1]
        
        #raise Exception("Temporarily suspend loading")
        netG.load_network(fn+".pth")
        if rank==0:
            print(f"Loaded network at {fn}", flush=True)
    except Exception as e:
        if rank==0:
            print(e, flush=True)
            print(f"Failed to load network at {fn}.pth. Starting from scratch.", flush=True)


    
    #test 512 boxes forward pass through model to see if we get OOM
    if False:#multi_gpu:
        if str(device)=="cuda:0":
            #save model print to text file 
            with open('out.txt', 'w') as f:
                print(netG.model, file=f)
        torch.distributed.barrier()
        print(f"[{str(device)}] Testing forward pass through model...", flush=True)
        with torch.no_grad():
            for box_len in [256,]:
                X = torch.randn(1,4,*(3*[box_len,])).to(device).requires_grad_(False)
                b,*d  = X[:,:1].shape
                batch_time_step = torch.tensor(b*[0.9]).view(b,*[1]*len(d))
                score = netG.model(x=X, noise_labels=batch_time_step.flatten(), class_labels=None, augment_labels=None) #999 from wrapper get_score_fn
                print(f"[{str(device)}] Passed {box_len}", flush=True)
                torch.distributed.barrier()
        print(f"[{str(device)}] Forward pass through model successful", flush=True)

        if str(device)=="cuda:0":
            #print memory summary
            with open('out.txt', 'a') as f:
                print(torch.cuda.memory_summary(device=device), file=f)

    if (str(device)=="cuda:0") and memory_profiling:
        torch.cuda.memory._record_memory_history()
        #prof.step()
    if rank==0:
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print(f"[{str(device)}] Starting training at {current_time}...", flush=True)

    not_saved = 0
    start_time_training = time.time()
    if True:#not (model_channels==8 and train_models==56):
        for e in range(total_epochs): #if False:#       
            
            if (str(device)=="cuda:0") or (str(device)=="cpu"):
                start_time = time.time()
            
            if True:
                if multi_gpu:
                    train_sampler.set_epoch(len(netG.loss)) #

                avg_loss = train_step(netG=netG, epoch=e, train_dataloader=train_dataloader, cut_factor=cut_factor, norm_factor=norm_factor, split_batch=True, sub_batch=4, one_box_per_epoch=True, device=device, multi_gpu=multi_gpu)

                if (str(device)=="cuda:0") or (str(device)=="cpu"):
                    print("[{0}]: Epoch {1} in {2:.2f}s | ".format(str(device), len(netG.loss), time.time()-start_time) +
                        "loss: {0:,}, mean(loss[-10:]): {1:,}, loss min: {2:,}, ".format(avg_loss,  torch.mean(torch.tensor(netG.loss[-10:])).item(), torch.min(torch.tensor(netG.loss)).item()) +
                        "learning rate: {0:.3e}".format(netG.optG.param_groups[0]['lr']), flush=True)

                if netG.scheduler is not False:
                    netG.scheduler.step()

                if (str(device)=="cuda:0") and memory_profiling:
                    torch.cuda.memory._dump_snapshot(f"memory_snap_16_2_{str(device)[-1]}.pickle")
                    #prof.step()
            
            
            if model_channels==32:
                validation_check_epoch = 1500
            elif model_channels==16:
                validation_check_epoch = 1500
            elif model_channels==8:
                validation_check_epoch = 1000#1500
            elif model_channels==4:
                validation_check_epoch = 1500
            
            if len(netG.loss)>=validation_check_epoch:
                if len(netG.loss)==validation_check_epoch:
                    for g in netG.optG.param_groups:
                        g['lr'] = 1e-4
                
                if len(netG.loss)%50==0 or avg_loss == torch.min(torch.tensor(netG.loss)).item():
                    start_time_validation = time.time()
                    #loss_validation, tensor_dict_validation = sample_model(netG=netG, dataloader=validation_dataloader, cut_factor=cut_factor, norm_factor=norm_factor, augment=1, split_batch = True, sub_batch = 4, n_boxes=1, num_steps=100, device=device, multi_gpu=multi_gpu)
                    #loss_validation, tensor_dict_validation = sample_model_v3(rank, netG=netG, dataloader=test_dataloader, cut_factor=cut_factor, norm_factor = norm_factor, augment=1, split_batch = True, sub_batch = 4, n_boxes = 1, num_steps=100, device=device, multi_gpu=multi_gpu)
                    if multi_gpu:
                        validation_dataloader.sampler.set_epoch(len(netG.loss))
                    loss_validation, tensor_dict_validation = sample_model_v3(rank, netG=netG, dataloader=validation_dataloader, cut_factor=0, norm_factor = norm_factor, augment=1, split_batch = False, sub_batch = 4, n_boxes = 1, num_steps=100, device=device, multi_gpu=multi_gpu)
                    validation_time = time.time()-start_time_validation
                        
                    loss_validation_min = torch.min(torch.tensor(netG.loss_validation["loss_validation"])).item()
                    if loss_validation < loss_validation_min:
                        if rank==0:
                            
                            path_plot = os.getcwd().split("/SR21cm")[0] + "/SR21cm/plots/vary_channels_nmodels_8/"
                            #plot_hist(T21_1=tensor_dict_validation["T21"], T21_2=tensor_dict_validation["T21_pred"], path=path_plot+f"hist_true_validation_during_{netG.model_name}.png", label="mean true-validation during")
                            plot_sigmas(**tensor_dict_validation, netG=netG, path = path_plot,  quantiles=[0.16,0.5,0.84]) #[(1-0.997)/2, (1-0.954)/2, 0.16, 0.5, 0.84, 1 - (1-0.954)/2, 1 - (1-0.997)/2])
                            #plot_checkpoint(**tensor_dict, netG=netG, MSE=loss_validation, epoch=len(netG.loss), path = path_plot, device=device)
                            
                            print(f"[{device}] Validation took {validation_time:.2f}s, validation rmse={loss_validation**0.5:.3f} smaller than minimum={loss_validation_min**0.5:.3f}", flush=True)
                            try:
                                print("Weights: ", netG.model.module.enc["128_conv_in4_out4"].weight[0,0,0], flush=True)
                            except Exception as e:
                                print(e, flush=True)
                        netG.save_network(fn+".pth")
                        saved_network_str = netG.model.module.state_dict().__str__()
                        
                        not_saved = 0
                        netG.loss_validation["loss_validation"].append(loss_validation)

                    else:
                        not_saved = not_saved + 1
                        if rank==0:
                            print(f"[{device}] Not saving... validation time={validation_time:.2f}s, validation rmse={loss_validation**0.5:.3f} larger than minimum={loss_validation_min**0.5:.3f}. Not saved={not_saved}", flush=True)

            #abort if last save was more than n validation tests ago
            #if False:#(not_saved>=20) or (len(netG.loss) == total_epochs-1):
            #    if rank==0:
            #        print("No improvement in 20 validation tests. Saving test data...", flush=True)
            if (time.time()-start_time_training > 12*60*60):
                if rank==0:
                    print("12 hours passed. Aborting training...", flush=True)
                break
    
    

    
    if (str(device)=="cuda:0") and memory_profiling:
        torch.cuda.memory._record_memory_history(enabled=None)

    if multi_gpu:#world_size > 1:
        torch.distributed.barrier()
        destroy_process_group()
###END main pytorch multi-gpu tutorial###



if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the optional arguments
    parser.add_argument("--channels", type=int, default=4, help="channels")
    parser.add_argument("--nmodels", type=int, default=1, help="nmodels")
    parser.add_argument("--id", type=int, default=4, help="id") #4

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
            cut_factor = 1 #train on 128
            #mp.spawn(main, args=(world_size, 10000, 1, n_models, channel, [1,2,4,8,16], cut_factor, 1., False, id), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
            print(f"Training with {n_models} models and {channel} channels took {(time.time()-training_time)/3600:.2f}hrs", flush=True)
            print("Sampling...", flush=True)
            sampling_time = time.time()
            cut_factor = 0 #sample 256
            mp.spawn(sample, args=(world_size, n_models, channel, [1,2,4,8,16], cut_factor, 1., id), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
            print(f"Sampling took {(time.time()-sampling_time)/3600:.2f}hrs", flush=True)
    else:
        print("Not using multi_gpu",flush=True)
        try:
            main(rank=0, world_size=0, total_epochs=1, batch_size=1, train_models=56, model_channels=4, channel_mult=[1,2,4,8,16], cut_factor=1, norm_factor=1., memory_profiling=False, model_id=4)#2*4)
        except KeyboardInterrupt:
            print('Interrupted', flush=True)
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)
    
        
 
