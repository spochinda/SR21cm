#import contextlib
import torch
import torch.distributed
import torch.utils
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from .utils import CustomDataset, get_subcubes, normalize, invert_normalization, augment_dataset, calculate_power_spectrum, sample_model_v3, initialize_model_directory, data_preprocess, get_paths
from .plotting import plot_sigmas, plot_input, plot_hist
from .diffusion import GaussianDiffusion
from .model_edm import SongUNet
from .loss import VPLoss

import os
import time
import psutil  # Import psutil
from contextlib import nullcontext
from tqdm import tqdm
from datetime import datetime


def ddp_setup(rank: int, world_size: int):
    try:
        os.environ["MASTER_ADDR"] #check if master address exists
        print("Found master address: ", os.environ["MASTER_ADDR"])
    except:
        print("Did not find master address variable. Setting manually...")
        os.environ["MASTER_ADDR"] = "localhost"

    os.environ["MASTER_PORT"] = "2595"#"12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="gloo", rank=rank, world_size=world_size) #backend gloo for cpus? nccl for gpus



def train_step(netG, train_dataloader, config, 
               device="cpu", multi_gpu = False,
               #cut_factor=1, norm_factor = 1., split_batch=True, sub_batch=4, one_box_per_epoch = True,
               ):
    """
    Train the model
    """
    netG.model.train()
    
    

    for i,(T21_, delta_, vbv_, labels) in enumerate(train_dataloader):
        if netG.network_opt["label_dim"] > 0:
            labels = labels
        else:
            labels = None #set labels to None for now (not used)

        #random cut = 0 or 1
        #cut = torch.randint(0,2,(1,)).item()
        
        for cut in [2,1]:
            
            avg_loss = torch.tensor(0.0, device=device)
            
            #T21 = get_subcubes(cubes=T21_, cut_factor=cut)
            #delta = get_subcubes(cubes=delta_, cut_factor=cut)
            #vbv = get_subcubes(cubes=vbv_, cut_factor=cut)
            #T21_lr = torch.nn.functional.interpolate(T21, scale_factor=1/4, mode='trilinear')#get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
            #            
            #T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
            #T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
            #T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)
            #
            #T21_lr, _,_ = normalize(T21_lr, mode="standard", factor=norm_factor)#, factor=2.)
            #T21, _,_ = normalize(T21, mode="standard", factor=norm_factor, x_mean=T21_lr_mean, x_std=T21_lr_std)
            #delta, _,_ = normalize(delta, mode="standard", factor=norm_factor)
            #vbv, _,_ = normalize(vbv, mode="standard", factor=norm_factor)
            #T21, delta, vbv , T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=1) #support device

            config["data_preprocess"]["cut_factor"] = cut #manually set cut factor
            T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std = data_preprocess(T21=T21_, delta=delta_, vbv=vbv_, **config["data_preprocess"])

            netG.optG.zero_grad()
            if False:#split_batch: #split subcube minibatch into smaller mini-batches for memory
                sub_data = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)
                sub_dataloader = torch.utils.data.DataLoader(sub_data, batch_size=sub_batch, shuffle=False, sampler = None) # (2**(cut_factor.item()-1))**3 // 2 #4
                #data_prep_etime = time.time() - data_prep_stime###
                #loop time
                #loop_stime = time.time()###
                for j,(T21, delta, vbv, T21_lr) in enumerate(sub_dataloader):
                    #print(f"Sub {j}", flush=True)

                    netG.optG.zero_grad()
                    #loss_stime = time.time()###
                    loss = netG.loss_fn(net=netG, images=T21, conditionals=[delta, vbv, T21_lr],
                                        labels=labels, augment_pipe=None,
                                        )
                    avg_loss = avg_loss + loss * T21.shape[0]  #add avg loss per mini-batch to accumulate total batch loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
                    netG.optG.step()        
                    netG.ema.update() #Update netG.model with exponential moving average  
            else:
                #make cut a label tensor
                if netG.network_opt["label_dim"] > 0:
                    labels = torch.tensor([cut], device=device)
                else:
                    labels = None
                #with torch.cuda.amp.autocast(enabled=netG.loss_fn.use_amp) if netG.loss_fn.use_amp else nullcontext():
                loss = netG.loss_fn(net=netG, images=T21, conditionals=[delta, vbv, T21_lr], labels=labels, augment_pipe=None,
                                        )
                    
                avg_loss = avg_loss + loss #* T21.shape[0]  #add avg loss per mini-batch to accumulate total batch loss
                #avg_loss = loss
                #print gradient 
                #for i,(name, param) in enumerate(netG.model.named_parameters()):
                #    if i<3 and torch.cuda.current_device() == 0:
                #        print(name, param.grad, flush=True)

                if netG.loss_fn.use_amp:
                    netG.scaler.scale(loss).backward()
                else:
                    loss.backward()
                #dev = torch.cuda.current_device()
                #if dev == 0:
                #    with torch.no_grad():
                #        params_test = []
                #        for i,(name, param) in enumerate(netG.model.named_parameters()):
                #            g = param.grad.detach().cpu().numpy()
                #            any_inf = np.any(np.isinf(g))
                #            any_nan = np.any(np.isnan(g))
                #            if any_inf or any_nan:
                #                print(f"[dev:{dev}] -- {name}", flush=True)
                #        params_test = np.array(params_test)
                #        #All none?
                #        print(f"[dev:{dev}] -- any None?", np.any(params_test==None), params_test, flush=True)
                #        #Any inf?
                #        #print(f"[dev:{dev}] -- any inf?", np.any(np.isinf(params_test)), flush=True)
                #        #Any nan?
                #        #print(f"[dev:{dev}] -- any nan?", np.any(np.isnan(params_test)), flush=True)
                #torch.distributed.barrier()
                if netG.loss_fn.use_amp:
                    netG.scaler.step(netG.optG)
                    netG.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
                    netG.optG.step()
                
                netG.ema.update() #Update netG.model with exponential moving average
                

        


        if True:#one_box_per_epoch:
            break #only do one box for now
    
    if multi_gpu:
        #all_reduce time
        #all_reduce_stime = time.time()###
        torch.distributed.all_reduce(tensor=avg_loss, op=torch.distributed.ReduceOp.SUM) #total loss=sum(average total batch loss per gpu)
        #all_reduce_etime = time.time() - all_reduce_stime###
    
    #if torch.cuda.current_device() == 0:
    #    print(f"Time: Data prep: {data_prep_etime:.3f}s, Loss: {loss_etime:.3f}s, Backward: {backward_etime:.3f}s, Clip: {clip_etime:.3f}s, Optimizer: {optimizer_etime:.3f}s, EMA: {ema_etime:.3f}s, loop: {loop_etime:.3f}s, all_reduce: {all_reduce_etime:.3f}s", flush=True)
    #Time: Data prep: 0.007s, Loss: 1.096s, Backward: 3.571s, Clip: 0.005s, Optimizer: 0.020s, EMA: 0.012s, loop: 4.707s, all_reduce: 0.001s

    netG.loss.append(avg_loss.item())
    
    return avg_loss.item(), labels





@torch.no_grad()
def save_test_data(netG, test_dataloader, norm_factor = 3., path="", cut_factor=2, device="cpu"):
    rank = torch.distributed.get_rank()
    #netG.model.eval() #already inside Euler_Maruyama_sampler
    for i,(T21, delta, vbv, labels) in tqdm(enumerate(test_dataloader), desc="Iterating over test data", total=len(test_dataloader), disable=False if rank==0 else True):
        T21, delta, vbv = T21.to(device), delta.to(device), vbv.to(device)

        T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
        delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
        vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
        T21_lr = torch.nn.functional.interpolate(T21, scale_factor=1/4, mode='trilinear') # get_subcubes(cubes=T21_lr, cut_factor=cut_factor)

        T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_orig = T21_lr #to save later

        T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)

        T21_lr, _, _ = normalize(T21_lr, mode="standard", factor=norm_factor) 
        T21, _, _ = normalize(T21, mode="standard", factor=norm_factor, x_mean=T21_lr_mean, x_std=T21_lr_std) #factor=2 for all inputs but should only be 2 for HR input in future
        delta, delta_mean, delta_std = normalize(delta, mode="standard", factor=norm_factor)
        vbv, vbv_mean, vbv_std = normalize(vbv, mode="standard", factor=norm_factor)

        sub_data = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr, T21_lr_orig, T21_lr_mean, T21_lr_std, delta_mean, delta_std, vbv_mean, vbv_std)
        sub_dataloader = torch.utils.data.DataLoader(sub_data, batch_size=4, shuffle=False, sampler = None)

        for j,(T21, delta, vbv, T21_lr, T21_lr_orig, T21_lr_mean, T21_lr_std, delta_mean, delta_std, vbv_mean, vbv_std) in tqdm(enumerate(sub_dataloader), desc="Iterating split batch", total=len(sub_dataloader), disable=False if rank==0 else True):
            if False:#(i==j==0) and (str(device)=='cuda:0'):
                print("mean and stds: ", T21_lr_mean.flatten(), T21_lr_std.flatten(), flush=True)
                plot_input(T21=T21, delta=delta, vbv=vbv, T21_lr=T21_lr, path=os.getcwd().split("/SR21cm")[0] + "/SR21cm/plots/vary_channels_nmodels_8/plot_input_save_test.png")
            with netG.ema.average_parameters():
                T21_pred_j = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=None, num_steps=100, eps=1e-3, clip_denoised=False, verbose=False)

            T21_pred_j = invert_normalization(T21_pred_j[:,-1:], mode="standard", factor=norm_factor, x_mean=T21_lr_mean, x_std=T21_lr_std)
            T21 = invert_normalization(T21, mode="standard", factor=norm_factor, x_mean=T21_lr_mean, x_std=T21_lr_std)

            #delta = invert_normalization(delta, mode="standard", x_mean=delta_mean, x_std=delta_std, factor=norm_factor)
            #vbv = invert_normalization(vbv, mode="standard", x_mean=vbv_mean, x_std=vbv_std, factor=norm_factor)
            
            if i==j==0:
                T21_pred_i = T21_pred_j[:,-1:]
                T21_i = T21

                #T21_lr_orig_all = T21_lr_orig
                #delta_all = delta
                #vbv_all = vbv
            else:
                T21_pred_i = torch.cat([T21_pred_i, T21_pred_j[:,-1:]], dim=0)
                T21_i = torch.cat([T21_i, T21], dim=0)

                #T21_lr_orig_all = torch.cat([T21_lr_orig_all, T21_lr_orig], dim=0)
                #delta_all = torch.cat([delta_all, delta], dim=0)
                #vbv_all = torch.cat([vbv_all, vbv], dim=0)

    T21_pred_tensor_list = [torch.zeros_like(T21_pred_i) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list=T21_pred_tensor_list, tensor=T21_pred_i)
    T21_pred = torch.cat(T21_pred_tensor_list, dim=0)

    T21_tensor_list = [torch.zeros_like(T21_i) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list=T21_tensor_list, tensor=T21_i)
    T21 = torch.cat(T21_tensor_list, dim=0)

    #T21_lr_orig_all_tensor_list = [torch.zeros_like(T21_lr_orig_all) for _ in range(torch.distributed.get_world_size())]
    #torch.distributed.all_gather(tensor_list=T21_lr_orig_all_tensor_list, tensor=T21_lr_orig_all)
    #T21_lr_orig_all = torch.cat(T21_lr_orig_all_tensor_list, dim=0)

    #delta_all_tensor_list = [torch.zeros_like(delta_all) for _ in range(torch.distributed.get_world_size())]
    #torch.distributed.all_gather(tensor_list=delta_all_tensor_list, tensor=delta_all)
    #delta_all = torch.cat(delta_all_tensor_list, dim=0)

    #vbv_all_tensor_list = [torch.zeros_like(vbv_all) for _ in range(torch.distributed.get_world_size())]
    #torch.distributed.all_gather(tensor_list=vbv_all_tensor_list, tensor=vbv_all)
    #vbv_all = torch.cat(vbv_all_tensor_list, dim=0)
    

    MSE_temp = torch.mean(torch.square(T21_pred - T21)).item()
    if rank==0:
        print(f"Test data RMSE={MSE_temp**0.5:.4f} ", flush=True)
        save_dict = dict(T21 = T21, T21_pred = T21_pred, )#delta=delta_all, vbv=vbv_all, T21_lr=T21_lr_orig_all)
        torch.save(obj=save_dict, 
                    f=path + "/analysis/model_6/" + netG.model_name + f"_test_data.pth")
        
    torch.distributed.barrier()

    return MSE_temp, dict(T21=T21, #delta=delta_all, vbv=vbv_all, T21_lr=T21_lr_orig_all, 
                          T21_pred=T21_pred)


def sample(rank, world_size, train_models = 56, model_channels = 32, channel_mult = [1,2,4,8,16], cut_factor=1, norm_factor = 1., model_id=1, **kwargs):
    multi_gpu = world_size > 1

    ddp_setup(rank, world_size=world_size)

    device = torch.device(f'cuda:{rank}')

    path = "/home/sp2053/rds/hpc-work" + "/SR21cm"

    
    #model_channels = int(fn.split("channels_")[1].split("_")[0])
    #channel_mult = [int(i) for i in fn.split("mult_")[1].split("_")[0].split("-")]
    network_opt = dict(img_resolution=128, in_channels=4, out_channels=1, label_dim=1, # (for tokens?), augment_dim,
                    model_channels=model_channels, channel_mult=channel_mult, num_blocks = 4, attn_resolutions=[8], mid_attn=True, #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                    embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                    )
    network = SongUNet
    noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  

    netG = GaussianDiffusion(
            network=network,
            network_opt=network_opt,
            noise_schedule_opt=noise_schedule_opt,
            loss_fn = None,
            learning_rate=1e-3,
            scheduler=False,
            mp=True,
            rank=rank,
        )
    

    #test_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/IC_cubes/", 
    #test_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
    #                                redshifts=[10,], IC_seeds=list(range(0,1)), Npix=512, device=device)
    #test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=1, shuffle=False if multi_gpu else True,
    #                                                sampler = DistributedSampler(test_data_module) if multi_gpu else None)


    fn = path + "/trained_models/model_6/DDPMpp_standard_channels_{0}_mult_{1}_tts_{2}_{3}_{4}_normfactor{5}".format(
        netG.network_opt["model_channels"],
        "".join(f"{m}-" for i,m in enumerate(channel_mult))[:-1], 
        train_models * 100 // 80,
        netG.noise_schedule_opt["schedule_type"], 
        model_id, int(norm_factor))
    
    netG.model_name = fn.split("/")[-1]

    netG.load_network(fn+".pth")

    test_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/", 
    #test_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
                                    redshifts=[10,], IC_seeds=list(range(72,80)), Npix=256, device=device) #IC_seeds=list(range(0,1)), cut_factor=0, Npix=512, 
    test_data_module.getFullDataset()
    test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=1, shuffle=False if multi_gpu else True,
                                                    sampler = DistributedSampler(test_data_module) if multi_gpu else None)
    torch.distributed.barrier()
    #augment?
    for steps in [100,]: #300,80,60,40,30,20,10
        loss_test, tensor_dict = sample_model_v3(rank, netG=netG, dataloader=test_dataloader, cut_factor=cut_factor, norm_factor = norm_factor, augment=24, split_batch = True, sub_batch = 4, n_boxes = -1, num_steps=steps, device=device, multi_gpu=multi_gpu) #augment=24
        torch.save(obj=tensor_dict,
                f=os.getcwd().split("/SR21cm")[0] + f"/SR21cm/analysis/model_6/save_data_tensors_{netG.model_name}_steps_{steps}_Npix256.pth")
        if rank==0:
            print(f"Test data loss={loss_test**0.5:.4f} ", flush=True)
            
            print(torch.cuda.memory_summary(device=device), flush=True)            
            virtual_memory = psutil.virtual_memory()
            total_ram = virtual_memory.total / (1024 ** 3)  # Convert bytes to GB
            used_ram = virtual_memory.used / (1024 ** 3)    # Convert bytes to GB
            available_ram = virtual_memory.available / (1024 ** 3)  # Convert bytes to GB
            print(f"Total RAM: {total_ram:.2f} GB", flush=True)
            print(f"Used RAM: {used_ram:.2f} GB", flush=True)
            print(f"Available RAM: {available_ram:.2f} GB", flush=True)


    destroy_process_group()


###START main pytorch multi-gpu tutorial###
def train(rank, 
          world_size=0, 
          config = None,
          #total_epochs = 1, #training_opt
          #batch_size = 1, #training_opt
          #train_models = 56, #datasets
          #model_channels = 32, #network_opt 
          #channel_mult = [1,2,4,8,16], #network_opt
          #cut_factor=1, #training_opt
          #norm_factor=1., #training_opt
          memory_profiling=False, 
          #model_id=1,
          **kwargs):
    #train_models = 56
    #model_channels = 8
    #model_id = 3
    torch.backends.cudnn.benchmark = True #find fastest algorithms

    multi_gpu = world_size > 1

    if multi_gpu:
        device = torch.device(f'cuda:{rank}')
        ddp_setup(rank, world_size=world_size)
    else:
        device = torch.device("cpu")
    

    #optimizer and model
    path = "/home/sp2053/venvs/SR21cmtest/lib/python3.8/site-packages" + "/SR21cm"


    #network_opt = dict(in_channel=4, out_channel=1, inner_channel=32, norm_groups=8, channel_mults=(1, 2, 4, 8, 8), attn_res=(16,8,), res_blocks=2, dropout = 0, with_attn=True, image_size=64, dim=3)
    #network_opt = dict(in_channel=4, out_channel=1, inner_channel=32, norm_groups=8, channel_mults=(1, 2, 4, 8, 8), attn_res=(8,), res_blocks=2, dropout = 0, with_attn=True, image_size=32, dim=3)
    #network = UNet
    #network_opt = dict(img_resolution=128, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
    #                model_channels=model_channels, channel_mult=channel_mult, num_blocks = 4, attn_resolutions=[8], mid_attn=True, #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
    #                embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
    #                )
    network_opt = config["network_opt"]
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
    #noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    noise_schedule_opt = config["noise_schedule_opt"]
    
    #loss_fn = VPLoss(beta_max=20., beta_min=0.1, epsilon_t=1e-5, use_amp=False)
    loss_fn = VPLoss(config["loss_opt"])
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
    #if torch.cuda.current_device()==0 and netG.loss_fn.use_amp:  print("Initial statedict: ", netG.scaler.state_dict(), flush=True)
    
    #generate seed to share across GPUs
    if multi_gpu:
        seed = torch.randint(0, 1420, (1,)) if config["seed"] is None else torch.tensor(config["seed"])
        torch.distributed.barrier()
        torch.distributed.broadcast(tensor=seed, src=0)
        seed = seed.item()
    if False:#multi_gpu:
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


    #train_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/", 
    ##train_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/", 
    #                                redshifts=[10,], IC_seeds=list(range(0,train_models)), Npix=256, device=device)
    #train_data_module.getFullDataset()
    #train_sampler = DistributedSampler(dataset=train_data_module, shuffle=True, seed=seed) if multi_gpu else None
    #train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=batch_size, shuffle=(train_sampler is None), sampler = train_sampler)#, num_workers=world_size*4)#, pin_memory=True) #rule of thumb 4*num_gpus
    #
    #validation_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/", 
    ##validation_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
    #                                redshifts=[10,], IC_seeds=list(range(train_models,72)), Npix=256, device=device)
    ##validation_data_module.getFullDataset()
    #validation_sampler = DistributedSampler(validation_data_module, shuffle=True, seed=seed) if multi_gpu else None
    #validation_dataloader = torch.utils.data.DataLoader(validation_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = validation_sampler)#, num_workers=world_size*4)#, pin_memory=True)
    #
    #test_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/", 
    ##test_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
    #                                redshifts=[10,], IC_seeds=list(range(72,80)), Npix=256, device=device)
    #test_sampler = DistributedSampler(test_data_module) if multi_gpu else None
    #test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = test_sampler)#, num_workers=world_size*4)#, pin_memory=True)
    train_data_module = CustomDataset(**config["datasets"]["train"], device=device)
    train_sampler = DistributedSampler(dataset=train_data_module, shuffle=True, seed=seed) if multi_gpu else None
    train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=train_data_module.batch_size, shuffle=(train_sampler is None), sampler = train_sampler)#, num_workers=world_size*4)#, pin_memory=True) #rule of thumb 4*num_gpus
    
    """
    validation_data_module = CustomDataset(**config["datasets"]["validation"])
    validation_sampler = DistributedSampler(validation_data_module, shuffle=True, seed=seed) if multi_gpu else None
    validation_dataloader = torch.utils.data.DataLoader(validation_data_module, batch_size=validation_data_module.batch_size, shuffle=False if multi_gpu else True, sampler = validation_sampler)#, num_workers=world_size*4)#, pin_memory=True)
    
    test_data_module = CustomDataset(**config["datasets"]["test"])
    test_sampler = DistributedSampler(test_data_module) if multi_gpu else None
    test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=test_data_module.batch_size, shuffle=False if multi_gpu else True, sampler = test_sampler)#, num_workers=world_size*4)#, pin_memory=True)

    """    

    config, model_path = initialize_model_directory(rank, config)

    if os.path.exists(model_path):
        try:
            netG.load_network(model_path)
            if rank==0:
                print(f"Loaded network at {model_path}", flush=True)
        except:
            if rank==0:
                print(f"Failed to load network at {model_path}. Starting from scratch.", flush=True)
    else:
        if rank==0:
            print(f"Model path does not exists at {model_path}. Starting from scratch.", flush=True)
    
    model_path, model_dir, plot_dir, data_dir = get_paths(config)

    

    
    #try:
    #    fn = path + "/trained_models/model_6/DDPMpp_standard_channels_{0}_mult_{1}_tts_{2}_{3}_{4}_normfactor{5}".format(
    #        netG.network_opt["model_channels"],
    #        "".join(f"{m}-" for i,m in enumerate(network_opt["channel_mult"]))[:-1], 
    #        len(train_data_module.IC_seeds) * 100 // 80,
    #        netG.noise_schedule_opt["schedule_type"], 
    #        model_id, int(norm_factor))
    #    
    #    netG.model_name = fn.split("/")[-1]
    #    
    #    #raise Exception("Temporarily suspend loading")
    #    netG.load_network(fn+".pth")
    #    if rank==0:
    #        print(f"Loaded network at {fn}", flush=True)
    #except Exception as e:
    #    if rank==0:
    #        print(e, flush=True)
    #        print(f"Failed to load network at {fn}.pth. Starting from scratch.", flush=True)


    
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

    if device.index == 0 and memory_profiling:
        torch.cuda.memory._record_memory_history()
        #prof.step()
    if rank==0:
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print(f"[{str(device)}] Starting training at {current_time}...", flush=True)

    not_saved = 0
    start_time_training = time.time()
    #netG.scaler = torch.cuda.amp.GradScaler(enabled=netG.loss_fn.use_amp)
    #if torch.cuda.current_device()==0 and netG.loss_fn.use_amp:  print("Loaded statedict: ", netG.scaler.state_dict(), flush=True)

    for e in range(config["total_epochs"]): #if False:#       
        
        if device.index==0 or device.type=="cpu":
            start_time = time.time()
        
        if True:
            if multi_gpu:
                train_sampler.set_epoch(len(netG.loss)) #
            #avg_loss, labels = train_step(netG=netG, epoch=e, train_dataloader=train_dataloader, cut_factor=cut_factor, norm_factor=norm_factor, split_batch=False, sub_batch=4, one_box_per_epoch=True, device=device, multi_gpu=multi_gpu)
            avg_loss, labels = train_step(netG=netG, train_dataloader=train_dataloader, config=config, device=device, multi_gpu = multi_gpu)
            #if e%4==0:  netG.save_network(model_path)
            if device.index==0 or device.type=="cpu":
                print("[{0}]: Epoch {1} in {2:.2f}s | ".format(device.type, len(netG.loss), time.time()-start_time) +
                    #"loss: {0:,}, mean(loss[-10:]): {1:,}, loss min: {2:,}, ".format(avg_loss,  torch.mean(torch.tensor(netG.loss[-10:])).item(), torch.min(torch.tensor(netG.loss)).item()) +
                    "loss: {0:.4f}, mean(loss[-10:]): {1:.4f}, loss min: {2:.4f}, ".format(avg_loss,  torch.mean(torch.tensor(netG.loss[-10:])).item(), torch.min(torch.tensor(netG.loss)).item()) +
                    "learning rate: {0:.3e}, ".format(netG.optG.param_groups[0]['lr']) +
                    "redshift: {0}".format(labels[0].item() if labels is not None else "None"), #labels is tensor([z])
                    flush=True)
                
                #if netG.loss_fn.use_amp:
                #    print("statedict: ", netG.scaler.state_dict(), flush=True)
            
            if netG.scheduler is not False:
                netG.scheduler.step()

            if device.index==0 and memory_profiling:
                memory_path = os.path.join(data_dir, "memory_snap.pickle")
                torch.cuda.memory._dump_snapshot(memory_path)
                #prof.step()
        
        
        validation_check_epoch = 1000#1500
        
        if False:# len(netG.loss)>=validation_check_epoch:
            if len(netG.loss)==validation_check_epoch:
                for g in netG.optG.param_groups:
                    g['lr'] = 1e-4
            
            if len(netG.loss)%50==0 or avg_loss == torch.min(torch.tensor(netG.loss)).item():
                start_time_validation = time.time()
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

        
        #every 100 epochs after epoch 2000 save network
        if len(netG.loss) >= 1000 and len(netG.loss)%100==0:
            if len(netG.loss) == 1000:
                for g in netG.optG.param_groups:
                            g['lr'] = 1e-4
            netG.save_network(model_path)
            if rank==0:
                print(f"[{device}] Saved network at {model_path}", flush=True)
        #abort if last save was more than n validation tests ago
        #if False:#(not_saved>=20) or (len(netG.loss) == total_epochs-1):
        #    if rank==0:
        #        print("No improvement in 20 validation tests. Saving test data...", flush=True)
        if (time.time()-start_time_training > 3*60*60):
            if rank==0:
                print("12 hours passed. Aborting training...", flush=True)
            break



    
    if device.index==0 and memory_profiling:
        torch.cuda.memory._record_memory_history(enabled=None)

    if multi_gpu:#world_size > 1:
        torch.distributed.barrier()
        destroy_process_group()
###END main pytorch multi-gpu tutorial###
        
