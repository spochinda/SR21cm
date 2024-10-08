import os
import matplotlib.pyplot as plt
import torch
import torch.distributed
from models.diffusion import *
from models.model import *
from models.utils import *
from models.model_edm import SongUNet

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

@torch.no_grad()
def sample_error(rank, ema=False, multi_gpu=False):
    if multi_gpu:
        world_size = torch.cuda.device_count()
        ddp_setup(rank, world_size=world_size)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    path = os.getcwd().split("/SR21cm")[0] + "/SR21cm"
    model_root = path + "/trained_models/model_13/"
    #model_files = [file for file in os.listdir(model_root) if file.endswith("_1.pth")][9:]
    model_files = ["DDPMpp_standard_channels_32_tts_70_VPSDE_2.pth",]
    
    batch_size = 1

    test_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/", 
                                        redshifts=[10,], IC_seeds=list(range(72,80)), upscale=4, cut_factor=0, transform=False, norm_lr=True, device=device)
    test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True,
                                                        sampler = DistributedSampler(test_data_module) if multi_gpu else None)                

    for ii,model_fn in tqdm(enumerate(model_files), desc="Model", total=len(model_files), disable=False if rank==0 else True):
        print(f"Model {ii}: {model_fn}", flush=True)

        model_channels = int(model_fn.split("channels_")[1].split("_")[0])
        
        #tts = int(model_fn.split("tts_")[1].split("_")[0])
        
        #n_validation_models = tts * 80 // 100 if tts > 1 else 1
        

        network_opt = dict(img_resolution=64, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
                        model_channels=model_channels, channel_mult=[2,2,2], attn_resolutions=[], #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                        embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                        )
        
        #cut_factor = 256^3 =  (2*2*64)^3 = (2^cut)^3 * 64^3 
        cut_factor = int(torch.log2(torch.tensor(256 // network_opt["img_resolution"])).item())

        network = SongUNet
        noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
        
        
        netG = GaussianDiffusion(
                network=network,
                network_opt=network_opt,
                noise_schedule_opt=noise_schedule_opt,
                loss_fn = None,
                learning_rate=1e-4,
                scheduler=False,
                rank=rank,
            )

        netG.load_network(model_root + model_fn)

        for i,(T21, delta, vbv, T21_lr, labels) in tqdm(enumerate(test_dataloader), desc=model_fn, total=len(test_dataloader), disable=False if rank==0 else True):
            T21_lr, T21, delta, vbv = T21_lr.to(device), T21.to(device), delta.to(device), vbv.to(device)

            T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
            delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
            vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
            T21_lr = get_subcubes(cubes=T21_lr, cut_factor=cut_factor)

            T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
            T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
            T21_lr_orig = T21_lr #to save later
            T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)

            T21_lr, T21_lr_ups_mean, T21_lr_ups_std = normalize(T21_lr, mode="standard") #model_12 trained with factor 2 for all inputs, but should only be 2 for HR inputs in future.
            T21, T21_mean, T21_std = normalize(T21, mode="standard", x_mean=T21_lr_mean, x_std=T21_lr_std, factor=2.)
            delta, delta_mean, delta_std = normalize(delta, mode="standard") #model_12 trained with factor 2 for all inputs, but should only be 2 for HR inputs in future.
            vbv, vbv_mean, vbv_std = normalize(vbv, mode="standard", factor=2.) #model_12 trained with factor 2 for all inputs, but should only be 2 for HR inputs in future.

            #print("Shapes: ", T21.shape, "T21_lr: ", T21_lr.shape, "delta: ", delta.shape, "vbv: ", vbv.shape, flush=True)

            sub_data = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr, T21_lr_orig, T21_lr_mean, T21_lr_std, delta_mean, delta_std, vbv_mean, vbv_std)
            sub_dataloader = torch.utils.data.DataLoader(sub_data, batch_size=4, shuffle=False, sampler = None)

            for j,(T21, delta, vbv, T21_lr, T21_lr_orig, T21_lr_mean, T21_lr_std, delta_mean, delta_std, vbv_mean, vbv_std) in tqdm(enumerate(sub_dataloader), desc="Sub", total=len(sub_dataloader), disable=False if rank==0 else True):
                #print("Shapes: ", T21.shape, "T21_lr: ", T21_lr.shape, "delta: ", delta.shape, "vbv: ", vbv.shape, flush=True)
                if ema:
                    with netG.ema.average_parameters():
                        if i==j==rank==0:
                            print("EMA evaluation ", flush=True)
                        sampling_time = time.time()
                        mean_x = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=100, eps=1e-3, clip_denoised=False, verbose=False)
                        #mean_x = netG.sample.ode_sampler(netG=netG, x_lr=T21_lr_norm, conditionals=[delta_norm, vbv_norm], class_labels=None, atol=nsteps, rtol=nsteps, eps=1e-3).to(rank)
                        sampling_time = time.time() - sampling_time
                else:
                    if i==j==rank==0:
                        print("Disable EMA evaluation ", flush=True)
                    sampling_time = time.time()
                    mean_x = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=100, eps=1e-3, clip_denoised=False, verbose=False)
                    #mean_x = netG.sample.ode_sampler(netG=netG, x_lr=T21_lr_norm, conditionals=[delta_norm, vbv_norm], atol=nsteps, rtol=nsteps, eps=1e-3).to(rank)
                    sampling_time = time.time() - sampling_time
                
                mean_x = invert_normalization(mean_x[:,-1:], mode="standard", x_mean=T21_lr_mean, x_std=T21_lr_std, factor=2.)
                T21 = invert_normalization(T21, mode="standard", x_mean=T21_lr_mean, x_std=T21_lr_std, factor=2.)
                
                delta = invert_normalization(delta, mode="standard", x_mean=delta_mean, x_std=delta_std)
                vbv = invert_normalization(vbv, mode="standard", x_mean=vbv_mean, x_std=vbv_std)

                if i==j==0:
                    T21_pred = mean_x[:,-1:]
                    T21_true = T21

                    T21_lr_orig_all = T21_lr_orig
                    delta_all = delta
                    vbv_all = vbv
                else:
                    T21_pred = torch.cat([T21_pred, mean_x[:,-1:]], dim=0)
                    T21_true = torch.cat([T21_true, T21], dim=0)

                    T21_lr_orig_all = torch.cat([T21_lr_orig_all, T21_lr_orig], dim=0)
                    delta_all = torch.cat([delta_all, delta], dim=0)
                    vbv_all = torch.cat([vbv_all, vbv], dim=0)

                if j==0:
                    sampling_time_j = torch.tensor(sampling_time, device=device).unsqueeze(0).unsqueeze(0)
                    MSE_j = torch.mean(torch.square(mean_x[:,-1:] - T21), dim=(1,2,3,4), keepdim=False).unsqueeze(0)
                else:
                    sampling_time_tmp = torch.tensor(sampling_time, device=device).unsqueeze(0).unsqueeze(0)
                    sampling_time_j = torch.cat([sampling_time_j, sampling_time_tmp], dim=1)
                    MSE_tmp = torch.mean(torch.square(mean_x[:,-1:] - T21), dim=(1,2,3,4), keepdim=False).unsqueeze(0)
                    MSE_j = torch.cat([MSE_j, MSE_tmp], dim=1)    
            if i==0:
                sampling_time_i = sampling_time_j
                MSE_i = MSE_j
            else:
                sampling_time_i = torch.cat([sampling_time_i, sampling_time_j], dim=0)
                MSE_i = torch.cat([MSE_i, MSE_j], dim=0)


        #print("Sampling time: ", sampling_time_i.shape, flush=True)
        
        if multi_gpu:
            sampling_time_tensor_list = [torch.zeros_like(sampling_time_i) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=sampling_time_tensor_list, tensor=sampling_time_i)
            sampling_time = torch.cat(sampling_time_tensor_list, dim=0)

            MSE_tensor_list = [torch.zeros_like(MSE_i) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=MSE_tensor_list, tensor=MSE_i)
            MSE = torch.cat(MSE_tensor_list, dim=0)
            torch.distributed.barrier()

            T21_pred_tensor_list = [torch.zeros_like(T21_pred) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=T21_pred_tensor_list, tensor=T21_pred)
            T21_pred = torch.cat(T21_pred_tensor_list, dim=0)

            T21_true_tensor_list = [torch.zeros_like(T21_true) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=T21_true_tensor_list, tensor=T21_true)
            T21_true = torch.cat(T21_true_tensor_list, dim=0)

            T21_lr_orig_all_tensor_list = [torch.zeros_like(T21_lr_orig_all) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=T21_lr_orig_all_tensor_list, tensor=T21_lr_orig_all)
            T21_lr_orig_all = torch.cat(T21_lr_orig_all_tensor_list, dim=0)

            delta_all_tensor_list = [torch.zeros_like(delta_all) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=delta_all_tensor_list, tensor=delta_all)
            delta_all = torch.cat(delta_all_tensor_list, dim=0)

            vbv_all_tensor_list = [torch.zeros_like(vbv_all) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=vbv_all_tensor_list, tensor=vbv_all)
            vbv_all = torch.cat(vbv_all_tensor_list, dim=0)

        else:
            sampling_time = sampling_time_i
            MSE = MSE_i
            T21_pred = T21_pred
            T21_true = T21_true
            T21_lr_orig_all = T21_lr_orig_all
            delta_all = delta_all
            vbv_all = vbv_all
        
        if device == "cuda:0" or device == "cpu":
            torch.save(
                obj = dict(MSE = MSE, sampling_time = sampling_time, T21_pred = T21_pred, T21_true = T21_true, T21_lr = T21_lr_orig_all, delta = delta_all, vbv = vbv_all),
                f=path + "/analysis/model_13/" + model_fn.split(".")[0] + "_mse_sampletime.pth"
                )
        torch.distributed.barrier()
        print("Sampling time: ", sampling_time.shape, flush=True)
        #print("MSE_i: ", MSE_i.shape, "MSE_j: ", MSE_j.shape, flush=True)
        print("MSE: ", MSE.shape, flush=True)
        
            
        
        
        if False:#rank==0:#False:#str(mse_all.device) == "cuda:0":
            
            

            print("Plotting...", flush=True)
            
            MSE_k = MSE.cpu().numpy()
            mse_all = MSE.cpu().numpy()
            mse_all_mean = mse_all.mean(axis=(0,-1), keepdims=True)
            #mse_all_resid = ((MSE_k[:1,:1,:] - mse_all_mean[:1,-1:,:1])**2)#.sum(axis=1, keepdims=True)
            #print("mse_all_resid.shape: ", mse_all_resid.shape, flush=True)
            #try:
            #    #model closest to mean in last batch MSE_k
            #    pass
            #except:
            #    print("Failed to calculate residuals for last batch...", flush=True)
            #    mse_all_resid = ((mse_all - mse_all_mean)**2).sum(axis=1, keepdims=True)
            
            #repeats_idx, model_idx = np.argmin(mse_all_resid, keepdims=False, axis=-1).flatten()
            #print("repeats_idx: ", repeats_idx, "model_idx: ", model_idx, "MSE_k: ", MSE_k.shape, flush=True)
            
            fig,ax = plt.subplots(1,5,figsize=(25,5))

            #mse_all_mean = mse_all.mean(axis=0)
            for i in range(repeats):
                ax[0].plot(steps, mse_all[i]**0.5, c="k", alpha=0.02)
            ax[0].plot(steps, mse_all_mean[0,:,0]**0.5, c="r", alpha=1., label="Mean")
            ax[0].set_xscale("log")

            ax[0].grid()
            ax[0].set_xlabel("steps")
            ax[0].set_ylabel("RMSE Validation Set")
            ax[0].legend()
            
            if not loaded:
                #show mean_x[model_idx,-1] example next to true T21
                model_idx = 0
                shape = mean_x.shape
                slice_idx = shape[-1]//2

                T21 = T21_norm #if invert_norm is not used else T21

                vmin = T21[model_idx,0].min().cpu().detach().numpy()
                vmax = T21[model_idx,0].max().cpu().detach().numpy()
                ax[1].imshow(T21[model_idx,0,slice_idx].cpu().detach().numpy(), vmin=vmin, vmax=vmax)
                ax[1].set_title("True (Example)")
                ax[2].imshow(mean_x[model_idx,-1,slice_idx].cpu().detach().numpy(), vmin=vmin, vmax=vmax)
                rmse = torch.nn.functional.mse_loss(mean_x[model_idx,-1], T21[model_idx,0]).item()**0.5
                ax[2].set_title(f"Predicted RMSE={rmse:.3f} (Example)")
                #clamp prediction
                mean_x_clamped = mean_x.clamp(min=T21.min().item())
                ax[3].imshow(mean_x_clamped[model_idx,-1,slice_idx].cpu().detach().numpy(), vmin=vmin, vmax=vmax)
                rmse_clamped = torch.nn.functional.mse_loss(mean_x_clamped[model_idx,-1], T21[model_idx,0]).item()**0.5
                ax[3].set_title(f"Predicted RMSE={rmse_clamped:.3f} (Clamped) (Example)")
                
                #show histograms
                ax[4].hist(T21[model_idx,0].cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="True", density=True)
                ax[4].hist(mean_x[model_idx,-1].cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="Predicted", density=True)
                ax[4].hist(mean_x_clamped[model_idx,-1].cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="Predicted (Clamped)", density=True)
                ax[4].set_xlabel("T21")
                ax[4].legend(title=f"Steps: {nsteps}")

                lab = shape[-1]
            else:
                lab = "None"
            
            fig_fn = path + "/plots/"+ model_pth.split("/")[-1].split(".")[0] + f"_sample_rmse_{lab}_2.png"
            plt.savefig(fig_fn)

    if multi_gpu:#world_size > 1:
        destroy_process_group()    

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    if multi_gpu:
        print("Spawning processes", flush=True)#100,300,500,1000
        start_time = time.time()
        #for i in [0,]:
        mp.spawn(sample_error, args=(True, multi_gpu), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
        #mp.spawn(sample_error, args=(1, [1e-5, ], i, True, multi_gpu), nprocs=world_size)
        print("Time taken: {0:.2f}".format(time.time()-start_time), flush=True)
    else:
        print("Running single process", flush=True)
        sample_error(rank=0, ema=True, multi_gpu=False)



    
    