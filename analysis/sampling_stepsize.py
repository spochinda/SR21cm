import os
import matplotlib.pyplot as plt
import torch
import torch.distributed
from models.diffusion import *
from models.model import *
from models.utils import *
from models.model_edm import SongUNet
#from models.train import plot_checkpoint

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
def sample_steps(rank, repeats=10, steps=[10,20,40,], cut_factor=0, ode_sampling=False, ema=False, multi_gpu=False):
    if multi_gpu:
        world_size = torch.cuda.device_count()
        ddp_setup(rank, world_size=world_size)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    path = os.getcwd().split("/21cmGen")[0] + "/21cmGen"
    model_pth = path + "/trained_models/model_14/DDPMpp_standard_channels_32_tts_70_VPSDE_3.pth"
    fn = path + "/analysis/model_14/" + model_pth.split("/")[-1].split(".")[0] + "_sample_steps.pth"

    try:
        print("Loading MSE...", flush=True)
        MSE = torch.load(fn, map_location=device)
        
        loaded = True
        #make error to go to except:
        raise Exception("Manually aborting loading MSE...")
    except:
        print("Either aborted or failed to load MSE: ", fn, flush=True)
        loaded = False
        batch_size = 1

        test_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/", 
                                            redshifts=[10,], IC_seeds=list(range(72,80)), upscale=4, cut_factor=0, transform=False, norm_lr=True, device=device)
        test_dataloader = torch.utils.data.DataLoader(test_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True,
                                                            sampler = DistributedSampler(test_data_module) if multi_gpu else None)                
            
        model_channels = int(model_pth.split("/")[-1].split("channels_")[1].split("_")[0])

        network_opt = dict(img_resolution=64, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
                        model_channels=model_channels, channel_mult=[2,2,2], attn_resolutions=[], #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                        embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                        )
        
        #cut_factor = int(torch.log2(torch.tensor(256 // network_opt["img_resolution"])).item())

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

        netG.load_network(model_pth)

        netG.model_name = model_pth.split("/")[-1].split(".")[0]


        
        for i in range(repeats):
            if rank==0:
                print("repeat: ", i, flush=True)
            
            for j,nsteps in tqdm(enumerate(steps), desc="Steps", total=len(steps), disable=False if rank==0 else True):
                mse = torch.tensor(0., device=rank)
                for k,(T21, delta, vbv, T21_lr, labels) in tqdm(enumerate(test_dataloader), desc=model_pth.split("/")[-1], total=len(test_dataloader), disable=True):
                    T21_lr, T21, delta, vbv = T21_lr.to(device), T21.to(device), delta.to(device), vbv.to(device)

                    T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
                    delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
                    vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
                    T21_lr = get_subcubes(cubes=T21_lr, cut_factor=cut_factor)

                    T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
                    T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
                    T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)

                    T21_lr, _, _ = normalize(T21_lr, mode="standard") 
                    T21, _, _ = normalize(T21, mode="standard", x_mean=T21_lr_mean, x_std=T21_lr_std, factor=2.) #factor=2 for all inputs but should only be 2 for HR input in future
                    delta, _, _ = normalize(delta, mode="standard")
                    vbv, _, _ = normalize(vbv, mode="standard")

                    if rank==0:
                        print("Shapes: ", T21.shape, delta.shape, vbv.shape, T21_lr.shape, flush=True)

                    sub_data = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std)
                    sub_dataloader = torch.utils.data.DataLoader(sub_data, batch_size=4, shuffle=False, sampler = None)

                    for z,(T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std) in tqdm(enumerate(sub_dataloader), desc="Sub", total=len(sub_dataloader), disable=False if rank==0 else True):

                        if ema:
                            with netG.ema.average_parameters():
                                if i==j==k==z==rank==0:
                                    print("EMA evaluation ", flush=True)
                                if not ode_sampling:
                                    T21_pred_z = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=nsteps, eps=1e-3, clip_denoised=False, verbose=False)
                                else:
                                    T21_pred_z = netG.sample.ode_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=None, atol=nsteps, rtol=nsteps, eps=1e-3).to(rank)
                        else:
                            if i==j==k==z==rank==0:
                                print("Disable EMA evaluation ", flush=True)
                            if not ode_sampling:
                                T21_pred_z = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=nsteps, eps=1e-3, clip_denoised=False, verbose=False)
                            else:
                                T21_pred_z = netG.sample.ode_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], atol=nsteps, rtol=nsteps, eps=1e-3).to(rank)

                        T21_pred_z = invert_normalization(T21_pred_z[:,-1:], mode="standard", x_mean=T21_lr_mean, x_std=T21_lr_std, factor=2.)
                        T21_z = invert_normalization(T21, mode="standard", x_mean=T21_lr_mean, x_std=T21_lr_std, factor=2.)


                        if z==0:
                            MSE_z = torch.mean(torch.square(T21_pred_z[:,-1:] - T21_z), dim=(1,2,3,4), keepdim=False).unsqueeze(0).unsqueeze(0)
                        else:
                            MSE_tmp = torch.mean(torch.square(T21_pred_z[:,-1:] - T21_z), dim=(1,2,3,4), keepdim=False).unsqueeze(0).unsqueeze(0)
                            MSE_z = torch.cat([MSE_z, MSE_tmp], dim=-1)
                        
                        if k==z==0:
                            T21_pred_k = T21_pred_z
                            T21_k = T21_z
                        else:
                            T21_pred_k = torch.cat([T21_pred_k, T21_pred_z], dim=0)
                            T21_k = torch.cat([T21_k, T21_z], dim=0)

                    if k==0:
                        MSE_k = MSE_z
                    else:
                        MSE_k = torch.cat([MSE_k, MSE_z], dim=-1)

                if j==0:
                    MSE_j = MSE_k
                else:
                    MSE_j = torch.cat([MSE_j, MSE_k], dim=-2)
                
                T21_pred_tensor_list = [torch.zeros_like(T21_pred_k) for _ in range(world_size)]
                torch.distributed.all_gather(tensor_list=T21_pred_tensor_list, tensor=T21_pred_k)
                T21_pred = torch.cat(T21_pred_tensor_list, dim=0)
                
                T21_tensor_list = [torch.zeros_like(T21_k) for _ in range(world_size)]
                torch.distributed.all_gather(tensor_list=T21_tensor_list, tensor=T21_k)
                T21 = torch.cat(T21_tensor_list, dim=0)
                torch.distributed.barrier()
                torch.save(obj=dict(T21 = T21, T21_pred = T21_pred), f=path + "/analysis/model_14/" + model_pth.split("/")[-1].split(".")[0] + f"_sample_steps_{nsteps}.pth")

            if i==0:
                MSE_i = MSE_j
            else:
                MSE_i = torch.cat([MSE_i, MSE_j], dim=-3)
        
        print("MSE_i: ", MSE_i.shape, "MSE_j: ", MSE_j.shape, "MSE_k: ", MSE_k.shape, flush=True)
        
        if multi_gpu:
            MSE_tensor_list = [torch.zeros_like(MSE_i) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=MSE_tensor_list, tensor=MSE_i)
            MSE = torch.cat(MSE_tensor_list, dim=-1)
            torch.distributed.barrier()
        else:
            MSE = MSE_i
        
        torch.save(obj=dict(MSE=MSE, steps=steps), f=fn)

            
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
        mp.spawn(sample_steps, args=(1, [10, 20, 30, 40, 60, 100, 300, 600, 1000], 2, False, True, multi_gpu), nprocs=world_size) #3.5 hrs
        #mp.spawn(sample_steps, args=(1, [1e-5, ], i, True, multi_gpu), nprocs=world_size)
        print("Time taken: {0:.2f}".format(time.time()-start_time), flush=True)
    else:
        print("Running single process", flush=True)
        sample_steps(rank=0, repeats=2, steps=[10,20,40,], cut_factor=0, ode_sampling=False, ema=True, multi_gpu=False)



    
    