#import contextlib
import contextlib
import torch
import socket
from datetime import datetime, timedelta
"""
def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
            ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=trace_handler,
        with_stack=True,
        ) as prof:
    with torch.autograd.profiler.record_function("## data prep in loop ##"):                
        torch.nn.Upsample(scale_factor=4, mode='trilinear')(torch.randn(1,1,64,64,64))


prof.export_memory_timeline("memory_trace.html")
"""
import torch 
import torch.distributed
import torch.utils

from utils import *
from diffusion import *
from model import *
from model_edm import SongUNet
from loss import *
from sde_lib import VPSDE


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS

import torch.multiprocessing as mp
import torch.utils
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import time
import sys
import os


#from torch_ema import ExponentialMovingAverage


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

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   #timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_"#{timestamp}"
   print("inside trace_handler, rank: ", torch.distributed.get_rank(), flush=True)
   # Construct the trace file.
   #prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")





def train_step(netG, epoch, train_dataloader, volume_reduction = False, split_batch=True, one_box_per_epoch = True, device="cpu", multi_gpu = False,):
    """
    Train the model
    """
    netG.model.train()
    
    avg_loss = torch.tensor(0.0, device=device)

    if multi_gpu:
        train_dataloader.sampler.set_epoch(epoch) #fix for ddp loaded checkpoint?

    for i,(T21, delta, vbv, T21_lr, labels) in enumerate(train_dataloader):

        if volume_reduction:
            if netG.network_opt["label_dim"] > 0:
                cut_factor = torch.randint(low=1, high=3, size=(1,), device=device)#.item()
            else:
                cut_factor = torch.randint(low=2, high=3, size=(1,), device=device)#always cut 2
                #cut_factor = torch.randint(low=1, high=2, size=(1,), device=device)#always cut 1

            T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
            delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
            vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
            T21_lr = get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
                        
            volume_frac = 1/(2**cut_factor)**3 #* torch.ones(size=(T21.shape[0],), device=device)
            multiple_redshifts = False
            labels = volume_frac #torch.tensor([volume_frac,], device=device) #torch.cat([labels, volume_frac], dim=1) if multiple_redshifts else

            

        if netG.network_opt["label_dim"] > 0:
            labels = labels
        else:
            labels = None
        
        #with torch.no_grad():
        #T21_lr_min = torch.amin(T21_lr, dim=(1,2,3,4), keepdim=True)
        #T21_lr_max = torch.amax(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)
        
        T21_lr, T21_lr_ups_mean, T21_lr_ups_std = normalize(T21_lr, mode="standard", factor=3.)#, factor=2.)
        T21, T21_mean, T21_std = normalize(T21, mode="standard", factor=3., x_mean=T21_lr_mean, x_std=T21_lr_std)#, factor=2.) ####
        delta, delta_mean, delta_std = normalize(delta, mode="standard", factor=3.)#, factor=2.)
        vbv, vbv_mean, vbv_std = normalize(vbv, mode="standard", factor=3.)#, factor=2.)
        T21, delta, vbv , T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=1) #support device
        if split_batch: #split subcube minibatch into smaller mini-batches for memory
            sub_data = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr)
            sub_dataloader = torch.utils.data.DataLoader(sub_data, batch_size=4, shuffle=False, sampler = None) # (2**(cut_factor.item()-1))**3 // 2 #4

            for j,(T21, delta, vbv, T21_lr) in enumerate(sub_dataloader):
                #print(f"Sub {j}", flush=True)

                netG.optG.zero_grad()
                loss = netG.loss_fn(net=netG, images=T21, conditionals=[delta, vbv, T21_lr],
                                    labels=labels, augment_pipe=None,
                                    )
                avg_loss = avg_loss + loss * T21.shape[0]  #add avg loss per mini-batch to accumulate total batch loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
                netG.optG.step()        
                netG.ema.update() #Update netG.model with exponential moving average  
                
        else:
            loss = netG.loss_fn(net=netG, images=T21, conditionals=[delta, vbv, T21_lr],
                                    labels=labels, augment_pipe=None,
                                    )
            avg_loss = avg_loss + loss * T21.shape[0]  #add avg loss per mini-batch to accumulate total batch loss
            #with torch.autograd.profiler.record_function("## backward ##") if False else contextlib.nullcontext(): 
            loss.backward()
            #with torch.autograd.profiler.record_function("## optimizer ##") if False else contextlib.nullcontext():            
            torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
            netG.optG.zero_grad()
            netG.optG.step()        
            #with torch.autograd.profiler.record_function("## ema ##") if False else contextlib.nullcontext():                
            netG.ema.update() #Update netG.model with exponential moving average

        
        if (str(device)=="cuda:0") or (str(device)=="cpu"):
            if False: #i%(len(train_data)//16) == 0:
                print(f"Batch {i} of {len(train_data)} batches")

        if one_box_per_epoch:
            break #only do one box for now
    
    if multi_gpu:
        torch.distributed.all_reduce(tensor=avg_loss, op=torch.distributed.ReduceOp.SUM) #total loss=sum(average total batch loss per gpu)

    netG.loss.append(avg_loss.item())
    
    return avg_loss.item()

@torch.no_grad()
def plot_sigmas(T21, T21_pred=None, netG=None, path = "", quantiles=[0.16, 0.5, 0.84], **kwargs):
    T21 = T21.detach().cpu()
    T21_pred = T21_pred.detach().cpu()


    RMSE = ((T21 - T21_pred)**2).mean(dim=(1,2,3,4))**0.5
    RMSE_slice = ((T21 - T21_pred)**2).mean(dim=(1,3,4))**0.5

    row = 4
    col = len(quantiles)

    fig = plt.figure(figsize=(5*col,5*row))
    gs = GS(row, col, figure=fig,)
    sgs = SGS(2,col, gs[-1,:], height_ratios=[4,1], hspace=0.)

    for i,quantile in enumerate(quantiles):
        q = torch.quantile(input=RMSE, q=quantile, dim=(-1),)
        #find model index closest to q
        idx = torch.argmin(torch.abs(RMSE - q), dim=-1)
        #find slice closest to q
        idx_slice = torch.argmin(torch.abs(RMSE_slice[idx] - q), dim=-1)
        
        ax_true = fig.add_subplot(gs[0,i])
        ax_pred = fig.add_subplot(gs[1,i])
        vmin = min(T21[idx,0, idx_slice].min().item(), T21_pred[idx,0,idx_slice].min().item())
        vmax = max(T21[idx,0, idx_slice].max().item(), T21_pred[idx,0,idx_slice].max().item())
        ax_true.imshow(T21[idx,0,idx_slice], vmin=vmin, vmax=vmax)
        ax_true.set_title("True")
        ax_pred.imshow(T21_pred[idx,0,idx_slice], vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"Predicted (q{quantile:.3f} RMSE={q:.4f})")

        sgs = SGS(2,col, gs[2,:], height_ratios=[4,1], hspace=0.)
        ax_hist = fig.add_subplot(sgs[0,i], sharey=None if i==0 else ax_hist)
        ax_hist_resid = fig.add_subplot(sgs[1,i], sharex=ax_hist, sharey=None if i==0 else ax_hist_resid)
        hist_min = min(T21[idx,0].min().item(), T21_pred[idx,0].min().item())
        hist_max = max(T21[idx,0].max().item(), T21_pred[idx,0].max().item())
        bins = np.linspace(hist_min, hist_max, 100)
        hist_true, _ = np.histogram(T21[idx,0,:,:,:].flatten(), bins=bins)
        hist_pred, _ = np.histogram(T21_pred[idx,0,:,:,:].flatten(), bins=bins)  # Reuse the same bins for consistency
        #hist_true = hist_true / np.sum(hist_true)
        #hist_pred = hist_pred / np.sum(hist_pred)
        ax_hist.bar(bins[:-1], hist_true, width=bins[1] - bins[0], alpha=0.5, label="T21 HR", )#color='orange')
        ax_hist.bar(bins[:-1], hist_pred, width=bins[1] - bins[0], alpha=0.5, label="T21 SR", )#color='blue')
        ax_hist.legend()
        hist_resid = np.abs(hist_true - hist_pred)
        #hist_resid = hist_resid / np.sum(hist_resid)
        ax_hist_resid.bar(bins[:-1], hist_resid, width=bins[1] - bins[0], alpha=0.5, label="|Residuals|", color='k')
        ax_hist_resid.legend()
        ax_hist_resid.set_xlabel("T21 [mK]")
        
        sgs = SGS(2,col, gs[3,:], height_ratios=[4,1], hspace=0.)
        ax_dsq = fig.add_subplot(sgs[0,i], sharey=None if i==0 else ax_dsq)
        ax_dsq_resid = fig.add_subplot(sgs[1,i], sharex=ax_dsq, sharey=None if i==0 else ax_dsq_resid)
        k_vals_true, dsq_true  = calculate_power_spectrum(T21[idx:idx+1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
        k_vals_pred, dsq_pred  = calculate_power_spectrum(T21_pred[idx:idx+1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
        ax_dsq.plot(k_vals_true, dsq_true[0,0], label="T21 HR", ls='solid', lw=2)
        ax_dsq.plot(k_vals_pred, dsq_pred[0,0], label="T21 SR", ls='solid', lw=2)
        ax_dsq.set_ylabel('$\Delta^2(k)_{{21}}$ [mK$^2$]')
        #ax_dsq.set_xlabel('$k$ [h/Mpc]')
        ax_dsq.set_yscale('log')
        ax_dsq.grid()
        ax_dsq.legend()

        dsq_resid = torch.abs(dsq_pred[0,0] - dsq_true[0,0])
        ax_dsq_resid.plot(k_vals_true, dsq_resid, lw=2, color='k')
        ax_dsq_resid.set_ylabel("|Residuals| [mK$^2$]")
        ax_dsq_resid.set_xlabel("$k$ [h/Mpc]")
        ax_dsq_resid.set_yscale('log')
        ax_dsq_resid.grid()
        
    plt.savefig(path + netG.model_name + f"_quantiles_epoch_{len(netG.loss)}.png")
    plt.close()


@torch.no_grad()
def plot_checkpoint(T21, delta, vbv, T21_lr, T21_pred, netG=None, MSE=None, epoch=None, path = None, device="cpu"):
    #find model_idx with MSE closest to MSE
    MSE_sample = torch.mean(torch.square(T21_pred - T21),dim=(1,2,3,4), keepdim=False)
    model_idx = torch.argmin(torch.abs(MSE_sample - MSE)).item()
    #print("Model_idx: ", model_idx, flush=True)

    k_vals_true, dsq_true  = calculate_power_spectrum(T21, Lpix=3, kbins=100, dsq = True, method="torch", device=device)
    k_vals_pred, dsq_pred  = calculate_power_spectrum(T21_pred, Lpix=3, kbins=100, dsq = True, method="torch", device=device)
    
    #detatch and send to cpu
    T21 = T21.detach().cpu()
    T21_lr = T21_lr.detach().cpu()
    delta = delta.detach().cpu()
    vbv = vbv.detach().cpu()
    T21_pred = T21_pred.detach().cpu()
    #T21_lr_stats = T21_lr_stats.detach().cpu()
    
    k_vals_true = k_vals_true.detach().cpu()
    dsq_true = dsq_true.detach().cpu()
    k_vals_pred = k_vals_pred.detach().cpu()
    dsq_pred = dsq_pred.detach().cpu()

    
    slice_idx = T21.shape[-3]//2

    fig = plt.figure(figsize=(15,15))
    gs = GS(3, 3, figure=fig,) #height_ratios=[1,1,1.5])

    #row 1 (inputs)
    ax_delta = fig.add_subplot(gs[0,0])#, wspace = 0.2)
    ax_vbv = fig.add_subplot(gs[0,1])
    ax_T21_lr = fig.add_subplot(gs[0,2])

    vmin = torch.amin(delta[model_idx,0,slice_idx]).item()
    vmax = torch.amax(delta[model_idx,0,slice_idx]).item()
    ax_delta.imshow(delta[model_idx,0,slice_idx], vmin=vmin, vmax=vmax)
    ax_delta.set_title("Delta (input)")

    vmin = torch.amin(vbv[model_idx,0,slice_idx]).item()
    vmax = torch.amax(vbv[model_idx,0,slice_idx]).item()
    ax_vbv.imshow(vbv[model_idx,0,slice_idx], vmin=vmin, vmax=vmax)
    ax_vbv.set_title("Vbv (input)")
    ax_T21_lr.imshow(T21_lr[model_idx,0,slice_idx],)
    ax_T21_lr.set_title("T21 LR (input)")

    #row 2 (outputs)
    sgs = SGS(1,4, gs[1,:], width_ratios=[10,10,10,1])
    #ax_dsq = fig.add_subplot(sgs_dsq[0])
    
    ax_T21 = fig.add_subplot(sgs[0])
    ax_T21_pred = fig.add_subplot(sgs[1])
    ax_T21_resid = fig.add_subplot(sgs[2])
    ax_T21_resid_cbar = fig.add_subplot(sgs[3])

    #ax_T21 = fig.add_subplot(gs[1,0])
    #ax_T21_pred = fig.add_subplot(gs[1,1])
    #ax_T21_resid = fig.add_subplot(gs[1,2])

    vmin = torch.amin(T21[model_idx,0,slice_idx]).item()
    vmax = torch.amax(T21[model_idx,0,slice_idx]).item()
    ax_T21.imshow(T21[model_idx,0,slice_idx], vmin=vmin, vmax=vmax)
    ax_T21.set_title("T21 HR (Real)")
    
    ax_T21_pred.imshow(T21_pred[model_idx,0,slice_idx], vmin=vmin, vmax=vmax)
    ax_T21_pred.set_title(f"T21 SR (Generated) epoch {len(netG.loss)}")

    T21_resid = T21_pred - T21
    slice_std  = torch.std(T21_resid[model_idx,0,slice_idx]).item()
    slice_mean = torch.mean(T21_resid[model_idx,0,slice_idx]).item()
    ax_T21_resid.imshow(T21_resid[model_idx,0,slice_idx], vmin=slice_mean-3*slice_std, vmax=slice_mean+3*slice_std)
    ax_T21_resid.set_title("T21 SR - T21 HR")

    cbar = plt.colorbar(ax_T21_resid.imshow(T21_resid[model_idx,0,slice_idx], vmin=slice_mean-3*slice_std, vmax=slice_mean+3*slice_std), cax=ax_T21_resid_cbar, label="Residuals [mK]")


    #row 3 (statistics)
    sgs = SGS(2,3, gs[2,:], height_ratios=[4,1], hspace=0.)

    ax_hist = fig.add_subplot(sgs[0,0])
    ax_hist_resid = fig.add_subplot(sgs[1,0], sharex=ax_hist)
    
    ax_hist_shift = fig.add_subplot(sgs[:,1])
    
    ax_dsq = fig.add_subplot(sgs[0,2])
    ax_dsq_resid = fig.add_subplot(sgs[1,2], sharex=ax_dsq)


    
    max_value = max(torch.amax(T21[model_idx,0,:,:,:]).item(), torch.amax(T21_pred[model_idx,0,:,:,:]).item())
    min_value = min(torch.amin(T21[model_idx,0,:,:,:]).item(), torch.amin(T21_pred[model_idx,0,:,:,:]).item())
    bins = np.linspace(min_value, max_value, 100)
    hist_true, _ = np.histogram(T21[model_idx,0,:,:,:].flatten(), bins=bins)
    hist_pred, _ = np.histogram(T21_pred[model_idx,0,:,:,:].flatten(), bins=bins)  # Reuse the same bins for consistency
    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)
    ax_hist.bar(bins[:-1], hist_true, width=bins[1] - bins[0], alpha=0.5, label="T21 HR", )#color='orange')
    ax_hist.bar(bins[:-1], hist_pred, width=bins[1] - bins[0], alpha=0.5, label="T21 SR", )#color='blue')

    hist_resid = np.abs(hist_true - hist_pred)
    hist_resid = hist_resid / np.sum(hist_resid)
    ax_hist_resid.bar(bins[:-1], hist_resid, width=bins[1] - bins[0], alpha=0.5, label="|Residuals|", color='k')

    #ax.bar(bins[:-1], diff, width=bins[1] - bins[0], color='skyblue')

    #ax_hist.hist(T21[model_idx,0,:,:,:].flatten(), bins=100, alpha=0.5, label="T21 HR", density=True)
    #ax_hist.hist(T21_pred[model_idx,0,:,:,:].flatten(), bins=100, alpha=0.5, label="T21 SR", density=True)
    ax_hist.set_xlabel("$T_{{21}}$ [mK]")
    ax_hist.set_ylabel("PDF")
    ax_hist.legend()
    ax_hist.set_title(f"Model {model_idx} RMSE={MSE_sample[model_idx].item()**0.5:.3f}mK \n(Sample RMSE={MSE**0.5:.3f}mK)")
    ax_hist.grid()

    ax_hist_resid.set_xlabel("$T_{{21}}$ [mK]")
    ax_hist_resid.set_ylabel("|Residuals| [mk]")
    ax_hist_resid.grid()



    mean_T_pred = torch.mean(T21_pred, dim=(1,2,3,4), keepdim=False)
    mean_T_true = torch.mean(T21, dim=(1,2,3,4), keepdim=False)
    ax_hist_shift.hist((mean_T_pred-mean_T_true).flatten(), bins=100, alpha=0.5, label="mean(Predicted)-mean(True)", density=True)
    ax_hist_shift.set_xlim(-1,1)
    ax_hist_shift.set_xlabel("Sample difference in mean [mK]")
    ax_hist_shift.set_ylabel("PDF")
    ax_hist_shift.grid()




    ax_dsq.plot(k_vals_true, dsq_true[model_idx,0], label="T21 HR", ls='solid', lw=2)
    ax_dsq.plot(k_vals_pred, dsq_pred[model_idx,0], label="T21 SR", ls='solid', lw=2)
    ax_dsq.set_ylabel('$\Delta^2(k)_{{21}}$ [mK$^2$]')
    #ax_dsq.set_xlabel('$k$ [h/Mpc]')
    ax_dsq.set_yscale('log')
    ax_dsq.grid()
    ax_dsq.legend()

    dsq_resid = torch.abs(dsq_pred[model_idx,0] - dsq_true[model_idx,0])
    ax_dsq_resid.plot(k_vals_true, dsq_resid, lw=2, color='k')
    ax_dsq_resid.set_ylabel("|Residuals| [mK$^2$]")
    ax_dsq_resid.set_xlabel("$k$ [h/Mpc]")
    ax_dsq_resid.set_yscale('log')
    ax_dsq_resid.grid()

    plt.savefig(path + netG.model_name + f"_epoch_{len(netG.loss)}.png".format(epoch))
    plt.close()

@torch.no_grad()
def validation_step_v2(netG, validation_dataloader, split_batch = True, device="cpu", multi_gpu=False):
    assert netG.noise_schedule_opt["schedule_type"] == "VPSDE", "Only VPSDE sampler supported for validation_step_v2"

    netG.model.eval()
    for i,(T21, delta, vbv, T21_lr, labels) in tqdm(enumerate(validation_dataloader), desc='validation loop', total=len(validation_dataloader), disable=False if str(device)=="cuda:0" else True):
        # Rest of the code
        # alternating cut_factor 1 and 2:
        #cut_factor = torch.tensor([i%2 + 1], device=device) #randomness comes from validation_dataloader shuffle so it is not the same cubes cut each time
        cut_factor = torch.tensor([2], device=device) #randomness comes from validation_dataloader shuffle so it is not the same cubes cut each time
        
        T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
        delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
        vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
        T21_lr = get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
                    
        volume_frac = 1/(2**cut_factor)**3 #* torch.ones(size=(T21.shape[0],), device=device)
        multiple_redshifts = False
        labels = volume_frac #torch.tensor([volume_frac,], device=device) #torch.cat([labels, volume_frac], dim=1) if multiple_redshifts else

        if netG.network_opt["label_dim"] > 0:
            labels = labels
        else:
            labels = None
        
        #T21_lr_min = torch.amin(T21_lr, dim=(1,2,3,4), keepdim=True)
        #T21_lr_max = torch.amax(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)
        
        T21_lr, T21_lr_ups_mean, T21_lr_ups_std = normalize(T21_lr, mode="standard", factor=3.)#, factor=2.)
        T21, T21_mean, T21_std = normalize(T21, mode="standard", factor=3., x_mean=T21_lr_mean, x_std=T21_lr_std)#, factor=2.) #####
        delta, delta_mean, delta_std = normalize(delta, mode="standard", factor=3.)#, factor=2.)
        vbv, vbv_mean, vbv_std = normalize(vbv, mode="standard", factor=3.)#, factor=2.)
        T21, delta, vbv , T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=1) #support device
    
        if split_batch: #split subcube minibatch into smaller mini-batches for memory
            sub_data = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std)
            sub_dataloader = torch.utils.data.DataLoader(sub_data, batch_size=(2**(cut_factor.item()-1))**3, shuffle=False, sampler = None) 
            
            for j,(T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std) in tqdm(enumerate(sub_dataloader), desc='validation loop', total=len(sub_dataloader), disable=False if str(device)=="cuda:0" else True):
                T21_pred_j = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=100, eps=1e-3, clip_denoised=False, verbose=False)
                
                T21_pred_j = invert_normalization(T21_pred_j[:,-1:], mode="standard", factor=3., x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)
                T21 = invert_normalization(T21, mode="standard", factor=3., x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)

                MSE_j = torch.mean(torch.square(T21_pred_j[:,-1:] - T21),dim=(1,2,3,4), keepdim=False)
                if j == 0:
                    MSE_i = MSE_j
                else:
                    MSE_i = torch.cat([MSE_i, MSE_j], dim=0)

                if i==j==0:
                    T21_i = T21
                    delta_i = delta
                    vbv_i = vbv
                    T21_lr_i = T21_lr
                    T21_pred_i = T21_pred_j
                else:
                    T21_i = torch.cat([T21_i, T21], dim=0)
                    delta_i = torch.cat([delta_i, delta], dim=0)
                    vbv_i = torch.cat([vbv_i, vbv], dim=0)
                    T21_lr_i = torch.cat([T21_lr_i, T21_lr], dim=0)
                    T21_pred_i = torch.cat([T21_pred_i, T21_pred_j], dim=0)
                #if j == 0:
                #    break #only do one subbatch for now
        
        else:
            T21_pred_i = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=100, eps=1e-3, clip_denoised=False, verbose=False)
            T21_pred_i = invert_normalization(T21_pred_i[:,-1:], mode="standard", factor=3., x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)
            T21_i = invert_normalization(T21, mode="standard", factor=3., x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)
            delta_i = delta
            vbv_i = vbv
            T21_lr_i = T21_lr
            MSE_i = torch.mean(torch.square(T21_pred_i[:,-1:] - T21),dim=(1,2,3,4), keepdim=False)
        
        if i == 0:
            MSE = MSE_i
        else:
            MSE = torch.cat([MSE, MSE_i], dim=0)

        break #only do one box for now
    
    if multi_gpu:
        MSE_tensor_list = [torch.zeros_like(MSE) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list=MSE_tensor_list, tensor=MSE)
        MSE = torch.cat(MSE_tensor_list, dim=0)

        T21_tensor_list = [torch.zeros_like(T21_i) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list=T21_tensor_list, tensor=T21_i)
        T21 = torch.cat(T21_tensor_list, dim=0)

        T21_lr_tensor_list = [torch.zeros_like(T21_lr_i) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list=T21_lr_tensor_list, tensor=T21_lr_i)
        T21_lr = torch.cat(T21_lr_tensor_list, dim=0)

        delta_tensor_list = [torch.zeros_like(delta_i) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list=delta_tensor_list, tensor=delta_i)
        delta = torch.cat(delta_tensor_list, dim=0)

        vbv_tensor_list = [torch.zeros_like(vbv_i) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list=vbv_tensor_list, tensor=vbv_i)
        vbv = torch.cat(vbv_tensor_list, dim=0)

        T21_pred_tensor_list = [torch.zeros_like(T21_pred_i) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensor_list=T21_pred_tensor_list, tensor=T21_pred_i)
        T21_pred = torch.cat(T21_pred_tensor_list, dim=0)
    
    
    MSE = torch.mean(MSE).item()

    return MSE, dict(T21=T21, delta=delta, vbv=vbv, T21_lr=T21_lr, T21_pred=T21_pred)


###START main pytorch multi-gpu tutorial###
def main(rank, world_size=0, total_epochs = 1, batch_size = 1, train_models = 56, model_channels = 32, memory_profiling=False, model_id=21):
    multi_gpu = world_size > 1

    if multi_gpu:
        device = torch.device(f'cuda:{rank}')
        print("Multi GPU: {0}, device: {1}".format(multi_gpu,device))
        ddp_setup(rank, world_size=world_size)


        
    else:
        device = "cpu"
        print("Multi GPU: {0}, device: {1}".format(multi_gpu,device))
    


    #optimizer and model
    path = os.getcwd().split("/21cmGen")[0] + "/21cmGen"

    with torch.autograd.profiler.record_function("## setup load data ##") if False else contextlib.nullcontext():
        #network_opt = dict(in_channel=4, out_channel=1, inner_channel=32, norm_groups=8, channel_mults=(1, 2, 4, 8, 8), attn_res=(16,8,), res_blocks=2, dropout = 0, with_attn=True, image_size=64, dim=3)
        #network_opt = dict(in_channel=4, out_channel=1, inner_channel=32, norm_groups=8, channel_mults=(1, 2, 4, 8, 8), attn_res=(8,), res_blocks=2, dropout = 0, with_attn=True, image_size=32, dim=3)
        #network = UNet
        network_opt = dict(img_resolution=64, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
                        model_channels=model_channels, channel_mult=[2,2,2], attn_resolutions=[], #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                        embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                        )
        
        #network = UNet
        network = SongUNet
        
        #noise_schedule_opt = {'schedule_type': "linear", 'schedule_opt': {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}} 
        #noise_schedule_opt = {'schedule_type': "cosine", 'schedule_opt': {"timesteps": 1000, "s" : 0.008}} 
        #noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
        noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
        
        #loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
        loss_fn = VPLoss(beta_max=20., beta_min=0.1, epsilon_t=1e-5)
        
        netG = GaussianDiffusion(
                network=network,
                network_opt=network_opt,
                noise_schedule_opt=noise_schedule_opt,
                loss_fn = loss_fn,
                learning_rate=1e-4,
                scheduler=False,
                rank=rank,
            )
        

        train_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/IC_cubes/", 
        #train_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/", 
                                        redshifts=[10,], IC_seeds=list(range(0,train_models)), upscale=4, cut_factor=0, transform=False, norm_lr=True, device=device)
        #train_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_128/", path_IC=path+"/outputs/IC_cubes_128/", 
        #                                redshifts=[10,], IC_seeds=list(range(1000,1008)), upscale=4, cut_factor=0, transform=False, norm_lr=True, device=device)
        #train_dataset, train_dataset_norm, train_dataset_extrema = train_data_module.getFullDataset()

        #train_dataloader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size, shuffle=False if multi_gpu else True, 
        #                                        sampler = DistributedSampler(train_dataset) if multi_gpu else None) #4
        #train_data_norm = torch.utils.data.DataLoader( train_dataset_norm, batch_size=batch_size, shuffle=False if multi_gpu else True, 
        #                                         sampler = DistributedSampler(train_dataset_norm) if multi_gpu else None) #4
        train_dataloader = torch.utils.data.DataLoader(train_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True,
                                                        sampler = DistributedSampler(train_data_module) if multi_gpu else None)
        
        
        validation_data_module = CustomDataset(path_T21="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/T21_cubes/", path_IC="/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/IC_cubes/", 
        #validation_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/",                                                
                                        redshifts=[10,], IC_seeds=list(range(train_models,72)), upscale=4, cut_factor=0, transform=False, norm_lr=True, device=device)
        validation_dataloader = torch.utils.data.DataLoader(validation_data_module, batch_size=batch_size, shuffle=False if multi_gpu else True,
                                                        sampler = DistributedSampler(validation_data_module) if multi_gpu else None)
        
        if False: #72-80 extra validation
            validation_data_module = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/", 
                                                redshifts=[10,], IC_seeds=list(range(56,72)), upscale=4, cut_factor=1, transform=False, norm_lr=True, device=device)
            validation_data_module_small = CustomDataset(path_T21=path+"/outputs/T21_cubes_256/", path_IC=path+"/outputs/IC_cubes_256/", 
                                                        redshifts=[10,], IC_seeds=list(range(56,80)), upscale=4, cut_factor=2, transform=False, norm_lr=True, device=device)

        
            validation_dataset, validation_dataset_norm, validation_dataset_extrema = validation_data_module.getFullDataset()
            validation_dataset_small, validation_dataset_norm_small, validation_dataset_extrema_small = validation_data_module_small.getFullDataset()
            validation_batch_size = 1
            validation_batch_size_small = validation_dataset_small.tensors[0].shape[0]//validation_dataset.tensors[0].shape[0]
            print("Validation batch size: ", validation_batch_size, validation_batch_size_small, flush=True)
            validation_data_norm = torch.utils.data.DataLoader( validation_dataset_norm, batch_size=validation_batch_size, shuffle=False if multi_gpu else True, 
                                                        sampler = DistributedSampler(validation_dataset_norm) if multi_gpu else None) #4
            validation_data_norm_small = torch.utils.data.DataLoader( validation_dataset_norm_small, batch_size=validation_batch_size_small, shuffle=False if multi_gpu else True,
                                                            sampler = DistributedSampler(validation_dataset_norm_small) if multi_gpu else None) #4
    
        try:
            fn = path + "/trained_models/model_1/DDPMpp_standard_channels_{0}_tts_{1}_{2}_{3}_fac33".format(netG.network_opt["model_channels"], 
                                                                                                   len(train_data_module.IC_seeds) * 100 // 80,
                                                                                                   #netG.scheduler.gamma,
                                                                                                   netG.noise_schedule_opt["schedule_type"], 
                                                                                                   model_id)
            netG.model_name = fn.split("/")[-1]
            
            raise Exception("Temporarily suspend loading")
            netG.load_network(fn+".pth")
            print("Loaded network at {0}".format(fn), flush=True)
        except Exception as e:
            print(e, flush=True)
            print("Failed to load network at {0}. Starting from scratch.".format(fn+".pth"), flush=True)
    

    if (str(device)=="cuda:0") or (str(device)=="cpu"):
        
        print(f"[{device}] (Mini)Batchsize: {train_dataloader.batch_size} | Steps (batches): {len(train_dataloader)}", flush=True)
    
    if (str(device)=="cuda:0") and memory_profiling:
        torch.cuda.memory._record_memory_history()
        #prof.step()
    
    not_saved = 0
    for e in range(total_epochs):        
        
        if (str(device)=="cuda:0") or (str(device)=="cpu"):
            start_time = time.time()
        
        
        avg_loss = train_step(netG=netG, epoch=e, train_dataloader=train_dataloader, volume_reduction=True, split_batch=True, one_box_per_epoch=True, device=device, multi_gpu=multi_gpu)


        if (str(device)=="cuda:0") or (str(device)=="cpu"):
            print("[{0}]: Epoch {1} in {2:.2f}s | loss: {3:.3f}, mean(loss[-10:]): {4:.3f}, loss min: {5:.3f}, learning rate: {6:.3e}".format(str(device), len(netG.loss), time.time()-start_time, 
                                                                                                                                              avg_loss,  torch.mean(torch.tensor(netG.loss[-10:])).item(), 
                                                                                                                                              torch.min(torch.tensor(netG.loss)).item(), netG.optG.param_groups[0]['lr']), flush=True)

            #if e<4 and multi_gpu: #memory snapshot
            #    try:
            #        torch.cuda.memory._dump_snapshot(f"memory_snapshot_{str(device)[-1]}.pickle")
            #    except Exception as E:
            #        print(f"Failed to capture memory snapshot {E}", flush=True)
            #elif e==4 and multi_gpu:
            #    torch.cuda.memory._record_memory_history(enabled=None)
                
        loss_min = torch.min( torch.tensor(netG.loss_validation["loss"]) ).item()

        

        if False:#avg_loss <= loss_min and e >= 4000:
            if rank ==0:
                print(f"[{device}] loss={avg_loss:.2f} smaller than saved loss={loss_min:.2f}, epoch {e}: validating...", flush=True)
            loss_validation = validation_step_v2(netG=netG, validation_data_norm=validation_data_norm, validation_data_norm_small=validation_data_norm_small, device=device, multi_gpu=multi_gpu)
            
            loss_validation_min = torch.min( torch.tensor(netG.loss_validation["loss_validation"]) ).item()
            if loss_validation < loss_validation_min:
                if rank==0:
                    print(f"[{device}] validation loss={loss_validation:.2f} smaller than validation minimum={loss_validation_min:.2f}", flush=True)
                #netG.save_network(fn+".pth")
                netG.loss_validation["loss"].append(avg_loss)
                netG.loss_validation["loss_validation"].append(loss_validation)
            else:
                if rank==0:
                    print(f"[{device}] Not saving... validation loss={loss_validation:.2f} larger than validation minimum={loss_validation_min:.2f}", flush=True)

        if False:#len(netG.loss)>=150:
            if (avg_loss == torch.min(torch.tensor(netG.loss)).item()) or len(netG.loss)%20==0:
                netG.save_network( fn+".pth" )
                #losses_validation, x_pred, k_and_dsq_and_idx = validation_step(netG=netG, validation_data=validation_data, validation_loss_type="dsq_voxel", device=device, multi_gpu=multi_gpu)
                
                if False: #(str(device)=="cuda:0") or (str(device)=="cpu"):
                    print("losses_validation: {0}, loss_validation minimum: {1}".format(losses_validation.item(), torch.min(torch.tensor(netG.loss_validation)).item()), flush=True)
                    
                    if losses_validation.item() == torch.min(torch.tensor(netG.loss_validation)).item():       
                        plot_checkpoint(validation_data, x_pred, k_and_dsq_and_idx=k_and_dsq_and_idx, epoch = e, path = fn+".png", device="cpu")
                        netG.save_network( fn+".pth" )
                    else:
                        print("Not saving model. Validaiton did not improve", flush=True)
        if netG.network_opt["model_channels"] >=16:
            validation_check_epoch = 250
        else:
            validation_check_epoch = 250
        if len(netG.loss)>=validation_check_epoch:
            if len(netG.loss)%20==0 or avg_loss == torch.min(torch.tensor(netG.loss)).item():
                start_time_validation = time.time()
                with netG.ema.average_parameters():
                    loss_validation, tensor_dict = validation_step_v2(netG=netG, validation_dataloader=validation_dataloader, split_batch = True, device=device, multi_gpu=multi_gpu)
                validation_time = time.time()-start_time_validation
                loss_validation_min = torch.min( torch.tensor(netG.loss_validation["loss_validation"]) ).item()
                if loss_validation < loss_validation_min:
                    if rank==0:
                        print(f"[{device}] Validation took {validation_time:.2f}s, validation rmse={loss_validation**0.5:.3f} smaller than minimum={loss_validation_min**0.5:.3f}", flush=True)
                        #print(f"[{device}] tensors[0].shape: {tensors[0].shape}", flush=True)
                        path_plot = os.getcwd().split("/21cmGen")[0] + "/21cmGen/plots/vary_channels_nmodels_3/"
                        plot_checkpoint(**tensor_dict, netG=netG, MSE=loss_validation, epoch=len(netG.loss), path = path_plot, device=device)
                        plot_sigmas(**tensor_dict, netG=netG, path = path_plot,  quantiles=[(1-0.997)/2, (1-0.954)/2, 0.16, 0.5, 0.84, 1 - (1-0.954)/2, 1 - (1-0.997)/2])
                    netG.save_network(fn+".pth")
                    not_saved = 0
                    netG.loss_validation["loss_validation"].append(loss_validation)

                else:
                    not_saved = not_saved + 1
                    if rank==0:
                        print(f"[{device}] Not saving... validation time={validation_time:.2f}s, validation rmse={loss_validation**0.5:.3f} larger than validation minimum={loss_validation_min**0.5:.3f}", flush=True)
            
        if netG.scheduler is not False:
            netG.scheduler.step()

        if (str(device)=="cuda:0") and memory_profiling:
            torch.cuda.memory._dump_snapshot(f"memory_snap_16_2_{str(device)[-1]}.pickle")
            #prof.step()

        #abort if last save was more than 10 validation tests ago
        if not_saved >= 10:
            print("No improvement in 10 validation tests. Aborting...", flush=True)
            break

    
    if (str(device)=="cuda:0") and memory_profiling:
        torch.cuda.memory._record_memory_history(enabled=None)

    if multi_gpu:#world_size > 1:
        torch.distributed.barrier()
        destroy_process_group()
###END main pytorch multi-gpu tutorial###



if __name__ == "__main__":
    
    
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
   
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    if multi_gpu:
        print("Using multi_gpu", flush=True)
        for i in range(torch.cuda.device_count()):
            print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)
        for channel in [32, 16, 8, 4]: #[32, 16, 8, 4]
            for n_models in [56, 28, 1]: #[56, 28, 1]
                mp.spawn(main, args=(world_size, 2000, 1, n_models, channel, False, 1), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
    else:
        print("Not using multi_gpu",flush=True)
        try:
            main(rank=0, world_size=0, total_epochs=1, batch_size=1, train_models=56, model_channels=32, memory_profiling=False, model_id=2)#2*4)
        except KeyboardInterrupt:
            print('Interrupted', flush=True)
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)
    
        
