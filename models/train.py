import torch 
import torch.nn as nn
from utils import *
from diffusion import *
from model import *
from sde_lib import VPSDE

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS

import torch.multiprocessing as mp
import torch.utils
from torch.utils.data.distributed import DistributedSampler
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
#from kymatio.scattering3d.backend.torch_skcuda_backend import TorchSkcudaBackend3D
from kymatio.torch import HarmonicScattering3D


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

def loss_fn(netG, batch_size, x_true, conditionals = [None]):
    if netG.beta_schedule_opt["schedule_type"] != "VPSDE":
        ts = torch.randint(low = 0, high = netG.timesteps, size = (batch_size // 2 + 1, ), device=x_true.device)
        ts = torch.cat([ts, netG.timesteps - ts - 1], dim=0)[:batch_size] # antithetic sampling
        
        alphas_cumprod = netG.alphas_cumprod[ts]     
        xt, target_noise = netG.q_sample(x_true, ts)

        X = torch.cat([xt, *conditionals], dim = 1)
        model_pred = netG.model(X, alphas_cumprod)
        loss = nn.MSELoss(reduction='mean')(target_noise, model_pred) # loss per x_true
    
    else:
        eps = 1e-5
        ts = torch.rand(x_true.shape[0], device=x_true.device) * (1. - eps) + eps  
        xt, target_noise, std = netG.q_sample(x0=x_true, t=ts, noise=None)
        
        X = torch.cat([xt, *conditionals], dim = 1)
        score = netG.model(X, ts)
        loss = torch.mean(torch.square(score  * std + target_noise)) # loss per x_true (weighting=lambda_t=sigma_t**2)
        #print("VPSDE t, mean loss:", ts, loss)
    #avg_batch_loss += loss / batch_size    
    return loss #avg_batch_loss

def train_step(netG, epoch, train_data, device="cpu", multi_gpu = False,
          ):
    """
    Train the model
    """
    netG.model.train()
    
    avg_loss = torch.tensor(0.0, device=device)

    if multi_gpu:
        train_data.sampler.set_epoch(epoch) #fix for ddp loaded checkpoint?
    if (str(device)=="cuda:0") or (str(device)=="cpu"):
        print(f"[{device}] Epoch {len(netG.loss)} | (Mini)Batchsize: {train_data.batch_size} | Steps (batches): {len(train_data)}", flush=True)
    for i,(T21_,delta_,vbv_, T21_lr_) in enumerate(train_data):
        #if (str(device)=="cpu") or (str(device)=="cuda:0"):

        T21_,delta_,vbv_, T21_lr_ = augment_dataset(T21_,delta_,vbv_, T21_lr_, n=1) #support device

        loss = loss_fn(netG=netG, batch_size=train_data.batch_size, x_true=T21_, conditionals = [delta_, vbv_, T21_lr_])

        #ts = torch.randint(low = 0, high = netG.timesteps, size = (train_data.batch_size // 2 + 1, ), device=device)
        #ts = torch.cat([ts, netG.timesteps - ts - 1], dim=0)[:train_data.batch_size] # antithetic sampling
        #alphas_cumprod = netG.alphas_cumprod[ts]     
        #xt, target_noise = netG.q_sample(T21_, ts)

        #X = torch.cat([xt, delta_, vbv_, T21_lr_], dim = 1)
        #predicted_noise = netG.model(X, alphas_cumprod)
        #loss = nn.MSELoss(reduction='mean')(target_noise, predicted_noise) # torch.nn.L1Loss(reduction='mean')(target_noise, predicted_noise) 
        avg_loss += loss * T21_.shape[0]

        

        netG.optG.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(netG.model.parameters(), 1.0)
        netG.optG.step()


        #losses.append(loss.item())
        if (str(device)=="cuda:0") or (str(device)=="cpu"):
            if False: #i%(len(train_data)//16) == 0:
                print(f"Batch {i} of {len(train_data)} batches")

        #netG.ema.update() #Update netG.model with exponential moving average
    
    if multi_gpu:
        torch.distributed.all_reduce(tensor=avg_loss, op=torch.distributed.ReduceOp.AVG)
        #print("Multigpu avg batch loss: ", avg_loss)
        #print("{0} loss: {1:.4f}".format(device, avg_loss.item()))
    #print("{0} loss: {1:.4f}".format(device, avg_loss.item()))

    netG.loss.append(avg_loss.item())
    
    return avg_loss.item()


def plot_checkpoint(validation_data, x_pred, k_and_dsq_and_idx=None, epoch=None, path = None, device="cpu"):
    x_true, delta, vbv, x_true_lr = validation_data.dataset.tensors 
    k_vals_true, dsq_true, k_vals_pred, dsq_pred, model_idx = k_and_dsq_and_idx

    try:
        if k_vals_true==dsq_true==k_vals_pred==dsq_pred==None:
            k_vals_true, dsq_true  = calculate_power_spectrum(x_true, Lpix=3, kbins=100, dsq = True, method="torch", device=device)
            k_vals_pred, dsq_pred  = calculate_power_spectrum(x_pred, Lpix=3, kbins=100, dsq = True, method="torch", device=device)
    except:
        pass
    print("Model idx: ", model_idx, flush=True)
    
    #send to cpu
    x_true = x_true.cpu()
    delta = delta.cpu()
    vbv = vbv.cpu()
    x_true_lr = x_true_lr.cpu()

    x_pred = x_pred.cpu()
    
    k_vals_true = k_vals_true.cpu()
    dsq_true = dsq_true.cpu()
    k_vals_pred = k_vals_pred.cpu()
    dsq_pred = dsq_pred.cpu()

    #model_idx = torch.randint(0, x_true.shape[0], (1,)).item() #0

    fig = plt.figure(figsize=(15,15))
    gs = GS(3, 3, figure=fig,) #height_ratios=[1,1,1.5])

    ax_delta = fig.add_subplot(gs[0,0])#, wspace = 0.2)
    ax_vbv = fig.add_subplot(gs[0,1])
    ax_x_true_lr = fig.add_subplot(gs[0,2])

    ax_delta.imshow(delta[model_idx,0,:,:,delta.shape[-1]//2], vmin=-1, vmax=1)
    ax_delta.set_title("Delta (input)")
    ax_vbv.imshow(vbv[model_idx,0,:,:,vbv.shape[-1]//2], vmin=-1, vmax=1)
    ax_vbv.set_title("Vbv (input)")
    ax_x_true_lr.imshow(x_true_lr[model_idx,0,:,:,x_true_lr.shape[-1]//2], vmin=-1, vmax=1)
    ax_x_true_lr.set_title("T21 LR (input)")



    ax_x_true = fig.add_subplot(gs[1,0])
    ax_x_pred = fig.add_subplot(gs[1,1])




    ax_x_true.imshow(x_true[model_idx,0,:,:,x_true.shape[-1]//2], vmin=-1, vmax=1)
    ax_x_true.set_title("T21 HR (Real)")
    
    ax_x_pred.imshow(x_pred[model_idx,0,:,:,x_pred.shape[-1]//2], vmin=-1, vmax=1)
    ax_x_pred.set_title(f"T21 SR (Generated) epoch {epoch}")

    sgs = SGS(1,2, gs[2,:])
    sgs_dsq = SGS(2,1, sgs[0], height_ratios=[4,1], hspace=0, )
    ax_dsq = fig.add_subplot(sgs_dsq[0])
    ax_dsq.get_xaxis().set_visible(False)
    ax_dsq_resid = fig.add_subplot(sgs_dsq[1], sharex=ax_dsq)
    ax_dsq_resid.set_ylabel("|Residuals|")#("$\Delta^2(k)_\\mathrm{{SR}} - \Delta^2(k)_\\mathrm{{HR}}$")
    ax_dsq_resid.set_xlabel("$k$")
    ax_dsq_resid.set_yscale('log')
    ax_dsq_resid.grid()
    
    #ax_dsq.plot(k_vals_true, dsq_pred[:,0].T, alpha=0.02, color='k', ls='solid')
    ax_dsq.plot(k_vals_true, dsq_true[model_idx,0], label="T21 HR", ls='solid', lw=2)
    ax_dsq.plot(k_vals_pred, dsq_pred[model_idx,0], label="T21 SR", ls='solid', lw=2)

    ax_dsq_resid.plot(k_vals_true, torch.abs(dsq_pred[:,0] - dsq_true[:,0]).T, color='k', alpha=0.02)
    ax_dsq_resid.plot(k_vals_true, torch.abs(dsq_pred[model_idx,0] - dsq_true[model_idx,0]), lw=2, )

    
    ax_dsq.set_ylabel('$\Delta^2(k)_\\mathrm{{norm}}$')
    #ax_dsq.set_xlabel('$k$')
    ax_dsq.set_yscale('log')
    ax_dsq.grid()
    ax_dsq.legend()
    ax_dsq.set_title("Power Spectrum (output)")


    ax_hist = fig.add_subplot(sgs[1])
    ax_hist.hist(x_pred[model_idx,0,:,:,:].flatten(), bins=100, alpha=0.5, label="T21 SR", density=True)
    ax_hist.hist(x_true[model_idx,0,:,:,:].flatten(), bins=100, alpha=0.5, label="T21 HR", density=True)
    
    ax_hist.set_xlabel("Norm. $T_{{21}}$")
    ax_hist.set_ylabel("PDF")
    ax_hist.legend()
    ax_hist.grid()
    ax_hist.set_title("Pixel Histogram (output)")

    plt.savefig(path)
    plt.close()




def validation_step(netG, validation_data, validation_type="DDIM", validation_loss_type="dsq_voxel", device="cpu", multi_gpu=False):
    netG.model.eval()
    
    #validation_type = "DDIM" # or "DDPM SR3" or "DDPM Classic" or "None" (to save at every minimum training loss)
    #validation_loss_type = "dsq" # or voxel

    #if len(netG.loss)>=validation_epoch: #only start checking voxel loss after n epochs #change this when it works 
    losses_validation_dsq = torch.tensor(0.0, device=device) #0 #avg_loss = torch.tensor(0.0, device=device)
    losses_validation_voxel = torch.tensor(0.0, device=device)
    losses_validation_WST = torch.tensor(0.0, device=device)
    #stime_ckpt = time.time()

    if validation_type == "DDIM":

        x_pred = []
        dsq_true = []
        dsq_pred = []
        
        dsq_mse = torch.tensor(0.0, device=device) #[]
        voxel_mse = torch.tensor(0.0, device=device) #[]
        wst_mse = torch.tensor(0.0, device=device) #[]
        
        k_vals_true_i = None
        k_vals_pred_i = None

        for i,(T21_validation, delta_validation, vbv_validation, T21_lr_validation) in enumerate(validation_data):
        #for i, (T21_validation, delta_validation, vbv_validation, T21_lr_validation) in tqdm(enumerate(validation_data), total=len(validation_data)):
            x_pred_i, x_slices, noises, pred_noises, x0_preds = netG.p_sample_loop(conditionals=[delta_validation, vbv_validation, T21_lr_validation], n_save=2, clip_denoised=True, sampler = "DDIM", save_slices=True, ema=False, ddim_n_steps = 20, verbose=False, device=device)
            x_pred.append(x_pred_i[:,-1:])

            if validation_loss_type == "dsq_voxel":
                k_vals_true_i, dsq_true_i  = calculate_power_spectrum(T21_validation, Lpix=3, kbins=100, dsq = True, method="torch", device=device)
                k_vals_pred_i, dsq_pred_i  = calculate_power_spectrum(x_pred_i[:,-1:], Lpix=3, kbins=100, dsq = True, method="torch", device=device)
                dsq_true.append(dsq_true_i)
                dsq_pred.append(dsq_pred_i)
                
                voxel_mse += torch.nanmean(torch.square(x_pred_i[:,-1:] - T21_validation)) / len(validation_data)
                dsq_mse += torch.nanmean(torch.square(dsq_pred_i - dsq_true_i)) / len(validation_data)
            
            elif validation_loss_type == "WST":
                gpu_backend = False
                Backend = TorchSkcudaBackend3D if gpu_backend else TorchBackend3D
                #print("devices: ", T21_lr_validation.contiguous().device, x_pred_i[:,-1].device, T21_validation.device, flush=True)

                scattering = HarmonicScattering3D(J=5, shape=T21_validation[0,0].shape, L=4, max_order=2, integral_powers=(1.0,), backend='torch_skcuda' if gpu_backend else 'torch') #(0.5, 1.0, 2.0))
                
                input = x_pred_i[:,-1].contiguous() if gpu_backend else x_pred_i[:,-1].contiguous().cpu()
                order_0_pred = Backend.compute_integrals(input, integral_powers=(1.0,),)#backend='torch_skcuda') # TorchSkcudaBackend3D
                order_1_2_pred = scattering(input)
                
                input = T21_validation[:,0].contiguous() if gpu_backend else T21_validation[:,0].contiguous().cpu()
                order_0_true = Backend.compute_integrals(input, integral_powers=(1.0,),)#backend='torch_skcuda')# TorchSkcudaBackend3D
                order_1_2_true = scattering(input)
                
                log_order_pred = torch.cat([order_0_pred, order_1_2_pred.mean(dim=(-2,-1))], dim=1).log10()
                log_order_true = torch.cat([order_0_true, order_1_2_true.mean(dim=(-2,-1))], dim=1).log10()

                
                wst_mse += torch.nanmean(torch.square(log_order_pred - log_order_true)) / len(validation_data)

                print("WST loss: {0:.4f}, device: {1}, i-loop: {2}".format(wst_mse.item(), str(device), i), flush=True)
            
            else:
                assert False, "Validation loss type not recognized"

        x_pred = torch.cat(x_pred, dim=0)
        dsq_true = torch.cat(dsq_true, dim=0) if validation_loss_type == "dsq_voxel" else None
        dsq_pred = torch.cat(dsq_pred, dim=0) if validation_loss_type == "dsq_voxel" else None
        
        model_idx = 0
        if multi_gpu:
            x_pred_tensor_list = [torch.zeros_like(x_pred) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensor_list=x_pred_tensor_list, tensor=x_pred)
            x_pred = torch.cat(x_pred_tensor_list, dim=0)

            if validation_loss_type == "dsq_voxel":
                dsq_true_tensor_list = [torch.zeros_like(dsq_true, device=device) for _ in range(torch.distributed.get_world_size())]
                dsq_pred_tensor_list = [torch.zeros_like(dsq_pred, device=device) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(tensor_list=dsq_true_tensor_list, tensor=dsq_true)
                torch.distributed.all_gather(tensor_list=dsq_pred_tensor_list, tensor=dsq_pred)
                dsq_true = torch.cat(dsq_true_tensor_list, dim=0)
                dsq_pred = torch.cat(dsq_pred_tensor_list, dim=0)
                torch.distributed.all_reduce(tensor=dsq_mse, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(tensor=voxel_mse, op=torch.distributed.ReduceOp.AVG)
                total_loss = dsq_mse + voxel_mse * 1e-1
                #get id of model with closest error to dsq_mse
                dsq_mse_i = torch.nanmean(torch.square(dsq_pred - dsq_true), dim=2)
                dsq_voxel_mse = dsq_mse_i + voxel_mse * 1e-1
                dsq_voxel_mse_mean = torch.nanmean(dsq_voxel_mse)
                model_idx = torch.argmin(torch.abs(dsq_voxel_mse - dsq_voxel_mse_mean)).item()
                if str(device)=="cuda:0":
                    print("Validation voxel_mse loss: {0:.4f}".format(voxel_mse.item()*1e-1), flush=True)
                    print("Validation dsq_mse loss: {0:.4f}".format(dsq_mse.item()), flush=True)
            
            elif validation_loss_type == "WST":
                torch.distributed.all_reduce(tensor=wst_mse, op=torch.distributed.ReduceOp.AVG)
                total_loss = wst_mse
                if True:#str(device)=="cuda:0":
                    print("[{1}] Validation wst_mse loss: {0:.4f}".format(wst_mse.item(), str(device)), flush=True)
            

        netG.losses_validation_history.append(total_loss)
    return total_loss, x_pred, [k_vals_true_i, dsq_true, k_vals_pred_i, dsq_pred, model_idx]
        #elif (validation_type == "DDPM SR3") or (validation_type=="DDPM Classic"):
        #    print(validation_type + " validation")
        #    T21_validation_, delta_validation, vbv_validation, T21_lr_validation = loader_validation.dataset.tensors
        #    #pick random int from batch shape 0 of validation data
        #    i = torch.randint(low=0, high=T21_validation_.shape[0], size=(1,)).item()
        #    x_sequence, x_slices, noises, pred_noises, x0_preds = netG.p_sample_loop(conditionals=[delta_validation[i:i+1], vbv_validation_[i:i+1], T21_lr_validation_[i:i+1]], n_save=100, clip_denoised=True, mean_approach = validation_type, save_slices=False, ema=True, ddim_n_steps = 10, verbose=True)
        #    
        #    if validation_loss_type == "dsq":
        #        k_vals_true, dsq_true  = calculate_power_spectrum(T21_validation_[i:i+1], Lpix=3, kbins=100, dsq = True, method="torch")
        #        k_vals_pred, dsq_pred  = calculate_power_spectrum(x_sequence[:,-1:], Lpix=3, kbins=100, dsq = True, method="torch")
        #        losses_validation += torch.nanmean(torch.square(dsq_pred - dsq_true)).item() #nn.MSELoss(reduction='mean')(dsq_pred, dsq_true).item()
        #    elif validation_loss_type == "voxel":
        #        losses_validation += torch.nanmean(torch.square(x_sequence[:,-1:] - T21_validation_[i:i+1])).item()
        #    else:
        #        assert False, "Validation loss type not recognized"
        #
        #
        #    netG.losses_validation_history.append(losses_validation)
        #    print("{0} voxel loss {1:.4f} and time {2:.2f}".format(validation_type, losses_validation, time.time()-stime_ckpt))
        #    save_bool = losses_validation == np.min(netG.losses_validation_history)
        #
        #elif validation_type == "None":
        #    save_bool = True
        #
        #else:
        #    assert False, "Validation type not recognized"
    #else:
        #save_bool = True
    

    #if save_bool:
        #print("Saving model now. Loss history is: ", netG.losses_validation_history)
        ##netG.save_network(model_path)

#test new repo name local
#test new repo name hpc

###START main pytorch multi-gpu tutorial###
def main(rank, world_size=0, total_epochs = 1, batch_size = 2*4, model_id=21):
    
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
    
    network_opt = dict(in_channel=4, out_channel=1, inner_channel=32, norm_groups=8, channel_mults=(1, 2, 4, 8, 8), attn_res=(8,), res_blocks=2, dropout = 0, with_attn=True, image_size=32, dim=3)
    #beta_schedule_opt = {'schedule_type': "linear", 'schedule_opt': {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}} 
    #beta_schedule_opt = {'schedule_type': "cosine", 'schedule_opt': {"timesteps": 1000, "s" : 0.008}} 
    beta_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}} 

    netG = GaussianDiffusion(
            network=UNet,
            network_opt=network_opt,
            beta_schedule_opt=beta_schedule_opt,
            #noise_schedule=cosine_beta_schedule,
            #noise_schedule=linear_beta_schedule,
            #noise_schedule_opt=noise_schedule_opt,
            learning_rate=1e-4,
            rank=rank,
        )
    
#    SDE = VPSDE(beta_min=0.1, beta_max=20, N=1000)

    try:
        netG.load_network(path + f"/trained_models/diffusion_model_test_{model_id}.pth")
        print("Loaded network at {0}".format(path + f"/trained_models/diffusion_model_test_{model_id}.pth"), flush=True)
    except:
        print("Failed to load network at {0}. Starting from scratch.".format(path + f"/trained_models/diffusion_model_test_{model_id}.pth"), flush=True)

    #train_data = prepare_dataloader(path=path, batch_size=batch_size, upscale=4, cut_factor=2, redshift=10, IC_seeds=list(range(1000,1008)), device=device, multi_gpu=multi_gpu)
    #validation_data = prepare_dataloader(path=path, batch_size=batch_size, upscale=4, cut_factor=2, redshift=10, IC_seeds=list(range(1008,1011)), device=device, multi_gpu=multi_gpu)
    train_data_module = CustomDataset(path=path, redshifts=[10,], IC_seeds=list(range(1000,1008)), upscale=4, cut_factor=2, transform=False, device=device)
    validation_data_module = CustomDataset(path=path, redshifts=[10,], IC_seeds=list(range(1008,1011)), upscale=4, cut_factor=2, transform=False, device=device)
    train_data = torch.utils.data.DataLoader( train_data_module.getFullDataset(), batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = DistributedSampler(train_data_module.getFullDataset()) if multi_gpu else None) #4
    validation_data = torch.utils.data.DataLoader( validation_data_module.getFullDataset(), batch_size=batch_size, shuffle=False if multi_gpu else True, sampler = DistributedSampler(validation_data_module.getFullDataset()) if multi_gpu else None) #4
    
    for e in range(total_epochs):
        avg_loss = train_step(netG=netG, epoch=e, train_data=train_data, device=device, multi_gpu=multi_gpu)
        #if (str(device)=="cuda:0") or (str(device)=="cpu"):
        print("device {2}: losses min: {0:.5f}, losses history minimum: {1:.5f}".format(avg_loss,  torch.min(torch.tensor(netG.loss)).item(),  str(device)))
        if (avg_loss == torch.min(torch.tensor(netG.loss)).item()) and (len(netG.loss)>=300):    
            pass
            #netG.save_network( path + "/trained_models/diffusion_model_{0}_{1}.pth".format(netG.beta_schedule_opt["schedule_type"], model_id)  )
        #barrier:
        if multi_gpu:
            torch.distributed.barrier()

            if len(netG.loss)>=300: #only start checking voxel loss after n epochs #change this when it works
                losses_validation, x_pred, k_and_dsq_and_idx = validation_step(netG=netG, validation_data=validation_data, validation_type="DDIM", validation_loss_type="dsq_voxel", device=device, multi_gpu=multi_gpu)
                #x_pred = x_pred.cpu() if multi_gpu else x_pred
                if (str(device)=="cuda:0") or (str(device)=="cpu"):
                    print("losses_validation: {0}, losses_validation_history minimum: {1}".format(losses_validation.item(), torch.min(torch.tensor(netG.losses_validation_history)).item()))
                    
                    if losses_validation.item() == torch.min(torch.tensor(netG.losses_validation_history)).item():
                            
                        #print("Saving model", flush=True)
                        #[x.cpu() for x in input_output]
                        plot_checkpoint(validation_data, x_pred, k_and_dsq_and_idx=k_and_dsq_and_idx, epoch = e, path = path + f"/trained_models/diffusion_saved_model_{model_id}.png", device="cpu")
                        #netG.save_network( path + f"/trained_models/diffusion_model_test_{model_id}.pth"  )
                    else:
                        print("Not saving model. Validaiton did not improve", flush=True)

    
    if multi_gpu:#world_size > 1:
        destroy_process_group()
###END main pytorch multi-gpu tutorial###



if __name__ == "__main__":
   
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1

    if multi_gpu:
        print("Using multi_gpu", flush=True)
        for i in range(torch.cuda.device_count()):
            print("Device {0}: ".format(i), torch.cuda.get_device_properties(i).name)
        mp.spawn(main, args=(world_size, 3000, 16, 34), nprocs=world_size) #wordlsize, total_epochs, batch size (for minibatch)
    else:
        print("Not using multi_gpu",flush=True)
        main(rank=0, world_size=0, total_epochs=1, batch_size=8, model_id=34)#2*4)
    
        
