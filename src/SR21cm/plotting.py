import os 

import torch
from torch.distributed import init_process_group, destroy_process_group

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS

from .utils import ddp_setup,calculate_power_spectrum

@torch.no_grad()
def plot_scales(rank, world_size, data_dir, plot_dir, **kwargs):
    """
    Plots the scales of the 21cm signal and its super-resolution predictions.
    This function loads the 21cm signal data and its super-resolution predictions from a given file path,
    and generates a series of plots comparing the high-resolution (HR) and super-resolution (SR) data at 
    different scales (512^3, 256^3, and 128^3). The plots include images of the 21cm signal, histograms 
    of the signal distributions, and power spectrum plots.
    Parameters:
        rank (int): The rank of the current process in a distributed setup.
        world_size (int): The total number of processes in the distributed setup.
        data_dir (str): The directory containing the data files to load.
        plot_dir (str): The directory to save the generated plots.
        **kwargs: Additional keyword arguments for customizing the plots.
            - fontsize (int): Font size for the plots. Default is 22.
            - use_tex (bool): Whether to use LaTeX for text rendering. Default is False.
            - rasterized (bool): Whether to rasterize the images. Default is True.
            - slice_idx (int): The index of the slice to plot. Default is 64.
            - plot_format (str): The format to save the plots. Default is ".png".
    Returns:
        None
    Notes:
        - This function uses PyTorch for loading the data and performing calculations.
        - The function supports multi-GPU setups using PyTorch's distributed data parallel (DDP) framework.
        - The generated plots are saved as image files in the specified save_path.
    """
    fontsize = kwargs.pop("fontsize", 22)
    use_tex = kwargs.pop("use_tex", False)

    rasterized = kwargs.pop("rasterized", True)
    slice_idx = kwargs.pop("slice_idx", 64)
    plot_format = kwargs.pop("plot_format", ".png")

    multi_gpu = world_size >= 1
    if multi_gpu:
        ddp_setup(rank, world_size=world_size)#multi_gpu = world_size > 1

    data_path = os.path.join(data_dir, f"T21_scales_{rank}.pth")
    loaded_state = torch.load(data_path, map_location=torch.device(rank) if multi_gpu else "cpu")

    filename = data_path.split("/")[-1]
    print(f"[dev:{rank}] Loaded file: {filename}", flush=True)
    
    T21_512 = loaded_state["T21_512"].cpu()
    T21_pred_512 = loaded_state["T21_pred_512"].cpu()
    T21_256 = loaded_state["T21_256"].cpu()
    T21_pred_256 = loaded_state["T21_pred_256"].cpu()
    T21_128 = loaded_state["T21_128"].cpu()
    T21_pred_128 = loaded_state["T21_pred_128"].cpu()

    

    plt.rcParams.update({'font.size': fontsize,
                        "text.usetex": use_tex,#True,
                        "font.family": "serif" if use_tex else "sans-serif",
                        "font.serif": "cm" if use_tex else "Arial",
                        })


    fig = plt.figure(figsize=(3*5,4*5))
    wspace = 0.2
    gs = GS(5, 3, figure=fig, height_ratios=[0.05, 1,1,0.75,0.75], hspace=0.3, wspace=wspace)

    sgs_cax = SGS(1, 3, gs[0,:], hspace=0.1, wspace=wspace)
    sgs_im = SGS(2, 3, gs[1:3,:], hspace=0.1, wspace=wspace)

    cax = [fig.add_subplot(sgs_cax[0,0]), fig.add_subplot(sgs_cax[0,1]), fig.add_subplot(sgs_cax[0,2])]
    axes_im = np.array([[fig.add_subplot(sgs_im[i,j]) for j in range(3)] for i in range(2)])
    for ax in axes_im.flatten():
        ax.tick_params(axis='both', labelleft=False, direction='in')


    vmin_512 = min([T21_512[0,0,slice_idx].min(),T21_pred_512[0,0,slice_idx].min()]).item()
    vmax_512 = max([T21_512[0,0,slice_idx].max(),T21_pred_512[0,0,slice_idx].max()]).item()
    vmin_256 = min([T21_256[0,0,slice_idx].min(),T21_pred_256[0,0,slice_idx].min()]).item()
    vmax_256 = max([T21_256[0,0,slice_idx].max(),T21_pred_256[0,0,slice_idx].max()]).item()
    vmin_128 = min([T21_128[0,0,slice_idx].min(),T21_pred_128[0,0,slice_idx].min()]).item()
    vmax_128 = max([T21_128[0,0,slice_idx].max(),T21_pred_128[0,0,slice_idx].max()]).item()

    #vmin = min(T21_512[0,0,slice_idx].min(), T21_256[0,0,slice_idx].min(), T21_128[0,0,slice_idx].min(), T21_pred_512[0,0,slice_idx].min(), T21_pred_256[0,0,slice_idx].min(), T21_pred_128[0,0,slice_idx].min())
    #vmax = max(T21_512[0,0,slice_idx].max(), T21_256[0,0,slice_idx].max(), T21_128[0,0,slice_idx].max(), T21_pred_512[0,0,slice_idx].max(), T21_pred_256[0,0,slice_idx].max(), T21_pred_128[0,0,slice_idx].max())
    #vmin_pred = min(T21_pred_512[0,0,slice_idx].min(), T21_pred_256[0,0,slice_idx].min(), T21_pred_128[0,0,slice_idx].min())
    #vmax_pred = max(T21_pred_512[0,0,slice_idx].max(), T21_pred_256[0,0,slice_idx].max(), T21_pred_128[0,0,slice_idx].max())


    img = axes_im[0,0].imshow(T21_512[0,0,slice_idx], vmin=vmin_512, vmax=vmax_512, rasterized=rasterized)
    #axes_im[0,0].set_title("$T_{{21}}$ HR $512^3$")
    rect = patches.Rectangle((0, 256), 256, -256, linewidth=2, edgecolor='r', linestyle='solid', facecolor='none')
    axes_im[0,0].add_patch(rect)
    rect = patches.Rectangle((0, 128), 128, -128, linewidth=2, edgecolor='r', linestyle='dashed', facecolor='none')
    axes_im[0,0].add_patch(rect)
    #divider = make_axes_locatable(axes_im[0,0])
    #cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = fig.colorbar(img, cax=cax[0], orientation='horizontal')
    cbar.ax.tick_params(labelsize=plt.rcParams['font.size'], labeltop=True, labelbottom=False, top=True, bottom=False)
    cbar.set_label("$T_{{21}}$ HR $512^3$", fontsize=plt.rcParams['font.size'])
    cbar.ax.xaxis.set_label_position('top')
    ax_pos = axes_im[0,0].get_position()
    cbar_pos = cax[0].get_position()
    cax[0].set_position([ax_pos.x0, cbar_pos.y0*0.96, ax_pos.width, cbar_pos.height])



    img = axes_im[0,1].imshow(T21_256[0,0,slice_idx], vmin=vmin_256, vmax=vmax_256, rasterized=rasterized)
    #axes_im[0,1].set_title("$T_{{21}}$ HR $256^3$")
    rect = patches.Rectangle((0, 128), 128, -128, linewidth=2, edgecolor='r', linestyle='dashed', facecolor='none')
    axes_im[0,1].add_patch(rect)
    #divider = make_axes_locatable(axes_im[0,1])
    #cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = fig.colorbar(img, cax=cax[1], orientation='horizontal')
    cbar.ax.tick_params(labelsize=plt.rcParams['font.size'], labeltop=True, labelbottom=False, top=True, bottom=False)
    cbar.set_label("$T_{{21}}$ HR $256^3$", fontsize=plt.rcParams['font.size'])
    cbar.ax.xaxis.set_label_position('top')
    ax_pos = axes_im[0,1].get_position()
    cbar_pos = cax[1].get_position()
    cax[1].set_position([ax_pos.x0, cbar_pos.y0*0.96, ax_pos.width, cbar_pos.height])

    img = axes_im[0,2].imshow(T21_128[0,0,slice_idx], vmin=vmin_128, vmax=vmax_128, rasterized=rasterized)
    #axes_im[0,2].set_title("$T_{{21}}$ HR $128^3$")
    #divider = make_axes_locatable(axes_im[0,2])
    #cax = divider.append_axes('top', size='5%', pad=0.05)
    cbar = fig.colorbar(img, cax=cax[2], orientation='horizontal')
    cbar.ax.tick_params(labelsize=plt.rcParams['font.size'], labeltop=True, labelbottom=False, top=True, bottom=False)
    cbar.set_label("$T_{{21}}$ HR $128^3$", fontsize=plt.rcParams['font.size'])
    cbar.ax.xaxis.set_label_position('top')
    ax_pos = axes_im[0,2].get_position()
    cbar_pos = cax[2].get_position()
    cax[2].set_position([ax_pos.x0, cbar_pos.y0*0.96, ax_pos.width, cbar_pos.height])
    #set colorbar ticks to 0,10,19
    cbar.set_ticks([0,10,19])


    #pos0 = axes_im[1,0].get_position()
    #new_height = pos0.height * (1 - 0.05 - 0.05)  # Adjust for the colorbar size and padding
    #axes_im[1,0].set_position([pos0.x0, pos0.y0, pos0.width, new_height])
    axes_im[1,0].imshow(T21_pred_512[0,0,slice_idx], vmin=vmin_512, vmax=vmax_512, rasterized=rasterized)
    axes_im[1,0].set_title("$T_{{21}}$ SR $512^3$", fontsize=plt.rcParams['font.size'])
    rect = patches.Rectangle((0, 256), 256, -256, linewidth=2, edgecolor='r', linestyle='solid', facecolor='none')
    axes_im[1,0].add_patch(rect)
    rect = patches.Rectangle((0, 128), 128, -128, linewidth=2, edgecolor='r', linestyle='dashed', facecolor='none')
    axes_im[1,0].add_patch(rect)

    axes_im[1,1].imshow(T21_pred_256[0,0,slice_idx], vmin=vmin_256, vmax=vmax_256, rasterized=rasterized)
    axes_im[1,1].set_title("$T_{{21}}$ SR $256^3$", fontsize=plt.rcParams['font.size'])
    rect = patches.Rectangle((0, 128), 128, -128, linewidth=2, edgecolor='r', linestyle='dashed', facecolor='none')
    axes_im[1,1].add_patch(rect)

    axes_im[1,2].imshow(T21_pred_128[0,0,slice_idx], vmin=vmin_128, vmax=vmax_128, rasterized=rasterized)
    axes_im[1,2].set_title("$T_{{21}}$ SR $128^3$", fontsize=plt.rcParams['font.size'])



    ##################
    legend_loc = "upper left"

    sgs = SGS(2, 3, gs[3,:], height_ratios=[0.75,0.25], hspace=0., wspace=wspace)
    axes_hist_512 = []
    axes_hist_512.append(fig.add_subplot(sgs[0,0]))
    axes_hist_512.append(fig.add_subplot(sgs[1,0], sharex=axes_hist_512[0]))
    axes_hist_512[0].tick_params(axis='both', labelbottom=False, direction='in')
    axes_hist_512[1].tick_params(axis='both', direction='in')

    axes_hist_256 = []
    axes_hist_256.append(fig.add_subplot(sgs[0,1], sharex=axes_hist_512[0], sharey=axes_hist_512[0]))
    axes_hist_256.append(fig.add_subplot(sgs[1,1], sharex=axes_hist_512[1], sharey=axes_hist_512[1]))
    axes_hist_256[0].tick_params(axis='both', labelbottom=False, labelleft=False, direction='in')
    axes_hist_256[1].tick_params(axis='both', labelleft=False, direction='in')

    axes_hist_128 = []
    axes_hist_128.append(fig.add_subplot(sgs[0,2], sharex=axes_hist_512[0], sharey=axes_hist_512[0]))
    axes_hist_128.append(fig.add_subplot(sgs[1,2], sharex=axes_hist_512[1], sharey=axes_hist_512[1]))
    axes_hist_128[0].tick_params(axis='both', labelbottom=False, labelleft=False, direction='in')
    axes_hist_128[1].tick_params(axis='both', labelleft=False, direction='in')

    hist_min = min(T21_512[0,0].min().item(), T21_pred_512[0,0].min().item())
    hist_max = max(T21_512[0,0].max().item(), T21_pred_512[0,0].max().item())
    bins = np.linspace(hist_min, hist_max, 200)
    hist_true, _ = np.histogram(T21_512[0,0].flatten(), bins=bins, density=True)
    hist_pred, _ = np.histogram(T21_pred_512[0,0].flatten(), bins=bins, density=True)
    axes_hist_512[0].bar(bins[:-1], hist_true, width=bins[1]-bins[0], alpha=0.5, label="$T_{{21}}$ HR", rasterized=rasterized)
    axes_hist_512[0].bar(bins[:-1], hist_pred, width=bins[1]-bins[0], alpha=0.5, label="$T_{{21}}$ SR", rasterized=rasterized)
    hist_resid_512 = np.abs(hist_true - hist_pred)
    axes_hist_512[1].bar(bins[:-1], hist_resid_512, width=bins[1] - bins[0], alpha=0.5, label="$|\mathrm{{Residuals}}|$", color='k', rasterized=rasterized)
    axes_hist_512[0].set_xlim(-5, 20)
    axes_hist_512[0].set_ylabel("PDF")
    axes_hist_512[1].set_ylabel("$|\mathrm{{Residuals}}|$")
    axes_hist_512[1].set_xlabel("$T_{{21}}$ [mK]")
    RMSE = torch.sqrt(torch.mean(torch.square(T21_pred_512[0,0]-T21_512[0,0]))).item()
    axes_hist_512[0].text(0.95, 0.95, f"RMSE: {RMSE:.3f}", ha='right', va='top', fontsize=plt.rcParams['font.size']-4, transform=axes_hist_512[0].transAxes)

    hist_min = min(T21_256[0,0].min().item(), T21_pred_256[0,0].min().item())
    hist_max = max(T21_256[0,0].max().item(), T21_pred_256[0,0].max().item())
    bins = np.linspace(hist_min, hist_max, 100)
    hist_true, _ = np.histogram(T21_256[0,0].flatten(), bins=bins, density=True)
    hist_pred, _ = np.histogram(T21_pred_256[0,0].flatten(), bins=bins, density=True)
    axes_hist_256[0].bar(bins[:-1], hist_true, width=bins[1]-bins[0], alpha=0.5, label="$T_{{21}}$ HR", rasterized=rasterized)
    axes_hist_256[0].bar(bins[:-1], hist_pred, width=bins[1]-bins[0], alpha=0.5, label="$T_{{21}}$ SR", rasterized=rasterized)
    hist_resid_256 = np.abs(hist_true - hist_pred)
    axes_hist_256[1].bar(bins[:-1], hist_resid_256, width=bins[1] - bins[0], alpha=0.5, label="$|\mathrm{{Residuals}}|$", color='k', rasterized=rasterized)
    axes_hist_256[1].set_xlabel("$T_{{21}}$ [mK]")
    RMSE = torch.sqrt(torch.mean(torch.square(T21_pred_256[0,0]-T21_256[0,0]))).item()
    axes_hist_256[0].text(0.95, 0.95, f"RMSE: {RMSE:.3f}", ha='right', va='top', fontsize=plt.rcParams['font.size']-4, transform=axes_hist_256[0].transAxes)

    hist_min = min(T21_128[0,0].min().item(), T21_pred_128[0,0].min().item())
    hist_max = max(T21_128[0,0].max().item(), T21_pred_128[0,0].max().item())
    bins = np.linspace(hist_min, hist_max, 100)
    hist_true, _ = np.histogram(T21_128[0,0].flatten(), bins=bins, density=True)
    hist_pred, _ = np.histogram(T21_pred_128[0,0].flatten(), bins=bins, density=True)
    axes_hist_128[0].bar(bins[:-1], hist_true, width=bins[1]-bins[0], alpha=0.5, label="$T_{{21}}$ HR", rasterized=rasterized)
    axes_hist_128[0].bar(bins[:-1], hist_pred, width=bins[1]-bins[0], alpha=0.5, label="$T_{{21}}$ SR", rasterized=rasterized)
    hist_resid_128 = np.abs(hist_true - hist_pred)
    axes_hist_128[1].bar(bins[:-1], hist_resid_128, width=bins[1] - bins[0], alpha=0.5, label="$|\mathrm{{Residuals}}|$", color='k', rasterized=rasterized)
    axes_hist_128[1].set_xlabel("$T_{{21}}$ [mK]")
    axes_hist_128[0].legend(fontsize=plt.rcParams['font.size']-4, loc=legend_loc)
    #axes_hist_128[1].legend(fontsize=plt.rcParams['font.size']-2, loc=legend_loc)
    RMSE = torch.sqrt(torch.mean(torch.square(T21_pred_128[0,0]-T21_128[0,0]))).item()
    axes_hist_128[0].text(0.95, 0.95, f"RMSE: {RMSE:.3f}", ha='right', va='top', fontsize=plt.rcParams['font.size']-4, transform=axes_hist_128[0].transAxes)

    fig.align_ylabels(axes_hist_512)
    fig.align_ylabels(axes_hist_256)
    fig.align_ylabels(axes_hist_128)

    ##################

    sgs = SGS(2, 3, gs[4,:], height_ratios=[0.75,0.25], hspace=0., wspace=wspace)
    axes_dsq_512 = []
    axes_dsq_512.append(fig.add_subplot(sgs[0,0]))
    axes_dsq_512.append(fig.add_subplot(sgs[1,0], sharex=axes_dsq_512[0]))
    axes_dsq_512[0].tick_params(axis='both', labelbottom=False, direction='in', which='both')

    axes_dsq_256 = []
    axes_dsq_256.append(fig.add_subplot(sgs[0,1], sharex=axes_dsq_512[0], sharey=axes_dsq_512[0]))
    axes_dsq_256.append(fig.add_subplot(sgs[1,1], sharex=axes_dsq_512[1], sharey=axes_dsq_512[1]))
    axes_dsq_256[0].tick_params(axis='both', labelbottom=False, labelleft=False, direction='in', which='both')
    axes_dsq_256[1].tick_params(axis='both', labelleft=False, direction='in', which='both')

    axes_dsq_128 = []
    axes_dsq_128.append(fig.add_subplot(sgs[0,2], sharex=axes_dsq_512[0], sharey=axes_dsq_512[0]))
    axes_dsq_128.append(fig.add_subplot(sgs[1,2], sharex=axes_dsq_512[1], sharey=axes_dsq_512[1]))
    axes_dsq_128[0].tick_params(axis='both', labelbottom=False, labelleft=False, direction='in', which='both')
    axes_dsq_128[1].tick_params(axis='both', labelleft=False, direction='in', which='both')


    k_vals_true_512, dsq_true_512  = calculate_power_spectrum(T21_512[0:1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
    k_vals_pred_512, dsq_pred_512  = calculate_power_spectrum(T21_pred_512[0:1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
    dsq_resid_512 = torch.abs(dsq_pred_512[0,0] - dsq_true_512[0,0])
    axes_dsq_512[0].plot(k_vals_true_512, dsq_true_512[0,0], label="$T_{{21}}$ HR", ls='solid', lw=2)
    axes_dsq_512[0].plot(k_vals_pred_512, dsq_pred_512[0,0], label="$T_{{21}}$ SR", ls='solid', lw=2)
    axes_dsq_512[1].plot(k_vals_true_512, dsq_resid_512, lw=2, color='k', label="$|\mathrm{{Residuals}}|$")
    axes_dsq_512[0].set_xscale('log')
    axes_dsq_512[0].set_yscale('log')
    axes_dsq_512[1].set_yscale('log')
    axes_dsq_512[0].set_ylabel("$\Delta^2_{{21}}\\ \\mathrm{{[mK^2]}}$ ")
    #absolute residuals in $$ math mode
    axes_dsq_512[1].set_ylabel("$|\mathrm{{Residuals}}|$")
    axes_dsq_512[1].set_xlabel("$k\\ [\\mathrm{{cMpc^{-1}}}]$")

    k_vals_true_256, dsq_true_256  = calculate_power_spectrum(T21_256[0:1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
    k_vals_pred_256, dsq_pred_256  = calculate_power_spectrum(T21_pred_256[0:1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
    dsq_resid_256 = torch.abs(dsq_pred_256[0,0] - dsq_true_256[0,0])
    axes_dsq_256[0].plot(k_vals_true_256, dsq_true_256[0,0], label="$T_{{21}}$ HR", ls='solid', lw=2)
    axes_dsq_256[0].plot(k_vals_pred_256, dsq_pred_256[0,0], label="$T_{{21}}$ SR", ls='solid', lw=2)
    axes_dsq_256[1].plot(k_vals_true_256, dsq_resid_256, lw=2, color='k', label="$|\mathrm{{Residuals}}|$")
    axes_dsq_256[1].set_xlabel("$k\\ [\\mathrm{{cMpc^{-1}}}]$")

    k_vals_true_128, dsq_true_128  = calculate_power_spectrum(T21_128[0:1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
    k_vals_pred_128, dsq_pred_128  = calculate_power_spectrum(T21_pred_128[0:1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
    dsq_resid_128 = torch.abs(dsq_pred_128[0,0] - dsq_true_128[0,0])
    axes_dsq_128[0].plot(k_vals_true_128, dsq_true_128[0,0], label="$T_{{21}}$ HR", ls='solid', lw=2)
    axes_dsq_128[0].plot(k_vals_pred_128, dsq_pred_128[0,0], label="$T_{{21}}$ SR", ls='solid', lw=2)
    axes_dsq_128[1].plot(k_vals_true_128, dsq_resid_128, lw=2, color='k', label="$|\mathrm{{Residuals}}|$")
    axes_dsq_128[1].set_xlabel("$k\\ [\\mathrm{{cMpc^{-1}}}]$")
    axes_dsq_128[0].legend(fontsize=plt.rcParams['font.size']-4, loc=legend_loc)
    #axes_dsq_128[1].legend(fontsize=plt.rcParams['font.size']-2, loc=legend_loc)


    fig.align_ylabels(axes_dsq_512)
    fig.align_ylabels(axes_dsq_256)
    fig.align_ylabels(axes_dsq_128)

    plot_path = os.path.join(plot_dir, f"T21_scales_{rank}")
    plt.savefig(plot_path+plot_format, bbox_inches='tight')

    plt.close(fig)

    if multi_gpu:
        destroy_process_group()


@torch.no_grad()
def plot_sigmas(T21, T21_pred=None, netG=None, path = "", quantiles=[0.16, 0.5, 0.84], rasterized=True, **kwargs):
    if True:
        plt.rcParams.update({'font.size': 14,
                             "text.usetex": True,
                             "font.family": "serif",
                             "font.serif": "cm",
                             })
    T21 = T21.detach().cpu()
    T21_pred = T21_pred.detach().cpu()


    RMSE = ((T21 - T21_pred)**2).mean(dim=(1,2,3,4))**0.5
    RMSE_slice = ((T21 - T21_pred)**2).mean(dim=(1,3,4))**0.5

    row = 5
    col = len(quantiles)

    fig = plt.figure(figsize=(5*col,5*row))
    wspace = 0.2
    hspace = 0.3
    gs = GS(row, col, figure=fig, wspace=wspace, hspace=hspace)

    sgs_im = SGS(3,col, gs[:2,:], height_ratios=[0.05,1,1], hspace=0.3, wspace=wspace)
    sgs_resid = SGS(2,col, gs[2,:], height_ratios=[0.05,1], hspace=0., wspace=wspace)
    sgs_hist = SGS(2,col, gs[3,:], height_ratios=[3,1], hspace=0., wspace=wspace)
    sgs_dsq = SGS(2,col, gs[4,:], height_ratios=[3,1], hspace=0., wspace=wspace)

    for i,quantile in enumerate(quantiles):
        q = torch.quantile(input=RMSE, q=quantile, dim=(-1),)
        #find model index closest to q
        idx = torch.argmin(torch.abs(RMSE - q), dim=-1)
        #find slice closest to q
        idx_slice = torch.argmin(torch.abs(RMSE_slice[idx] - q), dim=-1)
        
        cax_im = fig.add_subplot(sgs_im[0,i])
        ax_true = fig.add_subplot(sgs_im[1,i])
        ax_pred = fig.add_subplot(sgs_im[2,i])
        vmin = min(T21[idx,0, idx_slice].min().item(), T21_pred[idx,0,idx_slice].min().item())
        vmax = max(T21[idx,0, idx_slice].max().item(), T21_pred[idx,0,idx_slice].max().item())
        img = ax_true.imshow(T21[idx,0,idx_slice], vmin=vmin, vmax=vmax, rasterized=rasterized)
        ax_pred.imshow(T21_pred[idx,0,idx_slice], vmin=vmin, vmax=vmax, rasterized=rasterized)
        ax_pred.set_title("$T_{{21}}$ SR", fontdict={"fontsize":plt.rcParams['font.size']})
        ax_true.xaxis.set_tick_params(labelbottom=False)
        ax_pred.xaxis.set_tick_params(labelbottom=False)
        cbar = fig.colorbar(img, cax=cax_im, orientation='horizontal')
        cbar.ax.tick_params(labelsize=plt.rcParams['font.size'], labeltop=True, labelbottom=False, top=True, bottom=False)
        cbar.set_label("$T_{{21}}$ HR [mK]", fontsize=plt.rcParams['font.size'])
        cbar.ax.xaxis.set_label_position('top')
        ax_pos = ax_true.get_position()
        cbar_pos = cax_im.get_position()
        cax_im.set_position([ax_pos.x0, ax_pos.y0+ax_pos.height+5e-3, ax_pos.width, cbar_pos.height])
        if (i == 0) or (i == 1) or (i == len(quantiles)-1):
            cbar.set_ticks([0,10,19])
        
        cax_resid = fig.add_subplot(sgs_resid[0,i])
        ax_resid = fig.add_subplot(sgs_resid[1,i])
        resid = T21[idx,0,idx_slice] - T21_pred[idx,0,idx_slice]
        vmin = -1 #resid_mean-2*resid_std
        vmax = 1 #resid_mean+2*resid_std
        img = ax_resid.imshow(resid, vmin=vmin, vmax=vmax, cmap='viridis', rasterized=rasterized)
        cbar = fig.colorbar(img, cax=cax_resid, orientation='horizontal')
        cbar.ax.tick_params(labelsize=plt.rcParams['font.size'], labeltop=True, labelbottom=False, top=True, bottom=False)
        cbar.set_label("$\mathrm{{Residuals}}$ [mK]", fontsize=plt.rcParams['font.size'])
        cbar.ax.xaxis.set_label_position('top')
        ax_pos = ax_resid.get_position()
        cbar_pos = cax_resid.get_position()
        cax_resid.set_position([ax_pos.x0, cbar_pos.y0+5e-3, ax_pos.width, cbar_pos.height])
        ax_resid.xaxis.set_tick_params(labelbottom=False)

        
        ax_hist = fig.add_subplot(sgs_hist[0,i], sharey=None if i==0 else ax_hist)
        ax_hist_resid = fig.add_subplot(sgs_hist[1,i], sharex=ax_hist, sharey=None if i==0 else ax_hist_resid)
        hist_min = min(T21[idx,0].min().item(), T21_pred[idx,0].min().item())
        hist_max = max(T21[idx,0].max().item(), T21_pred[idx,0].max().item())
        bins = np.linspace(hist_min, hist_max, 100)
        hist_true, _ = np.histogram(T21[idx,0,:,:,:].flatten(), bins=bins, density=True)
        hist_pred, _ = np.histogram(T21_pred[idx,0,:,:,:].flatten(), bins=bins, density=True)  # Reuse the same bins for consistency
        #hist_true = hist_true / np.sum(hist_true)
        #hist_pred = hist_pred / np.sum(hist_pred)
        ax_hist.bar(bins[:-1], hist_true, width=bins[1] - bins[0], alpha=0.5, label="T21 HR", rasterized=rasterized)
        ax_hist.bar(bins[:-1], hist_pred, width=bins[1] - bins[0], alpha=0.5, label="T21 SR", rasterized=rasterized)
        if i==0:
            ax_hist.set_ylabel("PDF")
        ax_hist.legend()
        ax_hist.set_title(f"$\mathrm{{RMSE}}_{{Q={quantile:.3f}}}={q:.3f}$", fontdict={"fontsize":plt.rcParams['font.size']})
        #logfmt = LogFormatterExponent(base=10.0, labelOnlyBase=False)
        #ax_hist.yaxis.set_major_formatter(logfmt)
        ax_hist.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        #ax_hist.get_yaxis().get_offset_text().set_position((-0.1,0.9))
        hist_resid = np.abs(hist_true - hist_pred)
        #hist_resid = hist_resid / np.sum(hist_resid)
        ax_hist_resid.bar(bins[:-1], hist_resid, width=bins[1] - bins[0], alpha=0.5, label="$|\mathrm{{Residuals}}|$", color='k', rasterized=rasterized)
        ax_hist_resid.legend()
        ax_hist_resid.set_xlabel("T21 [mK]")
        #ax_hist_resid.yaxis.set_major_formatter(logfmt)
        ax_hist_resid.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if i==0:
            ax_hist_resid.set_ylabel("$|\mathrm{{Residuals}}|$")
        fig.canvas.draw()
        default_text = ax_hist_resid.get_yaxis().get_offset_text().get_text()
        default_position = ax_hist_resid.get_yaxis().get_offset_text().get_position()

        #transform default position to 0 to 1 coordinates
        #default_position = ax_hist_resid.transData.transform(default_position)
        #default_position = ax_hist_resid.transData.inverted().transform(default_position)

        #hide offset text
        ax_hist_resid.get_yaxis().get_offset_text().set_visible(False)
        #manually set offset text slightly below default position
        ax_hist_resid.text(0, 0.95, default_text, transform=ax_hist_resid.transAxes, ha='left', va='top')
        fig.align_ylabels([ax_hist, ax_hist_resid])
        
        
        ax_dsq = fig.add_subplot(sgs_dsq[0,i], sharey=None if i==0 else ax_dsq)
        ax_dsq_resid = fig.add_subplot(sgs_dsq[1,i], sharex=ax_dsq, sharey=None if i==0 else ax_dsq_resid)
        k_vals_true, dsq_true  = calculate_power_spectrum(T21[idx:idx+1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
        k_vals_pred, dsq_pred  = calculate_power_spectrum(T21_pred[idx:idx+1], Lpix=3, kbins=100, dsq = True, method="torch", device="cpu")
        ax_dsq.plot(k_vals_true, dsq_true[0,0], label="T21 HR", ls='solid', lw=2, rasterized=rasterized)
        ax_dsq.plot(k_vals_pred, dsq_pred[0,0], label="T21 SR", ls='solid', lw=2, rasterized=rasterized)
        if i==0:
            ax_dsq.set_ylabel('$\Delta^2(k)_{{21}}$ [mK$^2$]')
        #ax_dsq.set_xlabel('$k$ [h/Mpc]')
        ax_dsq.set_xscale('log')
        ax_dsq.set_yscale('log')
        ax_dsq.grid()
        ax_dsq.legend()
        ax_dsq.xaxis.set_tick_params(labelbottom=False)

        dsq_resid = torch.abs(dsq_pred[0,0] - dsq_true[0,0])
        ax_dsq_resid.plot(k_vals_true, dsq_resid, lw=2, color='k', rasterized=rasterized)
        if i==0:
            ax_dsq_resid.set_ylabel("$|\mathrm{{Residuals}}|$")
        ax_dsq_resid.set_xlabel("$k\\ [\\mathrm{{cMpc^{-1}}}]$")
        #ax_dsq_resid.set_yscale('log')
        ax_dsq_resid.set_xscale('log')
        ax_dsq_resid.set_yscale('log')
        ax_dsq_resid.grid()
        
    plt.savefig(path + netG.model_name + "_quantiles.pdf", bbox_inches='tight', dpi=300)
    plt.close()


@torch.no_grad()
def plot_input(T21, delta, vbv, T21_lr, path=""):
    batch = T21.shape[0]
    indices = list(range(0,batch*2,2))
    
    fig,axes = plt.subplots(batch*2,4, figsize=(4*5, batch*2*5))
    for k,(T21_cpu, delta_cpu, vbv_cpu, T21_lr_cpu) in enumerate(zip(T21, delta, vbv, T21_lr)):
        k = indices[k]
        T21_cpu = T21_cpu.detach().cpu()
        delta_cpu = delta_cpu.detach().cpu()
        vbv_cpu = vbv_cpu.detach().cpu()
        T21_lr_cpu = T21_lr_cpu.detach().cpu()

        slice_idx = T21_cpu.shape[-3]//2
        axes[k,0].imshow(T21_cpu[0,slice_idx], vmin=T21_cpu.min().item(), vmax=T21_cpu.max().item())
        axes[k,0].set_title("T21 HR")
        axes[k+1,0].hist(T21_cpu[0].flatten(), bins=100, alpha=0.5, label="T21 HR", density=True)

        axes[k,1].imshow(delta_cpu[0,slice_idx], vmin=delta_cpu.min().item(), vmax=delta_cpu.max().item())
        axes[k,1].set_title("Delta")
        axes[k+1,1].hist(delta_cpu[0].flatten(), bins=100, alpha=0.5, label="Delta", density=True)

        axes[k,2].imshow(vbv_cpu[0,slice_idx], vmin=vbv_cpu.min().item(), vmax=vbv_cpu.max().item())
        axes[k,2].set_title("Vbv")
        axes[k+1,2].hist(vbv_cpu[0].flatten(), bins=100, alpha=0.5, label="Vbv", density=True)

        axes[k,3].imshow(T21_lr_cpu[0,slice_idx], vmin=T21_lr_cpu.min().item(), vmax=T21_lr_cpu.max().item())
        axes[k,3].set_title("T21 LR")
        axes[k+1,3].hist(T21_lr_cpu[0].flatten(), bins=100, alpha=0.5, label="T21 LR", density=True)
        
    plt.savefig(path)
    plt.close()

@torch.no_grad()
def plot_hist(T21_1, T21_2, path="", label="true diff", **kwargs):
    T21_1 = T21_1.detach().cpu()
    T21_2 = T21_2.detach().cpu()

    fig,ax = plt.subplots(1,1, figsize=(5,5))

    mean_diff = (T21_1 - T21_2).mean(dim=(1,2,3,4))
    ax.hist(mean_diff, bins=20, alpha=0.5, label="Mean difference", density=True)
    ax.set_xlabel(label)
    # ax.set_xlabel("Mean T21_true - T21_pred [mK]")
    #ax.set_xlabel("Mean T21_validation before - T21_validation after [mK]")
    
    plt.savefig(path)
    plt.close()