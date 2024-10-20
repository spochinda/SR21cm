import os
import torch
import numpy as np
from scipy.io import loadmat
import pandas as pd
import torch.distributed
import torch.utils
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_T21, path_IC, 
                 redshifts=[10,], IC_seeds=list(range(1000,1008)), 
                 Npix=256, device='cpu'):
        self.device = device
        self.path_T21 = path_T21
        self.path_IC = path_IC

        self.redshifts = redshifts
        self.IC_seeds = IC_seeds
        self.Npix = Npix
        
        self.df = self.getDataFrame()
        try: 
            if torch.cuda.current_device() == 0:
                print("Loaded dataframe", flush=True)
                print(self.df, flush=True)
        except:
            pass
            
        #self.labels = pd.read_csv(annotations_file)
        #self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    @torch.no_grad()
    def __getitem__(self, idx):
        #if self.dataset exists
        if hasattr(self, 'dataset'):
            T21 = self.dataset.tensors[0][idx].to(self.device)
            delta = self.dataset.tensors[1][idx].to(self.device)
            vbv = self.dataset.tensors[2][idx].to(self.device)
            labels = self.dataset.tensors[3][idx].to(self.device)
        else:
            #load_stime = time.time()
            T21 = loadmat(self.path_T21 + self.df[["T21"]].iloc[idx].values[0])["Tlin"]
            delta = loadmat(self.path_IC + self.df[["delta"]].iloc[idx].values[0])["delta"]
            vbv = loadmat(self.path_IC + self.df[["vbv"]].iloc[idx].values[0])["vbv"]
            #load_etime = time.time()
            #convert time
            #convert_stime = time.time()
            T21 = torch.from_numpy(T21).to(torch.float32).unsqueeze(0)
            delta = torch.from_numpy(delta).to(torch.float32).unsqueeze(0)
            vbv = torch.from_numpy(vbv).to(torch.float32).unsqueeze(0)
            #convert_etime = time.time()

            #gpumove time
            #gpumove_stime = time.time()
            T21 = T21.to(self.device)
            delta = delta.to(self.device)
            vbv = vbv.to(self.device)
            #gpumove_etime = time.time()

            labels = torch.tensor(self.df[["labels (z)"]].iloc[idx].values[0]).to(torch.float32).to(self.device)
        #if torch.cuda.current_device() == 0:
        #    print(f"Time: Load: {load_etime - load_stime:.3f}, Convert: {convert_etime - convert_stime:.3f}, GpuMove: {gpumove_etime - gpumove_stime:.3f}", flush=True)
        # Time: Load: 1.907, Convert: 0.000, GpuMove: 0.094
        return T21, delta, vbv, labels#T21_lr, labels
    
    def getDataFrame(self):
        rows = []
        for IC_seed in self.IC_seeds:
            for redshift in self.redshifts:
                #for i in range((2**cut_factor)**3):
                row = [[IC_seed,], [redshift,]] #[[IC_seed, i], [redshift,]] #[f"IC={IC_seed}, z={redshift}"]
                for file in os.listdir(self.path_T21):
                    if ('T21_cube' in file) and (f'Npix{self.Npix}' in file):
                        z = int(file.split('z')[1].split('_')[0])
                        try:
                            IC = int(file.split('_')[7])
                        except:
                            IC = int(file.split('IC')[-1].split('.mat')[0])
                        if (z == redshift) and (IC == IC_seed):
                            row.append(file)
                for file in os.listdir(self.path_IC):
                    if ('delta' in file) and (f'Npix{self.Npix}' in file):
                        IC = int(file.split('IC')[1].split('.')[0])
                        if IC == IC_seed:
                            row.append(file)
                for file in os.listdir(self.path_IC):
                    if ('vbv' in file) and (f'Npix{self.Npix}' in file):
                        IC = int(file.split('IC')[1].split('.')[0])
                        if IC == IC_seed:
                            row.append(file)
                if len(row) == 5: #match number of columns
                    rows.append(row)
        df = pd.DataFrame(rows, columns=['IC,subcube', 'labels (z)', 'T21', 'delta', 'vbv'])
        
        return df

    @torch.no_grad()
    def getFullDataset(self):
        #load full dataset into CPU memory

        T21 = torch.empty(0,1, self.Npix, self.Npix, self.Npix, device='cpu')
        delta = torch.empty(0,1, self.Npix, self.Npix, self.Npix, device='cpu')
        vbv = torch.empty(0,1, self.Npix, self.Npix, self.Npix, device='cpu')
        labels = torch.empty(0, device='cpu')

        
        for index, row in self.df.iterrows():

            T21_file = row['T21']
            delta_file = row['delta']
            vbv_file = row['vbv']
            label = row['labels (z)']

            T21_cube = loadmat(self.path_T21 + T21_file)["Tlin"]
            T21_cube = torch.from_numpy(T21_cube).to(torch.float32).unsqueeze(0).unsqueeze(0)
            T21 = torch.cat([T21, T21_cube], dim=0)
            
            delta_cube = loadmat(self.path_IC + delta_file)["delta"]
            delta_cube = torch.from_numpy(delta_cube).to(torch.float32).unsqueeze(0).unsqueeze(0)
            delta = torch.cat([delta, delta_cube], dim=0)
            
            vbv_cube = loadmat(self.path_IC + vbv_file)["vbv"]
            vbv_cube = torch.from_numpy(vbv_cube).to(torch.float32).unsqueeze(0).unsqueeze(0)
            vbv = torch.cat([vbv, vbv_cube], dim=0)

            label = torch.as_tensor(label, dtype=torch.float32, device='cpu')
            labels = torch.cat([labels, label], dim=0)
            

            if torch.cuda.current_device() == 0:
                try:
                    #print every 10% loaded
                    if index % (len(self.df)//10) == 0:
                        print(f"Loaded {index}/{len(self.df)}", flush=True)
                except:
                    pass
            
        self.dataset = torch.utils.data.TensorDataset(T21, delta, vbv, labels)
        
        return self.dataset


def linear_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
    beta_schedule = torch.linspace(beta_start, beta_end, timesteps)
    return beta_schedule

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    Variance schedule. cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    beta_schedule = 1 - (alphas[1:] / alphas[: -1])
    result = torch.clip(input = beta_schedule, min = 0.0001, max = 0.9999)

    return result

def beta_schedule(schedule_type = "cosine", schedule_opt = {}):
    if schedule_type == "cosine":
        return cosine_beta_schedule(**schedule_opt)
    elif schedule_type == "linear":
        return linear_beta_schedule(**schedule_opt)
    elif schedule_type == "VPSDE":
        discrete_betas = torch.linspace(schedule_opt['beta_min'] / schedule_opt['timesteps'], 
                                        schedule_opt['beta_max'] / schedule_opt['timesteps'], 
                                        schedule_opt['timesteps'])
        return discrete_betas

    else:
        raise ValueError("schedule_type must be one of ['cosine', 'linear', 'VPSDE', 'EDM']")



def rot_onto_sides_torch(x):
    x1 = torch.rot90(x, k=1, dims=(3,4)).unsqueeze(0)
    x2 = torch.rot90(x, k=1, dims=(2,4)).unsqueeze(0)
    x3 = torch.rot90(x, k=1, dims=(2,3)).unsqueeze(0)
    x = torch.cat([x1,x2,x3], dim=0)
    return x

def rot_onto_sides(x):
    x1 = torch.rot90(x, k=1, dims=(3,4))
    x2 = torch.rot90(x, k=1, dims=(2,4))
    x3 = torch.rot90(x, k=1, dims=(2,3))
    return [x1,x2,x3]

def rot_on_base(cubes):
    x = []
    for cube in cubes:
        for i in range(4):
            x.append(torch.rot90(cube, k=i, dims=(2,3)))
    return x

def rot_to_opposite_corner(x):
    x = torch.rot90(x, k=2, dims=(2,4))
    x = torch.rot90(x, k=1, dims=(2,3))
    return x

def all_rotations(x):
    corner1 = rot_on_base(rot_onto_sides(x))
    corner2 = rot_on_base(rot_onto_sides(rot_to_opposite_corner(x)))
    #print("corner1[0] shape", corner1[0].shape, "corner2[0] shape", corner2[0].shape, flush=True)
    result = torch.cat([*corner1, *corner2], dim=0) #np.array(corner1 + corner2)
    #print("rotations shape: ", result.shape, flush=True)
    return result

def random_rotations(x, n=1):

    N = np.random.choice(np.arange(0,24), size=n,replace=False)
    N = torch.tensor(N, device=x.device)


    rotations = all_rotations(x)
    
    result = rotations[N]

    return result


@torch.no_grad()
def augment_dataset(T21, delta, vbv, T21_lr=None, n=8, broadcast=False):
    #dataset_x1 = []
    #dataset_x2 = []
    #dataset_x3 = []
    #dataset_x4 = []
    dataset_T21 = torch.empty(0,1, T21.shape[2], T21.shape[3], T21.shape[4], device=T21.device)
    dataset_delta = torch.empty(0,1, delta.shape[2], delta.shape[3], delta.shape[4], device=delta.device)
    dataset_vbv = torch.empty(0,1, vbv.shape[2], vbv.shape[3], vbv.shape[4], device=vbv.device)
    if T21_lr is not None:
        dataset_T21_lr = torch.empty(0,1, T21_lr.shape[2], T21_lr.shape[3], T21_lr.shape[4], device=T21_lr.device)
    #for i,(x1,x2,x3,x4) in enumerate(zip(T21, delta, vbv, T21_lr)):
    for i in range(T21.shape[0]):
        #x1 = x1.unsqueeze(0)
        #x2 = x2.unsqueeze(0)
        #x3 = x3.unsqueeze(0)
        #x4 = x4.unsqueeze(0)
        T21_i = T21[i].unsqueeze(0)
        delta_i = delta[i].unsqueeze(0)
        vbv_i = vbv[i].unsqueeze(0)
        if T21_lr is not None:
            T21_lr_i = T21_lr[i].unsqueeze(0)

        N = np.random.choice(np.arange(0,24), size=n,replace=False)
        N = torch.tensor(N, device=T21_i.device)
        if broadcast:
            torch.distributed.broadcast(N, src=0)
        #print(f"Rank {torch.cuda.current_device()} N: {N}", flush=True)
        #rotations_x1 = all_rotations(x1)[N]
        #rotations_x2 = all_rotations(x2)[N]
        #rotations_x3 = all_rotations(x3)[N]
        #rotations_x4 = all_rotations(x4)[N]
        T21_i = all_rotations(T21_i)[N]
        delta_i = all_rotations(delta_i)[N]
        vbv_i = all_rotations(vbv_i)[N]
        if T21_lr is not None:
            T21_lr_i = all_rotations(T21_lr_i)[N]

        #dataset_x1.append(rotations_x1)
        #dataset_x2.append(rotations_x2)
        #dataset_x3.append(rotations_x3)
        #dataset_x4.append(rotations_x4)
        dataset_T21 = torch.cat([dataset_T21, T21_i], dim=0)
        dataset_delta = torch.cat([dataset_delta, delta_i], dim=0)
        dataset_vbv = torch.cat([dataset_vbv, vbv_i], dim=0)
        if T21_lr is not None:
            dataset_T21_lr = torch.cat([dataset_T21_lr, T21_lr_i], dim=0)
        else:
            dataset_T21_lr = None

        
    #T21 = torch.cat(dataset_x1,dim=0)
    #delta = torch.cat(dataset_x2,dim=0)
    #vbv = torch.cat(dataset_x3,dim=0)
    #T21_lr = torch.cat(dataset_x4,dim=0)
    
    #return T21, delta, vbv, T21_lr
    return dataset_T21, dataset_delta, dataset_vbv, dataset_T21_lr

def calculate_power_spectrum(data_x, Lpix=3, kbins=100, dsq = False, method="torch", device="cpu"):
    #Simulation box variables
    batch, channel,*d = data_x.shape
    assert channel == 1, "Channel must be 1"
    Npix = d[-1]
    Vpix = Lpix**3
    Lbox = Npix * Lpix
    Vbox = Lbox**3

    if method == "numpy":
        kspace = np.fft.fftfreq(Npix, d=Lpix/(2*np.pi)) #Calculating wavevectors k for the simulation grid
        kx, ky, kz = np.meshgrid(kspace,kspace,kspace)
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        data_k = np.fft.fftn(data_x[0,0]) #Dont need to scipy.fft.fftshift since kspace isn't fftshift'ed
        k_bin_edges = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), endpoint=True, num=kbins+1) #Bin k values and calculate power spectrum
        k_vals = np.zeros(kbins)
        P_k = np.zeros(kbins)
        for i in range(kbins):
            cond = ((k >= k_bin_edges[i]) & (k < k_bin_edges[i+1]))
            k_vals[i] = (k_bin_edges[i+1] + k_bin_edges[i])/2
            P_k[i] = (Vpix/Vbox) * Vpix * np.average(np.absolute(data_k[cond])**2)
        P_k = P_k*k_vals**3/(2*np.pi**2) if dsq else P_k
        return k_vals, P_k
    elif method == "torch":
        kspace_torch = torch.fft.fftfreq(Npix, d=Lpix/(2*np.pi), device=device)
        kx_torch, ky_torch, kz_torch = torch.meshgrid(kspace_torch,kspace_torch,kspace_torch)#.view(batch,channel,*d)
        k_torch = torch.sqrt(kx_torch**2 + ky_torch**2 + kz_torch**2)#.unsqueeze(0).unsqueeze(0).repeat(batch,1,*(len(d)*[1]))
        data_k_torch = torch.fft.fftn(input=data_x, dim=(2,3,4))
        kmin_mask_torch = k_torch > 0
        kmin_torch = torch.min(k_torch[kmin_mask_torch])
        kmax_torch = torch.max(k_torch)
        k_bin_edges_torch = torch.logspace(start=torch.log10(kmin_torch), end=torch.log10(kmax_torch), steps=kbins+1, device=device)
        k_vals_torch = torch.zeros(kbins, device=device)
        P_k_torch = torch.zeros(batch,channel,kbins, device=device)
        
        conditions = [(k_torch >= k_bin_edges_torch[i]) & (k_torch < k_bin_edges_torch[i+1]) for i in range(kbins)]

        for i in range(kbins):
            cond_torch = conditions[i]
            k_vals_torch[i] = (k_bin_edges_torch[i+1] + k_bin_edges_torch[i])/2
            means = torch.mean(torch.abs(data_k_torch[...,cond_torch])**2,dim=2, keepdim=False)
            P_k_torch[:,:,i] = (Vpix/Vbox) * Vpix * means
        P_k_torch = P_k_torch*k_vals_torch**3/(2*np.pi**2) if dsq else P_k_torch
        return k_vals_torch, P_k_torch
    else:
        raise ValueError("Method must be numpy or torch")

@torch.no_grad()
def get_subcubes(cubes, cut_factor=0):
    if cut_factor > 0:
        batch, channel,*d = cubes.shape
        image_size = d[0]//(2**cut_factor)
        sub_cubes = []
        for cube in cubes:
            for i in range(2**cut_factor):
                for j in range(2**cut_factor):
                    for k in range(2**cut_factor):
                        sub_cube = cube[:,i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size,k*image_size:(k+1)*image_size]
                        sub_cubes.append(sub_cube)
        sub_cubes = torch.cat(sub_cubes, dim=0).unsqueeze(1)
        return sub_cubes
    else:
        return cubes
    
@torch.no_grad()
def normalize(x, mode = "standard", **kwargs):
    if mode == "standard":
        x_mean = kwargs.pop("x_mean", torch.mean(x, dim=(1,2,3,4), keepdim=True))
        x_std = kwargs.pop("x_std", torch.std(x, dim=(1,2,3,4), keepdim=True))
        factor = kwargs.pop("factor", 1.)

        x = (x - x_mean) / (factor * x_std) #should the 2 be there? LR std approx. 0.5*HR std

        return x, x_mean, x_std

    elif mode == "minmax":
        x_min = kwargs.pop("x_min", torch.amin(x, dim=(1,2,3,4), keepdim=True))
        x_max = kwargs.pop("x_max", torch.amax(x, dim=(1,2,3,4), keepdim=True))
        
        x = (x - x_min) / (x_max - x_min)
        x = 2 * x - 1

        return x, x_min, x_max
    
    
@torch.no_grad()
def invert_normalization(x, mode="standard", **kwargs):
    if mode == "standard":
        x_mean = kwargs.pop("x_mean", torch.mean(x, dim=(1,2,3,4), keepdim=True))
        x_std = kwargs.pop("x_std", torch.std(x, dim=(1,2,3,4), keepdim=True))
        factor = kwargs.pop("factor", 1.)
        
        x = x * (factor * x_std) + x_mean
        return x
    elif mode == "minmax":
        x_min = kwargs.pop("x_min", torch.amin(x, dim=(1,2,3,4), keepdim=True))
        x_max = kwargs.pop("x_max", torch.amax(x, dim=(1,2,3,4), keepdim=True))
        
        x = ((x + 1) * (x_max - x_min) / 2 ) + x_min
        return x

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
def sample_model_v3(rank, netG, dataloader, cut_factor=1, norm_factor = 1., augment=1, split_batch = True, sub_batch = 4, n_boxes = 1, num_steps=100, 
                    device="cpu", multi_gpu=False):
    assert netG.noise_schedule_opt["schedule_type"] == "VPSDE", "Only VPSDE supported"
    assert n_boxes > 0 or n_boxes == -1, "n_boxes has to be greater than 0 or -1 for all boxes"
    assert augment >= 0 or augment <= 24, "augment has to be between 0 and 24"
    
    world_size = torch.cuda.device_count()
    multi_gpu = world_size > 1
    
    T21_pred_cpu = torch.empty(0, device='cpu')
    T21_cpu = torch.empty(0, device='cpu')
    labels_cpu = torch.empty(0, device='cpu')
    for i,(T21, delta, vbv, labels) in tqdm(enumerate(dataloader), desc='sampling loop', total=len(dataloader), disable=False):
        #prepare data
        T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
        delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
        vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
        T21_lr = torch.nn.functional.interpolate(T21, scale_factor=1/4, mode='trilinear') # get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
        if augment:
            T21, delta, vbv , T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=augment) #support device

        T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
        
        T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)
        
        T21_lr, _,_ = normalize(T21_lr, mode="standard", factor=norm_factor)#, factor=2.)
        T21, _,_ = normalize(T21, mode="standard", factor=norm_factor, x_mean=T21_lr_mean, x_std=T21_lr_std)#, factor=2.) #####
        delta, _,_ = normalize(delta, mode="standard", factor=norm_factor)#, factor=2.)
        vbv, _,_ = normalize(vbv, mode="standard", factor=norm_factor)#, factor=2.)
        
        #T21_lr_mean = T21_lr_mean[:1]
        #T21_lr_std = T21_lr_std[:1]
        #T21_lr = T21_lr[:1]
        #T21 = T21[:1]
        #delta = delta[:1]
        #vbv = vbv[:1]
        
        if split_batch: #split subcube minibatch into smaller mini-batches for memory
            sub_data = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std)
            sub_dataloader = torch.utils.data.DataLoader(sub_data, batch_size=sub_batch, shuffle=False, sampler = None) 
            
            for j,(T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std) in tqdm(enumerate(sub_dataloader), desc='sampling subloop', total=len(sub_dataloader), disable=False if str(device)=="cuda:0" else True):
                T21_pred_j = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=None, num_steps=num_steps, eps=1e-3, use_amp=False, clip_denoised=False, verbose=True if str(device)=="cuda:0" else False)[:,-1:].to(device=device)
                
                T21_pred_j = invert_normalization(T21_pred_j, mode="standard", factor=norm_factor, x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)
                T21 = invert_normalization(T21, mode="standard", factor=norm_factor, x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)

                T21_pred_cpu = torch.cat((T21_pred_cpu, T21_pred_j.detach().cpu()), dim=0)
                T21_cpu = torch.cat((T21_cpu, T21.detach().cpu()), dim=0)
        else:
            #if torch.cuda.current_device() == 0:
            #    print("Validation without subbatching, shapes: ", T21.shape, delta.shape, vbv.shape, T21_lr.shape, flush=True)
            labels = labels #None to disable redshift-conditional generation
            T21_pred_i = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=num_steps, eps=1e-3, use_amp=False, clip_denoised=False, verbose=True)[:,-1:].to(device=device)
            T21_pred_i = invert_normalization(T21_pred_i, mode="standard", factor=norm_factor, x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)
            T21 = invert_normalization(T21, mode="standard", factor=norm_factor, x_mean = T21_lr_mean, x_std = T21_lr_std)

            try:
                RMSE_temp = torch.sqrt(torch.mean(torch.square(T21_pred_i - T21),dim=(1,2,3,4), keepdim=False))
                print(f"Rank {rank} finished sampling. RMSE: {RMSE_temp}, redshift: {labels}", flush=True)
            except Exception as e:
                print(f"Rank {rank} finished sampling. Error: {e}, redshift: {labels}", flush=True)

            T21_pred_cpu = torch.cat((T21_pred_cpu, T21_pred_i.detach().cpu()), dim=0)
            T21_cpu = torch.cat((T21_cpu, T21.detach().cpu()), dim=0)
            labels_cpu = torch.cat((labels_cpu, labels.detach().cpu()), dim=0)
        
        if multi_gpu:
            torch.distributed.barrier()

        if i==n_boxes-1:
            break #only do n_boxes (-1 or -random number for all)
    
    # Gather tensors from all ranks on the CPU
    if multi_gpu:
        T21_pred_list = [torch.empty_like(T21_pred_cpu) for _ in range(world_size)]
        T21_list = [torch.empty_like(T21_cpu) for _ in range(world_size)]
        labels_list = [torch.empty_like(labels_cpu) for _ in range(world_size)]
        
        #print date and time for each rank as it starts sampling
        #print(f"Rank {rank} finished sampling at {datetime.datetime.now()}. Waiting for all processes to finish...", flush=True)
        torch.distributed.barrier()
        torch.distributed.all_gather(T21_pred_list, T21_pred_cpu)
        torch.distributed.all_gather(T21_list, T21_cpu)
        torch.distributed.all_gather(labels_list, labels_cpu)
        
        # Concatenate the gathered tensors
        T21_pred = torch.cat(T21_pred_list, dim=0)
        T21 = torch.cat(T21_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
    else:
        T21_pred = T21_pred_cpu
        T21 = T21_cpu
        labels = labels_cpu

    MSE = torch.mean(torch.square(T21_pred - T21),dim=(1,2,3,4), keepdim=False).mean().item()
    
    return MSE, dict(T21=T21, T21_pred=T21_pred, labels=labels)
    
