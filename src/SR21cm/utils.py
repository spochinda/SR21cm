import os

import torch
import torch.distributed
import torch.utils
import torch.utils.data
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
import pandas as pd
import yaml


from .model_edm import SongUNet
from .loss import VPLoss
from .diffusion import GaussianDiffusion

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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_T21, path_IC, 
                 redshifts=[10,], IC_seeds=list(range(1000,1008)), 
                 Npix=256, batch_size=1, load_full_dataset=False,
                 growth_factor=False, device='cpu'):
        self.device = device
        self.path_T21 = path_T21
        self.path_IC = path_IC

        self.redshifts = redshifts
        self.IC_seeds = IC_seeds
        self.Npix = Npix
        self.batch_size = batch_size
        self.load_full_dataset = load_full_dataset
        self.df = self.getDataFrame()
        
        if growth_factor:
            self.g = get_growth_factor(device=device)

        if load_full_dataset:
            self.dataset = self.getFullDataset()


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
        if hasattr(self, 'g'):
            growth_factor = self.g.growth_factor(labels)
            delta = delta * growth_factor
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
    
class get_growth_factor():
    def __init__(self, path_D = None, path_z = None, device='cpu'):
        #get file path and load Dz.mat
        if path_D is None:
            self.path_D = os.path.dirname(__file__) + '/Dz.mat'
        else:
            self.path_D = path_D
        if path_z is None:    
            self.path_z = os.path.dirname(__file__) + '/zs_for_D.mat'
        else:
            self.path_z = path_z
        self.D = loadmat(self.path_D)['D'].flatten()
        self.z = loadmat(self.path_z)['zs'].flatten()
        self.idx40 = np.argmin((self.z - 40)**2)
        self.D40 = self.D[self.idx40]
                       
        self.device = device

    def growth_factor(self, z):
        growth_factor = np.interp(z, self.z, self.D) / self.D40
        growth_factor = torch.tensor(growth_factor, dtype=torch.float32, device=self.device)
        return growth_factor



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
def data_preprocess(T21, delta, vbv, cut_factor=0, scale_factor=4, norm_factor=1., n_augment=1, **kwargs):
    T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
    delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
    vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
    T21_lr = torch.nn.functional.interpolate(T21, scale_factor=1/scale_factor, mode='trilinear')#get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
                
    T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
    T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
    T21_lr = torch.nn.Upsample(scale_factor=scale_factor, mode='trilinear')(T21_lr)
    
    T21_lr, _,_ = normalize(T21_lr, mode="standard", factor=norm_factor)#, factor=2.)
    T21, _,_ = normalize(T21, mode="standard", factor=norm_factor, x_mean=T21_lr_mean, x_std=T21_lr_std)
    delta, _,_ = normalize(delta, mode="standard", factor=norm_factor)
    vbv, _,_ = normalize(vbv, mode="standard", factor=norm_factor)
    T21, delta, vbv , T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=n_augment) #support device
    return T21, delta, vbv, T21_lr #, T21_lr_mean, T21_lr_std

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






@torch.no_grad()
def sample_scales(rank, world_size, model_path, **kwargs):
    """
    Samples scales of T21 cubes using a Gaussian Diffusion model.
    Args:
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
        model_path (str): Path to the pre-trained model.
        **kwargs: Additional keyword arguments.
            save_path (str, optional): Path to save the output. Defaults to the base name of model_path.
            one_box (bool, optional): If True, only use sample first box index. Defaults to False.
    Returns:
        None. Saves the sampled T21 scales to the specified path.
    Notes:
        - This function sets up a distributed data parallel environment.
        - Loads pre-trained model and T21 cubes.
        - Normalizes and processes the T21 cubes and their corresponding delta and vbv cubes.
        - Uses the Gaussian Diffusion model to sample new T21 cubes.
        - Saves the original and predicted T21 cubes to a dictionary and saves it to disk.
    """
    save_path = kwargs.pop("save_path", model_path.split(".")[0])
    one_box = kwargs.pop("one_box", True)
    
    multi_gpu = world_size >= 1
    if multi_gpu:
        device = torch.device(f'cuda:{rank}')
        ddp_setup(rank, world_size=world_size)#multi_gpu = world_size > 1
    else:
        device = torch.device('cpu')

    network_opt = dict(img_resolution=128, in_channels=4, out_channels=1, label_dim=0, # (for tokens?), augment_dim,
                    model_channels=8, channel_mult=[1,2,4,8,16], num_blocks = 4, attn_resolutions=[8], mid_attn=True, #channel_mult_emb, num_blocks, attn_resolutions, dropout, label_dropout,
                    embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1,1], 
                    )
    
    network = SongUNet
    
    #noise_schedule_opt = {'schedule_type': "linear", 'schedule_opt': {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}} 
    #noise_schedule_opt = {'schedule_type': "cosine", 'schedule_opt': {"timesteps": 1000, "s" : 0.008}} 
    #noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    noise_schedule_opt = {'schedule_type': "VPSDE", 'schedule_opt': {"timesteps": 1000, "beta_min" : 0.1, "beta_max": 20.0}}  
    
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
    
    #netG.multi_gpu = False
    
    netG.load_network(model_path)

    netG.model_name = model_path.split("/")[-1].split(".")[0]

    T21_dict = {}
    for key1,key2,cut in zip(["T21_512", "T21_256", "T21_128"],["T21_pred_512", "T21_pred_256", "T21_pred_128"],[0,1,2]):
        T21 = loadmat("/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/T21_cubes/T21_cube_z10__Npix512_IC0.mat")["Tlin"]
        T21 = torch.from_numpy(T21).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device=device)

        delta = loadmat("/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/delta_Npix512_IC0.mat")["delta"]
        delta = torch.from_numpy(delta).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device=device)

        vbv = loadmat("/home/sp2053/rds/rds-cosmicdawnruns2-PJtLerV8oy0/JVD_diffusion_sims/varying_IC/IC_cubes/vbv_Npix512_IC0.mat")["vbv"]
        vbv = torch.from_numpy(vbv).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device=device)


        T21 = get_subcubes(cubes=T21, cut_factor=cut)
        delta = get_subcubes(cubes=delta, cut_factor=cut)
        vbv = get_subcubes(cubes=vbv, cut_factor=cut)
        if one_box:
            T21 = T21[:1]
            delta = delta[:1]
            vbv = vbv[:1]
        T21_lr = torch.nn.functional.interpolate(T21, scale_factor=1/4, mode='trilinear')#get_subcubes(cubes=T21_lr, cut_factor=cut_factor)
                    
        T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
        T21_lr = torch.nn.Upsample(scale_factor=4, mode='trilinear')(T21_lr)
        
        T21_lr, _,_ = normalize(T21_lr, mode="standard", factor=1.)#, factor=2.)
        T21, _,_ = normalize(T21, mode="standard", factor=1., x_mean=T21_lr_mean, x_std=T21_lr_std)
        delta, _,_ = normalize(delta, mode="standard", factor=1.)
        vbv, _,_ = normalize(vbv, mode="standard", factor=1.)

        labels = None

        #print("networkopt", netG.network_opt["label_dim"], flush=True)
        #torch.distributed.barrier()

        T21_pred = netG.sample.Euler_Maruyama_sampler(netG=netG, x_lr=T21_lr, conditionals=[delta, vbv], class_labels=labels, num_steps=100, eps=1e-3, use_amp=False, clip_denoised=False, verbose=True if device.index == 0 or device.type == 'cpu' else False)[:,-1:].to(device=device)
        T21_pred = invert_normalization(T21_pred, mode="standard", factor=1., x_mean = T21_lr_mean, x_std = T21_lr_std)#, factor=2.)
        T21 = invert_normalization(T21, mode="standard", factor=1., x_mean = T21_lr_mean, x_std = T21_lr_std)

        T21_dict[key1] = T21.detach().cpu()
        T21_dict[key2] = T21_pred.detach().cpu()

        del T21, T21_pred, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std

        print(f"[dev:{rank}] Finished {key1}, {key2}", flush=True)

    torch.save(T21_dict, save_path+f"T21_scales_{netG.model_name}_rank_{rank}.pth")

    if multi_gpu:
        destroy_process_group()

def initialize_model_directory(rank, config):
    path = config["path"]
    model_name = config["name"].split(".")[0]
    model_dir = os.path.join(path, model_name)
    plot_dir = os.path.join(model_dir, "plots")
    data_dir = os.path.join(model_dir, "data")
    model_path = os.path.join(model_dir, config["name"])
    config_path = os.path.join(model_dir, "config.yml")
    if not os.path.exists(model_dir):
        if rank==0:
            os.makedirs(model_dir)
            os.makedirs(plot_dir)
            os.makedirs(data_dir)
            #save config file
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
    else:
        if not os.path.exists(plot_dir):
            if rank==0:
                os.makedirs(plot_dir)
        if not os.path.exists(data_dir):
            if rank==0:
                os.makedirs(data_dir)
        #if config exists
        if not os.path.exists(config_path):
            #save config
            if rank==0:
                with open(config_path, 'w') as file:
                    yaml.dump(config, file)
        else:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        
        #does model_path exist
        #if os.path.exists(model_path):
        #    try:
        #        netG.load_network(model_path)
        #        if rank==0:
        #            print(f"Loaded network at {model_path}", flush=True)
        #    except:
        #        if rank==0:
        #            print(f"Error loading network at {model_path}. Starting from scratch.", flush=True)                
        #else:
        #    if rank==0:
        #        print(f"No model file found at {model_path}. Starting from scratch.", flush=True)
    return config, model_name, model_path, model_dir, plot_dir, data_dir