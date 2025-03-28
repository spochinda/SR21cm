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
                 device='cpu', **kwargs):
        self.device = device
        self.path_T21 = path_T21
        self.path_IC = path_IC

        self.redshifts = redshifts
        self.IC_seeds = IC_seeds
        self.Npix = Npix
        self.batch_size = batch_size
        self.load_full_dataset = load_full_dataset
        self.df = self.getDataFrame()
        

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
                        
        self.dataset = torch.utils.data.TensorDataset(T21, delta, vbv, labels)
        
        return self.dataset
    


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
    if n == 0:
        return T21, delta, vbv, T21_lr
    elif n>0 and n<24:
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
def data_preprocess(T21, delta, vbv, **kwargs):
    cut_factor = kwargs.pop("cut_factor", 1)
    norm_factor = kwargs.pop("norm_factor", 1.)
    n_augment = kwargs.pop("n_augment", 1)
    one_box = kwargs.pop("one_box", False)
    scale_max = kwargs.pop("scale_max", 4)
    scale_min = kwargs.pop("scale_min", 1)

    assert scale_min >= 1.1, "Scale min must be greater than 1.1"


    T21 = get_subcubes(cubes=T21, cut_factor=cut_factor)
    delta = get_subcubes(cubes=delta, cut_factor=cut_factor)
    vbv = get_subcubes(cubes=vbv, cut_factor=cut_factor)
    if one_box:
            T21 = T21[:1]
            delta = delta[:1]
            vbv = vbv[:1]

    b,c,h,w,d = T21.shape

    #draw random scale factor from 2 to 4. make sure it is divisible so we can Pixelshuffle
    scale_factor = np.random.rand(1)[0] * (scale_max - scale_min) + scale_min
    while (round(h/scale_factor) / 4) % 2 != 0:
        scale_factor = np.random.rand(1)[0] * (scale_max - scale_min) + scale_min
    h_lr = round(h/scale_factor)

    T21_lr = torch.nn.functional.interpolate(T21, size=h_lr, mode='trilinear')#get_subcubes(cubes=T21_lr, cut_factor=cut_factor)

    T21, delta, vbv , T21_lr = augment_dataset(T21, delta, vbv, T21_lr, n=n_augment)

    T21_lr_mean = torch.mean(T21_lr, dim=(1,2,3,4), keepdim=True)
    T21_lr_std = torch.std(T21_lr, dim=(1,2,3,4), keepdim=True)
    #T21_lr = torch.nn.Upsample(scale_factor=scale_factor, mode='trilinear')(T21_lr)
    
    T21_lr, _,_ = normalize(T21_lr, mode="standard", factor=norm_factor)
    T21, _,_ = normalize(T21, mode="standard", factor=norm_factor, x_mean=T21_lr_mean, x_std=T21_lr_std)
    delta, _,_ = normalize(delta, mode="standard", factor=norm_factor)
    vbv, _,_ = normalize(vbv, mode="standard", factor=norm_factor)
    
    return T21, delta, vbv, T21_lr, T21_lr_mean, T21_lr_std, scale_factor


def get_paths(config):
    print("path: ", config["path"])
    path = config["path"]
    model_name = config["name"].split(".")[0]
    model_dir = os.path.join(path, model_name)
    plot_dir = os.path.join(model_dir, "plots")
    data_dir = os.path.join(model_dir, "data")
    model_path = os.path.join(model_dir, config["name"])
    return model_path, model_dir, plot_dir, data_dir

def initialize_model_directory(rank, config):
    model_path, model_dir, plot_dir, data_dir = get_paths(config)
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
        if not os.path.exists(config_path):
            if rank==0:
                with open(config_path, 'w') as file:
                    yaml.dump(config, file)
        else:
            #check if config file is the same
            with open(config_path, 'r') as file:
                config_old = yaml.safe_load(file)
            if config_old != config:
                if rank==0:
                    print(f"Config file at {config_path} is different from the current config. Overwriting...", flush=True)
                    with open(config_path, 'w') as file:
                        yaml.dump(config, file)
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
    return config, model_path