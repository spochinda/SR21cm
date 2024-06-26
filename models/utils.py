import os
import torch
import numpy as np
from scipy.io import loadmat
import pandas as pd
import torch.utils
import torch.utils.data

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_T21, path_IC, 
                 redshifts=[10,], IC_seeds=list(range(1000,1008)), 
                 upscale=4, cut_factor=0, transform=False, norm_lr=False, device='cpu'):
        self.device = device
        self.path_T21 = path_T21
        self.path_IC = path_IC
        self.redshifts = redshifts
        self.IC_seeds = IC_seeds
        self.upscale = upscale
        self.cut_factor = cut_factor
        self.norm_lr = norm_lr
        self.df = self.getDataFrame(cut_factor=self.cut_factor)
        
        #self.labels = pd.read_csv(annotations_file)
        #self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @torch.no_grad()
    def __getitem__(self, idx):
        if True:
            #for new data
            T21 = torch.from_numpy(loadmat(self.path_T21 + self.df[["T21"]].iloc[idx].values[0])["Tlin"]).unsqueeze(0).to(torch.float32).to(self.device)
            delta = torch.from_numpy(loadmat(self.path_IC + self.df[["delta"]].iloc[idx].values[0])["delta"]).unsqueeze(0).to(torch.float32).to(self.device)
            vbv = torch.from_numpy(loadmat(self.path_IC + self.df[["vbv"]].iloc[idx].values[0])["vbv"]).unsqueeze(0).to(torch.float32).to(self.device)
            T21_lr = torch.nn.functional.interpolate(T21.unsqueeze(0), scale_factor=1/self.upscale, mode='trilinear')[0]
            labels = torch.tensor(self.df[["labels (z)"]].iloc[idx].values[0]).to(torch.float32).to(self.device)
        else:
            #for old data
            #use getFullDataset first to set self.dataset
            T21, delta, vbv, T21_lr, labels = self.dataset.tensors
            
            T21 = T21[idx]
            delta = delta[idx]
            vbv = vbv[idx]
            T21_lr = T21_lr[idx]
            labels = labels[idx]


    
        return T21, delta, vbv, T21_lr, labels
    
    def getDataFrame(self, cut_factor=0):
        rows = []
        for IC_seed in self.IC_seeds:
            for redshift in self.redshifts:
                for i in range((2**cut_factor)**3):
                    row = [[IC_seed, i], [redshift,]] #[f"IC={IC_seed}, z={redshift}"]
                    for file in os.listdir(self.path_T21):
                        if 'T21_cube' in file:
                            z = int(file.split('_')[2])
                            try:
                                IC = int(file.split('_')[7])
                            except:
                                IC = int(file.split('IC')[-1].split('.mat')[0])
                            if (z == redshift) and (IC == IC_seed):
                                row.append(file)
                    for file in os.listdir(self.path_IC):
                        if 'delta' in file:
                            IC = int(file.split('delta')[1].split('.')[0])
                            if IC == IC_seed:
                                row.append(file)
                    for file in os.listdir(self.path_IC):
                        if 'vbv' in file:
                            IC = int(file.split('vbv')[1].split('.')[0])
                            if IC == IC_seed:
                                row.append(file)
                    rows.append(row)
        df = pd.DataFrame(rows, columns=['IC,subcube', 'labels (z)', 'T21', 'delta', 'vbv'])
        
        return df

    def getFullDataset(self):
        labels = []
        for index, row in self.df.iterrows():
            labels.append(row['labels (z)'])

        self.df = self.getDataFrame(cut_factor=0)

        T21 = []
        delta = []
        vbv = []
        T21_lr = []
        
        for index, row in self.df.iterrows():
            T21_file = row['T21']
            delta_file = row['delta']
            vbv_file = row['vbv']

            T21_cube = torch.from_numpy(loadmat(self.path_T21 + T21_file)["Tlin"]).unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.device)
            delta_cube = torch.from_numpy(loadmat(self.path_IC + delta_file)["delta"]).unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.device)
            vbv_cube = torch.from_numpy(loadmat(self.path_IC + vbv_file)["vbv"]).unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.device)
            T21_lr_cube = torch.nn.functional.interpolate(T21_cube, scale_factor=1/self.upscale, mode='trilinear') #torch.nn.functional.interpolate( 
            #    ,
            #    scale_factor=self.upscale, mode='trilinear')
        
            T21.append(T21_cube)
            delta.append(delta_cube)
            vbv.append(vbv_cube)
            T21_lr.append(T21_lr_cube)
            

        T21 = torch.cat(T21, dim=0)
        delta = torch.cat(delta, dim=0)
        vbv = torch.cat(vbv, dim=0)
        T21_lr = torch.cat(T21_lr, dim=0)
        labels = torch.tensor(labels, device=self.device)

        T21 = get_subcubes(T21, self.cut_factor)
        delta = get_subcubes(delta, self.cut_factor)
        vbv = get_subcubes(vbv, self.cut_factor)
        T21_lr = get_subcubes(T21_lr, self.cut_factor)

        T21_lr_norm, T21_lr_extrema = normalize(T21_lr)
        T21_norm, T21_extrema = normalize(T21, x_min=T21_lr_extrema[:,:1] if self.norm_lr else None, x_max=T21_lr_extrema[:,1:2] if self.norm_lr else None)
        delta_norm, delta_extrema = normalize(delta)
        vbv_norm, vbv_extrema = normalize(vbv)

        dataset = torch.utils.data.TensorDataset(T21, delta, vbv, T21_lr, labels)
        self.dataset = dataset #save dataset for later use
        dataset_norm = torch.utils.data.TensorDataset(T21_norm, delta_norm, vbv_norm, T21_lr_norm, labels)
        dataset_extrema = torch.utils.data.TensorDataset(T21_extrema, delta_extrema, vbv_extrema, T21_lr_extrema)
        return dataset, dataset_norm, dataset_extrema


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

#def augment_dataset(T21, delta, vbv, T21_lr, n=8):
#    dataset = []
#    for i,(x1,x2,x3,x4) in enumerate(zip(T21, delta, vbv, T21_lr)):
#        x1 = x1.unsqueeze(0)
#        x2 = x2.unsqueeze(0)
#        x3 = x3.unsqueeze(0)
#        x4 = x4.unsqueeze(0)
#        data = torch.cat([x1,x2,x3,x4],dim=1) #cat data along channels so T21 and corresponding delta, vbv, T21_lr are rotated the same way
#        data = random_rotations(data[:1], n=n)#.tolist()
#        #data = torch.cat(data, dim=0)
#        dataset.append(data)
#        
#    dataset = torch.cat(dataset,dim=0)
#    
#    T21 = dataset[:,:1]
#    delta = dataset[:,1:2]
#    vbv = dataset[:,2:3]
#    T21_lr = dataset[:,3:]
#    return T21, delta, vbv, T21_lr

@torch.no_grad()
def augment_dataset(T21, delta, vbv, T21_lr, n=8):
    dataset_x1 = []
    dataset_x2 = []
    dataset_x3 = []
    dataset_x4 = []
    for i,(x1,x2,x3,x4) in enumerate(zip(T21, delta, vbv, T21_lr)):
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)
        x4 = x4.unsqueeze(0)

        N = np.random.choice(np.arange(0,24), size=n,replace=False)
        N = torch.tensor(N, device=x1.device)
        rotations_x1 = all_rotations(x1)[N]
        rotations_x2 = all_rotations(x2)[N]
        rotations_x3 = all_rotations(x3)[N]
        rotations_x4 = all_rotations(x4)[N]

        dataset_x1.append(rotations_x1)
        dataset_x2.append(rotations_x2)
        dataset_x3.append(rotations_x3)
        dataset_x4.append(rotations_x4)
        
    T21 = torch.cat(dataset_x1,dim=0)
    delta = torch.cat(dataset_x2,dim=0)
    vbv = torch.cat(dataset_x3,dim=0)
    T21_lr = torch.cat(dataset_x4,dim=0)
    
    return T21, delta, vbv, T21_lr

def calculate_power_spectrum(data_x, Lpix=3, kbins=100, dsq = False, method="torch", device="cpu"):
    #Simulation box variables
    batch, channel,(*d) = data_x.shape
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
            P_k[i] = (Vpix/Vbox) * Vpix * np.average(np.absolute(data_k[cond]))**2
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
            means = torch.mean(torch.abs(data_k_torch[...,cond_torch]),dim=2, keepdim=False)
            P_k_torch[:,:,i] = (Vpix/Vbox) * Vpix * means**2
        P_k_torch = P_k_torch*k_vals_torch**3/(2*np.pi**2) if dsq else P_k_torch
        return k_vals_torch, P_k_torch
    else:
        raise ValueError("Method must be numpy or torch")

@torch.no_grad()
def get_subcubes(cubes, cut_factor=0):
    if cut_factor > 0:
        batch, channel,(*d) = cubes.shape
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
    if mode == "minmax":
        if "x_min" not in kwargs: 
            x_min = torch.amin(x, dim=(1,2,3,4), keepdim=True)
        else:
            x_min = kwargs["x_min"]
        if "x_max" not in kwargs:
            x_max = torch.amax(x, dim=(1,2,3,4), keepdim=True)
        else:
            x_max = kwargs["x_max"]
        
        x_extrema = torch.cat([x_min, x_max], dim=1)
        
        x = (x - x_min) / (x_max - x_min)
        x = 2 * x - 1
        return x, x_extrema
    
    elif mode == "standard":
        if "x_mean" not in kwargs:
            x_mean = torch.mean(x, dim=(1,2,3,4), keepdim=True)
        else:
            x_mean = kwargs["x_mean"]
        if "x_std" not in kwargs:
            x_std = torch.std(x, dim=(1,2,3,4), keepdim=True)
        else:
            x_std = kwargs["x_std"]
        if "factor" not in kwargs:
            factor = 1
        else:
            factor = kwargs["factor"]

        x_stats = torch.cat([x_mean, x_std], dim=1)

        x = (x - x_mean) / (factor * x_std) #should the 2 be there? LR std approx. 0.5*HR std

        return x, x_stats
    
@torch.no_grad()
def invert_normalization(x, mode="standard", **kwargs):
    if mode == "standard":
        if "x_stats" in kwargs:
            x_mean = kwargs["x_stats"][:,:1]
            x_std = kwargs["x_stats"][:,1:2]
        else:
            print("x_mean and x_std not provided. Calculating...", flush=True)
            x_mean = torch.mean(x, dim=(1,2,3,4), keepdim=True)
            x_std = torch.std(x, dim=(1,2,3,4), keepdim=True) 
        if "factor" in kwargs:
            factor = kwargs["factor"]
        else:
            factor = 1.
        
        x = x * (factor * x_std) + x_mean
        return x
    elif mode == "minmax":
        if "x_extrema" in kwargs:
            x_min = kwargs["x_extrema"][:,:1]
            x_max = kwargs["x_extrema"][:,1:2]
        else:
            print("x_min and x_max not provided. Calculating...", flush=True)
            x_min = torch.amin(x, dim=(1,2,3,4), keepdim=True)
            x_max = torch.amax(x, dim=(1,2,3,4), keepdim=True)
        
        x = ((x + 1) * (x_max - x_min) / 2 ) + x_min
        return x
    
    
if __name__ == "__main__":
    pass