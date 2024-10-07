import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS
import pickle
from scipy.io import loadmat
import time
import os

path = os.getcwd()

class DataManager:
    def __init__(self, path, redshifts=[16], IC_seeds=[1000,1001]):
        self.path = path
        self.redshifts = redshifts
        self.IC_seeds = IC_seeds

    def get_file_lists(self):
        assert isinstance(self.redshifts, list), "redshifts must be a list"
        assert isinstance(self.IC_seeds, list), "IC_seeds must be a list"
        dir_T21 = os.listdir(self.path + '/outputs')
        T21_files = np.empty(shape=(len(self.IC_seeds), len(self.redshifts)), dtype=object)
        for file in dir_T21:
            if 'T21_cube' in file:
                z = int(file.split('_')[2])
                ID = int(file.split('_')[7])
                if (z in self.redshifts) and (ID in self.IC_seeds):
                    #print ID index in IC_seeds:
                    T21_files[self.IC_seeds.index(ID), self.redshifts.index(z)] = file
        
        dir_IC = os.listdir(self.path + '/IC')
        delta_files = np.empty(shape=(len(self.IC_seeds)), dtype=object)
        vbv_files = np.empty(shape=(len(self.IC_seeds)), dtype=object)
        for file in dir_IC:
            if 'delta' in file:
                ID = int(file.split('delta')[1].split('.')[0])
                if ID in self.IC_seeds:
                    delta_files[self.IC_seeds.index(ID)] = file
            elif 'vbv' in file:
                ID = int(file.split('vbv')[1].split('.')[0])
                if ID in self.IC_seeds:
                    vbv_files[self.IC_seeds.index(ID)] = file
        
        return T21_files, delta_files, vbv_files
    
    def generator_func(self, augment=False, augments=24, low_res=False):
        assert len(self.redshifts) == 1, "generator_func only works for one redshift at a time"
        T21_files, delta_files, vbv_files = self.get_file_lists()
        #print(T21_files,augments)
        if augment:
            augs = np.array([np.random.choice(24, size=augments, replace=False) for i in range(len(delta_files))]) #might need shuffling 
            #print(augs)
            #k=0
            for i in range(augments):
                #print("test: ", augs[:,i])
                #print("test2: ", T21_files[0],delta_files,vbv_files,augs[:,i])
                for j,(T21_file, delta_file, vbv_file, aug) in enumerate(zip(T21_files,delta_files,vbv_files,augs[:,i])):
                    #k+=1
                    #print(T21_file[0].split('_')[7], delta_file.split("delta")[1].split(".")[0], vbv_file.split("vbv")[1].split(".")[0], aug)
                    #print("j={0}, k={1}".format(j,k))
                    T21 = loadmat(self.path + '/outputs/' + T21_file[0])['Tlin']
                    delta = loadmat(self.path + '/IC/' + delta_file)['delta']
                    vbv = loadmat(self.path + '/IC/' + vbv_file)['vbv']

                    T21 = self.augment_data(T21, augments=aug).reshape(1,128,128,128,1)
                    delta = self.augment_data(delta, augments=aug).reshape(1,128,128,128,1)
                    vbv = self.augment_data(vbv, augments=aug).reshape(1,128,128,128,1)
                    if low_res:
                        T21_lr = tf.keras.layers.GaussianNoise(tf.reduce_mean(T21)*0.05)(T21)
                        T21_lr = tf.keras.layers.Conv3D(filters=1, kernel_size=(2, 2, 2),
                                                                       kernel_initializer=tf.keras.initializers.constant(value=1/8),
                                                                       use_bias=False, bias_initializer=None, #tf.keras.initializers.Constant(value=0.1),
                                                                       strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                                       activation=None,
                                                                       )(T21_lr)                        
                    else:
                        T21_lr = T21[:,:64,:64,:64,:]
                    
                    
                    T21 = tf.cast(tf.reshape(T21, (128,128,128,1)), dtype=tf.float32)
                    delta = tf.cast(tf.reshape(delta, (128,128,128,1)), dtype=tf.float32)
                    vbv = tf.cast(tf.reshape(vbv, (128,128,128,1)), dtype=tf.float32)
                    T21_lr = tf.cast(tf.reshape(T21_lr, (64,64,64,1)), dtype=tf.float32)
                    yield T21, delta, vbv, T21_lr
        else:
            for j,(T21_file, delta_file, vbv_file) in enumerate(zip(T21_files,delta_files,vbv_files)):
                T21 = loadmat(self.path + '/outputs/' + T21_file[0])['Tlin']
                delta = loadmat(self.path + '/IC/' + delta_file)['delta']
                vbv = loadmat(self.path + '/IC/' + vbv_file)['vbv']
                T21_lr = T21[:64,:64,:64]
                
                T21_lr = tf.cast(tf.reshape(T21_lr, (64,64,64,1)), dtype=tf.float32)
                T21 = tf.cast(tf.reshape(T21, (128,128,128,1)), dtype=tf.float32)
                delta = tf.cast(tf.reshape(delta, (128,128,128,1)), dtype=tf.float32)
                vbv = tf.cast(tf.reshape(vbv, (128,128,128,1)), dtype=tf.float32)
                print(j, T21.shape, delta.shape, vbv.shape, T21_lr.shape)
                yield T21, delta, vbv, T21_lr
            
    
        


    def load(self):
        T21_files, delta_files, vbv_files = self.get_file_lists()
        T21 = np.zeros((len(self.IC_seeds), 128, 128, 128, len(self.redshifts)), dtype=np.float32)
        delta = np.zeros((len(self.IC_seeds), 128, 128, 128), dtype=np.float32)
        vbv = np.zeros((len(self.IC_seeds), 128, 128, 128), dtype=np.float32)
        for i,file in enumerate(T21_files):
            delta[i] = loadmat(self.path + '/IC/' + delta_files[i])['delta']
            vbv[i] = loadmat(self.path + '/IC/' + vbv_files[i])['vbv']
            for j,file_ in enumerate(file):
                T21[i,:,:,:,j] = loadmat(self.path + '/outputs/' + file_)['Tlin']
        return T21, delta, vbv
    
    def data(self, augment=False, augments=23, low_res=False):
        #augments: number of augmented data per IC seed. Always includes the unaltered box
        T21, delta, vbv = self.load()
        if augment:
            assert (augments <= 23) and (augments>=1), "augments must be between 1 and 23"
            delta_augmented = np.empty(((augments+1)*len(self.IC_seeds), 128, 128, 128), dtype=np.float32)
            vbv_augmented = np.empty(((augments+1)*len(self.IC_seeds), 128, 128, 128), dtype=np.float32)
            T21_augmented = np.empty(((augments+1)*len(self.IC_seeds), 128, 128, 128, len(self.redshifts)), dtype=np.float32)

            for i in range(len(self.IC_seeds)):
                augs = [*np.random.choice(23, size=augments, replace=True), 23]
                delta_augmented[i*(augments+1):(i+1)*(augments+1)] = self.augment_data(delta[i], augments=augs)
                vbv_augmented[i*(augments+1):(i+1)*(augments+1)] = self.augment_data(vbv[i], augments=augs)
                for j in range(len(self.redshifts)):
                    T21_augmented[i*(augments+1):(i+1)*(augments+1),:,:,:,j] = self.augment_data(T21[i,:,:,:,j], augments=augs)
            T21 = tf.cast(T21_augmented, dtype=tf.float32)
            delta = tf.expand_dims(input=tf.cast(delta_augmented, dtype=tf.float32), axis=4)
            vbv = tf.expand_dims(input=tf.cast(vbv_augmented, dtype=tf.float32), axis=4)
        else:
            T21 = tf.cast(T21, dtype=tf.float32)
            delta = tf.expand_dims(input=tf.cast(delta, dtype=tf.float32), axis=4)
            vbv = tf.expand_dims(input=tf.cast(vbv, dtype=tf.float32), axis=4)

        if low_res:
            T21_lr = np.empty((T21.shape[0], 64, 64, 64, T21.shape[-1]), dtype=np.float32)
            for i in range(T21.shape[0]):
                for j in range(T21.shape[-1]):
                    temp = tf.reshape(T21[i,:,:,:,j], (1,128,128,128,1))
                    #add 5% gaussian noise
                    #temp = tf.keras.layers.GaussianNoise(tf.reduce_mean(temp)*0.1)(temp)
                    #T21_lr[i,:,:,:,j] = tf.keras.layers.Conv3D(filters=1, kernel_size=(2, 2, 2),
                    #                                           kernel_initializer=tf.keras.initializers.constant(value=1/8),#GaussianKernelInitializer(stddev=0.5, size=2)
                    #                                           use_bias=False, bias_initializer=None, #tf.keras.initializers.Constant(value=0.1),
                    #                                           strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                    #                                           activation=None,
                    #                                           )(temp).numpy().reshape(64,64,64)
                    #try average pooling 
                    T21_lr[i,:,:,:,j] = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format="channels_last")(temp).numpy().reshape(64,64,64)  
                    #T21_lr[i,:,:,:,j] = temp[0,::2,::2,::2,0].numpy()
                    
                    #fig,axes = plt.subplots(1,1,figsize=(10,5))
                    #axes[0].imshow(T21[i,:,:,64,j])
                    #axes[1].imshow(T21_lr[i,:,:,32,j])
                    #histograms instead
                    #axes.hist(T21[i,:,:,:,j].numpy().flatten(), density=True, bins=100, alpha=0.5, label="real")
                    #axes.hist(T21_lr[i,:,:,:,j].flatten(), density=True, bins=100, alpha=0.5, label="fake")
                    #axes.legend()
                    #plt.show()
                    
            T21_lr = tf.cast(T21_lr, dtype=tf.float32)
        else:
            T21_lr = None
        return T21, delta, vbv, T21_lr

    def augment_data(self, x, augments=23):
        y = np.empty((24,*x.shape))

        y[0,:,:,:] = x[::-1, ::-1, :]
        y[1,:,:,:] = x[::-1, :, ::-1]
        y[2,:,:,:] = x[:, ::-1, ::-1]
        
        y[3,:,:,:] = tf.transpose(x, (1, 0, 2))[::-1, :, :]
        y[4,:,:,:] = tf.transpose(x, (1, 0, 2))[::-1, :, ::-1]
        y[5,:,:,:] = tf.transpose(x, (1, 0, 2))[:, ::-1, :]
        y[6,:,:,:] = tf.transpose(x, (1, 0, 2))[:, ::-1, ::-1]

        y[7,:,:,:] = tf.transpose(x, (2, 1, 0))[::-1, :, :]
        y[8,:,:,:] = tf.transpose(x, (2, 1, 0))[::-1, ::-1, :]
        y[9,:,:,:] = tf.transpose(x, (2, 1, 0))[:, :, ::-1]
        y[10,:,:,:] = tf.transpose(x, (2, 1, 0))[:, ::-1, ::-1]

        y[11,:,:,:] = tf.transpose(x, (0, 2, 1))[:, ::-1, :]
        y[12,:,:,:] = tf.transpose(x, (0, 2, 1))[::-1, ::-1, :]
        y[13,:,:,:] = tf.transpose(x, (0, 2, 1))[:, :, ::-1]
        y[14,:,:,:] = tf.transpose(x, (0, 2, 1))[::-1, :, ::-1]

        y[15,:,:,:] = tf.transpose(x, (1, 2, 0))[::-1, ::-1, :]
        y[16,:,:,:] = tf.transpose(x, (1, 2, 0))[:, ::-1, ::-1]
        y[17,:,:,:] = tf.transpose(x, (1, 2, 0))[::-1, :, ::-1]
        y[18,:,:,:] = tf.transpose(x, (1, 2, 0))[::-1, ::-1, ::-1]
        
        y[19,:,:,:] = tf.transpose(x, (2, 0, 1))[::-1, ::-1, :]
        y[20,:,:,:] = tf.transpose(x, (2, 0, 1))[::-1, :, ::-1]
        y[21,:,:,:] = tf.transpose(x, (2, 0, 1))[:, ::-1, ::-1]
        y[22,:,:,:] = tf.transpose(x, (2, 0, 1))[::-1, ::-1, ::-1]
        y[23,:,:,:] = x
        
        return y[augments,:,:,:]

class Plotting:
    def __init__(self, path):
        self.path = path

def calculate_power_spectrum(data_x, Lpix=3, kbins=100):
    try:
        #print("tensor data types: ", data_x.dtype, Lpix.dtype, kbins.dtype)
        data_x = data_x.numpy()
        Lpix = Lpix.numpy()
        kbins = kbins.numpy()
    except:
        #print("non tensor types: ", type(data_x))
        pass
    #Simulation box variables
    Npix = data_x.shape[0]
    Vpix = Lpix**3
    Lbox = Npix * Lpix
    Vbox = Lbox**3
    
    #Calculating wavevectors k for the simulation grid
    kspace = np.fft.fftfreq(Npix, d=Lpix/(2*np.pi)).astype(np.float32)
    kx, ky, kz = np.meshgrid(kspace,kspace,kspace)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    
    #Dont need to scipy.fft.fftshift since kspace isn't fftshift'ed
    data_k = np.fft.fftn(data_x)

    #Bin k values and calculate power spectrum
    k_bin_edges = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), endpoint=True, num=kbins+1,dtype=np.float32)
    
    k_vals = np.zeros(kbins, dtype=np.float32)
    
    P_k = np.zeros(kbins, dtype=np.float32)
    
    #print("calculating power spectrum numpy")
    for i in range(kbins):
        s1 = time.time()
        cond = ((k >= k_bin_edges[i]) & (k < k_bin_edges[i+1]))
        #if i==0: print("cond time: {0:.2f}".format(1000*(time.time()-s1)))
        s2 = time.time()
        k_vals[i] = (k_bin_edges[i+1] + k_bin_edges[i])/2
        #if i==0: print("k_vals time: {0:.2f}".format(1000*(time.time()-s2)))
        s3 = time.time()
        P_k[i] = (Vpix/Vbox) * Vpix * np.average(np.absolute(data_k[cond]))**2
    
        #if i==0: print("P_k time: {0:.2f}".format(1000*(time.time()-s3)))
    
    return k_vals, P_k

def calculate_power_spectrum_tf(data_x, Lpix=3, kbins=100, **kwargs):
    
    #Simulation box variables
    Npix = data_x.shape[1]
    Vpix = Lpix**3
    Lbox = Npix * Lpix
    Vbox = Lbox**3

    #Calculating wavevectors k for the simulation grid
    kspace = np.fft.fftfreq(Npix, d=Lpix/(2*np.pi))
    kx, ky, kz = np.meshgrid(kspace,kspace,kspace)
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    #Dont need to scipy.fft.fftshift since kspace isn't fftshift'ed
    data_k = tf.signal.fft3d(tf.cast(data_x, dtype=tf.complex64))
    #tf.print(kwargs, "data_k dtype: ", data_k.dtype, "data_x dtype: ", data_x.dtype)

    #Bin k values and calculate power spectrum
    k_bin_edges = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), endpoint=True, num=kbins+1)
    
    k_vals = np.zeros(kbins)
    P_k = []
    for i in range(kbins):
        cond = ((k >= k_bin_edges[i]) & (k < k_bin_edges[i+1]))
        k_vals[i] = (k_bin_edges[i+1] + k_bin_edges[i])/2
        #if i==0: print("k_vals time: {0:.2f}".format(1000*(time.time()-s2)))
        #s3 = time.time()
        Pk = data_k[cond]
        #Pk = tf.boolean_mask(data_k, cond,)
        #if i==0: print("Pk time: {0:.2f}".format(1000*(time.time()-s3)))
        Pk = tf.abs(Pk)
        Pk = tf.reduce_mean(Pk)**2
        #sum_values = tf.reduce_sum(tf.where(cond, tf.abs(data_k), 0.0))
        #num_values = tf.reduce_sum(tf.cast(cond, tf.float32))
        #Pk = (sum_values / num_values)**2
        P_k.append((Vpix/Vbox) * Vpix * Pk)
    P_k = tf.stack(P_k)
        
    return k_vals, P_k

def calculate_power_spectrum_mse(generated_boxes, T21_batch, Lpix=3, kbins=100):
    dsq_mse = []    
    for i in range(generated_boxes.shape[0]):
        kwargs = {"data":"real", "index":i}
        k_real, Pk_real = calculate_power_spectrum_tf(data_x=T21_batch[i,:,:,:,0], Lpix=Lpix, kbins=kbins, **kwargs)
        kwargs["data"] = "generated"
        k_gen, Pk_gen = calculate_power_spectrum_tf(data_x=generated_boxes[i,:,:,:,0], Lpix=Lpix, kbins=kbins, **kwargs)
        dsq_gen = Pk_gen*k_gen**3/(2*np.pi**2)
        dsq_real = Pk_real*k_real**3/(2*np.pi**2)
        dsq_mse.append(tf.experimental.numpy.nanmean((dsq_gen-dsq_real)**2))
    dsq_mse = tf.experimental.numpy.nanmean(dsq_mse)
    return dsq_mse

def ionized_fraction(data):
    Vcube = tf.cast(data.shape[1]*data.shape[2]*data.shape[3],dtype=tf.float32)
    ionized_fraction = tf.reduce_sum(tf.cast(data==0, tf.float32),axis=[1,2,3],keepdims=True)/Vcube
    return ionized_fraction

def standardize(data, data_stats, subtract_mean = True):
    #subtract mean if non-zero minimum and divide by std, otherwise only divide by std
    mean, var = tf.nn.moments(data_stats, axes=[1,2,3], keepdims=True) #mean across xyz with shape=(batch,x,y,z,channels)
    mean = mean.numpy()#.flatten()
    var = var.numpy()#.flatten()  
    
    min_val = tf.reduce_min(data_stats,axis=[1,2,3],keepdims=True).numpy()#.flatten()  
    
    ionized_frac = ionized_fraction(data_stats)
    #tf.reduce_sum(tf.cast(data_stats==0, tf.float32),axis=[1,2,3],keepdims=True)/Vcube
    #if ionized_fraction >= 0.997:
    for i,(m,v,m_,ion_frac) in enumerate(zip(mean,var,min_val,ionized_frac)):
        if ion_frac >= 0.997:
            #mean[i] = 0
            var[i] = 1
        else:
            if not subtract_mean:
                mean[i] = 0

    #    if (m==0) and (v==0) and (m_==0):
    #        mean[i] = 0
    #        var[i] = 1
    #    elif (m!=0) and (v!=0) and (m_==0) and (subtract_mean):
    #        mean[i] = 0
    std = var**0.5

    return (data - mean) / std



def plot_and_save(generator, critic, learning_rate, 
                  IC_seeds, redshift, sigmas, step_skip_validation=1,
                  loss_file=None, plot_slice=True, subtract_mean=False, 
                  include_vbv = False, plot_loss=True, plot_loss_terms=True, 
                  seed=None, ncritic=None, savefig_path=None):

    if subtract_mean:
        ylabel = "Standardized"
    else:
        ylabel = "Std. norm."
    ncols = 6 + include_vbv
    nrows = len(IC_seeds)+plot_loss
    fig = plt.figure(tight_layout=True, figsize=(20,20//(ncols/nrows)))
    gs = GS(nrows, ncols, figure=fig)

    # Validation data
    Data_validation = DataManager(path, redshifts=[redshift,], IC_seeds=IC_seeds)
    T21, delta, vbv, T21_lr = Data_validation.data(augment=False, augments=9, low_res=True) 
    T21_standardized = standardize(T21, T21_lr, subtract_mean=subtract_mean)
    T21_lr_standardized = standardize(T21_lr, T21_lr, subtract_mean=subtract_mean)
    delta_standardized = standardize(delta, delta, subtract_mean=True)
    vbv_standardized = standardize(vbv, vbv, subtract_mean=True) if include_vbv else None
    if include_vbv:
        generated_boxes = generator.forward(T21_lr_standardized, delta_standardized, vbv_standardized).numpy()
    else:
        generated_boxes = generator.forward(T21_lr_standardized, delta_standardized).numpy()
        
    if plot_loss:
        
        if loss_file is not None:
            try: 
                with open(loss_file, "rb") as f: # Open the file in read mode and get data
                    generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch,generator_mse_losses_epoch,generator_dsq_mse_losses_epoch, generator_losses_epoch_validation, critic_losses_epoch_validation, gradient_penalty_epoch_validation,generator_mse_losses_epoch_validation,generator_dsq_mse_losses_epoch_validation = pickle.load(f)
            except:#from before I implemented saving validation losses
                with open(loss_file, "rb") as f: # Open the file in read mode and get data
                    generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch, generator_mse_losses_epoch, generator_dsq_mse_losses_epoch = pickle.load(f)
                    generator_losses_epoch_validation = len(generator_losses_epoch)*[np.NaN]
                    critic_losses_epoch_validation = len(generator_losses_epoch_validation)*[np.NaN]
                    gradient_penalty_epoch_validation = len(generator_losses_epoch_validation)*[np.NaN]
                    generator_mse_losses_epoch_validation  = len(generator_losses_epoch_validation)*[np.NaN]
                    generator_dsq_mse_losses_epoch_validation = len(generator_losses_epoch_validation)*[np.NaN]

        if not plot_loss_terms:
            ax_loss = fig.add_subplot(gs[0,:], wspace = 0.2)
        else:
            loss_gs = SGS(1,4, gs[0,:])
            ax_loss = fig.add_subplot(loss_gs[0,0])
            ax_loss_generator = fig.add_subplot(loss_gs[0,1])
            ax_loss_critic = fig.add_subplot(loss_gs[0,2])
            ax_loss_validation = fig.add_subplot(loss_gs[0,3])

        #loss row
        look_back = int(len(generator_losses_epoch)//(10/10)) #look back 40% of epochs
        
        ax_loss.plot(range(len(generator_losses_epoch)), generator_losses_epoch, label="Generator")
        ax_loss.plot(range(len(critic_losses_epoch)), critic_losses_epoch, label="Critic")
        ymin = min(np.nanmin(generator_losses_epoch[-look_back:]), np.min(critic_losses_epoch[-look_back:]))
        ymax = max(np.nanmax(generator_losses_epoch[-look_back:]), np.max(critic_losses_epoch[-look_back:]))
        if not (np.isnan(ymin) or np.isnan(ymax)):
            ax_loss.set_ylim(ymin,ymax) 
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        #ax_loss.set_yscale("symlog")
        ax_loss.grid()
        ax_loss.legend()

        plt.suptitle("$\lambda_\mathrm{{gp}}$={0:.2f}, $\lambda_\mathrm{{MSE}}$={1:.2f}, $\lambda_\mathrm{{\Delta^2, MSE}}$={2:.2f}, learning rate$\\times 10^{{-4}}$={3:.4f}, seed={4}, ncritic={5}".format(critic.lambda_gp, generator.lambda_mse, generator.lambda_dsq_mse, learning_rate*1e4, seed, ncritic))

        if plot_loss_terms:
            ax_loss_generator.plot(range(len(generator_losses_epoch)), generator_losses_epoch, label="$G_\mathrm{{total}}$")
            ax_loss_generator.plot(range(len(generator_losses_epoch)), generator_losses_epoch-generator.lambda_mse*np.array(generator_mse_losses_epoch)-generator.lambda_dsq_mse*np.array(generator_dsq_mse_losses_epoch), label="-(Wgen-Wreal)")
            ax_loss_generator.plot(range(len(generator_losses_epoch)), generator.lambda_mse*np.array(generator_mse_losses_epoch), label="$G_\mathrm{{MSE}}$")
            ax_loss_generator.plot(range(len(generator_losses_epoch)), generator.lambda_dsq_mse*np.array(generator_dsq_mse_losses_epoch), label="$G_\mathrm{{\Delta^2, MSE}}$")
            ymin = min(np.nanmin(generator_losses_epoch[-look_back:]), np.nanmin(generator_losses_epoch[-look_back:]-generator.lambda_mse*np.array(generator_mse_losses_epoch)[-look_back:]-generator.lambda_dsq_mse*np.array(generator_dsq_mse_losses_epoch)[-look_back:]), np.nanmin(generator.lambda_mse*np.array(generator_mse_losses_epoch)[-look_back:]), np.nanmin(generator.lambda_dsq_mse*np.array(generator_dsq_mse_losses_epoch)[-look_back:]))
            ymax = max(np.nanmax(generator_losses_epoch[-look_back:]), np.nanmax(generator_losses_epoch[-look_back:]-generator.lambda_mse*np.array(generator_mse_losses_epoch)[-look_back:]-generator.lambda_dsq_mse*np.array(generator_dsq_mse_losses_epoch)[-look_back:]), np.nanmax(generator.lambda_mse*np.array(generator_mse_losses_epoch)[-look_back:]), np.nanmax(generator.lambda_dsq_mse*np.array(generator_dsq_mse_losses_epoch)[-look_back:]))
            if not (np.isnan(ymin) or np.isnan(ymax)):
                ax_loss_generator.set_ylim(ymin,ymax)
            ax_loss_generator.set_xlabel("Epoch")
            ax_loss_generator.set_ylabel("Loss")
            ax_loss_generator.grid()
            ax_loss_generator.legend()

            ax_loss_critic.plot(range(len(critic_losses_epoch)), critic_losses_epoch, label="$C_\mathrm{{total}}$")
            ax_loss_critic.plot(range(len(critic_losses_epoch)), np.array(critic_losses_epoch)-np.array(gradient_penalty_epoch), label="$W_\mathrm{{gen}}-W_\mathrm{{real}}$")
            ax_loss_critic.plot(range(len(critic_losses_epoch)), gradient_penalty_epoch, label="Gradient Penalty")
            ymin = min(np.min(critic_losses_epoch[-look_back:]), np.min(np.array(critic_losses_epoch)[-look_back:]-np.array(gradient_penalty_epoch)[-look_back:]), np.min(np.array(gradient_penalty_epoch)[-look_back:]))
            ymax = max(np.max(critic_losses_epoch[-look_back:]), np.max(np.array(critic_losses_epoch)[-look_back:]-np.array(gradient_penalty_epoch)[-look_back:]), np.max(np.array(gradient_penalty_epoch)[-look_back:]))
            ax_loss_critic.set_ylim(ymin,ymax)
            ax_loss_critic.set_xlabel("Epoch")
            ax_loss_critic.set_ylabel("Loss")
            ax_loss_critic.grid()
            ax_loss_critic.legend()

            #print("MSE ratio before skippint: ", np.array(generator_mse_losses_epoch_validation)/np.array(generator_mse_losses_epoch))
            #print("dsq MSE ratio before skippint: ", np.array(generator_dsq_mse_losses_epoch_validation)/np.array(generator_dsq_mse_losses_epoch))
            mse_ratio = np.array(generator_mse_losses_epoch_validation)[::step_skip_validation]/np.array(generator_mse_losses_epoch)[::step_skip_validation]
            dsq_mse_ratio = np.array(generator_dsq_mse_losses_epoch_validation)[::step_skip_validation]/np.array(generator_dsq_mse_losses_epoch)[::step_skip_validation]
            #print("MSE ratio: ", mse_ratio)
            #print("dsq MSE ratio: ", dsq_mse_ratio)
            ax_loss_validation.plot(range(len(generator_losses_epoch_validation))[::step_skip_validation], mse_ratio, label="$G_\mathrm{{MSE,validation}}/G_\mathrm{{MSE,train}}$")
            ax_loss_validation.plot(range(len(generator_losses_epoch_validation))[::step_skip_validation], dsq_mse_ratio, label="$G_\mathrm{{\Delta^2, MSE,validation}}/G_\mathrm{{\Delta^2, MSE,train}}$")
            #ymin = min(np.min(generator_mse_losses_epoch_validation[-look_back:]), np.min(generator_dsq_mse_losses_epoch_validation[-look_back:]))
            #ymax = max(np.max(generator_mse_losses_epoch_validation[-look_back:]), np.max(generator_dsq_mse_losses_epoch_validation[-look_back:]))
            #ax_loss_validation.set_ylim(ymin,ymax)
            ax_loss_validation.set_xlabel("Epoch")
            ax_loss_validation.set_ylabel("Loss ratio")
            ax_loss_validation.grid()
            ax_loss_validation.legend()


    for i,IC in enumerate(IC_seeds):
        #calculate power spectra
        k_vals_gen, Pk_gen = calculate_power_spectrum(data_x=generated_boxes[i,:,:,:,0], Lpix=3, kbins=100)
        k_vals_real, Pk_real = calculate_power_spectrum(data_x=T21_standardized[i,:,:,:,0], Lpix=3, kbins=100)
        k_vals_real_lr, Pk_real_lr = calculate_power_spectrum(data_x=T21_lr_standardized[i,:,:,:,0], Lpix=6, kbins=100)

        # Plot histograms
        ax_hist = fig.add_subplot(gs[i+plot_loss,0])
        ax_hist.hist(generated_boxes[i, :, :, :, 0].flatten(), bins=100, alpha=0.5, label="Generated", density=True)
        ax_hist.hist(T21_standardized[i, :, :, :, 0].numpy().flatten(), bins=100, alpha=0.5, label="Real", density=True)
        ax_hist.set_ylim(0,2)
        ax_hist.set_xlabel(ylabel+" T21")
        ax_hist.set_title("Histograms of "+ylabel+" data")
        ax_hist.legend()
        
        # Plot power spectra
        ax_dsq = fig.add_subplot(gs[i+plot_loss,1])
        ax_dsq.plot(k_vals_gen, Pk_gen*k_vals_gen**3/(2*np.pi**2), label="Generated")
        ax_dsq.plot(k_vals_real, Pk_real*k_vals_real**3/(2*np.pi**2), label="Real")
        ax_dsq.plot(k_vals_real_lr, Pk_real_lr*k_vals_real_lr**3/(2*np.pi**2), label="Real LR")
        ax_dsq.set_xlabel("$k$")
        ax_dsq.set_ylabel(ylabel+" $\Delta^2_{21}$")
        ax_dsq.set_yscale("log")
        ax_dsq.set_title(ylabel+" power spectrum")
        ax_dsq.legend()

        # Plot real and generated data
        T21_std = np.std(T21_standardized[i, :, :, :, 0].numpy().flatten())
        T21_mean = np.mean(T21_standardized[i, :, :, :, 0].numpy().flatten())
        ax_real = fig.add_subplot(gs[i+plot_loss,2])
        ax_real.imshow(T21_standardized[i, :, :, T21_standardized.shape[-2]//2, 0], vmin=T21_mean-sigmas*T21_std, vmax=T21_mean+sigmas*T21_std)
        ax_real.set_title("Real")
        ax_gen = fig.add_subplot(gs[i+plot_loss,3])
        ax_gen.imshow(generated_boxes[i, :, :, generated_boxes.shape[-2]//2, 0], vmin=T21_mean-sigmas*T21_std, vmax=T21_mean+sigmas*T21_std)
        ax_gen.set_title("Generated")
        ax_real_lr = fig.add_subplot(gs[i+plot_loss,4])
        ax_real_lr.imshow(T21_lr_standardized[i, :, :, T21_lr_standardized.shape[-2]//2, 0], vmin=T21_mean-sigmas*T21_std, vmax=T21_mean+sigmas*T21_std)
        ax_real_lr.set_title("Real lr")
        

        if plot_slice:
            ax_delta = fig.add_subplot(gs[i+plot_loss,5])
            delta_std = np.std(delta_standardized[i, :, :, :, 0].numpy().flatten())
            delta_mean = np.mean(delta_standardized[i, :, :, :, 0].numpy().flatten())
            ax_delta.imshow(delta_standardized[i, :, :, delta_standardized.shape[-2]//2, 0], vmin=delta_mean-sigmas*delta_std, vmax=delta_mean+sigmas*delta_std)
            ax_delta.set_title("Standardized Delta IC ID={0}".format(IC))
            if include_vbv:
                ax_vbv = fig.add_subplot(gs[i+plot_loss,6])
                vbv_std = np.std(vbv_standardized[i, :, :, :, 0].numpy().flatten())
                vbv_mean = np.mean(vbv_standardized[i, :, :, :, 0].numpy().flatten())
                ax_vbv.imshow(vbv_standardized[i, :, :, vbv_standardized.shape[-2]//2, 0], vmin=vbv_mean-sigmas*vbv_std, vmax=vbv_mean+sigmas*vbv_std)
                ax_vbv.set_title("Standardized Vbv IC ID={0}".format(IC))
        else: #histogram delta and vbv_standardised
            ax_delta = fig.add_subplot(gs[i+plot_loss,5])
            ax_delta.hist(delta_standardized[i, :, :, :, 0].numpy().flatten(), bins=100, alpha=0.5, label="delta", density=True)
            ax_delta.set_title("Standardized delta IC ID={0}".format(IC))
            ax_delta.legend()
            if vbv_standardized is not None:
                ax_vbv = fig.add_subplot(gs[i+plot_loss,6])
                ax_vbv.hist(vbv_standardized[i, :, :, :, 0].numpy().flatten(), bins=100, alpha=0.5, label="vbv", density=True)
                ax_vbv.set_title("Standardized vbv IC ID={0}".format(IC))
                ax_vbv.legend()

    # Save figure
    plt.savefig(savefig_path)
    print("Saved figure in {0}".format(savefig_path))
    plt.close()


"""
#Examples of use:
Data = DataManager(path, redshifts=[10,], IC_seeds=list(range(1000,1008)))
dataset = Data.data(augment=True, augments=9, low_res=True)
#or
dataset = tf.data.Dataset.from_generator(Data.generator_func,
                                         args=(True, 6, True),
                                         output_signature=(
                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),  # Modify target_shape
                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),
                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),
                                             tf.TensorSpec(shape=(64,64,64,1), dtype=tf.float32)
                                             ))

"""


