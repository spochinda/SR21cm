import tensorflow as tf
import numpy as np
from scipy.io import loadmat
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
    
    #Simulation box variables
    Npix = data_x.shape[0]
    Vpix = Lpix**3
    Lbox = Npix * Lpix
    Vbox = Lbox**3

    #Calculating wavevectors k for the simulation grid
    kspace = np.fft.fftfreq(Npix, d=Lpix/(2*np.pi))
    kx, ky, kz = np.meshgrid(kspace,kspace,kspace)
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    #Dont need to scipy.fft.fftshift since kspace isn't fftshift'ed
    data_k = np.fft.fftn(data_x)

    #Bin k values and calculate power spectrum
    k_bin_edges = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), endpoint=True, num=kbins+1)
    k_vals = np.zeros(kbins)
    P_k = np.zeros(kbins)
    for i in range(kbins):
        cond = ((k >= k_bin_edges[i]) & (k < k_bin_edges[i+1]))
        k_vals[i] = (k_bin_edges[i+1] + k_bin_edges[i])/2
        P_k[i] = (Vpix/Vbox) * Vpix * np.average(np.absolute(data_k[cond]))**2
        
    return k_vals, P_k

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


