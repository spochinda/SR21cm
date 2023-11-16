import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import tqdm
import time
import os
import pickle

path = os.getcwd()


"""
#i=0, i*24:(i+1)*24 = 0:24, 
#1. Load data
dir_files = os.listdir(path + '/outputs')
rng =range(1000,1010)
z = np.arange(6,28.1,1)
T21_target = np.zeros((len(rng)*24, 128, 128, 128, len(z)))
T21_train = np.zeros((len(rng)*24, 64, 64, 64, len(z)))
test = np.zeros((len(rng)*24, 64, 64, 64, len(z)))
delta = np.zeros((len(rng)*24, 128, 128, 128))
vbv = np.zeros((len(rng)*24, 128, 128, 128))
files = []
for i,ID in enumerate(rng):
    files_ = [file for file in dir_files if (str(ID) in file) and ('T21_cube' in file)]
    #sort files by redshift
    files.append(sorted(files_, key=lambda x: float(x.split('_')[2])))

"""

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
                    temp = tf.keras.layers.GaussianNoise(tf.reduce_mean(temp)*0.05)(temp)
                    T21_lr[i,:,:,:,j] = tf.keras.layers.Conv3D(filters=1, kernel_size=(2, 2, 2),
                                                               kernel_initializer=tf.keras.initializers.constant(value=1/8),
                                                               use_bias=False, bias_initializer=None, #tf.keras.initializers.Constant(value=0.1),
                                                               strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                               activation=None,
                                                               )(temp).numpy().reshape(64,64,64)
                    
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

Data = DataManager(path, redshifts=list(np.arange(6,28,1)), IC_seeds=list(range(1000,1002)))

dataset = tf.data.Dataset.from_generator(Data.generator_func,
                                         args=(True, 6, True),
                                         output_signature=(
                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),  # Modify target_shape
                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),
                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),
                                             tf.TensorSpec(shape=(64,64,64,1), dtype=tf.float32)
                                             ))

# Take one batch from the dataset
#one_batch = dataset.batch(8)
#take = one_batch.take(1)
#print("take 1: ", take)
# Iterate over the batch
#for data1, data2, data3, data4 in one_batch:
#    print("data1 shape:", data1.shape)

   
"""


for i,ID in enumerate(rng):
    #load initial conditions
    delta[i*24:(i+1)*24,:,:,:] = augment_data(loadmat(path + '/IC/delta'+str(ID)+'.mat')['delta'])
    #delta[i,:,:,:] = loadmat(path + '/IC/delta'+str(ID)+'.mat')['delta']
    vbv[i*24:(i+1)*24,:,:,:] = augment_data(loadmat(path + '/IC/vbv'+str(ID)+'.mat')['vbv'])
    #vbv[i,:,:,:] = loadmat(path + '/IC/vbv'+str(ID)+'.mat')['vbv']
    for j,file in enumerate(files):
        data = loadmat(path+'/outputs/'+file)["Tlin"]
        #T21_target[i,:,:,:,j] = data
        T21_target[i*24:(i+1)*24,:,:,:,j] = augment_data(data)
        #T21_train[i,:,:,:,j] = data["Tlin"][:64,:64,:64]
        #temp = tf.cast(data.reshape(1,128,128,128,1), dtype=tf.float32)        
        #test[i,:,:,:,j] = tf.keras.layers.Conv3D(filters=1, kernel_size=(2, 2, 2), 
        #                                    kernel_initializer=tf.keras.initializers.constant(value=1/8),
        #                                    use_bias=False, bias_initializer=None, #tf.keras.initializers.Constant(value=0.1),
        #                                    strides=(2, 2, 2), padding='valid', data_format="channels_last", 
        #                                    activation=None,
        #                                    )(temp).numpy().reshape(64,64,64)
    

 
print(T21_target.shape, T21_target)
#print("original: ", data["Tlin"].shape)
#print("augmented: ", augment_data(data["Tlin"]))


for j in range(len(rng)):
    fig,axes = plt.subplots(2,6,figsize=(20,9))
    for i in range(6,12):
        #temp1 = tf.cast(T21_target[j,:,:,:,i].reshape(1,128,128,128,1), dtype=tf.float32)        
        #temp2 = tf.keras.layers.Conv3D(filters=1, kernel_size=(2, 2, 2), 
        #                                    kernel_initializer=tf.keras.initializers.constant(value=1/8),
        #                                    use_bias=False, bias_initializer=None, #tf.keras.initializers.Constant(value=0.1),
        #                                    strides=(2, 2, 2), padding='valid', data_format="channels_last", 
        #                                    activation=None
        #                                    )(temp1)
        #test[j,:,:,:,i] = temp2.numpy().reshape(64,64,64)

        axes[0,i-6].imshow(T21_target[j,:,:,30,i], )
        axes[0,i-6].set_title("z={0}, Target Mean: {1:.2f}".format(z[i], np.mean(T21_target[j,:,:,30,i])))
        axes[1,i-6].imshow(test[j,:,:,30,i], )
        axes[1,i-6].set_title("z={0}, Conv Mean: {1:.2f}".format(z[i], np.mean(test[j,:,:,30,i])))
    plt.show()

print("shapes loaded: ", T21_target.shape,T21_train.shape, delta.shape, vbv.shape)

T21_target = tf.expand_dims(input=tf.cast(T21_target[:,:,:,:,10], dtype=tf.float32), axis=4) #take 10th redshift slice
T21_train = tf.expand_dims(input=tf.cast(T21_train[:,:,:,:,10], dtype=tf.float32), axis=4) #take 10th redshift slice



delta = tf.expand_dims(input=tf.cast(delta,dtype=tf.float32), axis=4)
vbv = tf.expand_dims(input=tf.cast(vbv,dtype=tf.float32), axis=4)


"""
#2. Define critic
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.build_critic_model()

    def build_critic_model(self):
        conv1 = tf.keras.layers.Conv3D(filters=8, kernel_size=(7, 7, 7), 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                            bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                            strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                            )
        
        conv2 = tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 5), 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                            bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                            strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                            )
        
        conv3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                            bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                            strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                            )
        
        conv4 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 1), 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                            bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                            strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                            activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                            )
        flatten = tf.keras.layers.Flatten()
        out = tf.keras.layers.Dense(units=1,
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                         bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                         name='output_layer')

        self.model = tf.keras.Sequential([conv1, conv2, conv3, conv4, flatten, out])    
        return self.model

    @tf.function
    def critic_loss(self, T21_big, IC_delta, IC_vbv, generated_boxes):
        #wasserstein loss
        # Generate a batch of fake big boxes using the generator network
        T21_big = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(T21_big)
        IC_delta = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(IC_delta)
        IC_vbv = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(IC_vbv)

        # Evaluate the critic network on the real big boxes and the fake big boxes
        W_real = self.call(T21_big, IC_delta, IC_vbv)
        W_gen = self.call(generated_boxes, IC_delta, IC_vbv)

        epsilon = tf.random.uniform(shape=[T21_big.shape[0], 1, 1, 1, 1], minval=0., maxval=1., seed=None)
        # Compute the interpolated difference between the real and generated samples
        xhat = epsilon * T21_big + (1 - epsilon) * generated_boxes

        # Compute the gradients of the critic network with respect to the interpolated difference
        with tf.GradientTape() as tape:    
            tape.watch(xhat)
            critic_output = self.call(xhat, IC_delta, IC_vbv)
        gradients = tape.gradient(critic_output, xhat)
        l2_norm = tf.math.reduce_euclidean_norm(gradients, axis=[1,2,3])
        gp = 1. * tf.square(l2_norm - 1)
        
        #plotting: need to remove tf.function decorator to plot histograms and imshows (e is epoch and i is batch number)
        if False:
            if (e==0) and (i<1):
                fig,axes = plt.subplots(3,4,figsize=(15,5))
                if True:
                    for j in range(2):
                        density = True
                        axes[0,j].hist(T21_big[j,:,:,:,0].numpy().flatten(), density=density, bins=100, alpha=0.5, label="real")
                        axes[1,j].hist(generated_boxes[j,:,:,:,0].numpy().flatten(), density=density, bins=100, alpha=0.5, label="fake")
                        axes[2,j].hist(xhat[j,:,:,:,0].numpy().flatten(), density=density, bins=100, alpha=0.5, label="interpolated")
                        
                        axes[0,j+2].imshow(T21_big[j,:,:,10,0], vmin=-0.5, vmax=0.5)
                        axes[1,j+2].imshow(generated_boxes[j,:,:,10,0], vmin=-0.5, vmax=0.5)
                        axes[2,j+2].imshow(xhat[j,:,:,10,0], vmin=-0.5, vmax=0.5)
                    
                    for j in range(3):
                        for k in range(2):
                            axes[j,k].set_xlim(-5,5)
                            axes[j,k].legend()
                plt.show() 

        # Compute the approximate Wasserstein loss
        #print("W_real={0:.2f} shape={1}, \nW_gen={2:.2f} shape={3}, \ngp={4:.2f} shape={5}".format(tf.reduce_mean(W_real), W_real.shape, tf.reduce_mean(W_gen), W_gen.shape, tf.reduce_mean(gp), gp.shape))
        loss = tf.reduce_mean(W_gen - W_real + gp)

        return loss, gp 

    @tf.function
    def train_step_critic(self, T21_big, IC_delta, IC_vbv, T21_small, optimizer, generator):
        #
        #Function that performs one training step for the critic network.
        #The function calls the loss function for the critic network, computes the gradients,
        #and applies the gradients to the network's parameters.
        #

        with tf.GradientTape() as disc_tape:
            generated_boxes = generator(T21_small, IC_delta, IC_vbv)
            crit_loss, gp = self.critic_loss(T21_big, IC_delta, IC_vbv, generated_boxes)

        grad_disc = disc_tape.gradient(crit_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grad_disc, self.model.trainable_variables))
        
        return crit_loss, gp
    
    @tf.function
    def call(self, T21_target, IC_delta, IC_vbv):
        data_target = tf.concat((T21_target, IC_delta, IC_vbv), axis=4)
        x_out_model = self.model(data_target)
        return x_out_model

class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, filters_1x1x1_7x7x7=6, filters_7x7x7=6, filters_1x1x1_5x5x5=6, filters_5x5x5=6, filters_1x1x1_3x3x3=6, filters_3x3x3=6, filters_1x1x1=6):
        super(InceptionLayer, self).__init__()
        self.conv_1x1x1_7x7x7 = tf.keras.layers.Conv3D(filters=filters_1x1x1_7x7x7, kernel_size=(1, 1, 1),
                                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                       bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                                       strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                       activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                       )
        self.conv_7x7x7 = tf.keras.layers.Conv3D(filters=filters_7x7x7, kernel_size=(7, 7, 7),
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                 bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        self.conv_1x1x1_5x5x5 = tf.keras.layers.Conv3D(filters=filters_1x1x1_5x5x5, kernel_size=(1, 1, 1),
                                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                            bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                                            strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                            activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                            )
        self.conv_5x5x5 = tf.keras.layers.Conv3D(filters=filters_5x5x5, kernel_size=(5, 5, 5),
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                 bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        self.crop_5x5x5 = tf.keras.layers.Cropping3D(cropping=(1, 1, 1),data_format="channels_last")

        self.conv_1x1x1_3x3x3 = tf.keras.layers.Conv3D(filters=filters_1x1x1_3x3x3, kernel_size=(1, 1, 1),
                                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                       bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                                       strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                       activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                       )
        self.conv_3x3x3 = tf.keras.layers.Conv3D(filters=filters_3x3x3, kernel_size=(3, 3, 3),
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                 bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        self.crop_3x3x3 = tf.keras.layers.Cropping3D(cropping=(2, 2, 2),data_format="channels_last")
        
        self.conv_1x1x1 = tf.keras.layers.Conv3D(filters=filters_1x1x1, kernel_size=(1, 1, 1),
                                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                 bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        
        self.crop_1x1x1 = tf.keras.layers.Cropping3D(cropping=(3, 3, 3),data_format="channels_last")
        self.concat = tf.keras.layers.Concatenate(axis=4)#([x1, x2, x3, x4])
        
        self.crop_x = tf.keras.layers.Cropping3D(cropping=(3, 3, 3),data_format="channels_last")#(x[:,:,:,:,0:1])
        #self.tile_x = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 1, x_out.shape[-1]]))(x_out_)

    def call(self, x):
        x1 = self.conv_1x1x1_7x7x7(x)
        x1 = self.conv_7x7x7(x1)
        
        x2 = self.conv_1x1x1_5x5x5(x)
        x2 = self.conv_5x5x5(x2)
        x2 = self.crop_5x5x5(x2)
        
        x3 = self.conv_1x1x1_3x3x3(x)
        x3 = self.conv_3x3x3(x3)
        x3 = self.crop_3x3x3(x3)
        
        x4 = self.conv_1x1x1(x)
        x4 = self.crop_1x1x1(x4)
        x_out = self.concat([x1, x2, x3, x4])
        x_out_ = self.crop_x(x[:,:,:,:,0:1])
        x_out_ = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 1, x_out.shape[-1]]))(x_out_)
        return tf.add(x_out, x_out_)



class Generator(tf.keras.Model):
    def __init__(self, T21_shape=(1,64,64,64,1), delta_shape=(1,128,128,128,1), vbv_shape=(1,128,128,128,1)):
        super(Generator, self).__init__()
        self.T21_shape = T21_shape
        self.delta_shape = delta_shape
        self.vbv_shape = vbv_shape
        self.upsampling = int(delta_shape[1]/T21_shape[1])
        self.build_generator_model()

    def build_generator_model(self):
        inputs_T21 = tf.keras.layers.Input(shape=self.T21_shape[1:]) #not including the batch size according to docs
        inputs_delta = tf.keras.layers.Input(shape=self.delta_shape[1:])
        inputs_vbv = tf.keras.layers.Input(shape=self.vbv_shape[1:])

        T21 = tf.keras.layers.UpSampling3D(size=self.upsampling, data_format="channels_last")(inputs_T21)
        T21 = InceptionLayer()(T21) #tf.keras.layers.Lambda(self.inception__)(T21) #self.inception__(T21)
        T21 = tf.keras.layers.LeakyReLU(alpha=0.1)(T21) #nn.leaky_relu(T21, 0.1)
        
        delta = InceptionLayer()(inputs_delta) #tf.keras.layers.Lambda(self.inception__)(inputs_delta) #self.inception__(inputs_delta)
        delta = tf.keras.layers.LeakyReLU(alpha=0.1)(delta) #tf.nn.leaky_relu(delta, 0.1)

        vbv = InceptionLayer()(inputs_vbv) #tf.keras.layers.Lambda(self.inception__)(inputs_vbv)
        vbv = tf.keras.layers.LeakyReLU(alpha=0.1)(vbv) #tf.nn.leaky_relu(vbv, 0.1)

        data = tf.keras.layers.Concatenate(axis=4)([T21, delta, vbv])
        data = InceptionLayer()(data) #tf.keras.layers.Lambda(self.inception__)(data)
        data = tf.keras.layers.LeakyReLU(alpha=0.1)(data) #tf.nn.leaky_relu(data, 0.1)
        data = tf.keras.layers.Conv3D(filters=1,#data.shape[-1], 
                                      kernel_size=(1, 1, 1),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                      bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                      strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                      activation='relu'
                                      )(data)
        #data = tf.keras.layers.ReLU()(data) 
        
        self.model = tf.keras.Model(inputs=[inputs_T21, inputs_delta, inputs_vbv], outputs=data)
        return self.model

    @tf.function
    def generator_loss(self, T21_big, IC_delta, IC_vbv, generated_boxes, critic):
        T21_big = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(T21_big)
        IC_delta = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(IC_delta)
        IC_vbv = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(IC_vbv)
        
        #W_real = critic(T21_big, IC_delta, IC_vbv)
        W_gen = critic(generated_boxes, IC_delta, IC_vbv)

        loss = - tf.reduce_mean(W_gen) #- tf.reduce_mean(W_real - W_gen)
        return loss

    @tf.function
    def train_step_generator(self, T21_small, T21_big, IC_delta, IC_vbv, optimizer, critic):
        #
        #Function that performs one training step for the generator network.
        #The function calls the loss function for the generator network, computes the gradients,
        #and applies the gradients to the network's parameters.
        #

        with tf.GradientTape() as gen_tape: 
            generated_boxes = self.call(T21_small, IC_delta, IC_vbv)
            #generated_output = Critic(generated_boxes, IC_delta, IC_vbv)
            gen_loss = self.generator_loss(T21_big, IC_delta, IC_vbv, generated_boxes, critic)

        grad_gen = gen_tape.gradient(gen_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grad_gen, self.model.trainable_variables))

        return gen_loss
        
    @tf.function    
    def call(self, T21_train, IC_delta, IC_vbv):
        return self.model(inputs=[T21_train, IC_delta, IC_vbv])


generator = Generator()
critic = Critic()

#lbda=10
n_critic = 2
epochs = 10
learning_rate=1e-4
beta_1 = 0.5
beta_2 = 0.999

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

#model.summary()
#tf.keras.utils.plot_model(model, to_file=path+'/generator_model.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)




###Test Generator methods:
###generator.build_generator_model and call
#print(generator(test[0:2,:,:,:,10:11], delta[0:2,:,:,:,0:1], vbv[0:2,:,:,:,0:1]).shape) #passed
###generator.generator_loss
#print("gen loss", generator.generator_loss(T21_target[0:2,:,:,:,10:11], delta[0:2,:,:,:,0:1], vbv[0:2,:,:,:,0:1], generator(test[0:2,:,:,:,10:11], delta[0:2,:,:,:,0:1], vbv[0:2,:,:,:,0:1]) , critic)) #passed
###generator.train_step_generator
#print(generator.train_step_generator(test[0:2,:,:,:,10:11], T21_target[0:2,:,:,:,10:11], delta[0:2,:,:,:,0:1], vbv[0:2,:,:,:,0:1], generator_optimizer, critic) )




###Test Critic methods:
###critic.build_critic_model and call
#print(critic(T21_target[0:2,:,:,:,10:11], delta[0:2,:,:,:,0:1], vbv[0:2,:,:,:,0:1])) #passed
###critic.critic_loss
#print("loss: ", critic.critic_loss(T21_target[0:1,:,:,:,10:11], delta[0:1,:,:,:,0:1], vbv[0:1,:,:,:,0:1], generator(test[0:1,:,:,:,10:11], delta[0:1,:,:,:,0:1], vbv[0:1,:,:,:,0:1]) )) #passed
###critic.train_step_critic
#l,gp = critic.train_step_critic(T21_target[0:1,:,:,:,10:11], delta[0:1,:,:,:,0:1], vbv[0:1,:,:,:,0:1], test[0:1,:,:,:,10:11], critic_optimizer, generator)
#print("train loss: ", l,gp,l-gp )




Data = DataManager(path, redshifts=[10,], IC_seeds=list(range(1000,1002)))
#dataset = tf.data.Dataset.from_generator(Data.generator_func,
#                                         args=(True, 2, True),
#                                         output_signature=(
#                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),  # Modify target_shape
#                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),
#                                             tf.TensorSpec(shape=(128,128,128,1), dtype=tf.float32),
#                                             tf.TensorSpec(shape=(64,64,64,1), dtype=tf.float32)
#                                             ))
dataset = Data.data(augment=True, augments=2, low_res=True)
dataset = tf.data.Dataset.from_tensor_slices(dataset)

batches = dataset.batch(2)




def standardize(data, data_stats):
    mean, var = tf.nn.moments(data_stats, axes=[1,2,3], keepdims=True) #mean across xyz with shape=(batch,x,y,z,channels)
    mean = mean.numpy()
    var = var.numpy()
    for i,(m,v) in enumerate(zip(mean,var)):
        if m==0 and v==0:
            mean[i] = 0
            var[i] = 1
            print("mean and var both zero for i={0} j={1}, setting mean to {2} and var to {3}".format(i,np.nan,mean[i],var[i]))
    std = var**0.5
    return (data - mean) / std



model_path = path+"/trained_models/model_1"
ckpt = tf.train.Checkpoint(generator_model=generator.model, critic_model=critic.model, 
                           generator_optimizer=generator_optimizer, critic_optimizer=critic_optimizer,
                           )
manager = tf.train.CheckpointManager(ckpt, model_path+"/checkpoints", max_to_keep=5)

resume = False

if resume:
    weights_before = generator.model.get_weights()
    ckpt.restore(manager.latest_checkpoint)
    weights_after = generator.model.get_weights()
    are_weights_different = any([not np.array_equal(w1, w2) for w1, w2 in zip(weights_before, weights_after)])
    print("Are weights different after restoring from checkpoint: ", are_weights_different)

    if (os.path.exists(model_path+"/losses.pkl")==False) or (os.path.exists(model_path+"/checkpoints")==False) or (are_weights_different==False):
        assert False, "Resume=True: Checkpoints directory or losses file does not exist or weights are unchanged after restoring, cannot resume training."
else:
    print("Initializing from scratch.")
    if os.path.exists(model_path+"/losses.pkl") or os.path.exists(model_path+"/checkpoints"):
        assert False, "Resume=False: Loss file or checkpoints directory already exists, exiting..."
    print("Creating loss file...")
    with open(model_path+"/losses.pkl", "wb") as f:
        generator_losses_epoch = []
        critic_losses_epoch = []
        gradient_penalty_epoch = []
        pickle.dump((generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch), f)




for e in range(epochs):
    start = time.time()

    generator_losses = []
    critic_losses = []
    gradient_penalty = []
    for i, (T21, delta, vbv, T21_lr) in enumerate(batches):
        #print("shape inputs: ", T21.shape, delta.shape, vbv.shape, T21_lr.shape)
        start_start = time.time()
        T21_standardized = standardize(T21, T21_lr)
        T21_lr_standardized = standardize(T21_lr, T21_lr)
        vbv_standardized = standardize(vbv, vbv)
        
        crit_loss, gp = critic.train_step_critic(T21_standardized, delta, vbv_standardized, T21_lr_standardized, critic_optimizer, generator)
        critic_losses.append(crit_loss)
        gradient_penalty.append(gp)

        if i%n_critic == 0:
            gen_loss = generator.train_step_generator(T21_lr_standardized, T21_standardized, delta, vbv_standardized, generator_optimizer, critic)
            generator_losses.append(gen_loss)
        
        print("Time for batch {0} is {1:.2f} sec".format(i + 1, time.time() - start_start))
    
    #save losses
    with open(model_path+"/losses.pkl", "rb") as f: # Open the file in read mode and get data
        generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch = pickle.load(f)
    # Append the new values to the existing data
    generator_losses_epoch.append(np.mean(generator_losses))
    critic_losses_epoch.append(np.mean(critic_losses))
    gradient_penalty_epoch.append(np.mean(gradient_penalty))
    with open(model_path+"/losses.pkl", "wb") as f: # Open the file in write mode and dump the data
        pickle.dump((generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch), f)
    
    #checkpoint
    #if e%2 == 0:
    print("Saving checkpoint...")
    manager.save()
    print("Checkpoint saved!")

    print("Time for epoch {0} is {1:.2f} sec \nGenerator mean loss: {2:.2f}, \nCritic mean loss: {3:.2f}, \nGradient mean penalty: {4:.2f}".format(e + 1, time.time() - start, np.mean(generator_losses), np.mean(critic_losses), np.mean(gradient_penalty)))




with open(model_path+"/losses.pkl", "rb") as f:
    data = pickle.load(f)
print("Saved loss history: ", data)

weights_after = generator.model.get_weights()
print("weights[0] after loop: ", weights_after[0])

"""



#T21_train = tf.expand_dims(input=tf.cast(T21_train,dtype=tf.float32), axis=0)
#T21_target = tf.expand_dims(input=tf.cast(T21_target,dtype=tf.float32), axis=0)


for e in range(epochs):
    for t in range(n_critic):
        # Select a random batch of big boxes
        indices = np.random.choice(T21_target.shape[0], size=batch_size, replace=False)
        print("batch, target shape 0, indices: ", batch_size, T21_target.shape[0], indices)
        
        T21_big_batch = tf.gather(T21_target, indices, axis=0)#T21_target[indices,:,:,:,:] #10th redshift slice chosen earlier
        IC_delta_batch = tf.gather(delta, indices, axis=0) #delta[indices,:,:,:,:]
        IC_vbv_batch = tf.gather(vbv, indices, axis=0) #vbv[indices,:,:,:,:]

        # Select a random batch of small boxes
        indices = np.random.choice(T21_train.shape[0], size=batch_size, replace=False)
        T21_small_batch = tf.gather(T21_train, indices, axis=0) #T21_train[indices,:,:,:,:] #10th redshift slice chosen earlier
        
        print("shape inputs: ", T21_big_batch.shape, IC_delta_batch.shape, IC_vbv_batch.shape, T21_small_batch.shape)

        # Train the critic network on the batch of big boxes
        critic_loss_value = critic_train_step(T21_big_batch, IC_delta_batch, IC_vbv_batch, T21_small_batch)

    # Select a random batch of small boxes
    indices = np.random.choice(T21_train.shape[0], size=batch_size, replace=False)
    T21_small_batch = tf.gather(T21_train, indices, axis=0)# T21_train[indices,:,:,:,:] #10th redshift slice chosen earlier
    IC_delta_batch = tf.gather(delta, indices, axis=0) #delta[indices,:,:,:,:]
    IC_vbv_batch = tf.gather(vbv, indices, axis=0) #vbv[indices,:,:,:,:]

    # Train the generator network on the batch of small boxes
    generator_loss_value = generator_train_step(T21_small_batch, IC_delta_batch, IC_vbv_batch)
    if e%10000 == 0:
        batch_size *= 2
    # Print the loss values for the critic and generator networks
    print("Epoch: {}, Critic loss: {}, Generator loss: {}".format(e+1, critic_loss_value, generator_loss_value))




    

z = np.arange(6,29,1)[::-1]
Data = DataManager(path, redshifts=list(z[::-1]), IC_seeds=list(range(1000,1002)))
T21, delta, vbv, T21_lr = Data.data(augment=True, augments=2, low_res=True)


# Plotting

if False:

    #fig,ax = plt.subplots(1,2,figsize=(10,5))
    shapes = (2,3)
    fig,ax = plt.subplots(*shapes,figsize=(15,5))
    
    for i in range(T21_lr.shape[0]):
        ind = np.unravel_index(i, shapes)
        #ax[ind].imshow(T21_target[i,:,:,10,-1])
        data_standardized = standardize(T21_lr[i:i+1,:,:,:,-1], T21_lr[i:i+1,:,:,:,-1], i)
        ax[ind].hist(data_standardized[:,:,:,10].numpy().flatten(), bins=100)
    
    def update(i):
        print("z: ", z[i], i)
        for j in range(T21_lr.shape[0]):
            ind = np.unravel_index(j, shapes)
            #ax[ind].imshow(T21_target[j,:,:,10,-i-1])
            #ax[ind].imshow(test[j,:,:,10,-i-1])
            data_standardized = standardize(T21_lr[j:j+1,:,:,:,-i-1], T21_lr[j:j+1,:,:,:,-i-1],j)
            
            try:
                ax[ind].hist(data_standardized[:,:,:,10].numpy().flatten(), bins=100)
            except Exception as e:
                print("couldn't plot j={0}, i={1}, error: ".format(j,i), e)
            #ax[ind].set_title('z = '+str(z[-i-1]))
            #ax[1].imshow(T21_train[3,:,:,10,-i-1])
            #ax.set_axis_off()
            ax[ind].set_xlim(-5,5)
            ax[ind].set_title('z = '+str(z[i]))
        #ax[0].imshow(T21_target[3,:,:,10,-i-1])
        #ax[1].imshow(T21_train[3,:,:,10,-i-1])
        #ax.set_axis_off()

    anim = FuncAnimation(fig, update, frames=z.size, interval=800)

    #fig,axes = plt.subplots(1,3,figsize=(15,5))
    #slice_id = 10

    #im0 = axes[0].imshow(vbv[:,:,slice_id], animated=True)
    #im1 = axes[1].imshow(delta[:,:,slice_id], animated=True)
    #im2 = axes[2].imshow(T21_target[:,:,slice_id,-1], animated=True)

    # Define the animation function
    #def update(i):
    #    axes[2].set_title('z = '+str(z[-i-1]))
    #    axes[2].imshow(T21_target[:,:,slice_id,-i-1])
    #    #ax.set_axis_off()

    #anim = FuncAnimation(fig, update, frames=len(z), interval=800)
    ##plt.show()
    anim.save(path+"/hist_3.gif", dpi=300, writer=PillowWriter(fps=1))

#data_standardized = standardize(T21_lr[j:j+1,:,:,:,-i-1], T21_lr[j:j+1,:,:,:,-i-1])
#check T21_lr if all zeros
#print("tf count nonzero j=0 T21_lr sum: ", tf.math.count_nonzero(T21_lr[0:1,:,:,10,-22-1]))
#mean, var = tf.nn.moments(T21_lr[0:5,:,:,:,-22-1], axes=[1,2,3])

#set mean[i] to 0 and var[i] to 1 if both are zero:


#plt.imshow(T21_lr[0,:,:,10,-22-1])

#plt.title("z={0}, Mean: {1:.2f}, Std: {2:.2f}".format(z[22], mean[0], var[0]**0.5))
#plt.show()
"""
"""


#########RETIRED CODE#########
def critic(T21_target, IC_delta, IC_vbv):
    data_target = tf.concat((T21_target, IC_delta, IC_vbv), axis=4) #tf.expand_dims(input=tf.concat((T21_target, IC_delta, IC_vbv), axis=3), axis=0)

    # Define variable initializers
    


    # layer 1
    w1 = tf.Variable(name="W_w1", 
                    initial_value=w_initializer((7, 7, 7, 3, 8)), 
                    trainable=True, 
                    dtype=tf.float32)

    b1 = tf.Variable(name="W_b1",
                    initial_value=b_initializer((8,)), 
                    trainable=True, 
                    dtype=tf.float32)

    x1 = tf.nn.leaky_relu(tf.nn.conv3d(data_target, 
                                        w1, 
                                        strides=[1, 2, 2, 2, 1], 
                                        padding='VALID') + b1, 0.1)
    print("x1 shape: ", x1.shape)
    # layer 2
    w2 = tf.Variable(name="W_w2", 
                    initial_value=w_initializer((5, 5, 5, 8, 16)), 
                    trainable=True, 
                    dtype=tf.float32)

    b2 = tf.Variable(name="W_b2",
                    initial_value=b_initializer((16,)), 
                    trainable=True, 
                    dtype=tf.float32)

    x2 = tf.nn.leaky_relu(tf.nn.conv3d(x1, 
                                        w2, 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='VALID') + b2, 0.1)
    print("x2 shape: ", x2.shape)
    # layer 3
    w3 = tf.Variable(name="W_w3", 
                    initial_value=w_initializer((3, 3, 3, 16, 32)),
                    trainable=True, 
                    dtype=tf.float32)


    b3 = tf.Variable(name="W_b3",
                    initial_value=b_initializer((32,)), 
                    trainable=True, 
                    dtype=tf.float32)

    x3 = tf.nn.leaky_relu(tf.nn.conv3d(x2, 
                                        w3, 
                                        strides=[1, 2, 2, 2, 1], 
                                        padding='VALID') + b3, 0.1)
    print("x3 shape: ", x3.shape)
    # layer 4
    w4 = tf.Variable(name="W_w4", 
                    initial_value=w_initializer((1, 1, 1, 32, 64)),
                    trainable=True, 
                    dtype=tf.float32)


    b4 = tf.Variable(name="W_b4",
                    initial_value=b_initializer((64,)), 
                    trainable=True, 
                    dtype=tf.float32)

    x4 = tf.nn.leaky_relu(tf.nn.conv3d(x3, 
                                        w4, 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='VALID') + b4, 0.1)

    print("shape x4: ", x4.shape, x4.get_shape().as_list()[1:])
    #layer 5: output
    x5 = tf.reshape(x4, (-1, np.product(x4.get_shape().as_list()[1:])))
    print("shape x5: ", x5.shape)
    w5 = tf.Variable(name="W_w5", 
                    initial_value=w_initializer((x5.get_shape().as_list()[-1], 1)), 
                    trainable=True,
                    dtype=tf.float32)

    b5 = tf.Variable(name="W_b5", 
                    initial_value=b_initializer((1,)), 
                    trainable=True,
                    dtype=tf.float32, 
                    )

    x_out = tf.matmul(x5, w5) + b5
    print("shape x_out: ", x_out.shape)
    print("w5", w5.shape)
    print("b5", b5.shape)
    return x_out


"""
