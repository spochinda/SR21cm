import argparse
import itertools
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import tqdm
import time
import os
import pickle
from models.wgan import * #imports models and utils
from models.utils import *
#add in second inception module (done)
#layer normalisation

print("Inside GAN.py", flush=True)

parser = argparse.ArgumentParser(description="GAN to learn simulation cubes from initial conditions and smaller cubes")
parser.add_argument('--index', type=int, default=0, help='An index from 0 to 20 to pick a set of learning rates and penalty strengths')
args = parser.parse_args()
index = args.index

path = os.getcwd()


print("Available devices: ", tf.config.list_physical_devices(), flush=True)



def plot_and_save(IC_seeds, redshift, sigmas, plot_slice=True, subtract_mean=False, plot_loss_fraction=True):
    fig = plt.figure(tight_layout=True, figsize=(20,10))
    gs = gridspec.GridSpec(len(IC_seeds)+1, 7, figure=fig)
    if not plot_loss_fraction:
        ax_loss = fig.add_subplot(gs[0,:])
    else:
        ax_loss = fig.add_subplot(gs[0,:-3])
        ax_loss_fraction = fig.add_subplot(gs[0,-3:])
    #loss row
    ax_loss.plot(range(len(generator_losses_epoch)), generator_losses_epoch, label="generator")
    ax_loss.plot(range(len(critic_losses_epoch)), critic_losses_epoch, label="critic")
    ax_loss.plot(range(len(critic_losses_epoch)), np.array(critic_losses_epoch)-np.array(gradient_penalty_epoch), label="$W_{{gen}}-W_{{real}}$")
    ax_loss.plot(range(len(gradient_penalty_epoch)), gradient_penalty_epoch, label="gradient penalty")
    ax_loss.plot(range(len(generator_mse_losses_epoch)), generator_mse_losses_epoch, label="generator mse")
    print(lambda_mse, lambda_gp, learning_rate, flush=True)
    ax_loss.set_title("lambda_gp={0:.2f}, lambda_mse={1:.5f}, learning rate={2:.5f}".format(lambda_gp, lambda_mse, learning_rate))
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    #ax_loss.set_yscale("symlog")
    ax_loss.grid()
    ax_loss.legend()

    if plot_loss_fraction:
        ax_loss_fraction.plot(range(len(generator_losses_epoch)), lambda_mse*np.array(generator_mse_losses_epoch)/np.array(generator_losses_epoch), label="generator mse/Gloss")
        ax_loss_fraction.plot(range(len(critic_losses_epoch)), np.array(gradient_penalty_epoch)/np.array(critic_losses_epoch), label="gradient penalty/Wloss")
        ax_loss_fraction.set_xlabel("Epoch")
        ax_loss_fraction.set_ylabel("Loss fraction")
        ax_loss_fraction.grid()
        ax_loss_fraction.legend()

    # Validation data
    Data_validation = DataManager(path, redshifts=[redshift,], IC_seeds=IC_seeds)
    T21, delta, vbv, T21_lr = Data_validation.data(augment=False, augments=9, low_res=True) 
    T21_standardized = standardize(T21, T21_lr, subtract_mean=subtract_mean)
    T21_lr_standardized = standardize(T21_lr, T21_lr, subtract_mean=subtract_mean)
    delta_standardized = standardize(delta, delta, subtract_mean=subtract_mean)
    vbv_standardized = None #standardize(vbv, vbv)
    if vbv_standardized is not None:
        generated_boxes = generator.forward(T21_lr_standardized, delta_standardized, vbv_standardized).numpy()
    else:
        generated_boxes = generator.forward(T21_lr_standardized, delta_standardized).numpy()
        

    for i,IC in enumerate(IC_seeds):
        #calculate power spectra
        k_vals_gen, Pk_gen = calculate_power_spectrum(data_x=generated_boxes[i,:,:,:,0], Lpix=3, kbins=100)
        k_vals_real, Pk_real = calculate_power_spectrum(data_x=T21_standardized[i,:,:,:,0], Lpix=3, kbins=100)
        k_vals_real_lr, Pk_real_lr = calculate_power_spectrum(data_x=T21_lr_standardized[i,:,:,:,0], Lpix=6, kbins=100)

        # Plot histograms
        ax_hist = fig.add_subplot(gs[i+1,0])
        ax_hist.hist(generated_boxes[i, :, :, :, 0].flatten(), bins=100, alpha=0.5, label="generated", density=True)
        ax_hist.hist(T21_standardized[i, :, :, :, 0].numpy().flatten(), bins=100, alpha=0.5, label="real", density=True)
        ax_hist.set_ylim(0,2)
        ax_hist.set_xlabel("Standardized T21")
        ax_hist.set_title("Histograms of standardized data")
        ax_hist.legend()
        
        # Plot power spectra
        ax_dsq = fig.add_subplot(gs[i+1,1])
        ax_dsq.plot(k_vals_gen, Pk_gen*k_vals_gen**3/(2*np.pi**2), label="Generated")
        ax_dsq.plot(k_vals_real, Pk_real*k_vals_real**3/(2*np.pi**2), label="Real")
        ax_dsq.plot(k_vals_real_lr, Pk_real_lr*k_vals_real_lr**3/(2*np.pi**2), label="Real lr")
        ax_dsq.set_xlabel("$k$")
        ax_dsq.set_ylabel("Standardized $\Delta^2_{21}$")
        ax_dsq.set_yscale("log")
        ax_dsq.set_title("Standardized power spectrum")
        ax_dsq.legend()

        # Plot real and generated data
        T21_std = np.std(T21_standardized[i, :, :, :, 0].numpy().flatten())
        T21_mean = np.mean(T21_standardized[i, :, :, :, 0].numpy().flatten())
        ax_gen = fig.add_subplot(gs[i+1,2])
        ax_gen.imshow(generated_boxes[i, :, :, generated_boxes.shape[-2]//2, 0], vmin=T21_mean-sigmas*T21_std, vmax=T21_mean+sigmas*T21_std)
        ax_gen.set_title("Generated")
        ax_real = fig.add_subplot(gs[i+1,3])
        ax_real.imshow(T21_standardized[i, :, :, T21_standardized.shape[-2]//2, 0], vmin=T21_mean-sigmas*T21_std, vmax=T21_mean+sigmas*T21_std)
        ax_real.set_title("Real")
        ax_real_lr = fig.add_subplot(gs[i+1,4])
        ax_real_lr.imshow(T21_lr_standardized[i, :, :, T21_lr_standardized.shape[-2]//2, 0], vmin=T21_mean-sigmas*T21_std, vmax=T21_mean+sigmas*T21_std)
        ax_real_lr.set_title("Real lr")
        

        if plot_slice:
            ax_delta = fig.add_subplot(gs[i+1,5])
            delta_std = np.std(delta_standardized[i, :, :, :, 0].numpy().flatten())
            delta_mean = np.mean(delta_standardized[i, :, :, :, 0].numpy().flatten())
            ax_delta.imshow(delta_standardized[i, :, :, delta_standardized.shape[-2]//2, 0], vmin=delta_mean-sigmas*delta_std, vmax=delta_mean+sigmas*delta_std)
            ax_delta.set_title("Standardized Delta IC ID={0}".format(IC))
            if vbv_standardized is not None:
                ax_vbv = fig.add_subplot(gs[i+1,6])
                vbv_std = np.std(vbv_standardized[i, :, :, :, 0].numpy().flatten())
                vbv_mean = np.mean(vbv_standardized[i, :, :, :, 0].numpy().flatten())
                ax_vbv.imshow(vbv_standardized[i, :, :, vbv_standardized.shape[-2]//2, 0], vmin=vbv_mean-sigmas*vbv_std, vmax=vbv_mean+sigmas*vbv_std)
                ax_vbv.set_title("Standardized Vbv IC ID={0}".format(IC))
        else: #histogram delta and vbv_standardised
            ax_delta = fig.add_subplot(gs[i+1,5])
            ax_delta.hist(delta_standardized[i, :, :, :, 0].numpy().flatten(), bins=100, alpha=0.5, label="delta", density=True)
            ax_delta.set_title("Standardized delta IC ID={0}".format(IC))
            ax_delta.legend()
            if vbv_standardized is not None:
                ax_vbv = fig.add_subplot(gs[i+1,6])
                ax_vbv.hist(vbv_standardized[i, :, :, :, 0].numpy().flatten(), bins=100, alpha=0.5, label="vbv", density=True)
                ax_vbv.set_title("Standardized vbv IC ID={0}".format(IC))
                ax_vbv.legend()

    # Save figure
    plt.savefig(model_path+"/loss_history_and_validation_lambda_{0}_lr__.png".format(lambda_gp, learning_rate))
    plt.close()

def plot_lr(resume=False):
    # plot learning rate versus epoch
    current_generator_lr = learning_rate_g(generator_optimizer.iterations)
    current_critic_lr = learning_rate(critic_optimizer.iterations)
    lr_generator.append(current_generator_lr.numpy())
    lr_critic.append(current_critic_lr.numpy())
    _, ax = plt.subplots(1, 1, figsize=(10, 5))  # Fix: Unused variable 'fig'
    ax.plot(range(len(lr_generator)), lr_generator, label="generator", linewidth=2, ls="dashed")
    ax.plot(range(len(lr_critic)), lr_critic, label="critic")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(model_path + "/learning_rate.png")
    plt.close()

def plot_anim(generator, T21_big, T21_lr, IC_delta, IC_vbv, epoch=None, layer_name='concatenate_10', sigmas=3):
    generated_boxes = generator.forward(T21_lr, IC_delta, IC_vbv)
    intermediate_layer_model = tf.keras.Model(inputs=generator.model.inputs, outputs=generator.model.get_layer(layer_name).output)
    if IC_vbv is not None:
        intermediate_output = intermediate_layer_model([T21_lr, IC_delta, IC_vbv])
    else:
        intermediate_output = intermediate_layer_model([T21_lr, IC_delta])
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,10))
    if epoch is not None:
        fig.suptitle("Epoch {0}".format(epoch))

    def update(i):
        T21_std = np.std(T21_big[0, :, :, :, 0].numpy().flatten())
        ax[0].imshow(T21_big[0,:,:,T21_big.shape[-2]//2,0], vmin=-sigmas*T21_std, vmax=sigmas*T21_std)
        ax[0].set_title("T21 big box")
        ax[1].imshow(generated_boxes[0,:,:,generated_boxes.shape[-2]//2,0], vmin=-sigmas*T21_std, vmax=sigmas*T21_std)
        ax[1].set_title("T21_lr passed \nthrough full generator")
        ax[2].imshow(intermediate_output[0,:,:,intermediate_output.shape[-2]//2,i], vmin=-sigmas*T21_std, vmax=sigmas*T21_std)
        ax[2].set_title("T21_lr passed \nthrough {0} \nFeature map, channel {1}".format(layer_name,i))

    anim = FuncAnimation(fig, update, frames=intermediate_output.shape[-1])
    anim.save(model_path + "/anim.gif", dpi=300, writer=PillowWriter(fps=1))
    plt.close()

n_critic = 10
epochs = 10000
beta_1 = 0.5
beta_2 = 0.999
#calculate decay steps from T21  and epochs. I want the learning rate to decay to 1e-5 after 200 epochs
#learning_rate = 3e-2 #tf.keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=3e-3,#3e-3,
    #decay_steps=100,#100,
    #decay_rate=0.9)
#learning_rate_g = 3e-2#tf.keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=9e-3,#3e-3,
    #decay_steps=10,#10,
    #decay_rate=0.9)
learning_rate = np.logspace(-4,-2.5,4)[:-2]#0.01#np.logspace(-4,1,11).tolist()#[1e-2, 1e-3, 3e-4]#[3e-3, 1e-3, 3e-4] #np.logspace(-5,-5,1) #np.logspace(-6,-5,2) #np.logspace(-4,-1,4) #1e-4
learning_rate_g = learning_rate
lambda_gp = 10.#[10.,] #np.logspace(1,1,1) #np.logspace(0,0,1) #np.logspace(-4,0,5) #1e-2
lambda_mse = np.logspace(-4,8,8)



combinations = list(itertools.product(lambda_mse, learning_rate))
lambda_mse,learning_rate = combinations[index]

print("Params: ", lambda_mse, learning_rate, len(combinations),flush=True)

inception_kwargs = {
            #'input_channels': self.T21_shape[-1],
            #'filters_1x1x1_7x7x7': 4,
            #'filters_7x7x7': 4,
            #'filters_1x1x1_5x5x5': 4,
            #'filters_5x5x5': 4,
            #'filters_1x1x1_3x3x3': 4,
            #'filters_3x3x3': 4,
            #'filters_1x1x1': 4,
            'kernel_initializer': 'glorot_uniform', #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None), #
            'bias_initializer': 'zeros',#tf.keras.initializers.Constant(value=0.1), #
            
            'strides': (1,1,1), 
            'data_format': 'channels_last', 
            'padding': 'valid',
            }

generator = Generator(kernel_initializer='glorot_uniform', #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
                      bias_initializer='zeros', 
                      network_model='original_variable_output_activation',
                      lambda_mse=lambda_mse, 
                      inception_kwargs=inception_kwargs, vbv_shape=None)
critic = Critic(kernel_initializer='glorot_uniform', #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
                bias_initializer='zeros',
                lambda_gp=lambda_gp, vbv_shape=None, network_model='original_layer_norm')

if False:
    import tensorflow_addons as tfa
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=learning_rate_g, beta_1=beta_1, beta_2=beta_2),
        tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-1,
            decay_steps=10,
            decay_rate=0.9), 
            beta_1=beta_1, beta_2=beta_2)
            ]
    optimizers_and_layers = [(optimizers[0], generator.model.layers[:-1]), (optimizers[1], generator.model.layers[-1])]
    generator_optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2) #learning_rate_g
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)


#model.summary()
tf.keras.utils.plot_model(generator.model, 
                          to_file=path+'/plots/generator_model_skip_patches.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)
#tf.keras.utils.plot_model(critic.model,
#                          to_file=path+'/plots/critic_model_2.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)




Data = DataManager(path, redshifts=[10,], IC_seeds=list(range(1000,1002)))#1008
dataset = Data.data(augment=True, augments=9, low_res=True)
dataset = tf.data.Dataset.from_tensor_slices(dataset)

Data_validation = DataManager(path, redshifts=[10,], IC_seeds=[1008,1009,1010])
T21_validation, delta_validation, vbv_validation, T21_lr_validation = Data_validation.data(augment=False, augments=9, low_res=True) #T21, delta, vbv, T21_lr
T21_validation_standardized = standardize(T21_validation, T21_lr_validation, subtract_mean=False)
T21_lr_validation_standardized = standardize(T21_lr_validation, T21_lr_validation, subtract_mean=False)
delta_validation_standardized = standardize(delta_validation, delta_validation, subtract_mean=True)
vbv_validation_standardized = None #standardize(vbv_validation, vbv_validation)






model_path = path+"/trained_models/model_{0}".format(index)#index+20)#22
#make model directory if it doesn't exist:
if os.path.exists(model_path)==False:
    os.mkdir(model_path)
    os.mkdir(model_path+"/logs")
    train_summary_writer = tf.summary.create_file_writer(model_path+"/logs")

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
    print("Are weights different after restoring from checkpoint: ", are_weights_different, flush=True)

    if (os.path.exists(model_path+"/losses.pkl")==False) or (os.path.exists(model_path+"/checkpoints")==False) or (are_weights_different==False):
        assert False, "Resume=True: Checkpoints directory or losses file does not exist or weights are unchanged after restoring, cannot resume training."
else:
    print("Initializing from scratch.", flush=True)
    if os.path.exists(model_path+"/losses.pkl") or os.path.exists(model_path+"/checkpoints"):
        assert False, "Resume=False: Loss file or checkpoints directory already exists, exiting..."
    print("Creating loss file...", flush=True)
    with open(model_path+"/losses.pkl", "wb") as f:
        generator_losses_epoch = []
        critic_losses_epoch = []
        gradient_penalty_epoch = []
        generator_mse_losses_epoch = []
        pickle.dump((generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch, generator_mse_losses_epoch), f)
        



print("Starting training...", flush=True)
lr_generator = []
lr_critic = []
dsq_mse = 1e20
for e in range(epochs):
    start = time.time()
    
    generator_losses = []
    critic_losses = []
    gradient_penalty = []
    generator_mse = []
    batches = dataset.shuffle(buffer_size=len(dataset)).batch(4)
    for i, (T21, delta, vbv, T21_lr) in enumerate(batches):
        #print("shape inputs: ", T21.shape, delta.shape, vbv.shape, T21_lr.shape)
        start_start = time.time()
        T21_standardized = standardize(T21, T21_lr, subtract_mean=False)
        T21_lr_standardized = standardize(T21_lr, T21_lr, subtract_mean=False)
        delta_standardized = standardize(delta, delta, subtract_mean=True)
        vbv_standardized = None #vbv #standardize(vbv, vbv)
        
        crit_loss, gp = critic.train_step_critic(generator=generator, optimizer=critic_optimizer, T21_big=T21_standardized, 
                                                    T21_small=T21_lr_standardized, IC_delta=delta_standardized, IC_vbv=None)#vbv_standardized)
        critic_losses.append(crit_loss)
        gradient_penalty.append(gp)
        if i%n_critic == 0:
            #print("Output activation settings before optimizing:", generator.model.get_layer("pe_lu").get_config())
            gen_loss, mse = generator.train_step_generator(critic=critic, optimizer=generator_optimizer, T21_small=T21_lr_standardized, 
                                                      T21_big=T21_standardized, IC_delta=delta_standardized, IC_vbv=None)#vbv_standardized)
            generator_losses.append(gen_loss)
            generator_mse.append(mse)
        
        print("Time for batch {0} is {1:.2f} sec".format(i + 1, time.time() - start_start), flush=True)
    
    critic_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
    generator_loss = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
    gp_loss = tf.keras.metrics.Mean('gp_loss', dtype=tf.float32)
    critic_loss.update_state(critic_losses)
    generator_loss.update_state(generator_losses)
    gp_loss.update_state(gradient_penalty)
    with train_summary_writer.as_default():
        #merged_summary = tf.summary.merge(losses.values())
        tf.summary.scalar('loss/Wloss', critic_loss.result().numpy(), step=e)
        tf.summary.scalar('loss/Gloss', generator_loss.result().numpy(), step=e)
        tf.summary.scalar('loss/Wgp', gp_loss.result().numpy(), step=e)
    #reset state
    critic_loss.reset_states()
    generator_loss.reset_states()
    gp_loss.reset_states()

    
    #save losses
    with open(model_path+"/losses.pkl", "rb") as f: # Open the file in read mode and get data
        generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch, generator_mse_losses_epoch = pickle.load(f)
    # Append the new values to the existing data
    generator_losses_epoch.append(np.mean(generator_losses))
    critic_losses_epoch.append(np.mean(critic_losses))
    gradient_penalty_epoch.append(np.mean(gradient_penalty))
    generator_mse_losses_epoch.append(np.mean(generator_mse))
    with open(model_path+"/losses.pkl", "wb") as f: # Open the file in write mode and dump the data
        pickle.dump((generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch,generator_mse_losses_epoch), f)
    
    #checkpoint
    #if e%2 == 0:
    try:
        min_tester = np.min(generator_mse_losses_epoch[:-1])
    except:
        min_tester=1e20#np.NaN
    if (np.mean(generator_mse) < min_tester):
        print("Generator mse loss improved, checking dsq...", flush=True)
        generated_boxes_validation = generator.forward(T21_lr_validation_standardized, delta_validation_standardized, vbv_validation_standardized)
        dsq_mse_ = []
        
        for i in range(generated_boxes_validation.shape[0]):
            k_gen, Pk_gen = calculate_power_spectrum(data_x=generated_boxes_validation[i,:,:,:,0], Lpix=3, kbins=100)
            k_real, Pk_real = calculate_power_spectrum(data_x=T21_validation_standardized[i,:,:,:,0], Lpix=3, kbins=100)
            dsq_gen = Pk_gen*k_gen**3/(2*np.pi**2)
            dsq_real = Pk_real*k_real**3/(2*np.pi**2)
            dsq_mse_.append(np.nanmean((dsq_gen-dsq_real)**2))

        dsq_mse_ = np.nanmean(dsq_mse_)
        print("Current dsq_mse: {0:.2f}, Historic minimum dsq_mse: {1:.2f}".format(dsq_mse_, dsq_mse), flush=True)
        if (dsq_mse_ < dsq_mse):
            dsq_mse = dsq_mse_
            print("Saving checkpoint...", flush=True)
            print("Epoch: {0}\nGmse: {1:.2f}\ndsq_mse: {2:.2f}".format(e, np.mean(generator_mse), dsq_mse) , flush=True)
            manager.save()
            print("Checkpoint saved!", flush=True)
        else:
            print("dsq did not improve. Not saving checkpoint...", flush=True)
    else:
        print("Generator mse did not improve. Not saving checkpoint...", flush=True)
            
    #"validation: plot and savefig loss history, and histograms and imshows for two models for every 10th epoch"
    #with gridspec loss history should extend the whole top row and the histograms and imshows should fill one axes[i,j] for the bottom rows
    if e%1 == 0:
        plot_and_save(IC_seeds=[1008,1009,1010], redshift=10, sigmas=3, subtract_mean=False)
        #plot_lr()
    #if e%1 == 0:
        #plot_anim(generator=generator, T21_big=T21_standardized, 
                  #T21_lr=T21_lr_standardized, IC_delta=delta_standardized, IC_vbv=None, 
                  #epoch=e, layer_name='conv3d_72', sigmas=3)
    #if e%1 == 0:
        #plot_anim(generator=generator, T21_big=T21_standardized, 
                  #T21_lr=T21_lr_standardized, IC_delta=delta_standardized, IC_vbv=None, 
                  #epoch=e, layer_name='conv3d_72', sigmas=3)

    #try:
    #    print("Clipping layer: ", generator.model.layers[-1].trainable_variables[0].numpy()/generator.model.layers[-1].sensitivity)
    #except Exception as e:
    #    print("Failed printing clipping layer: ", e)
    print("Time for epoch {0} is {1:.2f} sec \nGenerator mean loss: {2:.2f}, \nCritic mean loss: {3:.2f}, \nGradient mean penalty: {4:.2f}".format(e + 1, time.time() - start, np.mean(generator_losses), np.mean(critic_losses), np.mean(gradient_penalty)), flush=True)
    #break

    



with open(model_path+"/losses.pkl", "rb") as f:
    generator_losses_epoch, critic_losses_epoch, gradient_penalty_epoch, generator_mse_losses_epoch = pickle.load(f)
#print last 10 losses and total number of epochs
print("Last 10 losses: \nGenerator: {0} \nCritic: {1} \nGradient penalty: {2}\nGenerator mse: {3}".format(generator_losses_epoch[-10:], critic_losses_epoch[-10:], gradient_penalty_epoch[-10:],generator_mse_losses_epoch[-10:]), flush=True)
"""
"""



"""
#animation

if True:
    z = list(np.arange(6,29,1)[::1])
    Data_hist = DataManager(path, redshifts=z, IC_seeds=[1005,1006,1007,1008,1009,1010])
    T21, delta, vbv, T21_lr = Data_hist.data(augment=False, augments=9, low_res=True) 

    #fig,ax = plt.subplots(1,2,figsize=(10,5))
    shapes = (2,3)
    fig,ax = plt.subplots(*shapes,figsize=(15,5))
    
    for i in range(T21_lr.shape[0]):
        ind = np.unravel_index(i, shapes)
        #ax[ind].imshow(T21_target[i,:,:,10,-1])
        data_standardized = standardize(T21_lr[i:i+1,:,:,:,-1], T21_lr[i:i+1,:,:,:,-1], i)
        #ax[ind].hist(data_standardized[:,:,:,-1].numpy().flatten(), bins=100)
    
    def update(i):
        for j in range(T21_lr.shape[0]):
            ind = np.unravel_index(j, shapes)
            #ax[ind].imshow(T21_target[j,:,:,10,-i-1])
            #ax[ind].imshow(test[j,:,:,10,-i-1])
            data_standardized = standardize(data=T21_lr[j:j+1,:,:,:,i], data_stats=T21_lr[j:j+1,:,:,:,i],subtract_mean=False)
            ax[ind].hist(data_standardized.numpy().flatten(), bins=100, density=True)
            #try:
            #    ax[ind].hist(data_standardized[:,:,:,10].numpy().flatten(), bins=100)
            #except Exception as e:
            #    print("couldn't plot j={0}, i={1}, error: ".format(j,i), e)
            
            #ax[ind].set_title('z = '+str(z[-i-1]))
            #ax[1].imshow(T21_train[3,:,:,10,-i-1])
            #ax.set_axis_off()
            ax[ind].set_xlim(-7,7)
            ax[ind].set_ylim(0,1)
            ax[ind].set_title('z = '+str(z[i]))
        #ax[0].imshow(T21_target[3,:,:,10,-i-1])
        #ax[1].imshow(T21_train[3,:,:,10,-i-1])
        #ax.set_axis_off()

    anim = FuncAnimation(fig, update, frames=len(z), interval=800)

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
    anim.save(path+"/plots/hist_3_1.gif", dpi=300, writer=PillowWriter(fps=2))

#data_standardized = standardize(T21_lr[j:j+1,:,:,:,-i-1], T21_lr[j:j+1,:,:,:,-i-1])
#check T21_lr if all zeros
#print("tf count nonzero j=0 T21_lr sum: ", tf.math.count_nonzero(T21_lr[0:1,:,:,10,-22-1]))
#mean, var = tf.nn.moments(T21_lr[0:5,:,:,:,-22-1], axes=[1,2,3])

#set mean[i] to 0 and var[i] to 1 if both are zero:


#plt.imshow(T21_lr[0,:,:,10,-22-1])

#plt.title("z={0}, Mean: {1:.2f}, Std: {2:.2f}".format(z[22], mean[0], var[0]**0.5))
#plt.show()



"""
