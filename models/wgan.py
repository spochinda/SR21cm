import tensorflow as tf
from models.utils import *
#from utils import *

class Critic(tf.keras.Model):
    def __init__(self,
                 delta_shape=(1,128,128,128,1), vbv_shape=(1,128,128,128,1),
                 kernel_sizes=[7,5,3,1],
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                 bias_initializer=tf.keras.initializers.Constant(value=0.1),
                 lambda_gp=1e1, activation='tanh',
                 network_model='modified'):
        super(Critic, self).__init__()
        #self.T21_shape = T21_shape
        self.delta_shape = delta_shape
        self.vbv_shape = vbv_shape
        self.kernel_sizes = kernel_sizes
        self.crop = int((max(self.kernel_sizes)-1))
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.lambda_gp = lambda_gp
        self.activation = activation
        self.network_model = network_model
        
        self.build_critic_model()
        #max kernel size must be odd
        assert max(self.kernel_sizes)%2==1, "max kernel size must be odd"

    def build_critic_model(self):
        if self.vbv_shape != None:
            input = tf.keras.layers.Input(shape=(self.delta_shape[1]-self.crop*2, self.delta_shape[2]-self.crop*2, self.delta_shape[3]-self.crop*2, 3)) #not including the batch size according to docs
        else:
            input = tf.keras.layers.Input(shape=(self.delta_shape[1]-self.crop*2, self.delta_shape[2]-self.crop*2, self.delta_shape[3]-self.crop*2, 2)) #not including the batch size according to docs
        
        if self.network_model == 'modified':
            input_ = tf.keras.layers.Conv3D(filters=3, kernel_size=(self.kernel_sizes[0], self.kernel_sizes[0], self.kernel_sizes[0]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input)
            input_pos = tf.keras.layers.Conv3D(filters=3, kernel_size=(self.kernel_sizes[0], self.kernel_sizes[0], self.kernel_sizes[0]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input)
            input_neg = -tf.keras.layers.Conv3D(filters=3, kernel_size=(self.kernel_sizes[0], self.kernel_sizes[0], self.kernel_sizes[0]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(-input)

            input_ = tf.keras.layers.Conv3D(filters=5, kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1], self.kernel_sizes[1]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input_)
            input_pos = tf.keras.layers.Conv3D(filters=5, kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1], self.kernel_sizes[1]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input_pos)
            input_neg = -tf.keras.layers.Conv3D(filters=5, kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1], self.kernel_sizes[1]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(-input_neg)
            
            input_ = tf.keras.layers.Conv3D(filters=9, kernel_size=(self.kernel_sizes[2], self.kernel_sizes[2], self.kernel_sizes[2]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input_)
            input_pos = tf.keras.layers.Conv3D(filters=9, kernel_size=(self.kernel_sizes[2], self.kernel_sizes[2], self.kernel_sizes[2]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input_pos)
            input_neg = -tf.keras.layers.Conv3D(filters=9, kernel_size=(self.kernel_sizes[2], self.kernel_sizes[2], self.kernel_sizes[2]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(-input_neg)
            
            input_ = tf.keras.layers.Conv3D(filters=21, kernel_size=(self.kernel_sizes[3], self.kernel_sizes[3], self.kernel_sizes[3]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input_)
            input_pos = tf.keras.layers.Conv3D(filters=21, kernel_size=(self.kernel_sizes[3], self.kernel_sizes[3], self.kernel_sizes[3]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input_pos)
            input_neg = -tf.keras.layers.Conv3D(filters=21, kernel_size=(self.kernel_sizes[3], self.kernel_sizes[3], self.kernel_sizes[3]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(-input_neg)
            
            output = tf.keras.layers.Concatenate(axis=4)([input_, input_pos, input_neg])
        elif self.network_model == 'original':
            output = tf.keras.layers.Conv3D(filters=8, kernel_size=(self.kernel_sizes[0], self.kernel_sizes[0], self.kernel_sizes[0]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input)

            output = tf.keras.layers.Conv3D(filters=16, kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1], self.kernel_sizes[1]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(output)
            
            output = tf.keras.layers.Conv3D(filters=32, kernel_size=(self.kernel_sizes[2], self.kernel_sizes[2], self.kernel_sizes[2]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(output)
            
            output = tf.keras.layers.Conv3D(filters=64, kernel_size=(self.kernel_sizes[3], self.kernel_sizes[3], self.kernel_sizes[3]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=tf.keras.layers.LeakyReLU(alpha=0.1)#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(output)
            
        elif self.network_model == 'original_layer_norm':
            layernorm = False
            output = tf.keras.layers.Conv3D(filters=8, kernel_size=(self.kernel_sizes[0], self.kernel_sizes[0], self.kernel_sizes[0]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=None#tf.keras.layers.LeakyReLU(alpha=0.1)
                                                )(input)
            output = tf.keras.layers.LayerNormalization()(output) if layernorm else output
            output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)

            output = tf.keras.layers.Conv3D(filters=16, kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1], self.kernel_sizes[1]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=None
                                                )(output)
            output = tf.keras.layers.LayerNormalization()(output) if layernorm else output
            output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
            
            output = tf.keras.layers.Conv3D(filters=32, kernel_size=(self.kernel_sizes[2], self.kernel_sizes[2], self.kernel_sizes[2]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(2, 2, 2), padding='valid', data_format="channels_last", 
                                                activation=None
                                                )(output)
            output = tf.keras.layers.LayerNormalization()(output) if layernorm else output
            output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
            
            output = tf.keras.layers.Conv3D(filters=64, kernel_size=(self.kernel_sizes[3], self.kernel_sizes[3], self.kernel_sizes[3]), 
                                                kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                                bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                                strides=(1, 1, 1), padding='valid', data_format="channels_last", 
                                                activation=None
                                                )(output)
            output = tf.keras.layers.LayerNormalization()(output) if layernorm else output
            output = tf.keras.layers.LeakyReLU(alpha=0.1)(output)
        
        output = tf.keras.layers.Flatten()(output)
        #flatten = tf.keras.layers.Flatten()
        output = tf.keras.layers.Dense(units=1,
                                         kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                         bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                         name='output_layer')(output)

        #self.model = tf.keras.Sequential([conv1, conv2, conv3, conv4, flatten, out])    
        self.model = tf.keras.Model(inputs=[input], outputs=output)
        return self.model
    
    

    #@tf.function
    def critic_loss(self, generator, T21_small, T21_big, IC_delta, IC_vbv=None):
        #wasserstein loss
        # Generate a batch of fake big boxes using the generator network

        epsilon = tf.random.uniform(shape=[T21_big.shape[0], 1, 1, 1, 1], minval=0., maxval=1., seed=None)
        # Compute the gradients of the critic network with respect to the interpolated difference
        with tf.GradientTape() as tape:
            # Evaluate the critic network on the real big boxes and the fake big boxes
            generated_boxes = generator.forward(T21_small, IC_delta, IC_vbv)

            # Crop to right size for critic
            T21_big = tf.keras.layers.Cropping3D(cropping=(self.crop,self.crop,self.crop),data_format="channels_last")(T21_big)
            IC_delta = tf.keras.layers.Cropping3D(cropping=(self.crop,self.crop,self.crop),data_format="channels_last")(IC_delta)
            if IC_vbv != None:
                IC_vbv = tf.keras.layers.Cropping3D(cropping=(self.crop,self.crop,self.crop),data_format="channels_last")(IC_vbv)
            
            #interpolated gp difference
            xhat = epsilon * T21_big + (1 - epsilon) * generated_boxes    

            critic_output = self.forward(xhat, IC_delta, IC_vbv)
        gradients = tape.gradient(critic_output, xhat)
        l2_norm = tf.math.reduce_euclidean_norm(gradients, axis=[1,2,3])
        gp = tf.reduce_mean(self.lambda_gp * tf.square(l2_norm - 1))
        
        W_real = tf.reduce_mean(self.forward(T21_big, IC_delta, IC_vbv))
        W_gen = tf.reduce_mean(self.forward(generated_boxes, IC_delta, IC_vbv))

        # Compute the approximate Wasserstein loss
        loss = W_gen - W_real + gp

        return loss, gp 

    @tf.function
    def train_step_critic(self, generator, optimizer, T21_big, T21_small, IC_delta, IC_vbv):
        if (IC_vbv == None) and (self.vbv_shape != None): 
            assert False, "Critic was initialized with vbv_shape=None, but IC_vbv in train_step_critic is not None"
        
        #gradient of loss
        with tf.GradientTape() as disc_tape:
            crit_loss, gp = self.critic_loss(generator, T21_small, T21_big, IC_delta, IC_vbv)

        grad_disc = disc_tape.gradient(crit_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grad_disc, self.model.trainable_variables))
        
        return crit_loss, gp
    
    @tf.function
    def forward(self, T21_target, IC_delta, IC_vbv=None):
        if IC_vbv != None:
            data_target = tf.concat((T21_target, IC_delta, IC_vbv), axis=4)
        else:
            data_target = tf.concat((T21_target, IC_delta), axis=4)
        #x_out_model = self.model(data_target)
        x_out_model = self.model(inputs=[data_target])
        return x_out_model
    #def call(self, T21_train, IC_delta, IC_vbv):
        #return self.model(inputs=[T21_train, IC_delta, IC_vbv])




class Generator(tf.keras.Model):
    def __init__(self, T21_shape=(1,64,64,64,1), delta_shape=(1,128,128,128,1), vbv_shape=(1,128,128,128,1),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                 bias_initializer=tf.keras.initializers.Constant(value=0.1),
                 network_model='original',
                 lambda_mse=1.,
                 lambda_dsq_mse=1.,
                 residual_block_kwargs= {'activation': [tf.keras.layers.LeakyReLU(alpha=0.1), tf.keras.layers.Activation('tanh'), tf.keras.layers.LeakyReLU(alpha=0.1)]},
                 inception_kwargs={}):
        super(Generator, self).__init__()
        self.T21_shape = T21_shape
        self.delta_shape = delta_shape
        self.vbv_shape = vbv_shape
        self.upsampling = delta_shape[1]//T21_shape[1]
        #self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.network_model = network_model
        self.lambda_mse = lambda_mse
        self.lambda_dsq_mse = lambda_dsq_mse
        self.residual_block_kwargs = residual_block_kwargs
        self.inception_kwargs = inception_kwargs
        self.build_generator_model()

    def build_generator_model(self):   
        inputs_T21 = tf.keras.layers.Input(shape=self.T21_shape[1:]) #not including the batch size according to docs
        inputs_delta = tf.keras.layers.Input(shape=self.delta_shape[1:])
        if self.vbv_shape != None:
            inputs_vbv = tf.keras.layers.Input(shape=self.vbv_shape[1:])
        else:
            inputs_vbv = None


        
        T21 = tf.keras.layers.UpSampling3D(size=self.upsampling, data_format="channels_last")(inputs_T21)
        
        if self.network_model == 'modified':
            #get ionized regions
            #T21_ion = tf.keras.layers.Lambda(
            #    lambda x: tf.where(x > tf.reduce_min(x, axis=[1,2,3],keepdims=True), 0., x))(T21)
            #T21_ion = InceptionLayer(use_bias = False, **self.inception_kwargs)(T21_ion)
            #T21_ion = tf.keras.layers.Cropping3D(cropping=(3, 3, 3),data_format="channels_last")(T21_ion)#crop to size

            T21_, T21_pos, T21_neg = ResidualBlock(InceptionLayer, filters=4, **self.residual_block_kwargs, **self.inception_kwargs)(T21)
            delta_, delta_pos, delta_neg = ResidualBlock(InceptionLayer, filters=4, **self.residual_block_kwargs, **self.inception_kwargs)(inputs_delta)

            if self.vbv_shape != None:
                vbv_, vbv_pos, vbv_neg = ResidualBlock(InceptionLayer, filters=4, **self.residual_block_kwargs, **self.inception_kwargs)(inputs_vbv)
            T21_, T21_pos, T21_neg = ResidualBlock(InceptionLayer, filters=4, **self.residual_block_kwargs, **self.inception_kwargs)(T21)
            delta_, delta_pos, delta_neg = ResidualBlock(InceptionLayer, filters=4, **self.residual_block_kwargs, **self.inception_kwargs)(inputs_delta)

            if self.vbv_shape != None:
                vbv_, vbv_pos, vbv_neg = ResidualBlock(InceptionLayer, filters=4, **self.residual_block_kwargs, **self.inception_kwargs)(inputs_vbv)

                data = tf.keras.layers.Concatenate(axis=4)([T21_, T21_pos, T21_neg, delta_, delta_pos, delta_neg, vbv_, vbv_pos, vbv_neg])
            else:
                data = tf.keras.layers.Concatenate(axis=4)([T21_, T21_pos, T21_neg, delta_, delta_pos, delta_neg])

            data_, data_pos, data_neg = ResidualBlock(InceptionLayer, filters=4, **self.residual_block_kwargs, **self.inception_kwargs)(data)

            data = tf.keras.layers.Concatenate(axis=4)([#T21_ion, 
                                                        data_, data_pos, data_neg])    
            data = tf.keras.layers.Conv3D(filters=1,
                                          kernel_size=(1, 1, 1),
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer,
                                          strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                          activation=None)(data)
        elif self.network_model =='original':
            T21 = InceptionLayer(filters=6, **self.inception_kwargs)(T21)
            T21 = tf.keras.layers.LeakyReLU(alpha=0.1)(T21)

            delta = InceptionLayer(filters=6, **self.inception_kwargs)(inputs_delta)
            delta = tf.keras.layers.LeakyReLU(alpha=0.1)(delta)

            if self.vbv_shape != None:
                vbv = InceptionLayer(filters=6, **self.inception_kwargs)(inputs_vbv)
                vbv = tf.keras.layers.LeakyReLU(alpha=0.1)(vbv)

                data = tf.keras.layers.Concatenate(axis=4)([T21, delta, vbv])
            else:
                data = tf.keras.layers.Concatenate(axis=4)([T21, delta])

            data = InceptionLayer(filters=6, **self.inception_kwargs)(data)
            data = tf.keras.layers.LeakyReLU(alpha=0.1)(data)
            data = tf.keras.layers.Conv3D(filters=1,
                                          kernel_size=(1, 1, 1),
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer,
                                          strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                          activation=None)(data)
        elif self.network_model =='original_variable_output_activation':
            layernorm = False
            T21 = InceptionLayer(filters=6, **self.inception_kwargs)(T21)
            #T21 = tf.keras.layers.LayerNormalization()(T21) if layernorm else T21
            T21 = tf.keras.layers.LeakyReLU(alpha=0.1)(T21)

            delta = InceptionLayer(filters=6, **self.inception_kwargs)(inputs_delta)
            delta = tf.keras.layers.LayerNormalization()(delta) if layernorm else delta
            delta = tf.keras.layers.LeakyReLU(alpha=0.1)(delta)

            if self.vbv_shape != None:
                vbv = InceptionLayer(filters=6, **self.inception_kwargs)(inputs_vbv)
                vbv = tf.keras.layers.LayerNormalization()(vbv) if layernorm else vbv
                vbv = tf.keras.layers.LeakyReLU(alpha=0.1)(vbv)

                data = tf.keras.layers.Concatenate(axis=4)([T21, delta, vbv])
            else:
                data = tf.keras.layers.Concatenate(axis=4)([T21, delta])

            data = InceptionLayer(filters=6, **self.inception_kwargs)(data)
            #data = tf.keras.layers.LayerNormalization()(data) if layernorm else data
            data = tf.keras.layers.LeakyReLU(alpha=0.1)(data)
            data = tf.keras.layers.Conv3D(filters=1,
                                          kernel_size=(1, 1, 1),
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer,
                                          strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                          activation=None)(data)
            ionized_frac = ionized_fraction(inputs_T21)
            mask = ionized_frac >= 0.#0.0005 #0.003
            data = tf.where(mask, tf.keras.layers.ReLU()(data), data)
        
        elif self.network_model =='skip_patches':
            T21_ion = tf.keras.layers.Lambda(lambda x: tf.where(x > tf.reduce_min(x, axis=[1,2,3],keepdims=True), 0., x))(T21)
            T21_ion = InceptionLayer(filters=6,kernel_initializer = 'glorot_uniform', #tf.keras.initializers.RandomNormal(mean=.02, stddev=0.005, seed=None),
                                     use_bias=False, strides= 1,data_format = 'channels_last', padding='valid',)(T21_ion) #**self.inception_kwargs)(T21_ion)
            #T21_ion = tf.keras.layers.ReLU()(T21_ion)###
            T21_ion= tf.keras.layers.Conv3D(filters=1,
                            kernel_size=(1, 1, 1),
                            kernel_initializer=self.kernel_initializer,
                            use_bias=False,#bias_initializer=self.bias_initializer,
                            strides=(1, 1, 1), padding='valid', data_format="channels_last",
                            activation=None)(T21_ion)
            T21_ion = tf.keras.layers.Cropping3D(cropping=(3, 3, 3))(T21_ion)
            
            T21 = InceptionLayer(filters=6, **self.inception_kwargs)(T21)
            T21 = tf.keras.layers.LeakyReLU(alpha=0.1)(T21)

            delta = InceptionLayer(filters=6, **self.inception_kwargs)(inputs_delta)
            delta = tf.keras.layers.LeakyReLU(alpha=0.1)(delta)

            if self.vbv_shape != None:
                vbv = InceptionLayer(filters=6, **self.inception_kwargs)(inputs_vbv)
                vbv = tf.keras.layers.LeakyReLU(alpha=0.1)(vbv)

                data = tf.keras.layers.Concatenate(axis=4)([T21, delta, vbv])
            else:
                data = tf.keras.layers.Concatenate(axis=4)([T21, delta])

            data = InceptionLayer(filters=6, **self.inception_kwargs)(data)
            data = tf.keras.layers.LeakyReLU(alpha=0.1)(data)
            data = tf.keras.layers.Conv3D(filters=1,
                            kernel_size=(1, 1, 1),
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            strides=(1, 1, 1), padding='valid', data_format="channels_last",
                            activation=None)(data)

            #data = tf.tile(data, [1,1,1,1,T21_ion.shape[-1]])
            data = tf.keras.layers.Add()([data, T21_ion])
            data = tf.keras.layers.Conv3D(filters=1,
                                          kernel_size=(1, 1, 1),
                                          kernel_initializer=self.kernel_initializer,
                                          bias_initializer=self.bias_initializer,
                                          strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                          activation=None)(data)
            
        
        #data = ClippingLayer(a = -4.5, sensitivity=1e-1, min_value = -5., max_value = -4.,trainable = True)(data) #array3
    
        
        if self.vbv_shape != None:
            self.model = tf.keras.Model(inputs=[inputs_T21, inputs_delta, inputs_vbv], outputs=data)
        else:
            self.model = tf.keras.Model(inputs=[inputs_T21, inputs_delta], outputs=data)
        return self.model

    @tf.function
    def generator_loss(self, critic, generated_boxes, T21_big, IC_delta, IC_vbv=None):
        T21_big = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(T21_big)
        IC_delta = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(IC_delta)
        if IC_vbv != None:
            IC_vbv = tf.keras.layers.Cropping3D(cropping=(6, 6, 6),data_format="channels_last")(IC_vbv)
        
        W_real = critic.forward(T21_big, IC_delta, IC_vbv)
        W_gen = critic.forward(generated_boxes, IC_delta, IC_vbv)

        mse = tf.keras.losses.MeanSquaredError()(T21_big, generated_boxes)
        #generated_boxes_validation = generator.forward(T21_lr_validation_standardized, delta_validation_standardized, vbv_validation_standardized)
        #dsq_mse = tf.py_function(func=calculate_power_spectrum_mse, inp=[generated_boxes, T21_big, 3, 100], Tout=tf.float32)
        dsq_mse = calculate_power_spectrum_mse(generated_boxes=generated_boxes, T21_batch=T21_big, Lpix=3, kbins=100) if self.lambda_dsq_mse!=0. else 0.
        
        loss = - tf.reduce_mean(W_gen - W_real) + self.lambda_mse * mse + self.lambda_dsq_mse * dsq_mse

        #tf.print("loss: ", loss, "mse: ", mse, "dsq_mse: ", dsq_mse)
        return loss, mse, dsq_mse

    @tf.function
    def train_step_generator(self, critic, optimizer, T21_small, T21_big, IC_delta, IC_vbv=None):
        #
        #Function that performs one training step for the generator network.
        #The function calls the loss function for the generator network, computes the gradients,
        #and applies the gradients to the network's parameters.
        #

        with tf.GradientTape() as gen_tape: 
            generated_boxes = self.forward(T21_small, IC_delta, IC_vbv)
            #generated_output = Critic(generated_boxes, IC_delta, IC_vbv)
            gen_loss, mse, dsq_mse= self.generator_loss(critic, generated_boxes, T21_big, IC_delta, IC_vbv)

        grad_gen = gen_tape.gradient(gen_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grad_gen, self.model.trainable_variables))

        return gen_loss, mse, dsq_mse
        
    @tf.function    
    def forward(self, T21_train, IC_delta, IC_vbv=None):
        if IC_vbv != None:
            return self.model(inputs=[T21_train, IC_delta, IC_vbv])
        else:
            return self.model(inputs=[T21_train, IC_delta])
    

class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, filters = 4, activation=tf.keras.layers.Activation('tanh'), **kwargs):
        super(InceptionLayer, self).__init__()
        self.filters = filters
        self.activation = activation
        self.kwargs = kwargs
        self.build()

    def build(self):
        #super().build(input_shape)
        self.conv_1x1x1_7x7x7 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1, 1, 1),
                                                       activation=None, **self.kwargs) #added kwargs
        self.conv_7x7x7 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(7, 7, 7),
                                                 activation=None, **self.kwargs)
        self.conv_1x1x1_5x5x5 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1, 1, 1),
                                                        activation=None, **self.kwargs)
        self.conv_5x5x5 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(5, 5, 5),
                                                 activation=None, **self.kwargs)
        self.crop_5x5x5 = tf.keras.layers.Cropping3D(cropping=(1, 1, 1))

        self.conv_1x1x1_3x3x3 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1, 1, 1),
                                                       activation=None, **self.kwargs)
        self.conv_3x3x3 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(3, 3, 3),
                                                 activation=None, **self.kwargs)
        self.crop_3x3x3 = tf.keras.layers.Cropping3D(cropping=(2, 2, 2))

        self.conv_1x1x1 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1, 1, 1),
                                                 activation=None, **self.kwargs)

        self.crop_1x1x1 = tf.keras.layers.Cropping3D(cropping=(3, 3, 3))
        self.concat = tf.keras.layers.Concatenate(axis=4)

        self.crop_x = tf.keras.layers.Cropping3D(cropping=(3, 3, 3))

        self.reduce_input_channels = tf.keras.layers.Conv3D(filters=self.filters*4, kernel_size=(1, 1, 1),
                                                                 activation=None, **self.kwargs)
        self.activation_layer = self.activation


    def __call__(self, x):
        x1 = self.conv_1x1x1_7x7x7(x)
        x1 = self.conv_7x7x7(x1)
        
        x2 = self.conv_1x1x1_5x5x5(x)
        x2 = self.conv_5x5x5(x2)
        x2 = self.crop_5x5x5(x2)
        
        x3 = self.conv_1x1x1_3x3x3(x)
        x3 = self.conv_3x3x3(x3)
        x3 = self.crop_3x3x3(x3)
        #filters_1x1x1_7x7x7=6, filters_7x7x7=6, filters_1x1x1_5x5x5=6, filters_5x5x5=6, filters_1x1x1_3x3x3=6, filters_3x3x3=6, filters_1x1x1=6
        x4 = self.conv_1x1x1(x)
        x4 = self.crop_1x1x1(x4)
        x_out = self.concat([x1, x2, x3, x4])
        
        x_out_ = self.crop_x(x[:,:,:,:,:])#crop input to right size, x[:,:,:,:,0:1] but not sure if this is the right way to do it
        #the issue is that the number of channels at the second pass through the inception
        #block is larger for the input than the output, so during the add step
        #I can't tile the input to match the output because it is larger.
        #Might have to implement change the number of filters to match the input 
        #on the second pass through, or do a 1x1x1 pointwise convolution to 
        #effectively reduce the number of channels to match the output. 
        #The 1x1x1 convolution is basically a weighted average of channels,
        # or linear transformation, with learned weights.
        #Update: implemented 1x1x1 convolution to reduce number of channels
        if x_out_.shape[-1] > x_out.shape[-1]: #if input channels larger than output channels
            x_out_ = self.reduce_input_channels(x_out_)
        else:
            x_out_ = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 1, x_out.shape[-1]]))(x_out_)

        x_out = tf.add(x_out, x_out_)
        #x_out = self.activation_layer(x_out) #need to comment out if network model is not modified
        return x_out
inception_kwargs = {
            #'input_channels': self.T21_shape[-1],
            #'filters_1x1x1_7x7x7': 4,
            #'filters_7x7x7': 4,
            #'filters_1x1x1_5x5x5': 4,
            #'filters_5x5x5': 4,
            #'filters_1x1x1_3x3x3': 4,
            #'filters_3x3x3': 4,
            #'filters_1x1x1': 4,
            #'filters': 4,
            'kernel_initializer': 'glorot_uniform',#tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), #
            'bias_initializer': 'zeros',#tf.keras.initializers.Constant(value=0.1), #
            
            'strides': (1,1,1), 
            'data_format': 'channels_last', 
            'padding': 'valid',
            #'activation': [tf.keras.layers.LeakyReLU(alpha=0.1), tf.keras.layers.Activation('tanh'), tf.keras.layers.LeakyReLU(alpha=0.1)]
            }

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, layer, filters=4,
                 activation=[tf.keras.layers.LeakyReLU(alpha=0.1), 
                             tf.keras.layers.Activation('tanh'),
                             tf.keras.layers.LeakyReLU(alpha=0.1)],
                **kwargs):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.filters = filters
        self.activation = activation
        self.kwargs = kwargs  # Save kwargs in an instance variable
        self.build()
    
    def build(self):
        self.x1 = self.layer(self.filters, activation=self.activation[0], **self.kwargs)
        self.x2 = self.layer(self.filters, activation=self.activation[1], **self.kwargs)
        self.x3 = self.layer(self.filters, activation=self.activation[2], **self.kwargs)
        
    def __call__(self, input):
        x1 = self.x1(input)
        x2 = self.x2(input)
        x3 = -self.x3(-input)
        return x1,x2,x3

class PeLU(tf.keras.layers.Layer):
    """``PeLU``."""
    def __init__(self,
                a: float = 2.512,
                b: float = 1.,
                c: float = 1.,
                trainable: bool = False,
                **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        self.a_factor = tf.Variable(
            self.a,
            dtype=tf.float32,
            trainable=False, #self.trainable,
            name="a_factor")

        self.b_factor = tf.Variable(
            self.b,
            dtype=tf.float32,
            trainable=self.trainable,
            name="b_factor")

        self.c_factor = tf.Variable(
            self.c,
            dtype=tf.float32,
            trainable=False, #self.trainable,
            name="c_factor")

    def call(self, inputs):
        res = tf.where(inputs >= 0, self.c_factor * inputs, self.a_factor * (tf.exp(inputs / self.b_factor) - 1))
        return res

    def get_config(self):
        config = {
            "a": self.get_weights()[0] if self.trainable else self.a,
            "b": self.get_weights()[1] if self.trainable else self.b,
            "c": self.get_weights()[2] if self.trainable else self.c,
            "trainable": self.trainable
        }
        #base_config = super().get_config()
        return config #dict(list(base_config.items()) + list(config.items()))
    
class CustomActivation(tf.keras.layers.Layer):
    """``PeLU``."""
    def __init__(self,
                a: float = 1.,
                trainable: bool = False,
                **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        self.a_factor = tf.Variable(
            self.a,
            dtype=tf.float32,
            trainable=self.trainable,
            name="a_factor")
        
    def call(self, inputs):
        res = tf.where(inputs >= self.a_factor, inputs, self.a_factor)
        return res

    def get_config(self):
        config = {
            "a": self.get_weights()[0] if self.trainable else self.a,
            "trainable": self.trainable
        }
        #base_config = super().get_config()
        return config #dict(list(base_config.items()) + list(config.items()))

class ClippingLayer(tf.keras.layers.Layer):
    def __init__(self,
                a: float = -4.5,
                sensitivity: float = 1e-1,
                min_value: float = -6.,
                max_value: float = -4.,
                trainable: bool = False,
                **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.sensitivity = sensitivity
        self.min_value = min_value
        self.max_value = max_value
        self.trainable = trainable

    def build(self, input_shape):
        super().build(input_shape)
        self.a_factor = tf.Variable(
            self.a*self.sensitivity,
            dtype=tf.float32,
            trainable=self.trainable,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=self.min_value*self.sensitivity ,max_value=self.min_value*self.sensitivity,axis=None),
            name="a_factor")

    def call(self, inputs):
        #print("Clipping values: ", self.a_factor/self.sensitivity, self.min_value, self.max_value, self.sensitivity,flush=True)
        res = tf.where(inputs > self.a_factor / self.sensitivity, inputs, self.a_factor / self.sensitivity)
        return res
"""
generator = Generator(kernel_initializer='glorot_uniform', #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
                      bias_initializer='zeros', 
                      network_model='original_variable_output_activation', inception_kwargs=inception_kwargs, vbv_shape=None)
critic = Critic(kernel_initializer='glorot_uniform', #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
                bias_initializer='zeros',
                lambda_gp=10., vbv_shape=None, network_model='original_layer_norm')
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5, beta_2=0.999)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5, beta_2=0.999)
Data_validation = DataManager(path, redshifts=[10,], IC_seeds=[1008,1009,1010])
T21, delta, vbv, T21_lr = Data_validation.data(augment=False, augments=9, low_res=True) 
T21_standardized = standardize(T21, T21_lr, subtract_mean=False)
T21_lr_standardized = standardize(T21_lr, T21_lr, subtract_mean=False)
delta_standardized = standardize(delta, delta, subtract_mean=False)

crit_loss, gp = critic.train_step_critic(generator=generator,optimizer=critic_optimizer, T21_big=T21_standardized,
                                         T21_small=T21_lr_standardized, IC_delta=delta_standardized, IC_vbv=None)
gen_loss, mse = generator.train_step_generator(critic=critic, optimizer=generator_optimizer, T21_small=T21_lr_standardized, 
                                                      T21_big=T21_standardized, IC_delta=delta_standardized, IC_vbv=None)#vbv_standardized)
"""