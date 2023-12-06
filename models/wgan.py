import tensorflow as tf

class Critic(tf.keras.Model):
    def __init__(self,
                 delta_shape=(1,128,128,128,1), vbv_shape=(1,128,128,128,1),
                 kernel_sizes=[7,5,3,1],
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                 bias_initializer=tf.keras.initializers.Constant(value=0.1),
                 lbda=1e1, activation='tanh'):
        super(Critic, self).__init__()
        #self.T21_shape = T21_shape
        self.delta_shape = delta_shape
        self.vbv_shape = vbv_shape
        self.kernel_sizes = kernel_sizes
        self.crop = int((max(self.kernel_sizes)-1))
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.lbda = lbda
        self.activation = activation
        self.build_critic_model()
        #max kernel size must be odd
        assert max(self.kernel_sizes)%2==1, "max kernel size must be odd"

    def build_critic_model(self):
        if self.vbv_shape != None:
            input = tf.keras.layers.Input(shape=(self.delta_shape[1]-self.crop*2, self.delta_shape[2]-self.crop*2, self.delta_shape[3]-self.crop*2, 3)) #not including the batch size according to docs
        else:
            input = tf.keras.layers.Input(shape=(self.delta_shape[1]-self.crop*2, self.delta_shape[2]-self.crop*2, self.delta_shape[3]-self.crop*2, 2)) #not including the batch size according to docs
        
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
        #input_, input_pos, input_neg = ResidualBlock(tf.keras.layers.Conv3D, **self.residual_block_kwargs, **kwargs)(input)
        if True:
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
        
        #conv1 = tf.keras.layers.Conv3D(filters=8, kernel_size=(self.kernel_sizes[0], self.kernel_sizes[0], self.kernel_sizes[0]), 
        #                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
        #                                    bias_initializer=tf.keras.initializers.Constant(value=0.1),
        #                                    strides=(2, 2, 2), padding='valid', data_format="channels_last", 
        #                                    activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
        #                                    )
        # 
        #conv2 = tf.keras.layers.Conv3D(filters=16, kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1], self.kernel_sizes[1]), 
        #                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
        #                                    bias_initializer=tf.keras.initializers.Constant(value=0.1),
        #                                    strides=(1, 1, 1), padding='valid', data_format="channels_last", 
        #                                    activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
        #                                    )                                    
        #conv3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(self.kernel_sizes[2], self.kernel_sizes[2], self.kernel_sizes[2]), 
        #                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
        #                                    bias_initializer=tf.keras.initializers.Constant(value=0.1),
        #                                    strides=(2, 2, 2), padding='valid', data_format="channels_last", 
        #                                    activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
        #                                    )
        
        #conv4 = tf.keras.layers.Conv3D(filters=64, kernel_size=(self.kernel_sizes[3], self.kernel_sizes[3], self.kernel_sizes[3]), 
        #                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
        #                                    bias_initializer=tf.keras.initializers.Constant(value=0.1),
        #                                    strides=(1, 1, 1), padding='valid', data_format="channels_last", 
        #                                    activation=tf.keras.layers.Activation(self.activation)#tf.keras.layers.LeakyReLU(alpha=0.1)
        #                                    )
        output = tf.keras.layers.Concatenate(axis=4)([input_, input_pos, input_neg])
        
        output = tf.keras.layers.Flatten()(output)
        #flatten = tf.keras.layers.Flatten()
        output = tf.keras.layers.Dense(units=1,
                                         kernel_initializer=self.kernel_initializer, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                         bias_initializer=self.bias_initializer, #tf.keras.initializers.Constant(value=0.1),
                                         name='output_layer')(output)

        #self.model = tf.keras.Sequential([conv1, conv2, conv3, conv4, flatten, out])    
        self.model = tf.keras.Model(inputs=[input], outputs=output)
        return self.model

    @tf.function
    def critic_loss(self, generated_boxes, T21_big, IC_delta, IC_vbv=None):
        #wasserstein loss
        # Generate a batch of fake big boxes using the generator network
        T21_big = tf.keras.layers.Cropping3D(cropping=(self.crop,self.crop,self.crop),data_format="channels_last")(T21_big)
        IC_delta = tf.keras.layers.Cropping3D(cropping=(self.crop,self.crop,self.crop),data_format="channels_last")(IC_delta)
        if IC_vbv != None:
            IC_vbv = tf.keras.layers.Cropping3D(cropping=(self.crop,self.crop,self.crop),data_format="channels_last")(IC_vbv)
        

        # Evaluate the critic network on the real big boxes and the fake big boxes
        W_real = self.forward(T21_big, IC_delta, IC_vbv)
        W_gen = self.forward(generated_boxes, IC_delta, IC_vbv)

        epsilon = tf.random.uniform(shape=[T21_big.shape[0], 1, 1, 1, 1], minval=0., maxval=1., seed=None)
        # Compute the interpolated difference between the real and generated samples
        xhat = epsilon * T21_big + (1 - epsilon) * generated_boxes

        # Compute the gradients of the critic network with respect to the interpolated difference
        with tf.GradientTape() as tape:    
            tape.watch(xhat)
            critic_output = self.forward(xhat, IC_delta, IC_vbv)
        gradients = tape.gradient(critic_output, xhat)
        l2_norm = tf.math.reduce_euclidean_norm(gradients, axis=[1,2,3])
        gp = self.lbda * tf.square(l2_norm - 1)
        
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
    def train_step_critic(self, generator, optimizer, T21_big, T21_small, IC_delta, IC_vbv):
        if (IC_vbv == None) and (self.vbv_shape != None): 
            assert False, "Critic was initialized with vbv_shape=None, but IC_vbv in train_step_critic is not None"
        #
        #Function that performs one training step for the critic network.
        #The function calls the loss function for the critic network, computes the gradients,
        #and applies the gradients to the network's parameters.
        #

        with tf.GradientTape() as disc_tape:
            generated_boxes = generator.forward(T21_small, IC_delta, IC_vbv)
            crit_loss, gp = self.critic_loss(generated_boxes, T21_big, IC_delta, IC_vbv)

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
        self.inception_kwargs["input_channels"] = inputs_T21.shape[-1]
        T21_, T21_pos, T21_neg = ResidualBlock(InceptionLayer, **self.residual_block_kwargs, **self.inception_kwargs)(T21)
        if False:
            T21_ = InceptionLayer(input_channels=inputs_T21.shape[-1], 
                                filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4, 
                                filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                activation=self.activation)(T21)
            T21_pos = InceptionLayer(input_channels=inputs_T21.shape[-1], 
                                filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4, 
                                filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1))(T21)
            T21_neg = -InceptionLayer(input_channels=inputs_T21.shape[-1],
                                    filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4,
                                    filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1))(-T21)
            
        self.inception_kwargs["input_channels"] = inputs_delta.shape[-1]
        delta_, delta_pos, delta_neg = ResidualBlock(InceptionLayer, **self.residual_block_kwargs, **self.inception_kwargs)(inputs_delta)
        if False:
            delta_ = InceptionLayer(input_channels=inputs_delta.shape[-1],
                                filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4, 
                                filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4, 
                                activation=self.activation)(inputs_delta)
            delta_pos = InceptionLayer(input_channels=inputs_delta.shape[-1],
                                    filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4,
                                    filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1))(inputs_delta)
            delta_neg = -InceptionLayer(input_channels=inputs_delta.shape[-1],
                                    filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4,
                                    filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1))(-inputs_delta)
        if self.vbv_shape != None:
            self.inception_kwargs["input_channels"] = inputs_vbv.shape[-1]
            vbv_, vbv_pos, vbv_neg = ResidualBlock(InceptionLayer, **self.residual_block_kwargs, **self.inception_kwargs)(inputs_vbv)
            if False:
                vbv_ = InceptionLayer(input_channels=inputs_vbv.shape[-1], 
                                    filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4, 
                                    filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4, 
                                    activation=self.activation)(inputs_vbv) #tf.keras.layers.Lambda(self.inception__)(inputs_vbv)
                vbv_pos = InceptionLayer(input_channels=inputs_vbv.shape[-1],
                                        filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4,
                                        filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                        activation=tf.keras.layers.LeakyReLU(alpha=0.1))(inputs_vbv)
                vbv_neg = -InceptionLayer(input_channels=inputs_vbv.shape[-1],
                                        filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4,
                                        filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                        activation=tf.keras.layers.LeakyReLU(alpha=0.1))(-inputs_vbv)

            data = tf.keras.layers.Concatenate(axis=4)([T21_, T21_pos, T21_neg, delta_, delta_pos, delta_neg, vbv_, vbv_pos, vbv_neg])
        else:
            data = tf.keras.layers.Concatenate(axis=4)([T21_, T21_pos, T21_neg, delta_, delta_pos, delta_neg])
        self.inception_kwargs["input_channels"] = data.shape[-1]
        data_, data_pos, data_neg = ResidualBlock(InceptionLayer, **self.residual_block_kwargs, **self.inception_kwargs)(data)
        if False:
            data_ = InceptionLayer(input_channels=data.shape[-1], 
                                filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4, 
                                filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4, 
                                activation=self.activation)(data) 
            data_pos = InceptionLayer(input_channels=data.shape[-1],
                                    filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4,
                                    filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1))(data)
            data_neg = -InceptionLayer(input_channels=data.shape[-1],
                                    filters_1x1x1_7x7x7=4, filters_7x7x7=4, filters_1x1x1_5x5x5=4,
                                    filters_5x5x5=4, filters_1x1x1_3x3x3=4, filters_3x3x3=4, filters_1x1x1=4,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1))(-data)
        data = tf.keras.layers.Concatenate(axis=4)([data_, data_pos, data_neg])

        data = tf.keras.layers.Conv3D(filters=1,
                          kernel_size=(1, 1, 1),
                          kernel_initializer=self.kernel_initializer,
                          bias_initializer=self.bias_initializer,
                          strides=(1, 1, 1), padding='valid', data_format="channels_last",
                          activation=None)(data)
        
        PeLU_activation = PeLU(a=10.0, b=10.0, c=1.0, trainable=True)
        data = PeLU_activation(data)
        
        #alpha = tf.Variable(initial_value=2, trainable=True, dtype=tf.float32, name='alpha_elu_hyperparam')# added 3/12
        #data = tf.keras.layers.ELU(alpha=alpha.numpy())(data)  # added 3/12
        #data = tf.keras.layers.LeakyReLU(alpha=0.1)(data) ##added 3/12
        #data = tf.keras.layers.Conv3D(filters=1,#added 2/12
        #                              kernel_size=(1, 1, 1),
        #                              kernel_initializer=self.kernel_initializer,
        #                              #bias_initializer=self.bias_initializer,
        #                              use_bias=False,
        #                              strides=(1, 1, 1), padding='valid', data_format="channels_last",
        #                              activation=None,
        #                              )(data)

        #add a trainable constant with a labda layer
        #constant1 = tf.Variable(initial_value=1.5, trainable=True, dtype=tf.float32)
        #data = tf.keras.layers.Lambda(lambda x: x + constant1)(data)
        #data = tf.keras.layers.LeakyReLU(alpha=0.1)(data)
        #constant2 = tf.Variable(initial_value=-1.5, trainable=True, dtype=tf.float32)
        #data = tf.keras.layers.Lambda(lambda x: x + constant2)(data)
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

        loss = - tf.reduce_mean(W_gen - W_real) #- tf.reduce_mean(- W_gen)
        #loss = tf.reduce_mean(tf.math.reduce_euclidean_norm(T21_big-generated_boxes, axis=[1,2,3])) #l2 norm loss
        return loss

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
            gen_loss = self.generator_loss(critic, generated_boxes, T21_big, IC_delta, IC_vbv)

        grad_gen = gen_tape.gradient(gen_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grad_gen, self.model.trainable_variables))

        return gen_loss
        
    @tf.function    
    def forward(self, T21_train, IC_delta, IC_vbv=None):
        if IC_vbv != None:
            return self.model(inputs=[T21_train, IC_delta, IC_vbv])
        else:
            return self.model(inputs=[T21_train, IC_delta])
    

class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, input_channels=1, filters_1x1x1_7x7x7=6, filters_7x7x7=6, filters_1x1x1_5x5x5=6, filters_5x5x5=6, filters_1x1x1_3x3x3=6, filters_3x3x3=6, filters_1x1x1=6,
                 activation=tf.keras.layers.Activation('tanh'),
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                 bias_initializer=tf.keras.initializers.Constant(value=0.1)
                 ):
        super(InceptionLayer, self).__init__()
        self.input_channels = input_channels
        self.conv_1x1x1_7x7x7 = tf.keras.layers.Conv3D(filters=filters_1x1x1_7x7x7, kernel_size=(1, 1, 1),
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                       activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                       )
        self.conv_7x7x7 = tf.keras.layers.Conv3D(filters=filters_7x7x7, kernel_size=(7, 7, 7),
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        self.conv_1x1x1_5x5x5 = tf.keras.layers.Conv3D(filters=filters_1x1x1_5x5x5, kernel_size=(1, 1, 1),
                                                            kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer,
                                                            strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                            activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                            )
        self.conv_5x5x5 = tf.keras.layers.Conv3D(filters=filters_5x5x5, kernel_size=(5, 5, 5),
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        self.crop_5x5x5 = tf.keras.layers.Cropping3D(cropping=(1, 1, 1),data_format="channels_last")

        self.conv_1x1x1_3x3x3 = tf.keras.layers.Conv3D(filters=filters_1x1x1_3x3x3, kernel_size=(1, 1, 1),
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                       activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                       )
        self.conv_3x3x3 = tf.keras.layers.Conv3D(filters=filters_3x3x3, kernel_size=(3, 3, 3),
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        self.crop_3x3x3 = tf.keras.layers.Cropping3D(cropping=(2, 2, 2),data_format="channels_last")
        
        self.conv_1x1x1 = tf.keras.layers.Conv3D(filters=filters_1x1x1, kernel_size=(1, 1, 1),
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        
        self.crop_1x1x1 = tf.keras.layers.Cropping3D(cropping=(3, 3, 3),data_format="channels_last")
        self.concat = tf.keras.layers.Concatenate(axis=4)#([x1, x2, x3, x4])
        
        self.crop_x = tf.keras.layers.Cropping3D(cropping=(3, 3, 3),data_format="channels_last")#(x[:,:,:,:,0:1])
        #self.tile_x = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 1, x_out.shape[-1]]))(x_out_)

        self.conv_1x1x1_reduce_channels = tf.keras.layers.Conv3D(filters=filters_7x7x7+filters_5x5x5+filters_3x3x3+filters_1x1x1, kernel_size=(1, 1, 1), #warning: number of filters not generalized
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer,
                                                 strides=(1, 1, 1), padding='valid', data_format="channels_last",
                                                 activation=None #tf.keras.layers.LeakyReLU(alpha=0.1)
                                                 )
        self.activation_layer = activation

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
        if self.input_channels > x_out.shape[-1]:
            x_out_ = self.conv_1x1x1_reduce_channels(x_out_)
        else:
            x_out_ = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 1, x_out.shape[-1]]))(x_out_)

        x_out = tf.add(x_out, x_out_)
        x_out = self.activation_layer(x_out)
        return x_out

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, layer,
                 activation=[tf.keras.layers.LeakyReLU(alpha=0.1), 
                             tf.keras.layers.Activation('tanh'),
                             tf.keras.layers.LeakyReLU(alpha=0.1)],
                **kwargs):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.activation = activation
        self.kwargs = kwargs  # Save kwargs in an instance variable
        self.build()
    
    def build(self):
        self.x1 = self.layer(activation=self.activation[0], **self.kwargs)
        self.x2 = self.layer(activation=self.activation[1], **self.kwargs)
        self.x3 = self.layer(activation=self.activation[2], **self.kwargs)
        
    def __call__(self, input):
        x1 = self.x1(input)
        x2 = self.x2(input)
        x3 = -self.x3(-input)
        return x1,x2,x3

class PeLU(tf.keras.layers.Layer):
    """``PeLU``."""
    def __init__(self,
                a: float = 1.,
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
            trainable=self.trainable,
            name="a_factor")

        self.b_factor = tf.Variable(
            self.b,
            dtype=tf.float32,
            name="b_factor")

        self.c_factor = tf.Variable(
            self.c,
            dtype=tf.float32,
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
#test generator and critic on noise
inception_kwargs = {
            #'input_channels': self.T21_shape[-1],
            'filters_1x1x1_7x7x7': 4,
            'filters_7x7x7': 4,
            'filters_1x1x1_5x5x5': 4,
            'filters_5x5x5': 4,
            'filters_1x1x1_3x3x3': 4,
            'filters_3x3x3': 4,
            'filters_1x1x1': 4,
            'kernel_initializer': 'glorot_uniform',#tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), #
            'bias_initializer': 'zeros',#tf.keras.initializers.Constant(value=0.1), #
            #'activation': [tf.keras.layers.LeakyReLU(alpha=0.1), tf.keras.layers.Activation('tanh'), tf.keras.layers.LeakyReLU(alpha=0.1)]
            }

#T21_lr = tf.random.normal(shape=(3,64,64,64,1), mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32, name=None)
#IC_delta = tf.random.normal(shape=(3,128,128,128,1), mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32, name=None)   
#IC_vbv = tf.random.normal(shape=(3,128,128,128,1), mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32, name=None) 
#generator = Generator(T21_shape=T21_lr.shape, delta_shape=IC_delta.shape, vbv_shape=None,#IC_vbv.shape, 
#                      inception_kwargs=inception_kwargs)
#generated_boxes = generator.forward(T21_lr, IC_delta, )

#tf.keras.utils.plot_model(generator.model, 
#                          to_file='generator_model_3.png', 
#                          show_shapes=True, show_layer_names=True, 
#                          show_layer_activations=True, expand_nested=False,
#                          show_trainable=True)

#T21_target = tf.random.normal(shape=(3,116,116,116,1), mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32, name=None)
#IC_delta = tf.random.normal(shape=(3,116,116,116,1), mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32, name=None)   
#IC_vbv = None #tf.random.normal(shape=(3,116,116,116,1), mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32, name=None) 
#critic = Critic(vbv_shape=None)
#W_real = critic.forward(T21_target, IC_delta, IC_vbv)
#W_gen = critic.forward(generated_boxes, IC_delta, IC_vbv)
"""
import matplotlib.pyplot as plt
import numpy as np
def standardize(data, data_stats):
    mean, var = tf.nn.moments(data_stats, axes=[1,2,3], keepdims=True) #mean across xyz with shape=(batch,x,y,z,channels)
    mean = mean.numpy()
    var = var.numpy()
    for i,(m,v) in enumerate(zip(mean,var)):
        if m==0 and v==0:
            mean[i] = 0
            var[i] = 1
            #print("mean and var both zero for i={0} j={1}, setting mean to {2} and var to {3}".format(i,np.nan,mean[i],var[i]))
    std = var**0.5
    return (data - mean) / std

fig, axes = plt.subplots(1,4,figsize=(20,5))

T21_target_standardized = standardize(T21_target, T21_target)
T21_std = np.std(T21_target_standardized[0,:,:,:,:].numpy().flatten())
axes[0].imshow(T21_target_standardized[0,:,:,T21_target_standardized.shape[3]//2,0].numpy().squeeze(), vmin=-3*T21_std, vmax=3*T21_std)

#pelu activation
act = PeLU(a=10.0, b=10.0, c=1.0, trainable=False)
axes[1].imshow(act(T21_target_standardized)[0,:,:,T21_target_standardized.shape[3]//2,0].numpy().squeeze(), vmin=-3*T21_std, vmax=3*T21_std)

axes[2].hist(T21_target_standardized[0,:,:,:,:].numpy().flatten(), density=True, bins=100, alpha=0.5, label="real")
axes[3].hist(act(T21_target_standardized)[0,:,:,:,:].numpy().flatten(), density=True, bins=100, alpha=0.5, label="fake")
plt.savefig("pelu_activation3.png")
"""
#print("a b and c pelu parameters: ", act.get_config())