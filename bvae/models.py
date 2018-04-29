'''
models.py
contains models for use with the BVAE experiments.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D, Dense,
            Reshape, GlobalAveragePooling2D)
from tensorflow.python.keras.models import Model

from model_utils import ResConvBnLRelu, ConvBnLRelu, TransConvBnLRelu, SampleLayer, FullyConnectedRelu
import math

class Architecture(object):
    '''
    generic architecture template
    '''
    def __init__(self, inputShape=None, batchSize=None, latentSize=None):
        '''
        params:
        ---------
        inputShape : tuple
            the shape of the input, expecting 3-dim images (h, w, 3)
        batchSize : int
            the number of samples in a batch
        latentSize : int
            the number of dimensions in the two output distribution vectors -
            mean and std-deviation
        '''
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.latentSize = latentSize

        self.model = self.Build()

    def Build(self):
        raise NotImplementedError('architecture must implement Build function')

class FCEncoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=1,
                 latentSize=1000, latentConstraints='bvae', beta=100., capacity=0.,
                 randomSample=True):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use 
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.latentCapacity = capacity
        self.randomSample = randomSample
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        
        net = Reshape((-1,))(inLayer)
        net = FullyConnectedRelu(1200*3)(net)
        net = FullyConnectedRelu(1200*3)(net)
        net = FullyConnectedRelu(1200*3)(net)

        # variational encoder output (distributions)
        mean = FullyConnectedRelu(self.latentSize, linear=True)(net)
        # mean = Reshape((self.latentSize,))(mean)
        stddev = FullyConnectedRelu(self.latentSize, linear=True)(net)
        # stddev = Reshape((self.latentSize,))(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                            self.latentCapacity, self.randomSample)([mean, stddev])

        return Model(inputs=inLayer, outputs=sample)

class FCDecoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=1, latentSize=1000):
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((self.latentSize,))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        # net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        net = FullyConnectedRelu(1200*3)(net)
        net = FullyConnectedRelu(1200*3)(net)
        net = FullyConnectedRelu(1200*3)(net)
        
        # net = ConvBnLRelu(3, kernelSize=1)(net)
        net = FullyConnectedRelu(self.inputShape[0]*self.inputShape[1]*self.inputShape[2], linear=True)(net)
        net = Reshape(self.inputShape)(net)

        return Model(inLayer, net)

class BetaEncoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=1,
                 latentSize=1000, latentConstraints='bvae', beta=100., capacity=0.,
                 randomSample=True):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use 
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.latentCapacity = capacity
        self.randomSample = randomSample
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        net = ConvBnLRelu(32, kernelSize=3, strides=2)(inLayer)
        for i in range(int(math.log(self.inputShape[0],2)) - 1):
            net = ConvBnLRelu(32 * math.pow(2, i), kernelSize=3, strides=2)(net)
        
        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        stddev = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        stddev = GlobalAveragePooling2D()(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                            self.latentCapacity, self.randomSample)([mean, stddev])

        return Model(inputs=inLayer, outputs=sample)

class BetaDecoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=1, latentSize=1000):
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        # net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)

        power = int(math.log(self.inputShape[0],2)-1)
        for i in range(power):
            net = TransConvBnLRelu(32 * int(math.pow(2, power-i-1)), kernelSize=3, strides=2)(net)
        net = TransConvBnLRelu(32, kernelSize=3, strides=2)(net)
        # net = ConvBnLRelu(3, kernelSize=1)(net)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same')(net)

        return Model(inLayer, net)

class Darknet19Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=1,
                 latentSize=1000, latentConstraints='bvae', beta=100., capacity=0.,
                 randomSample=True):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use 
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.latentCapacity = capacity
        self.randomSample = randomSample
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        net = ConvBnLRelu(32, kernelSize=3)(inLayer) # 1
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(64, kernelSize=3)(net) # 2
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(128, kernelSize=3)(net) # 3
        net = ConvBnLRelu(64, kernelSize=1)(net) # 4
        net = ConvBnLRelu(128, kernelSize=3)(net) # 5
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(256, kernelSize=3)(net) # 6
        net = ConvBnLRelu(128, kernelSize=1)(net) # 7
        net = ConvBnLRelu(256, kernelSize=3)(net) # 8
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(512, kernelSize=3)(net) # 9
        net = ConvBnLRelu(256, kernelSize=1)(net) # 10
        net = ConvBnLRelu(512, kernelSize=3)(net) # 11
        net = ConvBnLRelu(256, kernelSize=1)(net) # 12
        net = ConvBnLRelu(512, kernelSize=3)(net) # 13
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ConvBnLRelu(1024, kernelSize=3)(net) # 14
        net = ConvBnLRelu(512, kernelSize=1)(net) # 15
        net = ConvBnLRelu(1024, kernelSize=3)(net) # 16
        net = ConvBnLRelu(512, kernelSize=1)(net) # 17
        net = ConvBnLRelu(1024, kernelSize=3)(net) # 18

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        stddev = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        stddev = GlobalAveragePooling2D()(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                            self.latentCapacity, self.randomSample)([mean, stddev])

        return Model(inputs=inLayer, outputs=sample)

class Darknet19Decoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=1, latentSize=1000):
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        # TODO try inverting num filter arangement (e.g. 512, 1204, 512, 1024, 512)
        # and also try (1, 3, 1, 3, 1) for the filter shape
        net = ConvBnLRelu(1024, kernelSize=3)(net)
        net = ConvBnLRelu(512, kernelSize=1)(net)
        net = ConvBnLRelu(1024, kernelSize=3)(net)
        net = ConvBnLRelu(512, kernelSize=1)(net)
        net = ConvBnLRelu(1024, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net)
        net = ConvBnLRelu(128, kernelSize=1)(net)
        net = ConvBnLRelu(256, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net)
        net = ConvBnLRelu(64, kernelSize=1)(net)
        net = ConvBnLRelu(128, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=1)(net)
        net = ConvBnLRelu(32, kernelSize=3)(net)
        # net = ConvBnLRelu(3, kernelSize=1)(net)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same')(net)

        return Model(inLayer, net)

class ResDarknet19Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=1,
                 latentSize=1000, latentConstraints='bvae', beta=100., capacity=0.,
                 randomSample=True):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use 
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.latentCapacity = capacity
        self.randomSample = randomSample
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        net = ResConvBnLRelu(32, kernelSize=3)(inLayer) # 1
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ResConvBnLRelu(64, kernelSize=3)(net) # 2
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ResConvBnLRelu(128, kernelSize=3)(net) # 3
        net = ResConvBnLRelu(64, kernelSize=1)(net) # 4
        net = ResConvBnLRelu(128, kernelSize=3)(net) # 5
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ResConvBnLRelu(256, kernelSize=3)(net) # 6
        net = ResConvBnLRelu(128, kernelSize=1)(net) # 7
        net = ResConvBnLRelu(256, kernelSize=3)(net) # 8
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ResConvBnLRelu(512, kernelSize=3)(net) # 9
        net = ResConvBnLRelu(256, kernelSize=1)(net) # 10
        net = ResConvBnLRelu(512, kernelSize=3)(net) # 11
        net = ResConvBnLRelu(256, kernelSize=1)(net) # 12
        net = ResConvBnLRelu(512, kernelSize=3)(net) # 13
        net = MaxPool2D((2, 2), strides=(2, 2))(net)

        net = ResConvBnLRelu(1024, kernelSize=3)(net) # 14
        net = ResConvBnLRelu(512, kernelSize=1)(net) # 15
        net = ResConvBnLRelu(1024, kernelSize=3)(net) # 16
        net = ResConvBnLRelu(512, kernelSize=1)(net) # 17
        net = ResConvBnLRelu(1024, kernelSize=3)(net) # 18

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        stddev = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        stddev = GlobalAveragePooling2D()(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                            self.latentCapacity, self.randomSample)([mean, stddev])

        return Model(inputs=inLayer, outputs=sample)

class ResDarknet19Decoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=1, latentSize=1000):
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        # TODO try inverting num filter arangement (e.g. 512, 1204, 512, 1024, 512)
        # and also try (1, 3, 1, 3, 1) for the filter shape
        net = ResConvBnLRelu(1024, kernelSize=3)(net)
        net = ResConvBnLRelu(512, kernelSize=1)(net)
        net = ResConvBnLRelu(1024, kernelSize=3)(net)
        net = ResConvBnLRelu(512, kernelSize=1)(net)
        net = ResConvBnLRelu(1024, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ResConvBnLRelu(512, kernelSize=3)(net)
        net = ResConvBnLRelu(256, kernelSize=1)(net)
        net = ResConvBnLRelu(512, kernelSize=3)(net)
        net = ResConvBnLRelu(256, kernelSize=1)(net)
        net = ResConvBnLRelu(512, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ResConvBnLRelu(256, kernelSize=3)(net)
        net = ResConvBnLRelu(128, kernelSize=1)(net)
        net = ResConvBnLRelu(256, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ResConvBnLRelu(128, kernelSize=3)(net)
        net = ResConvBnLRelu(64, kernelSize=1)(net)
        net = ResConvBnLRelu(128, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ResConvBnLRelu(64, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ResConvBnLRelu(64, kernelSize=1)(net)
        net = ResConvBnLRelu(32, kernelSize=3)(net)
        # net = ResConvBnLRelu(3, kernelSize=1)(net)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same')(net)

        return Model(inLayer, net)

class StrideDarknet19Encoder(Architecture):
    '''
    This encoder predicts distributions then randomly samples them.
    Regularization may be applied to the latent space output

    a simple, fully convolutional architecture inspried by 
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    '''
    def __init__(self, inputShape=(256, 256, 3), batchSize=1,
                 latentSize=1000, latentConstraints='bvae', beta=100., capacity=0.,
                 randomSample=True):
        '''
        params
        -------
        latentConstraints : str
            Either 'bvae', 'vae', or 'no'
            Determines whether regularization is applied
                to the latent space representation.
        beta : float
            beta > 1, used for 'bvae' latent_regularizer
            (Unused if 'bvae' not selected, default 100)
        capacity : float
            used for 'bvae' to try to break input down to a set number
                of basis. (e.g. at 25, the network will try to use 
                25 dimensions of the latent space)
            (unused if 'bvae' not selected)
        randomSample : bool
            whether or not to use random sampling when selecting from distribution.
            if false, the latent vector equals the mean, essentially turning this into a
                standard autoencoder.
        '''
        self.latentConstraints = latentConstraints
        self.beta = beta
        self.latentCapacity = capacity
        self.randomSample = randomSample
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # create the input layer for feeding the netowrk
        inLayer = Input(self.inputShape, self.batchSize)
        net = ConvBnLRelu(32, kernelSize=3, strides=2)(inLayer) # 1

        net = ConvBnLRelu(64, kernelSize=3, strides=2)(net) # 2

        net = ConvBnLRelu(128, kernelSize=3)(net) # 3
        net = ConvBnLRelu(64, kernelSize=1)(net) # 4
        net = ConvBnLRelu(128, kernelSize=3, strides=2)(net) # 5

        net = ConvBnLRelu(256, kernelSize=3)(net) # 6
        net = ConvBnLRelu(128, kernelSize=1)(net) # 7
        net = ConvBnLRelu(256, kernelSize=3, strides=2)(net) # 8

        net = ConvBnLRelu(512, kernelSize=3)(net) # 9
        net = ConvBnLRelu(256, kernelSize=1)(net) # 10
        net = ConvBnLRelu(512, kernelSize=3)(net) # 11
        net = ConvBnLRelu(256, kernelSize=1)(net) # 12
        net = ConvBnLRelu(512, kernelSize=3, strides=2)(net) # 13

        net = ConvBnLRelu(1024, kernelSize=3)(net) # 14
        net = ConvBnLRelu(512, kernelSize=1)(net) # 15
        net = ConvBnLRelu(1024, kernelSize=3)(net) # 16
        net = ConvBnLRelu(512, kernelSize=1)(net) # 17
        net = ConvBnLRelu(1024, kernelSize=3)(net) # 18

        # variational encoder output (distributions)
        mean = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                      padding='same')(net)
        mean = GlobalAveragePooling2D()(mean)
        stddev = Conv2D(filters=self.latentSize, kernel_size=(1, 1),
                        padding='same')(net)
        stddev = GlobalAveragePooling2D()(stddev)

        sample = SampleLayer(self.latentConstraints, self.beta,
                            self.latentCapacity, self.randomSample)([mean, stddev])

        return Model(inputs=inLayer, outputs=sample)

class StrideDarknet19Decoder(Architecture):
    def __init__(self, inputShape=(256, 256, 3), batchSize=1, latentSize=1000):
        super().__init__(inputShape, batchSize, latentSize)

    def Build(self):
        # input layer is from GlobalAveragePooling:
        inLayer = Input([self.latentSize], self.batchSize)
        # reexpand the input from flat:
        net = Reshape((1, 1, self.latentSize))(inLayer)
        # darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
        net = UpSampling2D((self.inputShape[0]//32, self.inputShape[1]//32))(net)

        # TODO try inverting num filter arangement (e.g. 512, 1204, 512, 1024, 512)
        # and also try (1, 3, 1, 3, 1) for the filter shape
        net = ConvBnLRelu(1024, kernelSize=3)(net)
        net = ConvBnLRelu(512, kernelSize=1)(net)
        net = ConvBnLRelu(1024, kernelSize=3)(net)
        net = ConvBnLRelu(512, kernelSize=1)(net)
        net = ConvBnLRelu(1024, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)
        net = ConvBnLRelu(256, kernelSize=1)(net)
        net = ConvBnLRelu(512, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(256, kernelSize=3)(net)
        net = ConvBnLRelu(128, kernelSize=1)(net)
        net = ConvBnLRelu(256, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(128, kernelSize=3)(net)
        net = ConvBnLRelu(64, kernelSize=1)(net)
        net = ConvBnLRelu(128, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=3)(net)

        net = UpSampling2D((2, 2))(net)
        net = ConvBnLRelu(64, kernelSize=1)(net)
        net = ConvBnLRelu(32, kernelSize=3)(net)
        # net = ConvBnLRelu(3, kernelSize=1)(net)
        net = Conv2D(filters=self.inputShape[-1], kernel_size=(1, 1),
                      padding='same')(net)

        return Model(inLayer, net)

class Darknet53Encoder(Architecture):
    '''
    a larger, fully convolutional architecture inspried by
        pjreddie's darknet architecture
    https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    https://pjreddie.com/media/files/papers/YOLOv3.pdf
    '''
    def __init__(self, inputShape=(None, None, None, None), name='darkent53_encoder'):
        '''
        input shape for the network, a name for the scope, and a data format.
        '''
        super().__init__(inputShape, name)
        self.Build()

    def Build(self):
        '''
        builds darknet53 encoder network
        '''
        raise NotImplementedError('this architecture is not complete')

    def ConvBlock(self):
        '''
        adds a darknet conv block to the net
        '''
        raise NotImplementedError('this architecture is not complete')


def test():
    model_shape = (32, 32, 3)
    batchSize = 8
    latentSize = 16
    d19e = BetaEncoder(model_shape, batchSize, latentSize, None)
    d19e.model.summary()
    d19d = BetaDecoder(model_shape, batchSize, latentSize)
    d19d.model.summary()

if __name__ == '__main__':
    test()
