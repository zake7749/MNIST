from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model


class DeepAutoEncoder(object):

    def __init__(self, input_size):
        self.input_size = input_size

        self.autoencoder = None
        self.encoder = None


    def build(self):

        input_img = Input(shape=(self.input_size,))

        # Encoder
        encoder = Dense(1000, activation='relu')(input_img)
        encoder.add(Dense(500, activation='relu'))
        encoder.add(Dense(250, activation='relu'))
        encoder.add(Dense(30, activation='relu'))

        # Decoder
        decoder = Dense(250, activation='relu')(encoder)
        decoder = Dense(500, activation='relu')(decoder)
        decoder = Dense(1000, activation='relu')(decoder)
        decoder = Dense(784, activation='sigmoid')(decoder)

        self.autoencoder = Model(input=input_img, output=decoder)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.encoder = Model(input=input_img, output=encoder)
        self.encoder.compile(optimizer='adam', loss='binary_crossentropy')
