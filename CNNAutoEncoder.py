from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from os.path import isfile

def CNNAutoEncoder(features):

    if isfile("CNNAutoEncoder.h5") and isfile("CNNEncoder.h5"):
        autoencoder = load_model("CNNAutoEncoder.h5")
        encoder = load_model("CNNEncoder.h5")
    else:
        input_img = Input(shape=(28, 28, 1))

        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3 ,3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        autoencoder.fit(features, features,
                        nb_epoch=50,
                        batch_size=128,
                        shuffle=True)

        autoencoder.save("CNNAutoEncoder.h5")
        encoder.save("CNNEncoder.h5")

    return autoencoder, encoder
