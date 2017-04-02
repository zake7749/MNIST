import numpy

from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model,Sequential

from DeepAutoEncoder import DeepAutoEncoder
from CNNAutoEncoder import *
from preprocessing import *

def main():

    # feature shape: (SAMPLE_NUM, IMG_WIDTH, IMG_HEIGHT)
    # label   shape: (SAMPLE_NUM. 10)
    train_features, train_labels, test_features = getDataset()

    print("Training the autoencoder.")
    autoencoder, encoder = CNNAutoEncoder(train_features)

    encoded_trian_features = encoder.predict(train_features)
    encoded_test_features = encoder.predict(test_features)


    model = Sequential()

    cl1_nb_filter = 100
    cl1_nb_size = 2
    drop_prob = 0.5

    # Convolution only

    model.add(Convolution2D(cl1_nb_filter, cl1_nb_size, cl1_nb_size, input_shape=(4, 4, 8),
                            activation='relu'))
    model.add(Flatten())

    # Dense layer

    model.add(Dense(output_dim=512, activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(output_dim=256, activation='relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(output_dim=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    for i in range(35):
        model.fit(
            encoded_trian_features,
            train_labels,
            verbose=2,
            batch_size=128,
            nb_epoch=10,
            validation_split=0.15
        )

    test_labels = model.predict_classes(encoded_test_features)

    result_output = pd.DataFrame({
                     "ImageId" : list( range(1,len(test_labels)+1) ),
                     "Label" : test_labels}
                    )
    result_output.to_csv("MNIST.csv", index=False, header=True)

if __name__ == "__main__":
    main()
