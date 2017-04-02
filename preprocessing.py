import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical

def getDataset():

    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")

    train = np.array(train)
    test = np.array(test)

    train_features = train[:, 1:]
    train_labels = train[:, 0]

    test_features = test

    img_width = 28
    img_height = 28
    channel = 1  # black or white

    train_features = train_features.reshape(-1, img_width, img_height, channel)
    train_features = train_features.astype("float32") / 255.  # normalization

    test_features = test_features.reshape(-1, img_width, img_height, channel)
    test_features = test_features.astype("float32") / 255.

    train_labels = to_categorical(train_labels, 10)

    return train_features,train_labels,test_features

