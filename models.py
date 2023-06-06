from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten, Permute
from tensorflow.keras.constraints import max_norm
import tensorflow.keras.backend as K


def CNN_1D_model(num_classes=4, chans=16, samples=60):
    input_shape = (samples, chans)
    model = Sequential(
        layers=[
            Conv1D(16, (3), activation='relu', input_shape=input_shape),
            MaxPooling1D((3)),
            Flatten(),
            Dense(num_classes, activation="softmax")
        ]
    )
    return model


def small_CNN_2D_model(num_classes=4, chans=16, samples=60):
    input_shape = (1, samples, chans)
    model = Sequential(
        layers=[
            Conv2D(16, (1, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((1, 3)),
            Flatten(),
            Dense(num_classes, activation="softmax")
        ]
    )
    return model


def CNN_2D_model(num_classes=4, chans=16, samples=60):
    input_shape = (1, samples, chans)
    model = Sequential(
        layers=[
            Conv2D(16, (5, 5), padding="same", activation='relu', input_shape=input_shape),
            MaxPooling2D((5, 5), padding="same"),
            Conv2D(32, (5, 5), padding="same", activation='relu'),
            MaxPooling2D((5, 5), padding="same"),
            Conv2D(128, (5, 5), padding="same", activation='relu'),
            MaxPooling2D((5, 5), padding="same"),
            Flatten(),
            Dense(128),
            Dense(64),
            Dense(num_classes, activation="softmax")
        ]
    )
    return model


# Source: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
def EEGNet(num_classes=4, chans=16, samples=60,
           dropout_rate=0.5, kernel_length=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropout_type='Dropout'):
    if dropout_type == 'SpatialDropout2D':
        dropout_type = SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    # Original: (Chans, Samples, 1)
    input_shape = (chans, samples, 1)
    input1 = Input(shape=input_shape)

    ##################################################################
    block1 = Conv2D(F1, (1, kernel_length), padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropout_type(dropout_rate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropout_type(dropout_rate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(num_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def DeepConvNet(num_classes=4, chans=16, samples=60,
                dropout_rate=0.5):

    input_shape = (chans, samples, 1)
    input_main = Input(input_shape)
    block1 = Conv2D(25, (1, 5),
                    input_shape=(chans, samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropout_rate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropout_rate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropout_rate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropout_rate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(num_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# Source: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
def ShallowConvNet(num_classes=4, chans=16, samples=60, dropout_rate=0.5):
    # (Chans, Samples, 1)
    input_shape = (chans, samples, 1)
    input_main = Input(input_shape)
    block1 = Conv2D(40, (1, 13),
                    input_shape=(chans, samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropout_rate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(num_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))
