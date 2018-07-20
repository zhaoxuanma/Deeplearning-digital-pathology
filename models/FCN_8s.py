from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Activation, Add, UpSampling2D, Dropout
from keras.models import Model
from keras import regularizers
from keras.initializers import Constant, glorot_normal

# global constants
IMG_SIZE = 256
DROPOUT = 0.0
data_format = 'channels_last'

def conv_layer(x, nb_filter, nb_kernel,
               strides=(1, 1), activation='relu',
               border_mode='same', weight_decay=False):

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Conv2D(filters=nb_filter, kernel_size=(nb_kernel, nb_kernel),
               strides=strides, padding=border_mode,kernel_initializer=glorot_normal(), bias_initializer=Constant(0.2),
               activation=activation, kernel_regularizer=W_regularizer,
               bias_regularizer=b_regularizer)(x)

    x = Dropout(rate=DROPOUT)(x)


    return x


def create_model(NB_CLASS):

    # Define image input layer
    if data_format == 'channels_first':
        INP_SHAPE = (3, IMG_SIZE, IMG_SIZE)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
    elif data_format == 'channels_last':
        INP_SHAPE = (IMG_SIZE, IMG_SIZE, 3)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
    else:
        raise Exception('Invalid dim ordering: ' + str(data_format))

    conv1_1 = conv_layer(img_input, nb_kernel=3, nb_filter=64, strides=(1,1))
    conv1_2 = conv_layer(conv1_1, nb_kernel=3, nb_filter=64, strides=(1,1))

    pool_1 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv1_2)

    conv2_1 = conv_layer(pool_1, nb_kernel=3, nb_filter=128, strides=(1, 1))
    conv2_2 = conv_layer(conv2_1, nb_kernel=3, nb_filter=128, strides=(1, 1))

    pool_2 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv2_2)


    conv3_1 = conv_layer(pool_2, nb_kernel=3, nb_filter=256, strides=(1, 1))
    conv3_2 = conv_layer(conv3_1, nb_kernel=3, nb_filter=256, strides=(1, 1))
    conv3_3 = conv_layer(conv3_2, nb_kernel=3, nb_filter=256, strides=(1, 1))

    pool_3 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv3_3)
    score_pool3 = conv_layer(pool_3, nb_kernel=1, nb_filter=21, strides=(1, 1))


    conv4_1 = conv_layer(pool_3, nb_kernel=3, nb_filter=512, strides=(1, 1))
    conv4_2 = conv_layer(conv4_1, nb_kernel=3, nb_filter=512, strides=(1, 1))
    conv4_3 = conv_layer(conv4_2, nb_kernel=3, nb_filter=512, strides=(1, 1))

    pool_4 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv4_3)
    score_pool4 = conv_layer(pool_4, nb_kernel=1, nb_filter=21, strides=(1, 1))


    conv5_1 = conv_layer(pool_4, nb_kernel=3, nb_filter=512, strides=(1, 1))
    conv5_2 = conv_layer(conv5_1, nb_kernel=3, nb_filter=512, strides=(1, 1))
    conv5_3 = conv_layer(conv5_2, nb_kernel=3, nb_filter=512, strides=(1, 1))

    pool_5 = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv5_3)

    fc6 = conv_layer(pool_5, nb_kernel=7, nb_filter=4096, strides=(1, 1))
    fc7 = conv_layer(fc6, nb_kernel=1, nb_filter=4096, strides=(1, 1))

    score_fr = conv_layer(fc7, nb_kernel=1, nb_filter=21, strides=(1, 1))

    upscore2 = UpSampling2D(size=(2, 2))(score_fr)

    fuse_pool4 = Add()([score_pool4,upscore2])

    upscore2_pool4 = UpSampling2D(size=(2, 2))(fuse_pool4)

    fuse_pool3 = Add()([score_pool3,upscore2_pool4])

    upscore8 = UpSampling2D(size=(8, 8))(fuse_pool3)

    score_final = conv_layer(upscore8, nb_kernel=16, nb_filter=NB_CLASS, strides=(1,1))
    softmax = Activation("softmax")(score_final)

    model = Model(inputs=img_input,outputs=[softmax])

    return model

