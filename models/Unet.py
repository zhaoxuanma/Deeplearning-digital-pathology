from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Activation, Concatenate, UpSampling2D, Dropout
from keras.models import Model
from keras import regularizers
from keras.initializers import Constant, glorot_normal

# global constants
IMG_SIZE = 256
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


    return x

def deconv_layer(x, nb_filter, nb_kernel, size):


    x = UpSampling2D(size)(x)
    x = conv_layer(x, nb_filter, nb_kernel)

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

    conv_d0a_b = conv_layer(img_input, nb_kernel=3, nb_filter=64, strides=(1,1))
    conv_d0b_c = conv_layer(conv_d0a_b, nb_kernel=3, nb_filter=64, strides=(1,1))

    pool_d0c_1a = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv_d0b_c)
    conv_d1a_b = conv_layer(pool_d0c_1a, nb_kernel=3, nb_filter=128, strides=(1,1))
    conv_d1b_c = conv_layer(conv_d1a_b, nb_kernel=3, nb_filter=128, strides=(1,1))

    pool_d1c_2a = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv_d1b_c)
    conv_d2a_b = conv_layer(pool_d1c_2a, nb_kernel=3, nb_filter=256, strides=(1,1))
    conv_d2b_c = conv_layer(conv_d2a_b, nb_kernel=3, nb_filter=256, strides=(1,1))

    pool_d2c_3a = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv_d2b_c)
    conv_d3a_b = conv_layer(pool_d2c_3a, nb_kernel=3, nb_filter=512, strides=(1,1))
    conv_d3b_c = conv_layer(conv_d3a_b, nb_kernel=3, nb_filter=512, strides=(1,1))
    conv_d3b_c = Dropout(rate=0.5)(conv_d3b_c)

    pool_d3c_4a = MaxPooling2D(strides=(2, 2), pool_size=(2, 2), padding="same")(conv_d3b_c)
    conv_d4a_b = conv_layer(pool_d3c_4a, nb_kernel=3, nb_filter=1024, strides=(1,1))
    conv_d4b_c = conv_layer(conv_d4a_b, nb_kernel=3, nb_filter=1024, strides=(1,1))
    conv_d4b_c = Dropout(rate=0.5)(conv_d4b_c)

    upconv_d4c_u3a = deconv_layer(conv_d4b_c,nb_kernel=3, nb_filter=512, size=(2,2))
    relu_u3a = Activation('relu')(upconv_d4c_u3a)
    concat_d3cc_u3a_b = Concatenate()([conv_d3b_c, relu_u3a])
    conv_u3b_c = conv_layer(concat_d3cc_u3a_b, nb_kernel=3, nb_filter=512, strides=(1,1))
    conv_u3c_d = conv_layer(conv_u3b_c, nb_kernel=3, nb_filter=512, strides=(1,1))

    upconv_u3d_u2a = deconv_layer(conv_u3c_d,nb_kernel=3, nb_filter=256, size=(2,2))
    relu_u2a = Activation('relu')(upconv_u3d_u2a)
    concat_d2cc_u2a_b = Concatenate()([conv_d2b_c, relu_u2a])
    conv_u2b_c = conv_layer(concat_d2cc_u2a_b, nb_kernel=3, nb_filter=256, strides=(1,1))
    conv_u2c_d = conv_layer(conv_u2b_c, nb_kernel=3, nb_filter=256, strides=(1,1))

    upconv_u2d_u1a = deconv_layer(conv_u2c_d,nb_kernel=3, nb_filter=128, size=(2,2))
    relu_u1a = Activation('relu')(upconv_u2d_u1a)
    concat_d1cc_u1a_b = Concatenate()([conv_d1b_c, relu_u1a])
    conv_u1b_c = conv_layer(concat_d1cc_u1a_b, nb_kernel=3, nb_filter=128, strides=(1,1))
    conv_u1c_d = conv_layer(conv_u1b_c, nb_kernel=3, nb_filter=128, strides=(1,1))

    upconv_u1d_u0a = deconv_layer(conv_u1c_d,nb_kernel=3, nb_filter=64, size=(2,2))
    relu_u0a = Activation('relu')(upconv_u1d_u0a)
    concat_d0cc_u0a_b = Concatenate()([conv_d0b_c, relu_u0a])
    conv_u0b_c = conv_layer(concat_d0cc_u0a_b, nb_kernel=3, nb_filter=128, strides=(1,1))
    conv_u0c_d = conv_layer(conv_u0b_c, nb_kernel=3, nb_filter=128, strides=(1,1))

    conv_u0d_score =  conv_layer(conv_u0c_d, nb_kernel=1, nb_filter=NB_CLASS, strides=(1,1))

    softmax = Activation("softmax")(conv_u0d_score)

    model = Model(inputs=img_input,outputs=[softmax])

    return model



