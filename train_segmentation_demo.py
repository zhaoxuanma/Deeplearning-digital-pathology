from models import FCN_8s, Unet

from KerasLayers.ImageDataGeneratorEXT import ImageDataGeneratorEXT

from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


import math

NB_CLASS = 5  # number of catagories of the training

train_data_dir = "Path_to_training_data_folder" # image folder of training data
validation_data_dir = "Path_to_validation_data_folder" # image folder of validation data
img_width = Unet.IMG_SIZE # image size
img_height = Unet.IMG_SIZE
batch_size = 16 # batch size per iteration
epochs = 50 # epochs count
learning_rate = 0.001 # initial learning rate
gamma = 0.95 # decay gamma for learning rate
log_filepath = './log' # logging folder
model_filepath = './Model' # result model path


def step_decay(epoch):
    lr = learning_rate * math.pow(gamma,epoch)
    return lr


def main():

    train_datagen = ImageDataGeneratorEXT(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory_segmentation(
        directory = train_data_dir, image_subfolder='image', mask_subfolder='mask',
        target_size=(img_width, img_height),
        classes=NB_CLASS,
        batch_size=batch_size, shuffle=True)

    validation_datagen = ImageDataGeneratorEXT()

    validation_generator = validation_datagen.flow_from_directory_segmentation(
        directory=validation_data_dir, image_subfolder='image', mask_subfolder='mask',
        target_size=(img_width, img_height),
        classes=NB_CLASS,
        batch_size=batch_size, shuffle=True)

    # Create model
    model = FCN_8s.create_model(NB_CLASS)
    # model = Unet.create_model(NB_CLASS)
    model.summary()

    sgd = SGD(lr = learning_rate, momentum=0.9, decay= 0.0, nesterov=True)
    model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])

    tb_cb = TensorBoard(log_dir=log_filepath)
    lrate = LearningRateScheduler(step_decay)

    modelcheckpoint_cbk = ModelCheckpoint(model_filepath + '/weights.{epoch:02d}-{acc:.2f}.h5', monitor='acc', period=10)
    cbks = [tb_cb, modelcheckpoint_cbk, lrate]

    img_count_train = train_datagen.samples
    steps_per_epoch_train = img_count_train/batch_size
    img_count_validation = validation_datagen.samples
    steps_per_epoch_validation = img_count_validation/batch_size

    # Train
    model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch_train, epochs=epochs,
                        validation_data=validation_generator, validation_steps=steps_per_epoch_validation,
                        max_queue_size = batch_size,callbacks = cbks,verbose=1)


if __name__ == '__main__':
    main()