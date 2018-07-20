from utils.image_utils import Imageobject

from utils.tf_utils_queue import TFobject
from utils.caffe_utils_queue import Caffeobject

import time

image_folder = "Path_to_images_for_model"
result_folder = "Path_to_result_folder"
model_path = "Path_to_Keras_model"
# caffe_model_path = "Path_to_caffe_model_file(.caffemodel)"
# caffe_prototxt_path = "Path_to_caffe_prototxt_file(.prototxt)"

IMG_SIZE = 256 # model input image size
BATCH_SIZE = 32 # model batch size


def main():

    # Create model class
    model_helper = TFobject(model_path=model_path)
    # model_helper = Caffeobject(caffemodel_path=caffe_model_path, prototxt_path=caffe_prototxt_path, output_layer='softmax')

    # Create slide class
    image_helper = Imageobject(image_folder,
                               batch_size=BATCH_SIZE,
                               target_img_size=IMG_SIZE,
                               queue_size = 256,
                               ext ='.png', channel_order =model_helper.channel_order,
                               data_format =model_helper.data_format)

    # Retrieve batch ready for neural network and put in queue
    data_queue = image_helper.retrieve_images_to_queue_thread(rotation=False, thread_num=16)

    t0 = time.time()

    # Push batch for neural network and put results in queue
    result_queue = model_helper.forward_from_queue_to_queue(data_queue=data_queue)

    # Reconstruction of results and save
    # image_helper.reconstruct_classification_queue_to_file(result_queue, result_folder, result_suffix='test', save_raw=False)
    image_helper.reconstruct_segmentation_queue_to_file(result_queue, result_folder, result_suffix='test', save_raw=False)

    print('Time elapsed: ', time.time() - t0)


if __name__ == '__main__':
    main()

