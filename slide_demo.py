from utils.slide_utils import Slideobject

from utils.tf_utils_queue import TFobject
from utils.caffe_utils_queue import Caffeobject

import time
import scipy.io as sio
import glob
import os
from shutil import copyfile

import cv2

slide_folder = "Path_to_slides_files"
model_path = "Path_to_Keras_model"
# caffe_model_path = "Path_to_caffe_model_file(.caffemodel)"
# caffe_prototxt_path = "Path_to_caffe_prototxt_file(.prototxt)"

RAW_TILE_SIZE = 600 # image size to retrieve from slide
IMG_SIZE = 256 # model input image size
BATCH_SIZE = 32 #model batch size


def main():

    # Create model class
    model_helper = TFobject(model_path=model_path)
    # model_helper = Caffeobject(caffemodel_path=caffe_model_path, prototxt_path=caffe_prototxt_path, output_layer='softmax')

    slide_path_list = glob.glob(slide_folder+ '/*.svs')

    for ind, slide_path in enumerate(slide_path_list):

        print("{0:d} of {1:d} - {2}".format(ind+1, len(slide_path_list),slide_path))

        filename = os.path.basename(slide_path)
        filepath = '/dev/shm/'+filename
        copyfile(slide_path, filepath)

        # Create slide class
        slide_helper = Slideobject(filepath,
                                   retrieve_img_size=RAW_TILE_SIZE,
                                   target_img_size=IMG_SIZE,
                                   batch_size=BATCH_SIZE,
                                   level=0,queue_size = 256,
                                   data_format=model_helper.data_format,
                                   channel_order=model_helper.channel_order)

        # Retrieve batch ready for neural network and put in queue
        data_queue = slide_helper.retrieve_tiles_to_queue_thread(voting=False, rotation=False, thread_num=16)

        t0 = time.time()

        # Push batch for neural network and put results in queue
        result_queue = model_helper.forward_from_queue_to_queue(data_queue=data_queue)

        # Reconstruction of results into full slide
        # result_rgb, result_mask, result_data = slide_helper.reconstruct_segmentation_queue_to_level(
        #     queue=result_queue, result_level=2, save_raw=True) # Save raw results

        # result_rgb, result_mask= slide_helper.reconstruct_classification_queue_to_level(
        #     data_queue=result_queue, result_level=2, save_raw=False) # classification
        result_rgb, result_mask= slide_helper.reconstruct_segmentation_queue_to_level(
            data_queue=result_queue, result_level=2, save_raw=False) # segmentation

        print('Time elapsed: ', time.time() - t0)

        cv2.imwrite(os.path.splitext(slide_path)[0] + '_result_rgb.jpg',cv2.cvtColor(result_rgb,cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.splitext(slide_path)[0] + '_result_mask.tif',result_mask)

        os.remove(filepath)


if __name__ == '__main__':
    main()

