import numpy as np
import cv2
import os
import glob
import scipy.io as sio

import Queue as queue
import threading

from math import ceil

from threadsafe_iter import threadsafe_generator


def round(number, places=0):

    place = 10**places
    rounded = (int(number*place + 0.5if number>=0 else -0.5))/place
    if rounded == int(rounded):
        rounded = int(rounded)
    return rounded

class Imageobject(object):


    def __init__(self, image_folder, batch_size = 50, target_img_size = 256, queue_size = 10,
                 ext ='.png', channel_order ='RGB',
                 data_format ='channels_last'):

        """ Image class.

        Args:
            image_folder: path to image file.
            batch_size: image number per batch for neural network
            target_img_size: target image size after resizing as the input of the neural network.
            ext: valid image file extension, should be string or tuple of string
            queue_size: queue size for retrieving batch ready for neural network.
            channel_order: r, g, b order of color image. ('RGB' or 'BGR')
            data_format: batch dimension format
                channels_first: [batch, channel, height, width] (Caffe format)
                channels_last: [batch, height, width, channel] (Tensorflow format)


        Return:
            slide class for retrieving image batches and result reconstruction

        """

        self.image_folder = image_folder
        self.batch_size = batch_size
        self.target_img_size = target_img_size
        self.queue_size = queue_size
        self.channel_order = channel_order
        self.data_format = data_format
        self.ext = ext

        self.q = queue.Queue(maxsize=self.queue_size)
        self.threads = None


    # multi thread
    def retrieve_images_to_queue_thread(self, rotation=False, thread_num=16):

        """ Generate image batch in multiple threads with specified augmentations and put in queue.

        Args:
            rotation: whether to add rotation augmentation in image batch.
            thread_num: threads for retrieving.

        Return:
            queue of image batch.

        """

        if type(self.ext) is str:
            image_path_list = glob.glob(self.image_folder + os.path.sep + '*' + self.ext)

        elif(type(self.ext) is tuple):
            image_path_list = []
            for ext_t in self.ext:
                if type(ext_t) is str:
                    image_path_list.extend(glob.glob(self.image_folder + os.path.sep + '*' + ext_t))

        print('Found '+ str(len(image_path_list)) + ' images.')

        # threading
        self.threads = []
        # split into chunks of thread_num
        if (len(image_path_list) < thread_num):
            path_list_chunks = [image_path_list[index::len(image_path_list)] for index in range(len(image_path_list))]
        else:
            path_list_chunks = [image_path_list[index::thread_num] for index in range(thread_num)]

        for i in range(len(path_list_chunks)):
            thread = threading.Thread(target=self.retrieve_images_to_queue_thread_target,
                                      args=(path_list_chunks[i], rotation, threading.Lock()))
            thread.start()
            self.threads.append(thread)

        return self.q

    def retrieve_images_to_queue_thread_target(self, image_path_list, rotation, lock):

        self._retrieve_images_from_paths(image_path_list, rotation, lock)

        # thread finished control
        lock.acquire()
        self.threads.remove(threading.current_thread())
        if len(self.threads) == 0:
            self.q.put(None) # put None for queue iter stop
        lock.release()

    @threadsafe_generator
    def _retrieve_images_from_paths(self,image_path_list, rotation, lock):

        img_batch = []
        path_batch = []

        for ind, image_path in enumerate(image_path_list):

            if not rotation:  # multi tile batch

                batch_count = ceil(len(image_path_list) / float(self.batch_size))

                lock.acquire()
                img = cv2.imread(image_path)
                lock.release()

                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.target_img_size, self.target_img_size))
                img = self.preprocess_img(img)
                if img is None:  # white tile
                    continue

                if self.channel_order is 'RGB':
                    pass
                elif self.channel_order is 'BGR':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    raise Exception('Invalid channel order!')

                img_batch.append(img)
                path_batch.append(image_path)

                if len(img_batch) == self.batch_size or ind == len(image_path_list) - 1:

                    batch_ind = (ind + 1) // self.batch_size

                    img_batch_ = np.stack(img_batch, axis=0)
                    path_batch_ = path_batch

                    if self.data_format == 'channels_first':
                        img_batch_ = np.moveaxis(img_batch_, 3, 1)

                    self.q.put((img_batch_, path_batch_, batch_ind, batch_count))

                    img_batch = []
                    path_batch = []

            else:  # same tile batch

                batch_count = len(image_path_list)

                lock.acquire()
                img = cv2.imread(image_path)
                lock.release()

                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.target_img_size, self.target_img_size))
                img = self.preprocess_img(img)

                if img is None:  # white tile
                    continue

                # rotation
                rotation_time = 0
                if rotation:
                    rotation_time = 4

                if self.channel_order is 'RGB':
                    pass
                elif self.channel_order is 'BGR':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    raise Exception('Invalid channel order!')

                for rot in range(rotation_time):
                    img_batch.append(np.rot90(img, rot))

                img_batch_ = np.stack(img_batch, axis=0)
                path_batch_ = [image_path]

                if self.data_format == 'channels_first':
                    img_batch_ = np.moveaxis(img_batch_, 3, 1)

                self.q.put((img_batch_, path_batch_, ind, batch_count))

                img_batch = []
                path_batch = []

        # last batch due to None img
        if len(img_batch) is not 0:
            img_batch_ = np.stack(img_batch, axis=0)
            path_batch_ = np.stack(path_batch, axis=0)

            if self.data_format == 'channels_first':
                img_batch_ = np.moveaxis(img_batch_, 3, 1)

            self.q.put((img_batch_, path_batch_, batch_count, batch_count))


    @staticmethod
    def preprocess_img(img):

        """ Preprocess image here.

        Args:
            img: image before preprocessing.

        Return:
            image after preprocessing.

        """

        return img

    def reconstruct_classification_queue_to_file(self, data_queue, result_folder, suffix, save_raw=True):

        """ Reconstruct classification results on top of images.

        Args:
            data_queue: queue contains the results after neural network (same format as the retrieving).
            result_folder: folder to save the results.
            result_suffix: suffix for result file.
            save_raw: flag if to save the raw results before argmax.

        """

        for data_out in iter(data_queue.get, None):

            results, paths= data_out

            if len(paths) == 1:  # same-tile_augmentation

                results_mean = np.mean(results, axis=0)
                predictions = np.expand_dims(np.argmax(results_mean,-1),0)

            elif len(paths) == len(results):  # multi-tiles
                pass

            else:
                raise Exception('Invalid result dimension!')

            for ind in range(len(predictions)):

                pred = predictions[ind]
                path = paths[ind]

                raw = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

                tile_gray_coded = np.zeros((raw.shape[0],raw.shape[1]), np.uint8)
                tile_gray_coded[:] = self.gray_code(pred)

                tile_color_coded = np.zeros(raw.shape, np.uint8)
                tile_color_coded[:] = self.color_code(pred)

                result_path_base = result_folder + os.path.sep + os.path.splitext(os.path.basename(path))[0]

                rgb_result = cv2.cvtColor((tile_color_coded*0.2+raw*0.8).astype(np.uint8), cv2.COLOR_RGB2BGR)

                cv2.imwrite(result_path_base + '_result_rgb_' + suffix + self.ext, rgb_result)
                cv2.imwrite(result_path_base + '_result_mask_' + suffix + self.ext, tile_gray_coded)

                if save_raw:
                    mdict = {}
                    mdict['Raw'] = results

                    sio.savemat(result_path_base + '_result_data_' + suffix + '.mat', mdict)

    def reconstruct_segmentation_queue_to_file(self, data_queue, result_folder, result_suffix, save_raw=True):

        """ Reconstruct segmentation results on top of images.

        Args:
            data_queue: queue contains the results after neural network (same format as the retrieving).
            result_folder: folder to save the results.
            result_suffix: suffix for result file.
            save_raw: flag if to save the raw results before argmax.

        """

        for data_out in iter(data_queue.get, None):

            results, paths= data_out

            if self.data_format == 'channels_first':
                results = np.moveaxis(results, 1, -1)

            if len(paths) == 1:  # same-tile_augmentation

                for i in range(4):
                    results[i, :, :, :] = np.rot90(results[i, :, :, :], -i)

                results_mean = np.mean(results, axis=0)
                predictions = np.expand_dims(np.argmax(results_mean,-1),0)

            elif len(paths) == len(results):  # multi-tiles
                predictions = np.argmax(results, -1)

            else:
                raise Exception('Invalid result dimension!')

            for ind in range(len(predictions)):

                pred = predictions[ind]
                path = paths[ind]

                raw = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

                tile_gray_coded = cv2.resize(self.gray_code(pred),(raw.shape[1],raw.shape[0]), interpolation=cv2.INTER_NEAREST)
                tile_color_coded = cv2.resize(self.color_code(pred), (raw.shape[1],raw.shape[0]), interpolation=cv2.INTER_NEAREST)

                result_path_base = result_folder + os.path.sep + os.path.splitext(os.path.basename(path))[0]

                rgb_result = cv2.cvtColor((tile_color_coded*0.2+raw*0.8).astype(np.uint8), cv2.COLOR_RGB2BGR)

                cv2.imwrite(result_path_base + '_result_rgb_' + result_suffix + self.ext, rgb_result)
                cv2.imwrite(result_path_base + '_result_mask_' + result_suffix + self.ext, tile_gray_coded)

                if save_raw:
                    mdict = {}
                    mdict['Raw'] = results

                    sio.savemat(result_path_base + '_result_data_' + result_suffix + '.mat', mdict)

    @staticmethod
    def gray_code(index):

        color_list = {'0':43,
                      '1':172,
                      '2':86,
                      '3':215,
                      '4':129}

        if np.isscalar(index): # classification

            if str(index) in color_list:
                return color_list[str(index)]
            else:
                return 255

        else: # segmentation

            result = np.zeros(index.shape)
            inds = np.unique(index)
            for ind in inds:

                if str(ind) in color_list:
                    result[index == ind] = color_list[str(ind)]
                else:
                    result[index == ind] = 255

            return result
    @staticmethod
    def color_code(index):

        color_list = {'0':np.array((255, 255, 0), np.uint8),
                      '1':np.array((255, 0, 0), np.uint8),
                      '2':np.array((0, 255, 0), np.uint8),
                      '3':np.array((0, 0, 255), np.uint8)}

        if np.isscalar(index): # classification

            if str(index) in color_list:
                return color_list[str(index)]
            else:
                return np.array((0,0,0), np.uint8)

        else: # segmentation

            result = np.zeros(index.shape+(3,),np.uint8)
            inds = np.unique(index)
            for ind in inds:

                if str(ind) in color_list:
                    result[index == ind] = color_list[str(ind)]
                else:
                    result[index == ind] = np.array((0, 0, 0), np.uint8)

            return result