import openslide

import numpy as np
import cv2

import Queue as queue
import threading

from math import ceil

from threadsafe_iter import threadsafe_generator


class Slideobject(object):

    def __init__(self, slide_path, batch_size=32, retrieve_img_size=600, target_img_size=256, queue_size=32,
                 level=0, channel_order='RGB', data_format='channels_last'):

        """ Slide class.

        Args:
            slide_path: path to slide file.
            batch_size: image number per batch for neural network
            retrieve_img_size: retrieve image size of the selected resolution layer.
            target_img_size: target image size after resizing as the input of the neural network.
            queue_size: queue size for retrieving batch ready for neural network.
            level: selected resolution layer.
            channel_order: r, g, b order of color image. ('RGB' or 'BGR')
            data_format: batch dimension format
                channels_first: [batch, channel, height, width] (Caffe format)
                channels_last: [batch, height, width, channel] (Tensorflow format)


        Return:
            slide class for retrieving image batches and result reconstruction

        """

        self.slide_path = slide_path
        self.batch_size = batch_size
        self.retrieve_img_size = retrieve_img_size
        self.target_img_size = target_img_size
        self.queue_size = queue_size
        self.level = level
        self.channel_order = channel_order
        self.data_format = data_format

        self.slide = openslide.OpenSlide(slide_path)

        self.q = queue.Queue(maxsize=self.queue_size)
        self.threads = None

    # multi thread
    def retrieve_tiles_to_queue_thread(self, rotation=False, voting=False, subsize=3, thread_num=16):

        """ Generate image batch in multiple threads with specified augmentations and put in queue.

        Args:
            rotation: whether to add rotation augmentation in image batch.
            voting: whether to add voting augmentation in image batch.
            subsize: subsize of voting.
            thread_num: threads for retrieving.

        Return:
            queue of image batch.

        """

        size = self.slide.level_dimensions[self.level]

        print('Size: ' + str(size))

        size_w = size[0]
        size_h = size[1]

        w_ind = int(float(size_w) / self.retrieve_img_size)
        h_ind = int(float(size_h) / self.retrieve_img_size)

        coord_list = [(w * self.retrieve_img_size, h * self.retrieve_img_size) for w in range(w_ind) for h in
                      range(h_ind)]

        # threading
        self.threads = []
        # split into chunks of thread_num
        if (len(coord_list) < thread_num):
            coord_list_chunks = [coord_list[index::len(coord_list)] for index in range(len(coord_list))]
        else:
            coord_list_chunks = [coord_list[index::thread_num] for index in range(thread_num)]

        for i in range(len(coord_list_chunks)):

            thread = threading.Thread(target=self._retrieve_tiles_to_queue_thread_target,
                                      args=(coord_list_chunks[i], rotation, voting, subsize, threading.Lock()))
            thread.start()
            self.threads.append(thread)

        return self.q

    def _retrieve_tiles_to_queue_thread_target(self, coord_list, rotation, voting, subsize, lock):

        self._retrieve_tiles_from_coords(coord_list, rotation, voting, subsize, lock)

        # thread finished control
        lock.acquire()
        self.threads.remove(threading.current_thread())
        if len(self.threads) == 0:
            self.q.put(None) # put None for queue iter stop
        lock.release()

    @threadsafe_generator
    def _retrieve_tiles_from_coords(self, coord_list, rotation, voting, subsize, lock):

        img_batch = []
        coord_batch = []

        for ind, coord in enumerate(coord_list):

            if not rotation and not voting:  # multi tile batch

                batch_count = ceil(len(coord_list) / float(self.batch_size))

                lock.acquire()
                img = self.slide.read_region(coord, self.level, (self.retrieve_img_size, self.retrieve_img_size))
                lock.release()

                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
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
                coord_batch.append(coord)

                if len(img_batch) == self.batch_size or ind == len(coord_list) - 1:

                    batch_ind = (ind + 1) // self.batch_size

                    img_batch_ = np.stack(img_batch, axis=0)
                    coord_batch_ = np.stack(coord_batch, axis=0)

                    if self.data_format == 'channels_first':
                        img_batch_ = np.moveaxis(img_batch_, 3, 1)

                    self.q.put((img_batch_, coord_batch_, batch_ind, batch_count))

                    img_batch = []
                    coord_batch = []

            else:  # same tile batch

                batch_count = len(coord_list)

                lock.acquire()
                img = self.slide.read_region(coord, self.level, (self.retrieve_img_size, self.retrieve_img_size))
                lock.release()

                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
                img = cv2.resize(img, (self.target_img_size, self.target_img_size))
                img = self.preprocess_img(img)

                if img is None:  # white tile
                    continue

                # voting
                coord_delta_grid = [(0, 0)]  # default no voting
                if voting:

                    # read bigger region for sub-sampling
                    coord_shift = self.retrieve_img_size // subsize * (subsize // 2)
                    voting_target_img_size = self.target_img_size + self.target_img_size // subsize * (subsize // 2 * 2)

                    lock.acquire()
                    img = self.slide.read_region((coord[0]-coord_shift,coord[1]-coord_shift),
                                                 self.level, (self.retrieve_img_size + coord_shift*2, self.retrieve_img_size+ coord_shift*2))
                    lock.release()

                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
                    img = cv2.resize(img, (voting_target_img_size, voting_target_img_size))
                    img = self.preprocess_img(img)

                    if img is None:  # white tile
                        continue

                    coord_delta = [self.target_img_size // subsize * dl for dl in
                                   range(0, subsize)]
                    g = np.meshgrid(coord_delta, coord_delta)
                    coord_delta_grid = zip(*(x.flat for x in g))

                # rotation
                rotation_time = 1
                if rotation:
                    rotation_time = 4

                if self.channel_order is 'RGB':
                    pass
                elif self.channel_order is 'BGR':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    raise Exception('Invalid channel order!')

                for xx, yy in coord_delta_grid:

                    img_sub = img[xx:xx+self.target_img_size,yy:yy+self.target_img_size,:]

                    for rot in range(rotation_time):
                        img_batch.append(np.rot90(img_sub, rot))

                coord_batch.append(coord)

                img_batch_ = np.stack(img_batch, axis=0)
                coord_batch_ = np.stack(coord_batch, axis=0)

                if self.data_format == 'channels_first':
                    img_batch_ = np.moveaxis(img_batch_, 3, 1)

                self.q.put((img_batch_, coord_batch_, ind, batch_count))

                img_batch = []
                coord_batch = []

        # last batch due to None img
        if len(img_batch) is not 0:
            img_batch_ = np.stack(img_batch, axis=0)
            coord_batch_ = np.stack(coord_batch, axis=0)

            if self.data_format == 'channels_first':
                img_batch_ = np.moveaxis(img_batch_, 3, 1)

            self.q.put((img_batch_, coord_batch_, batch_count, batch_count))

    @staticmethod
    def preprocess_img(img):

        """ Preprocess image here.

        Args:
            img: image before preprocessing.

        Return:
            image after preprocessing.

        """

        return img

    def reconstruct_segmentation_queue_to_level(self, data_queue, result_level, save_raw=False):

        """ Reconstruct segmentation results on top of selected layer image.

        Args:
            data_queue: queue contains the results after neural network (same format as the retrieving).
            result_level: selected layer image of the virtual slide (openslide backend).
            save_raw: flag if to save the raw results before argmax.

        Return:
            result_img: results overlayed on original image with colorcoding defined in 'color_code'.
            result_mosaic: results with colorcoding defined in 'gray_code'.

            if save_raw:
                result_raw: raw results before argmax.

        """

        size1 = self.slide.level_dimensions[self.level]
        size2 = self.slide.level_dimensions[result_level]

        # edit the coords to match new level
        ratio = float(size1[0] / size2[0])
        new_tile_size = int(round(self.retrieve_img_size / ratio))

        img = np.array(self.slide.read_region((0, 0), result_level, size2))

        output_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        result_raw = None

        for data_out in iter(data_queue.get, None):

            results, coords = data_out

            if self.data_format == 'channels_first':
                results = np.moveaxis(results,1,-1)

            if result_raw is None:
                result_raw = np.zeros((output_img.shape[0],output_img.shape[1],results.shape[-1]), np.float)

            if len(np.squeeze(coords).shape)==1 and len(results) == 4: # same-tile_augmentation (only rotation supported)

                for i in range(4):
                    results[i, :, :, :] = np.rot90(results[i, :, :, :], -i)

                results = np.mean(results, axis=0)
                predictions = np.expand_dims(np.argmax(results,2),0)
                results = np.expand_dims(results,0)

            elif len(np.squeeze(coords).shape) == 2 and np.squeeze(coords).shape[0] == results.shape[0]:  # multi-tiles
                predictions = np.argmax(results,-1)

            else:
                raise Exception('Invalid predictions!')

            for ind in range(len(predictions)):

                coord = coords[ind]
                result = results[ind]

                new_coords = [int(round(cd / ratio)) for cd in coord]

                result_raw_block = result_raw[new_coords[1]:new_coords[1] + new_tile_size,
                new_coords[0]:new_coords[0] + new_tile_size, ...]

                result = cv2.resize(result, (result_raw_block.shape[1],result_raw_block.shape[0]))

                result_raw[new_coords[1]:new_coords[1] + new_tile_size,
                new_coords[0]:new_coords[0] + new_tile_size, ...] += result # avoid border overflow

        final_prediction = np.argmax(result_raw, 2)

        result_mosaic = self.gray_code(final_prediction)
        result_mask = self.color_code(final_prediction)
        result_img = (output_img*0.8 + result_mask*0.2).astype(np.uint8)

        print('Rebuild result finished')
        if save_raw:
            return result_img, result_mosaic, result_raw
        else:
            return result_img, result_mosaic

    def reconstruct_classification_queue_to_level(self, data_queue, result_level, save_raw=False):

        """ Reconstruct classification results on top of selected layer image.

        Args:
            data_queue: queue contains the results after neural network (same format as the retrieving).
            result_level: selected layer image of the virtual slide (openslide backend).
            save_raw: flag if to save the raw results before argmax.

        Return:
            result_img: results overlayed on original image with colorcoding defined in 'color_code'.
            result_mosaic: results with colorcoding defined in 'gray_code'.

            if save_raw:
                result_raw: raw results before argmax.

        """

        size1 = self.slide.level_dimensions[self.level]
        size2 = self.slide.level_dimensions[result_level]

        # edit the coords to match new level
        ratio = float(size1[0] / size2[0])
        new_tile_size = int(round(self.retrieve_img_size / ratio))

        img = np.array(self.slide.read_region((0, 0), result_level, size2))

        output_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        result_raw = None

        for data_out in iter(data_queue.get, None):

            results, coords = data_out

            if self.data_format == 'channels_first':
                results = np.moveaxis(results, 1, -1)

            if result_raw is None:
                result_raw = np.zeros((output_img.shape[0], output_img.shape[1], results.shape[-1]), np.float)

            if len(np.squeeze(coords).shape)==1: # same-tile_augmentation

                results = np.mean(results, axis=0)
                predictions = np.expand_dims(np.argmax(results,0),0)
                results = np.expand_dims(results, 0)

            elif len(np.squeeze(coords).shape) == 2 and np.squeeze(coords).shape[0] == results.shape[0]:  # multi-tiles
                predictions = np.argmax(results,1)
                pass

            else:
                raise Exception('Invalid predictions!')

            for ind in range(len(predictions)):

                coord = coords[ind]
                result = results[ind]

                new_coords = [int(round(cd / ratio)) for cd in coord]

                result_raw[new_coords[1]:new_coords[1] + new_tile_size,
                new_coords[0]:new_coords[0] + new_tile_size, ...] += result  # avoid border overflow

        final_prediction = np.argmax(result_raw, 2)

        result_mosaic = self.gray_code(final_prediction)
        result_mask = self.color_code(final_prediction)
        result_img = (output_img * 0.8 + result_mask * 0.2).astype(np.uint8)

        print('Rebuild result finished')
        if save_raw:
            return result_img, result_mosaic, result_raw
        else:
            return result_img, result_mosaic


    def clean_up_background(self,raw_image,*args):

        re = ()

        grayref = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
        bgmask = (grayref > 215) | (grayref < 20)

        for image in args:
            if (not np.isscalar(image)) and (image.shape[0:2] == bgmask.shape):

                if len(image.shape) == 2:
                    image[bgmask] = self.gray_code(-1) # -1 as background
                else:
                    image[bgmask] = self.color_code(-1)

            re = re + (image,)

        return re

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
                      '1':np.array((0, 0, 255), np.uint8),
                      '2':np.array((0, 255, 0), np.uint8),
                      '3':np.array((255, 0, 0), np.uint8)}

        if np.isscalar(index): # classification

            if str(index) in color_list:
                return color_list[str(index)]
            else:
                return np.array((0,0,0), np.uint8)

        else: # segmentation

            result = np.zeros(index.shape+(3,), np.uint8)
            inds = np.unique(index)
            for ind in inds:

                if str(ind) in color_list:
                    result[index == ind] = color_list[str(ind)]
                else:
                    result[index == ind] = np.array((0, 0, 0), np.uint8)

            return result

