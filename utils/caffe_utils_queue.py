import os
os.environ['GLOG_minloglevel'] = '3'

import caffe

from threadsafe_iter import threadsafe_generator

import Queue as queue
import threading

import numpy as np


class Caffeobject(object):

    channel_order = 'BGR'   # [blue, green, red] (Caffe uses OpenCV)
    data_format = 'channels_first'     # [batch, channel, height, width]

    def __init__(self, caffemodel_path, prototxt_path, queue_size=10, output_layer='softmax'):

        """ Caffe class.

        Args:
            caffemodel_path: path of caffe model file (.caffemodel).
            prototxt_path: path of caffe model structure file (.prototxt).
            queue_size: queue size for result.
            output_layer: result layer name of the model.

        Return:
            Caffe class.

        """

        self.caffemodel_path = caffemodel_path
        self.prototxt_path = prototxt_path
        self.queue_size = queue_size
        self.output_layer = output_layer

        self.model= caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
        caffe.set_mode_gpu()

        self.q = queue.Queue(maxsize=self.queue_size)

    def forward_batch(self,data_batch):

        """ Forward batch to neural network and get results.

        Args:
            data_batch: image batch ready for Caffe.

        Return:
            results after neural network.

        """

        self.model.blobs['data'].reshape(*data_batch.shape)
        self.model.blobs['data'].data[...] = data_batch
        result = self.model.forward()[self.output_layer]

        return result

    def forward_from_queue_to_queue(self, data_queue):

        """ Forward batch in queue to neural network and get results.

        Args:
            data_queue: queue containing image batch ready for Caffe.

        Return:
            queue containing results after neural network.

        """

        thread = threading.Thread(target=forward_from_queue,
                                  args=(self.caffemodel_path,self.prototxt_path,self.output_layer,data_queue,self.q))
        thread.daemon = True
        thread.start()

        return self.q

@threadsafe_generator
def forward_from_queue(caffemodel_path, prototxt_path, output_layer, data_queue, result_queue):

    progress_float = 0

    caffe.set_mode_gpu()
    MODEL = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    for data_out in iter(data_queue.get, None):
        data_batch, coord, batch_ind, batch_count = data_out

        if float(batch_ind)/batch_count - progress_float > 0.01:

            progress_float = float(batch_ind) / batch_count
            print("\rProgress: [{0:50s}] {1:.0f}%".
                  format('#' * int(progress_float * 50) + '-' * (50 - int(progress_float * 50)), progress_float * 100))

        if data_batch is None:
            continue

        MODEL.blobs['data'].reshape(*data_batch.shape)
        MODEL.blobs['data'].data[...] = data_batch
        result = MODEL.forward()[output_layer]

        result_queue.put((np.copy(result), np.array(coord)))

    print("\rProgress: [{0:50s}] 100%\n".format('#' * 50))

    result_queue.put(None)

