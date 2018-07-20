from keras.models import load_model
from threadsafe_iter import threadsafe_generator

import tensorflow as tf

import Queue as queue
import threading

import numpy as np
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class TFobject(object):

    channel_order = 'RGB'   # [red, blue, green] (Tensorflow uses PIL)
    data_format = 'channels_last'     # [batch, height, width, channel]

    def __init__(self, model_path, queue_size=10):

        """ Tensorflow (Keras) class.

        Args:
            model_path: path of Keras model file.
            queue_size: queue size for result.

        Return:
            Tensorflow (Keras) class.

        """

        self.model_path = model_path
        self.queue_size = queue_size

        with suppress_stdout():
            self.model = load_model(self.model_path)
            self.model.summary()
            self.graph = tf.get_default_graph()

        self.q = queue.Queue(maxsize=self.queue_size)

    def forward_batch(self, data_batch):

        """ Forward batch to neural network and get results.

        Args:
            data_batch: image batch ready for Tensorflow.

        Return:
            results after neural network.

        """
        return self.model.predict_on_batch(data_batch)

    def forward_from_queue_to_queue(self, data_queue):

        """ Forward batch in queue to neural network and get results.

        Args:
            data_queue: queue containing image batch ready for Tensorflow.

        Return:
            queue containing results after neural network.

        """

        thread = threading.Thread(target=forward_from_queue,
                                  args=(self.model,self.graph, data_queue, self.q))
        thread.daemon = True
        thread.start()

        return self.q

@threadsafe_generator
def forward_from_queue(model, graph, data_queue, result_queue):


    progress_float = 0

    for data_out in iter(data_queue.get, None):
        data_batch, coord, batch_ind, batch_count = data_out

        if float(batch_ind)/batch_count - progress_float > 0.01:

            progress_float = float(batch_ind) / batch_count
            print("\rProgress: [{0:50s}] {1:.0f}%".
                  format('#' * int(progress_float * 50) + '-' * (50 - int(progress_float * 50)), progress_float * 100))

        if data_batch is None:
            continue

        data_batch = np.asarray(data_batch, dtype=np.float)
        with graph.as_default():
            result = model.predict_on_batch(data_batch)

        result_queue.put((result, np.array(coord)))

    print("\rProgress: [{0:50s}] 100%\n".format('#' * 50))

    result_queue.put(None)

