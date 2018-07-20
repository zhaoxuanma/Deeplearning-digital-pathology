from keras.preprocessing.image import *
import keras.backend as K

import numpy as np
import os

class ImageDataGeneratorEXT(ImageDataGenerator):

    def __init__(self, *args, **kwargs):
        super(ImageDataGeneratorEXT, self).__init__(*args, **kwargs)

    def convert_to_onehot(self, x, classes, data_format):

        if np.amax(x) > classes:
            raise AssertionError('Mask data doesn\' match classes.')

        x = np.squeeze(x)
        x_ = (np.arange(classes) == x[..., None]).astype(dtype=K.floatx())

        if data_format == 'channels_first':
            x_ = np.rollaxis(x_,2,0)

        return x_


    def flow_from_directory_segmentation(self, directory,image_subfolder, mask_subfolder,
                                         target_size=(256, 256), color_mode='rgb',
                                         batch_size=32, shuffle=True, seed=None,
                                         save_to_dir=None,classes=None, 
                                         save_prefix='',
                                         save_format='png',
                                         follow_links=False):

        """Iterator capable of reading images from a directory on disk.

        Args
            directory: Path to the directory to read images from.
            image_subfolder: folder name containing images. # new
            mask_subfolder: folder name containing masks. # new
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
            batch_size: Integer, size of a batch.
            shuffle: Boolean, whether to shuffle the data between epochs.
            seed: Random seed for data shuffling.
            save_to_dir: Optional directory where to save the pictures being yielded, in a viewable format. This is
                useful for visualizing the random transformations being applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images (if `save_to_dir` is set).
            follow_links: Whether to follow symlinks inside class subdirectories (default: False).

        """

        return DirectoryIteratorSegmentation(directory,
                                             self,
                                             target_size=target_size, color_mode=color_mode,
                                             classes=classes,
                                             data_format=self.data_format,
                                             batch_size=batch_size, shuffle=shuffle, seed=seed,
                                             save_to_dir=save_to_dir,
                                             save_prefix=save_prefix,
                                             save_format=save_format,
                                             follow_links=follow_links,
                                             image_subfolder=image_subfolder,
                                             mask_subfolder=mask_subfolder
                                             )

class DirectoryIteratorSegmentation(Iterator):

    def __init__(self, directory, image_data_generator, image_subfolder, mask_subfolder,classes,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
                self.mask_shape = self.target_size + (classes,)
            else:
                self.image_shape = (3,) + self.target_size
                self.mask_shape = (classes,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
                self.mask_shape = self.target_size + (classes,)
            else:
                self.image_shape = (1,) + self.target_size
                self.mask_shape = (classes,) + self.target_size
        self.classes = classes
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        self.image_directory = os.path.join(directory, image_subfolder)
        self.mask_directory = os.path.join(directory, mask_subfolder)

        # first, count the number of samples and classes
        self.samples, self.image_filenames, self.mask_filenames = _list_valid_filenames_in_directory_segmentation(
            directory=directory,white_list_formats=white_list_formats,follow_links=follow_links,
            image_subfolder = image_subfolder, mask_subfolder= mask_subfolder)

        print('Found %d images' % (self.samples))
        image_data_generator.samples = self.samples

        super(DirectoryIteratorSegmentation, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array),)+ self.mask_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.image_filenames[j]
            img = load_img(os.path.join(self.image_directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            # x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            fname = self.mask_filenames[j]
            img = load_img_indexed(os.path.join(self.mask_directory, fname),
                                   target_size=self.target_size)
            y = img_to_array(img, data_format=self.data_format)
            y = self.image_data_generator.convert_to_onehot(y, self.classes, self.data_format)
            batch_y[i] = y

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in range(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

def _list_valid_filenames_in_directory_segmentation(directory, image_subfolder, mask_subfolder, white_list_formats, follow_links):
    """Count files with extension in `white_list_formats` contained in a directory.

    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0
    image_filenames = []
    mask_filenames = []

    imagedirectory = os.path.join(directory, image_subfolder)
    maskdirectory = os.path.join(directory, mask_subfolder)

    image_ext = None
    mask_ext = None
    for root, _, files in _recursive_list(imagedirectory):
        for fname in files:
            is_valid1 = False
            if image_ext is None:
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        image_ext = extension
                        is_valid1 = True
                        break
            else:
                if fname.lower().endswith('.' + image_ext):
                    is_valid1 = True

            is_valid2 = False
            if is_valid1:
                if mask_ext is None:
                    for extension_ in white_list_formats:
                        if os.path.isfile(os.path.join(maskdirectory, os.path.splitext(fname)[0] + '.' + extension_)):
                            mask_ext = extension_
                            is_valid2 = True
                            break
                else:
                    if os.path.isfile(os.path.join(maskdirectory, os.path.splitext(fname)[0] + '.' + mask_ext)):
                        is_valid2 = True

            if is_valid1 and is_valid2:
                image_filenames.append(fname)
                mask_filenames.append(os.path.splitext(fname)[0] + '.' + mask_ext)
                samples += 1

    return samples, image_filenames, mask_filenames

def load_img_indexed(path, target_size=None):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img

