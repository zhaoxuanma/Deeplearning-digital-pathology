# Deeplearning-digital-pathology
This repository contains utilities for [virtual slides](https://openslide.org/) and images classification and semantic segmentation with [Keras](https://keras.io/) and [Caffe](http://caffe.berkeleyvision.org/) and an extension class of `ImageDataGenerator`  of Keras to generate batches of images with data augmentation for segmentation. 
Demo code is provided for reference.

# Requirement

```
Python 2.7
OpenCV 3.4
Numpy 1.14
Tensorflow 1.7
Keras 2.1
OpenSlide 1.1
Caffe 0.15 (optional)
```
# Getting started
**[slide_demo.py](slide_demo.py)**
It shows an example of using the Keras or Caffe model to segment a whole virtual slide or classify slide in grid and saving the results.

__Segmentation:__
![Alt text](Workflow_segmentation.png?raw=true "segmentation workflow")

__Classification:__
![Alt text](Workflow_classification.png?raw=true "classification workflow")

**[image_demo.py](image_demo.py)**
It shows an example of using the Keras or Caffe model to segment or classify images from a folder and saving the results.

**[train_segmentation_demo.py](train_segmentation_demo.py)**
It shows an example of training a segmentation model from scratch.


# Contents
__See comments or use `help` for detailed function usage.__

**[ImageDataGeneratorEXT.py](KerasLayers/ImageDataGeneratorEXT.py)**

An extension of `ImageDataGenerator` of Keras for semgentation images iteration , with similiar api of `flow_from_directory`.

The structure of the image directory would be like:

```
"directory"
├── "image_subfolder"
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── "mask_subfolder"
    ├── image1.png
    ├── image2.png
    └── ...
```

**[slide_utils.py](utils/slide_utils.py)**

Virtual slide class (`Slideobject`) for retrieving image batches and results reconstruction. 

+ `retrieve_tiles_to_queue_thread`
Retrieve image batches and put into a queue waiting for analysis. 

+ `reconstruct_segmentation_queue_to_level`
Reconstruct segmentation results on top of selected layer image.

+ `reconstruct_classification_queue_to_level`
Reconstruct classification results on top of selected layer image.

+ `preprocess_img`
Rewrite to include customized image preprocessing.

+ `gray_code`
Rewrite based on given code to change color for each category for `result_mask`.

+ `color_code`
Rewrite based on given code to change color for each category  for `result_rgb`.

**[image_utils.py](utils/image_utils.py)**

Image class (`Imageobject`) for retrieving image batches and results reconstruction. 

+ `retrieve_tiles_to_queue_thread`
Retrieve image batches and put into a queue waiting for analysis. 

+ `reconstruct_segmentation_queue_to_file`
Reconstruct segmentation results on top of original image and save.

+ `reconstruct_classification_queue_to_file`
Reconstruct classification results on top of original image and save.

+ `preprocess_img`
Rewrite to include customized image pre-processing.

+ `gray_code`
Rewrite based on given code to change color for each category for `result_mask`.

+ `color_code`
Rewrite based on given code to change color for each category  for `result_rgb`.


**[caffe_utils_queue.py](utils/caffe_utils_queue.py)**

Caffe class (`Caffeobject`) for forwarding images batch to neural network.

+ `forward_from_queue_to_queue`
Forward batch in data queue to neural network and put results in result queue.

**[tf_utils_queue.py](utils/tf_utils_queue.py)**

Tensorflow (Keras) class (`TFobject`) for forwarding images batch to neural network.

+ `forward_from_queue_to_queue`
Forward batch in data queue to neural network and put results in result queue.


# Related publications
+ Semantic Segmentation of Colon Glands in Inflammatory Bowel Disease Biopsies. Z Ma, Z Swiderska-Chadaj, N Ing, H Salemi, D McGovern, B Knudsen, A Gertych. Information Technologies in Biomedicine, Kamień Śląski, Poland, June 2018.
