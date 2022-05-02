__author__ = 'Brian M Anderson'
# Created on 05/02/2022

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image, plt


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


class ImageProcessor(object):
    def parse(self, *args, **kwargs):
        return args, kwargs


class Decoder(object):
    def __init__(self, d_type_dict=None):
        self.d_type_dict = d_type_dict


class DecodeImagesAnnotations(Decoder):
    def parse(self, image_features, *args, **kwargs):
        all_keys = list(image_features.keys())
        is_modern = False
        for key in image_features.keys():
            if key.find('size') != -1:
                continue
            size_keys = [i for i in all_keys if i.find('size') != -1 and i.split('_size')[0] == key]  # All size keys
            size_keys.sort(key=lambda x: x.split('_')[-1])
            if size_keys:
                dtype = 'float'
                if key in self.d_type_dict:
                    dtype = self.d_type_dict[key]
                out_size = tuple([image_features[i] for i in size_keys])
                image_features[key] = tf.reshape(tf.io.decode_raw(image_features[key], out_type=dtype),
                                                 out_size)
                is_modern = True
        if not is_modern:  # To retain backwards compatibility
            print('Please update to the latest versions of the TFRecord maker')
            image_dtype = 'float'
            if 'image' in self.d_type_dict:
                image_dtype = self.d_type_dict['image']
            annotation_dtype = 'int8'
            if 'annotation' in self.d_type_dict:
                annotation_dtype = self.d_type_dict['annotation']
            if 'z_images' in image_features:
                if 'image' in image_features:
                    image_features['image'] = tf.reshape(tf.io.decode_raw(image_features['image'], out_type=image_dtype),
                                                         (image_features['z_images'], image_features['rows'],
                                                          image_features['cols']))
                if 'annotation' in image_features:
                    if 'num_classes' in image_features:
                        image_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                                   out_type=annotation_dtype),
                                                                  (image_features['z_images'], image_features['rows'],
                                                                   image_features['cols'], image_features['num_classes']))
                    else:
                        image_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                                   out_type=annotation_dtype),
                                                                  (image_features['z_images'], image_features['rows'],
                                                                   image_features['cols']))
            else:
                image_features['image'] = tf.reshape(tf.io.decode_raw(image_features['image'], out_type=image_dtype),
                                                     (image_features['rows'], image_features['cols']))
                if 'num_classes' in image_features:
                    image_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                               out_type=annotation_dtype),
                                                              (image_features['rows'], image_features['cols'],
                                                               image_features['num_classes']))
                else:
                    image_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                               out_type=annotation_dtype),
                                                              (image_features['rows'], image_features['cols']))
            if 'spacing' in image_features:
                spacing = tf.io.decode_raw(image_features['spacing'], out_type='float32')
                image_features['spacing'] = spacing
            if 'dose' in image_features:
                dose_dtype = 'float'
                if 'dose' in self.d_type_dict:
                    dose_dtype = self.d_type_dict['dose']
                image_features['dose'] = tf.reshape(tf.io.decode_raw(image_features['dose'], out_type=dose_dtype),
                                                    (image_features['dose_images'], image_features['dose_rows'],
                                                     image_features['dose_cols']))
        return image_features


class Decode_Images_Annotations(DecodeImagesAnnotations):
    def __init__(self, **kwargs):
        print('Please move from using Decode_Images_Annotations to DecodeImagesAnnotations, same arguments are passed')
        super().__init__(**kwargs)


class MaskOneBasedOnOther(ImageProcessor):
    def __init__(self, guiding_keys=('annotation',), changing_keys=('image',), guiding_values=(1,), mask_values=(-1,),
                 methods=('equal_to',), on_channel=False):
        """
        :param guiding_keys: keys which will guide the masking of another key
        :param changing_keys: keys which will be masked
        :param guiding_values: values which will define the mask
        :param mask_values: values which will be changed
        :param methods: method of masking, 'equal_to', 'less_than', 'greater_than'
        :param on_channel: binary, should we look at values or channels?
        """
        self.guiding_keys, self.changing_keys = guiding_keys, changing_keys
        self.guiding_values, self.mask_values = guiding_values, mask_values
        for method in methods:
            assert method in ('equal_to', 'less_than', 'greater_than'), 'Only provide a method of equal_to, ' \
                                                                        'less_than, or greater_than'
        self.methods = methods
        self.on_channel = on_channel

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.guiding_keys)
        _check_keys_(input_features=input_features, keys=self.changing_keys)
        for guiding_key, changing_key, guiding_value, mask_value, method in zip(self.guiding_keys, self.changing_keys,
                                                                                self.guiding_values, self.mask_values,
                                                                                self.methods):
            mask_value = tf.constant(mask_value, dtype=input_features[changing_key].dtype)
            if self.on_channel:
                val = tf.constant(1, dtype=input_features[guiding_key].dtype)
                if method == 'equal_to':
                    input_features[changing_key] = tf.where(input_features[guiding_key][..., guiding_value] == val,
                                                            mask_value, input_features[changing_key])
                elif method == 'less_than':
                    input_features[changing_key] = tf.where(input_features[guiding_key] < val,
                                                            mask_value, input_features[changing_key])
                elif method == 'greater_than':
                    input_features[changing_key] = tf.where(input_features[guiding_key] > val,
                                                            mask_value, input_features[changing_key])
            else:
                guiding_value = tf.constant(guiding_value, dtype=input_features[guiding_key].dtype)
                if method == 'equal_to':
                    input_features[changing_key] = tf.where(input_features[guiding_key] == guiding_value,
                                                            mask_value, input_features[changing_key])
                elif method == 'less_than':
                    input_features[changing_key] = tf.where(input_features[guiding_key] < guiding_value,
                                                            mask_value, input_features[changing_key])
                elif method == 'greater_than':
                    input_features[changing_key] = tf.where(input_features[guiding_key] > guiding_value,
                                                            mask_value, input_features[changing_key])
        return input_features


if __name__ == '__main__':
    pass
