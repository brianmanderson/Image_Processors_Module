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


class NormalizeBasedOnOther(ImageProcessor):
    def __init__(self, guiding_keys=('annotation',), changing_keys=('image',), reference_method=('reduce_max',),
                 changing_methods=('divide',)):
        self.guiding_keys, self.changing_keys = guiding_keys, changing_keys
        for method in reference_method:
            assert method in ('reduce_max', 'reduce_min'), 'Only provide a method of argmax, or argmin'
        for method in changing_methods:
            assert method in ('divide', 'multiply'), 'Only provide a method of argmax, or argmin'
        self.methods = reference_method
        self.changing_methods = changing_methods

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.guiding_keys)
        _check_keys_(input_features=input_features, keys=self.changing_keys)
        for guiding_key, changing_key, ref_method, change_method in zip(self.guiding_keys, self.changing_keys,
                                                                        self.methods, self.changing_methods):
            if ref_method == 'reduce_max':
                value = tf.reduce_max(input_features[guiding_key])
                if change_method == 'divide':
                    input_features[changing_key] = tf.divide(input_features[changing_key], value)
                elif change_method == 'multiply':
                    input_features[changing_key] = tf.multiply(input_features[changing_key], value)
            elif ref_method == 'reduce_min':
                value = tf.reduce_min(input_features[guiding_key])
                if change_method == 'divide':
                    input_features[changing_key] = tf.divide(input_features[changing_key], value)
                elif change_method == 'multiply':
                    input_features[changing_key] = tf.multiply(input_features[changing_key], value)
        return input_features


class DivideBasedOnOther(ImageProcessor):
    def __init__(self, guiding_keys=('annotation',), changing_keys=('image',)):
        self.guiding_keys, self.changing_keys = guiding_keys, changing_keys

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.guiding_keys)
        _check_keys_(input_features=input_features, keys=self.changing_keys)
        for guiding_key, changing_key in zip(self.guiding_keys, self.changing_keys):
            input_features[changing_key] = input_features[changing_key] / input_features[guiding_key]
        return input_features


if __name__ == '__main__':
    pass
