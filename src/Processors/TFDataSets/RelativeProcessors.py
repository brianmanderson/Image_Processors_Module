__author__ = 'Brian M Anderson'
# Created on 05/02/2022

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import tensorflow as tf
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
        parsed_features = {key: value for key, value in input_features.items()}
        for guiding_key, changing_key, guiding_value, mask_value, method in zip(self.guiding_keys, self.changing_keys,
                                                                                self.guiding_values, self.mask_values,
                                                                                self.methods):
            mask_value = tf.constant(mask_value, dtype=parsed_features[changing_key].dtype)
            if self.on_channel:
                val = tf.constant(1, dtype=parsed_features[guiding_key].dtype)
                if method == 'equal_to':
                    parsed_features[changing_key] = tf.where(parsed_features[guiding_key][..., guiding_value] == val,
                                                            mask_value, parsed_features[changing_key])
                elif method == 'less_than':
                    parsed_features[changing_key] = tf.where(parsed_features[guiding_key] < val,
                                                            mask_value, parsed_features[changing_key])
                elif method == 'greater_than':
                    parsed_features[changing_key] = tf.where(parsed_features[guiding_key] > val,
                                                            mask_value, parsed_features[changing_key])
            else:
                guiding_value = tf.constant(guiding_value, dtype=parsed_features[guiding_key].dtype)
                if method == 'equal_to':
                    parsed_features[changing_key] = tf.where(parsed_features[guiding_key] == guiding_value,
                                                            mask_value, parsed_features[changing_key])
                elif method == 'less_than':
                    parsed_features[changing_key] = tf.where(parsed_features[guiding_key] < guiding_value,
                                                            mask_value, parsed_features[changing_key])
                elif method == 'greater_than':
                    parsed_features[changing_key] = tf.where(parsed_features[guiding_key] > guiding_value,
                                                            mask_value, parsed_features[changing_key])
        return parsed_features


class AddMetricBasedOnImage(ImageProcessor):
    def __init__(self, image_keys=('annotation',), methods=('reduce_max',), out_key_names=('annotation_max',)):
        self.image_keys = image_keys
        self.methods = methods
        self.out_key_names = out_key_names

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        parsed_features = {key: value for key, value in input_features.items()}
        for image_key, ref_method, out_key in zip(self.image_keys, self.methods, self.out_key_names):
            if ref_method == 'reduce_max':
                value = tf.reduce_max(input_features[image_key])
                parsed_features[out_key] = value
            elif ref_method == 'reduce_min':
                value = tf.reduce_min(input_features[image_key])
                parsed_features[out_key] = value
        return parsed_features


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
        parsed_features = {key: value for key, value in input_features.items()}
        for guiding_key, changing_key, ref_method, change_method in zip(self.guiding_keys, self.changing_keys,
                                                                        self.methods, self.changing_methods):
            if ref_method == 'reduce_max':
                value = tf.reduce_max(parsed_features[guiding_key])
                if change_method == 'divide':
                    parsed_features[changing_key] = tf.divide(parsed_features[changing_key], value)
                elif change_method == 'multiply':
                    parsed_features[changing_key] = tf.multiply(parsed_features[changing_key], value)
            elif ref_method == 'reduce_min':
                value = tf.reduce_min(parsed_features[guiding_key])
                if change_method == 'divide':
                    parsed_features[changing_key] = tf.divide(parsed_features[changing_key], value)
                elif change_method == 'multiply':
                    parsed_features[changing_key] = tf.multiply(parsed_features[changing_key], value)
        return parsed_features


class DivideBasedOnOther(ImageProcessor):
    def __init__(self, guiding_keys=('annotation',), changing_keys=('image',)):
        self.guiding_keys, self.changing_keys = guiding_keys, changing_keys

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.guiding_keys)
        _check_keys_(input_features=input_features, keys=self.changing_keys)
        parsed_features = {key: value for key, value in input_features.items()}
        for guiding_key, changing_key in zip(self.guiding_keys, self.changing_keys):
            parsed_features[changing_key] = parsed_features[changing_key] / input_features[guiding_key]
        return parsed_features


if __name__ == '__main__':
    pass
