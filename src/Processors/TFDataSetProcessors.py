__author__ = 'Brian M Anderson'
# Created on 3/5/2021

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
    @tf.function
    def parse(self, image_features, *args, **kwargs):
        parsed_features = {}
        all_keys = list(image_features.keys())
        is_modern = False
        for key in image_features.keys():
            if key.find('size') != -1:
                continue

            # Find size keys for the current key
            size_keys = [i for i in all_keys if i.find('size') != -1 and i.split('_size')[0] == key]
            size_keys.sort(key=lambda x: x.split('_')[-1])

            if size_keys:
                dtype = 'float'
                if key in self.d_type_dict:
                    dtype = self.d_type_dict[key]

                out_size = tuple([image_features[i] for i in size_keys])

                # Add the decoded feature to the new dictionary
                parsed_features[key] = tf.reshape(tf.io.decode_raw(image_features[key], out_type=dtype), out_size)
                is_modern = True
            else:
                # If no size keys, copy the original feature
                parsed_features[key] = image_features[key]

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
                    parsed_features['image'] = value = tf.reshape(tf.io.decode_raw(image_features['image'], out_type=image_dtype),
                                                         (image_features['z_images'], image_features['rows'],
                                                          image_features['cols']))
                if 'annotation' in image_features:
                    if 'num_classes' in image_features:
                        parsed_features['annotation'] = value = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                                   out_type=annotation_dtype),
                                                                  (image_features['z_images'], image_features['rows'],
                                                                   image_features['cols'], image_features['num_classes']))
                    else:
                        parsed_features['annotation'] = value = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                                   out_type=annotation_dtype),
                                                                  (image_features['z_images'], image_features['rows'],
                                                                   image_features['cols']))
            else:
                parsed_features['image'] = tf.reshape(tf.io.decode_raw(image_features['image'], out_type=image_dtype),
                                                     (image_features['rows'], image_features['cols']))
                if 'num_classes' in image_features:
                    parsed_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                               out_type=annotation_dtype),
                                                              (image_features['rows'], image_features['cols'],
                                                               image_features['num_classes']))
                else:
                    parsed_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                               out_type=annotation_dtype),
                                                              (image_features['rows'], image_features['cols']))
            if 'spacing' in image_features:
                spacing = tf.io.decode_raw(image_features['spacing'], out_type='float32')
                parsed_features['spacing'] = spacing
            if 'dose' in image_features:
                dose_dtype = 'float'
                if 'dose' in self.d_type_dict:
                    dose_dtype = self.d_type_dict['dose']
                parsed_features['dose'] = tf.reshape(tf.io.decode_raw(image_features['dose'], out_type=dose_dtype),
                                                    (image_features['dose_images'], image_features['dose_rows'],
                                                     image_features['dose_cols']))
        return parsed_features


class Decode_Images_Annotations(DecodeImagesAnnotations):
    def __init__(self, **kwargs):
        print('Please move from using Decode_Images_Annotations to DecodeImagesAnnotations, same arguments are passed')
        super().__init__(**kwargs)


class Random_Noise(ImageProcessor):
    def __init__(self, max_noise=2.5, wanted_keys=['image']):
        '''
        Return the image feature with an additive noise randomly weighted between [0.0, max_noise)
        :param max_noise: maximum magnitude of the noise in HU (apply before normalization)
        '''
        self.max_noise = max_noise
        self.wanted_keys = wanted_keys

    def parse(self, image_features, *args, **kwargs):
        for key in self.wanted_keys:
            if key in image_features:
                data = image_features[key]
                dtype = data.dtype
                data = tf.cast(data, 'float32')
                data += tf.random.uniform(shape=[], minval=0.0, maxval=self.max_noise,
                                          dtype='float32') * tf.random.normal(tf.shape(image_features['image']),
                                                                              mean=0.0, stddev=1.0, dtype='float32')
                data = tf.cast(data, dtype)
                image_features[key] = data
        return image_features


class CombineKeys(ImageProcessor):
    def __init__(self, image_keys=('primary_image', 'secondary_image'), output_key='combined', axis=-1):
        self.image_keys = image_keys
        self.output_key = output_key
        self.axis = axis

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.image_keys)
        combine_images = [image_features[i] for i in self.image_keys]
        image_features[self.output_key] = tf.concat(combine_images, axis=self.axis)
        return image_features


class Combine_image_RT_Dose(ImageProcessor):
    def parse(self, input_features, *args, **kwargs):
        image = input_features['image']
        rt = input_features['annotation']
        dose = input_features['dose']
        output = tf.concat([image, rt, dose], axis=-1)
        input_features['combined'] = output
        return input_features


class Fuzzy_Segment_Liver_Lobes(ImageProcessor):
    def __init__(self, min_val=0, max_val=None, num_classes=9):
        '''
        :param variation: margin to expand region, mm. np.arange(start=0, stop=1, step=1), in mm
        '''
        self.min_val = min_val
        self.max_val = max_val
        self.num_classes = num_classes

    def parse(self, image_features, *args, **kwargs):
        if type(image_features) is dict:
            annotation = image_features['annotation']
        else:
            annotation = image_features[-1][-1]
        annotation = tf.cast(annotation, dtype=tf.dtypes.float32)
        filter_size = tf.random.uniform([2], minval=self.min_val, maxval=self.max_val)
        filter_shape = tuple(tf.cast(tf.divide(filter_size, image_features['spacing'][:2]), dtype=tf.dtypes.float32))

        # Explicitly pad the image

        filter_height, filter_width = filter_shape
        pad_top = (filter_height - 1) // 2
        pad_bottom = filter_height - 1 - pad_top
        pad_left = (filter_width - 1) // 2
        pad_right = filter_width - 1 - pad_left
        paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
        annotation = tf.pad(annotation, paddings, mode='CONSTANT')

        # Filter of shape (filter_width, filter_height, in_channels, 1)
        # has the value of 1 for each element.
        area = tf.math.reduce_prod(filter_shape)
        filter_shape += (tf.shape(annotation)[-1], 1)
        kernel = tf.ones(shape=filter_shape, dtype='float32')

        annotation = tf.nn.depthwise_conv2d(annotation, kernel, strides=(1, 1, 1, 1), padding="VALID")
        annotation = tf.divide(annotation, area)
        annotation = tf.divide(annotation, tf.expand_dims(tf.reduce_sum(annotation, axis=-1), axis=-1))
        image_features['annotation'] = annotation
        return image_features


class ReturnOutputs(ImageProcessor):
    """
    This should be your final image processor, this will turn your dictionary into a set of tensors
    """
    def __init__(self, input_keys=('image',), output_keys=('annotation',)):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def parse(self, image_features, *args, **kwargs):
        inputs = []
        outputs = []
        _check_keys_(input_features=image_features, keys=self.input_keys + self.output_keys)
        for key in self.input_keys:
            inputs.append(image_features[key])
        for key in self.output_keys:
            outputs.append(image_features[key])
        del image_features
        return tuple(inputs), tuple(outputs)


class Return_Outputs(ImageProcessor):
    def __init__(self, wanted_keys_dict={'inputs': ('image',), 'outputs': ('annotation',)}):
        assert type(wanted_keys_dict) is dict, 'You need to pass a dictionary to Return_Outputs in the form of ' \
                                               '{"inputs":["image"],"outputs":["annotation"]}, etc.'
        self.wanted_keys_dict = wanted_keys_dict
        print('Return_Outputs is deprecated! Please move to ReturnOutputs and specifically define desired '
              'input_keys and output_keys')

    def parse(self, image_features, *args, **kwargs):
        inputs = []
        outputs = []
        for key in self.wanted_keys_dict['inputs']:
            if key in image_features:
                inputs.append(image_features[key])
            else:
                print('WARNING\n\n\n{} not in image_features\n\n\n'.format(key))
        for key in self.wanted_keys_dict['outputs']:
            if key in image_features:
                outputs.append(image_features[key])
            else:
                print('WARNING\n\n\n{} not in image_features\n\n\n'.format(key))
        del image_features
        return tuple(inputs), tuple(outputs)


class Resize_Images(ImageProcessor):
    def __init__(self, image_rows=512, image_cols=512):
        self.image_rows = tf.constant(image_rows)
        self.image_cols = tf.constant(image_cols)

    def parse(self, image_features, *args, **kwargs):
        assert len(image_features['image'].shape) > 2, 'You should do an expand_dimensions before this!'
        image_features['image'] = tf.image.resize(image_features['image'], size=(self.image_rows, self.image_cols),
                                                  method='bilinear', preserve_aspect_ratio=True)
        image_features['annotation'] = tf.image.resize(image_features['annotation'],
                                                       size=(self.image_rows, self.image_cols),
                                                       method='nearest', preserve_aspect_ratio=True)
        return image_features


class Pad_Z_Images_w_Reflections(ImageProcessor):
    '''
    This will not work for parallelized.. because the z dimension is None unknown to start
    '''

    def __init__(self, z_images=32):
        self.z_images = tf.constant(z_images)

    def parse(self, image_features, *args, **kwargs):
        dif = tf.subtract(self.z_images, image_features['image'].shape[0])
        image_features['image'] = tf.concat(
            [image_features['image'], tf.reverse(image_features['image'], axis=[0])[:dif]], axis=0)
        image_features['annotation'] = tf.concat(
            [image_features['annotation'], tf.reverse(image_features['annotation'], axis=[0])[:dif]], axis=0)
        return image_features


class RandomCrop(ImageProcessor):
    def __init__(self, keys_to_crop=('image_array', 'annotation_array'), crop_dimensions=((32, 32, 32, 1),
                                                                                          (32, 32, 32, 2))):
        self.keys_to_crop = keys_to_crop
        self.crop_dimensions = crop_dimensions

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.keys_to_crop)
        seed = tf.random.uniform(shape=[2], maxval=99999, dtype=tf.dtypes.int32)
        for key, crop_dimensions in zip(self.keys_to_crop, self.crop_dimensions):
            image = input_features[key]
            image = tf.image.stateless_random_crop(value=image, size=crop_dimensions, seed=seed)
            input_features[key] = image
        return input_features


class Ensure_Image_Proportions(ImageProcessor):
    def __init__(self, image_rows=512, image_cols=512, preserve_aspect_ratio=False):
        self.image_rows = tf.constant(image_rows)
        self.image_cols = tf.constant(image_cols)
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def parse(self, image_features, *args, **kwargs):
        assert len(image_features['image'].shape) > 2, 'You should do an expand_dimensions before this!'
        image_features['image'] = tf.image.resize(image_features['image'], (self.image_rows, self.image_cols),
                                                  preserve_aspect_ratio=self.preserve_aspect_ratio)
        image_features['image'] = tf.image.resize_with_crop_or_pad(image_features['image'],
                                                                   target_width=self.image_cols,
                                                                   target_height=self.image_rows)
        annotation = image_features['annotation']
        method = 'bilinear'
        if annotation.dtype.name.find('int') != -1:
            method = 'nearest'
        annotation = tf.image.resize(annotation, (self.image_rows, self.image_cols),
                                     preserve_aspect_ratio=self.preserve_aspect_ratio, method=method)
        annotation = tf.image.resize_with_crop_or_pad(annotation, target_width=self.image_cols,
                                                      target_height=self.image_rows)
        if annotation.shape[-1] != 1:
            annotation = annotation[..., 1:]  # remove background
            background = tf.expand_dims(1 - tf.reduce_sum(annotation, axis=-1), axis=-1)
            annotation = tf.concat([background, annotation], axis=-1)
        image_features['annotation'] = annotation
        return image_features


class Ensure_Annotation_Range(ImageProcessor):
    def parse(self, image_features, *args, **kwargs):
        annotation = image_features['annotation']
        annotation = tf.divide(annotation, tf.expand_dims(tf.reduce_sum(annotation, axis=-1), axis=-1))
        image_features['annotation'] = annotation
        return image_features


class Return_Add_Mult_Disease(ImageProcessor):
    def __init__(self, on_disease=True, change_background=False, cast_to_min=False):
        self.on_disease = on_disease
        self.cast_to_min = cast_to_min
        self.change_background = change_background

    def parse(self, image_features, *args, **kwargs):
        annotation = image_features['annotation']
        if annotation.shape[-1] != 1:
            mask = tf.expand_dims(tf.where(tf.cast(tf.reduce_sum(annotation[..., 1:], axis=-1), 'float16') > .99, 1, 0),
                                  axis=-1)
            if self.on_disease:
                annotation = tf.expand_dims(annotation[..., 2], axis=-1)  # Kick out everything except for the disease
                image_features['annotation'] = annotation
        else:
            mask = tf.where(annotation > 0, 1, 0)
            if self.on_disease:
                annotation = tf.where(annotation == 2, 1, 0)
                image_features['annotation'] = annotation
        image_features['mask'] = mask
        if self.change_background:
            value = 0
            if self.cast_to_min:
                value = tf.reduce_min(image_features['image'])
            image_features['image'] = tf.where(mask == 0, tf.cast(value, dtype=image_features['image'].dtype),
                                               image_features['image'])
        return image_features


class Combine_Liver_Lobe_Segments(ImageProcessor):
    '''
    Combines segments 5, 6, 7 and 8 into 5
    '''

    def parse(self, image_features, *args, **kwargs):
        annotation = image_features['annotation']
        output = [tf.expand_dims(annotation[..., i], axis=-1) for i in range(5)]
        output.append(tf.expand_dims(tf.reduce_sum(annotation[..., 5:], axis=-1), axis=-1))
        output = tf.concat(output, axis=-1)
        image_features['annotation'] = output
        return image_features


class ToCategorical(ImageProcessor):
    def __init__(self, annotation_keys=('annotation',), number_of_classes=(2,)):
        self.annotation_keys = annotation_keys
        self.number_of_classes = number_of_classes

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.annotation_keys)
        for key, num_classes in zip(self.annotation_keys, self.number_of_classes):
            y = input_features[key]
            input_features[key] = tf.cast(tf.one_hot(tf.cast(y, tf.int32), num_classes), dtype=y.dtype)
        return input_features


class Squeeze(ImageProcessor):
    def __init__(self, image_keys=('image',)):
        """
        Designed to squeeze tf arrays
        :param image_keys:
        """
        self.image_keys = image_keys
    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for key in self.image_keys:
            image_features[key] = tf.squeeze(image_features[key])
        return image_features


class ExpandDimension(ImageProcessor):
    def __init__(self, axis=-1, image_keys=('image', 'annotation')):
        self.axis = axis
        self.image_keys = image_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for key in self.image_keys:
            image_features[key] = tf.expand_dims(image_features[key], axis=self.axis)
        return image_features


class Expand_Dimensions(ImageProcessor):
    def __init__(self, axis=-1, on_images=True, on_annotations=False):
        self.axis = axis
        self.on_images = on_images
        self.on_annotations = on_annotations

    def parse(self, image_features, *args, **kwargs):
        if self.on_images:
            image_features['image'] = tf.expand_dims(image_features['image'], axis=self.axis)
        if self.on_annotations:
            image_features['annotation'] = tf.expand_dims(image_features['annotation'], axis=self.axis)
        return image_features


class Repeat_Channel(ImageProcessor):
    def __init__(self, axis=-1, repeats=3, on_images=True, on_annotations=False):
        '''
        :param axis: axis to expand
        :param repeats: number of repeats
        :param on_images: expand the axis on the images
        :param on_annotations: expand the axis on the annotations
        '''
        self.axis = axis
        self.repeats = repeats
        self.on_images = on_images
        self.on_annotations = on_annotations

    def parse(self, image_features, *args, **kwargs):
        if self.on_images:
            image_features['image'] = tf.repeat(image_features['image'], axis=self.axis, repeats=self.repeats)
        if self.on_annotations:
            image_features['annotation'] = tf.repeat(image_features['annotation'], axis=self.axis, repeats=self.repeats)
        return image_features


class Return_Lung(ImageProcessor):
    def __init__(self, dual_output=False):
        self.dual_output = dual_output

    def parse(self, image_features, *args, **kwargs):
        if self.dual_output:
            image_features['lung'] = tf.cast(image_features['annotation'] > 0, dtype='float32')
        return image_features


class MultiplyImagesByConstant(ImageProcessor):
    def __init__(self, keys=('image',), values=(1,)):
        """
        :param keys: tuple of keys for addition
        :param values: tuple of values for addition
        """
        self.keys = keys
        self.values = values

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key, value in zip(self.keys, self.values):
            image_features[key] = tf.multiply(image_features[key], tf.cast(value, image_features[key].dtype))
        return image_features


class Add_Constant(ImageProcessor):
    def __init__(self, keys=('image',), values=(0,)):
        """
        :param keys: tuple of keys for addition
        :param values: tuple of values for addition
        """
        self.keys = keys
        self.values = values

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key, value in zip(self.keys, self.values):
            image_features[key] = tf.add(image_features[key], tf.cast(value, image_features[key].dtype))
        return image_features


class AddConstantToImages(Add_Constant):
    def __init__(self, keys=('image',), values=(0,)):
        """
        :param keys: tuple of keys for addition
        :param values: tuple of values for addition
        """
        super().__init__(keys=keys, values=values)


class V3Normalize(ImageProcessor):
    def __init__(self):
        '''
        Normalizes a 255. image to values trained on pascal
        '''
        self.means = tf.constant([-123.68, -116.779, -103.939])

    def parse(self, image_features, *args, **kwargs):
        image_features['image'] = tf.add(image_features['image'], self.means)
        return image_features


class Normalize_Images(ImageProcessor):
    def __init__(self, keys=('image',), mean_values=(0,), std_values=(1,),):
        """
        :param keys: tuple of image keys
        :param mean_values: tuple of mean values
        :param std_values: tuple of standard deviations
        """
        self.keys = keys
        self.mean_values = mean_values
        self.std_values = std_values

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        for key, mean_val, std_val in zip(self.keys, self.mean_values, self.std_values):
            mean_val = tf.constant(mean_val, dtype=image_features[key].dtype)
            std_val = tf.constant(std_val, dtype=image_features[key].dtype)
            image_features[key] = (image_features[key] - mean_val) / std_val
        return image_features


class ArgMax(ImageProcessor):
    def __init__(self, annotation_keys=('annotation',), axis=-1):
        """
        :param annotation_keys: tuple of keys to perform arg_max across
        :param axis: axis across which to arg max
        """
        self.axis = axis
        self.annotation_keys = annotation_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.annotation_keys)
        for key in self.annotation_keys:
            image_features[key] = tf.argmax(image_features[key], axis=self.axis)
        return image_features


class CombineAnnotations(ImageProcessor):
    def __init__(self, key_tuple=('annotation',), from_values_tuple=(2,), to_values_tuple=(1,)):
        """
        :param key_tuple: tuple of key names that will be present in image_features
        :param from_values_tuple: tuple of values that we will change from
        :param to_values_tuple: tuple of values that we will change to
        """
        self.key_list = key_tuple
        self.from_list = from_values_tuple
        self.to_list = to_values_tuple

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.key_list)
        for key, from_value, to_value in zip(self.key_list, self.from_list, self.to_list):
            from_tensor = tf.constant(from_value, dtype=image_features[key].dtype)
            to_tensor = tf.constant(to_value, dtype=image_features[key].dtype)
            image_features[key] = tf.where(image_features[key] == from_tensor, to_tensor, image_features[key])
        return image_features


class ReturnSingleChannel(ImageProcessor):
    def __init__(self, image_keys=('annotation', ), channels=(1, ), out_keys=('annotation', )):
        """
        :param image_keys: tuple of image keys
        :param channels: tuple of channels to withdraw
        :param out_keys: tuple of image keys to be named. Same name will rewrite
        """
        self.image_keys = image_keys
        self.channels = channels
        self.out_keys = out_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.image_keys)
        for key, channel, new_key in zip(self.image_keys, self.channels, self.out_keys):
            image_features[new_key] = image_features[key][..., channel]
        return image_features


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


class MaskKeys(CombineAnnotations):
    def __init__(self, key_tuple=('annotation',), from_values_tuple=(2,), to_values_tuple=(1,)):
        super().__init__(key_tuple=key_tuple, from_values_tuple=from_values_tuple, to_values_tuple=to_values_tuple)


class Combined_Annotations(ImageProcessor):
    def __init__(self, values=[tf.constant(1, dtype='int8'), tf.constant(2, dtype='int8')]):
        self.values = values

    def parse(self, image_features, *args, **kwargs):
        for value in self.values:
            value = tf.constant(value, dtype=image_features['annotation'].dtype)
            image_features['annotation'] = tf.where(image_features['annotation'] == value,
                                                    tf.constant(1, dtype=image_features['annotation'].dtype),
                                                    image_features['annotation'])
        return image_features


class CreateDiseaseKey(ImageProcessor):
    def parse(self, image_features, *args, **kwargs):
        value = tf.constant(1, dtype=image_features['primary_liver'].dtype)
        image_features['disease'] = tf.where(image_features['primary_liver'] > value,
                                             value,
                                             tf.constant(0, dtype=image_features['primary_liver'].dtype))
        return image_features


class CreateNewKeyFromArgSum(ImageProcessor):
    def __init__(self, guiding_keys=('annotation', ), new_keys=('mask',)):
        """
        :param guiding_keys: keys which will guide the masking of another key
        :param new_keys: keys which will be masked
        """
        self.guiding_keys = guiding_keys
        self.new_keys = new_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.guiding_keys)
        for guiding_key, new_key in zip(self.guiding_keys, self.new_keys):
            annotation = image_features[guiding_key]
            mask = tf.expand_dims(tf.reduce_sum(annotation[..., 1:], axis=-1), axis=-1)
            image_features[new_key] = mask
        return image_features


class Cast_Data(ImageProcessor):
    def __init__(self, keys=('image', 'annotation',), dtypes=('float16', 'float16')):
        """
        :param keys: tuple of keys
        :param dtypes: tuple of datatypes
        """
        self.keys = keys
        self.dtypes = dtypes

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key, dtype in zip(self.keys, self.dtypes):
            image_features[key] = tf.cast(image_features[key], dtype=dtype)
        return image_features


def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.

    Args:
    image: original image size
    result: flipped or transformed image

    Returns:
    An image whose shape is at least (None, None, None).
    """
    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result


def _random_flip(image, flip_index, seed, scope_name, flip_3D_together=False):
    """Randomly (50% chance) flip an image along axis `flip_index`.

    Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    flip_index: Dimension along which to flip the image.
      Vertical: 0, Horizontal: 1
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.
    scope_name: Name of the scope in which the ops are added.

    Returns:
    A tensor of the same type and shape as `image`.

    Raises:
    ValueError: if the shape of `image` not supported.
    """
    with ops.name_scope(None, scope_name, [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        shape = image.get_shape()
        if shape.ndims == 3 or shape.ndims is None:
            uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
            mirror_cond = math_ops.less(uniform_random, .5)
            result = control_flow_ops.cond(
                mirror_cond,
                lambda: array_ops.reverse(image, [flip_index]),
                lambda: image,
                name=scope)
            return fix_image_flip_shape(image, result)
        elif shape.ndims == 4:
            batch_size = array_ops.shape(image)[0]
            if flip_3D_together:
                uniform_random = array_ops.repeat(random_ops.random_uniform([1], 0, 1.0, seed=seed), batch_size)
            else:
                uniform_random = random_ops.random_uniform([batch_size], 0, 1.0, seed=seed)
            flips = math_ops.round(
                array_ops.reshape(uniform_random, [batch_size, 1, 1, 1]))
            flips = math_ops.cast(flips, image.dtype)
            flipped_input = array_ops.reverse(image, [flip_index + 1])
            return flips * flipped_input + (1 - flips) * image
        else:
            raise ValueError('\'image\' must have either 3 or 4 dimensions.')


class Flip_Images(ImageProcessor):
    def __init__(self, keys=('image', 'annotation'), flip_lr=True, flip_up_down=True, flip_z=False,
                 flip_3D_together=False, threshold=0.5):
        self.flip_lr = flip_lr
        self.flip_z = flip_z
        self.flip_up_down = flip_up_down
        self.keys = keys
        self.flip_3D_together = flip_3D_together
        self.threshold = threshold

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        if self.flip_lr:
            uniform_random = None
            flip_index = 1
            for key in self.keys:
                assert key in image_features.keys(), 'You need to pass correct keys in dictionary!'
                image = image_features[key]
                shape = image.get_shape()
                if shape.ndims != 3 and shape.ndims is not None:
                    flip_index = 2
                if uniform_random is None:
                    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=None)
                mirror_cond = math_ops.less(uniform_random, self.threshold)
                result = control_flow_ops.cond(
                    mirror_cond,
                    lambda: array_ops.reverse(image, [flip_index]),
                    lambda: image)
                image_features[key] = fix_image_flip_shape(image, result)
        if self.flip_up_down:
            uniform_random = None
            flip_index = 0
            for key in self.keys:
                assert key in image_features.keys(), 'You need to pass correct keys in dictionary!'
                image = image_features[key]
                shape = image.get_shape()
                if shape.ndims != 3 and shape.ndims is not None:
                    flip_index = 1
                if uniform_random is None:
                    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=None)
                mirror_cond = math_ops.less(uniform_random, self.threshold)
                result = control_flow_ops.cond(
                    mirror_cond,
                    lambda: array_ops.reverse(image, [flip_index]),
                    lambda: image)
                image_features[key] = fix_image_flip_shape(image, result)
        if self.flip_z:
            uniform_random = None
            for key in self.keys:
                assert key in image_features.keys(), 'You need to pass correct keys in dictionary!'
                image = image_features[key]
                if uniform_random is None:
                    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=None)
                mirror_cond = math_ops.less(uniform_random, self.threshold)
                result = control_flow_ops.cond(
                    mirror_cond,
                    lambda: array_ops.reverse(image, [0]),
                    lambda: image)
                image_features[key] = fix_image_flip_shape(image, result)
        return image_features


class Threshold_Images(ImageProcessor):
    def __init__(self, keys=('image',), lower_bounds=(-np.inf,), upper_bounds=(np.inf,), divides=(True,)):
        """
        :param keys: tuple of image keys
        :param lower_bounds: tuple of bounds
        :param upper_bounds: tuple of bounds
        :param divides: boolean if you want to divide
        """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.keys = keys
        self.divides = divides

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        for key, lower_bound, upper_bound, divide in zip(self.keys, self.lower_bounds, self.upper_bounds, self.divides):
            image_features[key] = tf.where(image_features[key] > tf.cast(upper_bound, dtype=image_features[key].dtype),
                                           tf.cast(upper_bound, dtype=image_features[key].dtype), image_features[key])
            image_features[key] = tf.where(image_features[key] < tf.cast(lower_bound, dtype=image_features[key].dtype),
                                           tf.cast(lower_bound, dtype=image_features[key].dtype), image_features[key])
            if divide:
                image_features[key] = tf.divide(image_features[key], tf.cast(tf.subtract(upper_bound, lower_bound),
                                                                             dtype=image_features[key].dtype))
        return image_features


class Resize_with_crop_pad(ImageProcessor):
    def __init__(self, keys=('image', 'annotation'), image_rows=(512, 512), image_cols=(512, 512),
                 is_mask=(False, True)):
        """
        :param keys: tuple of keys
        :param image_rows: tuple of image rows
        :param image_cols: tuple of image columns
        :param is_mask: boolean for ensuring mask is correct
        """
        print("Be careful..Resize with crop/pad can severely slow down data retrieval, best to do this and cache")
        self.keys = keys
        self.is_mask = is_mask
        self.image_rows = image_rows
        self.image_cols = image_cols

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key, image_rows, image_cols, is_mask in zip(self.keys, self.image_rows, self.image_cols, self.is_mask):
            image_rows = tf.constant(image_rows)
            image_cols = tf.constant(image_cols)
            image_features[key] = tf.image.resize_with_crop_or_pad(image_features[key],
                                                                   target_width=image_cols,
                                                                   target_height=image_rows)
            if is_mask and image_features[key].shape[-1] != 1:
                array = image_features[key]
                array = array[..., 1:]  # remove background
                background = tf.expand_dims(1 - tf.reduce_sum(array, axis=-1), axis=-1)
                array = tf.concat([background, array], axis=-1)
                image_features[key] = array
        return image_features


class Clip_Images(ImageProcessor):
    def __init__(self, annotations_index=None, bounding_box_expansion=(10, 10, 10), power_val_z=1, power_val_r=1,
                 power_val_c=1, min_images=0, min_rows=0, min_cols=0):
        self.annotations_index = annotations_index
        self.bounding_box_expansion = tf.convert_to_tensor(bounding_box_expansion)
        self.power_val_z, self.power_val_r, self.power_val_c = tf.constant(power_val_z), tf.constant(
            power_val_r), tf.constant(power_val_c)
        self.min_images, self.min_rows, self.min_cols = tf.constant(min_images), tf.constant(min_rows), tf.constant(
            min_cols)

    def parse(self, image_features, *args, **kwargs):
        zero = tf.constant(0)
        image = image_features['image']
        annotation = image_features['annotation']
        img_shape = tf.shape(image)
        if self.annotations_index:
            bounding_box = image_features['bounding_boxes_{}'.format(self.annotations_index)][
                0]  # Assuming one bounding box
            c_start, r_start, z_start = bounding_box[0], bounding_box[1], bounding_box[2]
            c_stop, r_stop, z_stop = c_start + bounding_box[3], r_start + bounding_box[4], z_start + bounding_box[5]
            z_start = tf.maximum(zero, z_start - self.bounding_box_expansion[0])
            z_stop = tf.minimum(z_stop + self.bounding_box_expansion[0], img_shape[0])
            r_start = tf.maximum(zero, r_start - self.bounding_box_expansion[1])
            r_stop = tf.minimum(img_shape[1], r_stop + self.bounding_box_expansion[1])
            c_start = tf.maximum(zero, c_start - self.bounding_box_expansion[2])
            c_stop = tf.minimum(img_shape[2], c_stop + self.bounding_box_expansion[2])
        else:
            z_stop, r_stop, c_stop = img_shape[:3]
            z_start, r_start, c_start = 0, 0, 0
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start

        remainder_z = tf.math.floormod(self.power_val_z - z_total, self.power_val_z) if tf.math.floormod(z_total,
                                                                                                         self.power_val_z) != zero else zero
        remainder_r = tf.math.floormod(self.power_val_r - r_total, self.power_val_r) if tf.math.floormod(r_total,
                                                                                                         self.power_val_r) != zero else zero
        remainder_c = tf.math.floormod(self.power_val_c - r_total, self.power_val_c) if tf.math.floormod(r_total,
                                                                                                         self.power_val_c) != zero else zero
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        min_images = tf.maximum(self.min_images, min_images)
        min_rows = tf.maximum(self.min_rows, min_rows)
        min_cols = tf.maximum(self.min_cols, min_cols)
        output_dims = tf.convert_to_tensor([min_images, min_rows, min_cols])
        image_cube = image[z_start:z_start + min_images, r_start:r_start + min_rows, c_start:c_start + min_cols, ...]
        annotation_cube = annotation[z_start:z_start + min_images, r_start:r_start + min_rows,
                          c_start:c_start + min_cols]
        img_shape = tf.shape(image_cube)
        size_dif = output_dims - img_shape[:3]
        if tf.reduce_max(size_dif) > 0:
            paddings = tf.convert_to_tensor(
                [[size_dif[0], zero], [size_dif[1], zero], [size_dif[2], zero], [zero, zero]])
            image_cube = tf.pad(image_cube, paddings=paddings, constant_values=tf.reduce_min(image))
            annotation_cube = tf.pad(annotation_cube, paddings=paddings)
        image_features['image'] = image_cube
        image_features['annotation'] = annotation_cube
        return image_features


class Pull_Subset(ImageProcessor):
    def __init__(self, max_batch=32):
        self.max_batch = max_batch

    def parse(self, images, annotations, *args, **kwargs):
        num_images = images.shape[0]
        if num_images > self.max_batch:
            random_index = tf.random.uniform(shape=[], minval=0, maxval=num_images - self.max_batch, dtype='int32')
            images = images[random_index:random_index + self.max_batch, ...]
            annotations = annotations[random_index:random_index + self.max_batch, ...]
        return images, annotations


class Pull_Bounding_Box(ImageProcessor):
    def __init__(self, annotation_index=None, max_cubes=10, z_images=16, rows=100, cols=100, min_volume=0, min_voxels=0,
                 max_volume=np.inf, max_voxels=np.inf):
        '''
        annotation_index = scalar value referring to annotation value for bbox
        '''
        self.annotation_index = annotation_index
        self.max_cubes = tf.constant(max_cubes)
        self.z_images, self.rows, self.cols = tf.constant(z_images), tf.constant(rows), tf.constant(cols)
        if min_volume != 0 and min_voxels != 0:
            raise AssertionError('Cannot have both min_volume and min_voxels specified')
        if max_volume != np.inf and min_voxels != np.inf:
            raise AssertionError('Cannot have both max_volume and max_voxels specified')
        self.min_volume = tf.constant(min_volume * 1000, dtype='float')
        self.min_voxels = tf.constant(min_voxels, dtype='float')
        self.max_volume, self.max_voxels = tf.constant(max_volume * 1000, dtype='float'), tf.constant(max_voxels,
                                                                                                      dtype='float')

    def parse(self, image_features, *args, **kwargs):
        if self.annotation_index is not None:
            keys = ['bounding_boxes_{}_{}_{}'.format(i, j, self.annotation_index) for i in ['r', 'z', 'c'] for j in
                    ['start', 'stop']]
            for key in keys:
                if key not in image_features:
                    return image_features
            z_start = image_features['bounding_boxes_z_start_{}'.format(self.annotation_index)]
            r_start = image_features['bounding_boxes_r_start_{}'.format(self.annotation_index)]
            c_start = image_features['bounding_boxes_c_start_{}'.format(self.annotation_index)]
            z_stop = image_features['bounding_boxes_z_stop_{}'.format(self.annotation_index)]
            r_stop = image_features['bounding_boxes_r_stop_{}'.format(self.annotation_index)]
            c_stop = image_features['bounding_boxes_c_stop_{}'.format(self.annotation_index)]
            image_features['image'] = image_features['image'][z_start:z_stop, r_start:r_stop, c_start:c_stop, ...]
            image_features['annotation'] = image_features['annotation'][z_start:z_stop, r_start:r_stop, c_start:c_stop,
                                           ...]
        return image_features


if __name__ == '__main__':
    pass
