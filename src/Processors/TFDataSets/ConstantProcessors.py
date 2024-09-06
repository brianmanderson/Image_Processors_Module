__author__ = 'Brian M Anderson'
# Created on 3/5/2021
import copy
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


class Random_Noise(ImageProcessor):
    def __init__(self, max_noise=2.5, wanted_keys=('image',)):
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


class ResizeAndPad(ImageProcessor):
    def __init__(self, input_keys=('primary_image', 'secondary_image'),
                 output_keys=('primary_image_mag', 'secondary_image_mag'), resize_row_col=(133, 133),
                 output_size=(256, 256)):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.resize_row_col = resize_row_col
        self.output_size = output_size

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        for image_key, resize_row_col, output_key, output_size in zip(self.input_keys, self.resize_row_col,
                                                                      self.output_keys, self.output_size):
            x = image_features[image_key]
            new_image = tf.image.resize(x, [resize_row_col, resize_row_col])
            padded_image = tf.image.pad_to_bounding_box(new_image, (output_size - resize_row_col) // 2,
                                                        (output_size - resize_row_col) // 2, output_size, output_size)
            image_features[output_key] = padded_image
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


class FixOutputShapes(ImageProcessor):
    def __init__(self, keys=('ct_array', 'mask_array'),
                 image_shapes=([1, None, None, None, 1], [1, None, None, None, 2])):
        self.keys = keys
        self.image_shapes = image_shapes

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key, shape in zip(self.keys, self.image_shapes):
            image_features[key] = image_features[key].set_shape(shape)
        return image_features


class DefineShape(ImageProcessor):
    def __init__(self, keys=('image_array', 'mask_array'), image_shapes=([None, None, None, 1], [None, None, None, 2])):
        self.keys = keys
        self.image_shapes = image_shapes

    def parse(self, input_features):
        _check_keys_(input_features, self.keys)
        for key, shape in zip(self.keys, self.image_shapes):
            image = input_features[key]
            image.set_shape(shape)
            input_features[key] = image
        return input_features


class FixOutputShapesPostOutput(ImageProcessor):
    def __init__(self, image_shapes=([1, None, None, None, 1], [1, None, None, None, 2]),
                 as_tuple=True):
        self.image_shapes = image_shapes
        self.as_tuple = as_tuple

    def parse(self, images, labels, *args, **kwargs):
        if self.as_tuple:
            images[0].set_shape(self.image_shapes[0])
            labels[0].set_shape(self.image_shapes[1])
            return tuple(images), tuple(labels)
        else:
            images.set_shape(self.image_shapes[0])
            labels.set_shape(self.image_shapes[1])
            return images, labels


class ReturnOutputs(ImageProcessor):
    """
    This should be your final image processor, this will turn your dictionary into a set of tensors
    """
    def __init__(self, input_keys=('image',), output_keys=('annotation',), as_tuple=True):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.as_tuple = as_tuple

    def parse(self, image_features, *args, **kwargs):
        inputs = []
        outputs = []
        _check_keys_(input_features=image_features, keys=self.input_keys + self.output_keys)
        for key in self.input_keys:
            inputs.append(image_features[key])
        for key in self.output_keys:
            outputs.append(image_features[key])
        del image_features
        if self.as_tuple:
            return tuple(inputs), tuple(outputs)
        return inputs[0], outputs[0]


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
    def __init__(self, image_rows=512, image_cols=512, image_keys=('image',)):
        print("Does not work")
        self.image_rows = tf.constant(image_rows)
        self.image_cols = tf.constant(image_cols)
        self.image_keys = image_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for key in self.image_keys:
            image_features[key] = tf.image.resize(image_features[key], size=(self.image_rows, self.image_cols),
                                                  method='bilinear')
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


class ToCategorical(ImageProcessor):
    def __init__(self, annotation_keys=('annotation',), number_of_classes=(2,)):
        self.annotation_keys = annotation_keys
        self.number_of_classes = number_of_classes

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features=input_features, keys=self.annotation_keys)
        for key, num_classes in zip(self.annotation_keys, self.number_of_classes):
            y = input_features[key]
            y = tf.squeeze(y)
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


class RepeatChannel(ImageProcessor):
    def __init__(self, axis=-1, repeats=3, input_keys=('image',)):
        """
        :param axis: axis to expand
        :param repeats: number of repeats
        :param input_keys: tuple of the keys that you want to change
        """
        self.axis = axis
        self.repeats = repeats
        self.input_keys = input_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        for key in self.input_keys:
            image_features[key] = tf.repeat(image_features[key], axis=self.axis, repeats=self.repeats)
        return image_features


class Return_Lung(ImageProcessor):
    def __init__(self, dual_output=False):
        self.dual_output = dual_output

    def parse(self, image_features, *args, **kwargs):
        if self.dual_output:
            image_features['lung'] = tf.cast(image_features['annotation'] > 0, dtype='float32')
        return image_features


class ShiftImages(ImageProcessor):
    def __init__(self, keys=('image', 'mask'), channel_dimensions=(1, 1), fill_value=None, fill_mode="reflect", interpolation="bilinear",
                 seed=None, height_factor=0.0, width_factor=0.0, on_global_3D=True, num_images=32):
        """
    Args:
        height_factor: a float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound for shifting vertically. A
            negative value means shifting image up, while a positive value means
            shifting image down. When represented as a single positive float,
            this value is used for both the upper and lower bound. For instance,
            `height_factor=(-0.2, 0.3)` results in an output shifted by a random
            amount in the range `[-20%, +30%]`. `height_factor=0.2` results in
            an output height shifted by a random amount in the range
            `[-20%, +20%]`.
        width_factor: a float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound for shifting horizontally.
            A negative value means shifting image left, while a positive value
            means shifting image right. When represented as a single positive
            float, this value is used for both the upper and lower bound. For
            instance, `width_factor=(-0.2, 0.3)` results in an output shifted
            left by 20%, and shifted right by 30%. `width_factor=0.2` results
            in an output height shifted left or right by 20%.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about the edge of the last
                pixel.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The input is extended by wrapping around to the opposite edge.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
            Note that when using torch backend, `"reflect"` is redirected to
            `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not
            support `"reflect"`.
            Note that torch backend does not support `"wrap"`.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside the
            boundaries when `fill_mode="constant"`.
        """
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.random_translation = tf.keras.layers.RandomTranslation(height_factor=height_factor,
                                                                    width_factor=width_factor,
                                                                    interpolation=interpolation,
                                                                    fill_value=fill_value, seed=seed,
                                                                    fill_mode=fill_mode)
        self.translations = []
        if on_global_3D and num_images > 0:
            for _ in range(num_images):
                self.translations.append(
                    tf.keras.layers.RandomTranslation(height_factor=height_factor,
                                                      width_factor=width_factor,
                                                      interpolation=interpolation,
                                                      fill_value=fill_value, seed=seed,
                                                      fill_mode=fill_mode)
                )
        self.keys = keys
        self.channel_dimensions = channel_dimensions
        self.global_3D = on_global_3D

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        combine_images = [image_features[i] for i in self.keys]
        shift_image = tf.concat(combine_images, axis=-1)
        if not self.global_3D:
            shifted_image = self.random_translation(shift_image)
        else:
            shifted_image = []
            for i in range(len(self.translations)):
                shifted_image.append(tf.expand_dims(self.translations[i](shift_image[i]), axis=0))
            shifted_image = tf.concat(shifted_image, axis=0)
        start_dim = 0
        for key in self.keys:
            dimension = image_features[key].shape[-1]
            end_dim = start_dim + dimension
            new_image = shifted_image[..., start_dim:end_dim]
            start_dim += dimension
            image_features[key] = new_image
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


class TakeExpOfKey(ImageProcessor):
    def __init__(self, input_keys=('pdos_array',)):
        self.input_keys = input_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        for input_key in self.input_keys:
            image_features[input_key] = tf.exp(image_features[input_key])
        return image_features


class CreateNewKey(ImageProcessor):
    def __init__(self, input_keys=('pdos_array',), output_keys=('new_pdos_array',)):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            image_features[output_key] = tf.add(image_features[input_key], tf.cast(0, image_features[input_key].dtype))
        return image_features


class AddImagesTogether(ImageProcessor):
    def __init__(self, keys=('pdos_array', 'drr_array'), out_key='pdos_drr_combined'):
        """
        :param keys:
        """
        self.keys = keys
        self.out_key = out_key

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        image_features[self.out_key] = tf.add(image_features[self.keys[0]], image_features[self.keys[1]])
        return image_features


class MultiplyImagesTogether(ImageProcessor):
    def __init__(self, keys=('pdos_array', 'drr_array'), out_key='pdos_drr_combined'):
        """
        :param keys:
        """
        self.keys = keys
        self.out_key = out_key

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        image_features[self.out_key] = tf.multiply(image_features[self.keys[0]], image_features[self.keys[1]])
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


class Change_Data_Type(ImageProcessor):
    def __init__(self, keys=('image_array',), new_dtypes=(tf.float32,)):
        """
        :param keys: tuple of image keys
        :param lower_bounds: tuple of bounds
        :param upper_bounds: tuple of bounds
        :param divides: boolean if you want to divide
        """
        self.image_keys = keys
        self.new_dtypes = new_dtypes

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for image_key, new_dtype in zip(self.image_keys, self.new_dtypes):
            image_features[image_key] = tf.cast(image_features[image_key], dtype=new_dtype)
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


class PadImages(ImageProcessor):
    def __init__(self, keys=('image', 'annotation'), pad_left=(100, 100), pad_top=(100, 100), out_size=(256, 256)):
        self.keys = keys
        self.pad_left = pad_left
        self.pad_top = pad_top
        self.out_size = out_size

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        for key, pad_left, pad_top, out_size in zip(self.keys, self.pad_left, self.pad_top, self.out_size):
            image = image_features[key]
            new_image = tf.image.pad_to_bounding_box(image, pad_left, pad_top, out_size, out_size)
            image_features[key] = new_image
        return image_features


class Resize_with_crop_pad(ImageProcessor):
    def __init__(self, keys=('image', 'annotation'), image_rows=(512, 512), image_cols=(512, 512),
                 is_mask=(False, True), out_keys=('image', 'annotation')):
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
        self.out_keys = out_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key, image_rows, image_cols, is_mask, out_key in zip(self.keys, self.image_rows, self.image_cols,
                                                                 self.is_mask, self.out_keys):
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
                image_features[out_key] = array
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
