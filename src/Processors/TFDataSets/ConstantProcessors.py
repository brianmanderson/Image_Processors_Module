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
    @tf.function
    def parse(self, *args, **kwargs):
        return args, kwargs


class Decoder(object):
    def __init__(self, d_type_dict=None):
        self.d_type_dict = d_type_dict


class DecodeImagesAnnotations(Decoder):
    def parse(self, image_features, *args, **kwargs):
        parsed_features = {key: value for key, value in image_features.items()}
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
                parsed_features[key] = tf.reshape(tf.io.decode_raw(image_features[key], out_type=dtype),
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
                    parsed_features['image'] = tf.reshape(tf.io.decode_raw(image_features['image'], out_type=image_dtype),
                                                         (image_features['z_images'], image_features['rows'],
                                                          image_features['cols']))
                if 'annotation' in image_features:
                    if 'num_classes' in image_features:
                        parsed_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
                                                                                   out_type=annotation_dtype),
                                                                  (image_features['z_images'], image_features['rows'],
                                                                   image_features['cols'], image_features['num_classes']))
                    else:
                        parsed_features['annotation'] = tf.reshape(tf.io.decode_raw(image_features['annotation'],
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


class RandomNoise(ImageProcessor):
    def __init__(self, max_noise=2.5, wanted_keys=('image',), indexes=(0,), tuple_length=2,
                 image_input_size=(5, 64, 256, 256, 1)):
        '''
        Return the image feature with an additive noise randomly weighted between [0.0, max_noise)
        :param max_noise: maximum magnitude of the noise in HU (apply before normalization)
        '''
        self.max_noise = max_noise
        self.wanted_keys = wanted_keys
        self.indexes = indexes
        self.tuple_length = tuple_length
        self.image_input_size = image_input_size

    def parse(self, image_features, *args, **kwargs):
        if isinstance(image_features, dict):
            _check_keys_(input_features=image_features, keys=self.wanted_keys)
            if self.max_noise == 0.0:
                return image_features
            parsed_features = {key: value for key, value in image_features.items()}
            for key in self.wanted_keys:
                if key in image_features:
                    data = image_features[key]
                    dtype = data.dtype
                    data = tf.cast(data, 'float32')
                    data += tf.random.uniform(shape=[], minval=0.0, maxval=self.max_noise,
                                            dtype='float32') * tf.random.normal(tf.shape(image_features[key]),
                                                                                mean=0.0, stddev=1.0, dtype='float32')
                    data = tf.cast(data, dtype)
                    parsed_features[key] = data
            return parsed_features
        else:
            noisy_outputs = tuple(tf.cast(image_features[i], 'float32') +
                                  tf.random.uniform(shape=[],
                                                    minval=0.0, maxval=self.max_noise,
                                                    dtype='float32') *
                                                    tf.random.normal(tf.shape(image_features[i]),
                                                                     mean=0.0, stddev=1.0, dtype='float32')
                if i in self.indexes else image_features[i]
                for i in range(self.tuple_length)
            )
            return noisy_outputs


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
        parsed_features = {key: value for key, value in image_features.items()}
        for image_key, resize_row_col, output_key, output_size in zip(self.input_keys, self.resize_row_col,
                                                                      self.output_keys, self.output_size):
            x = image_features[image_key]
            new_image = tf.image.resize(x, [resize_row_col, resize_row_col])
            padded_image = tf.image.pad_to_bounding_box(new_image, (output_size - resize_row_col) // 2,
                                                        (output_size - resize_row_col) // 2, output_size, output_size)
            parsed_features[output_key] = padded_image
        return parsed_features


class CombineKeys(ImageProcessor):
    def __init__(self, image_keys=('primary_image', 'secondary_image'), output_key='combined', axis=-1):
        self.image_keys = image_keys
        self.output_key = output_key
        self.axis = axis

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.image_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        combine_images = [image_features[i] for i in self.image_keys]
        parsed_features[self.output_key] = tf.concat(combine_images, axis=self.axis)
        return parsed_features


class FixOutputShapes(ImageProcessor):
    def __init__(self, keys=('ct_array', 'mask_array'),
                 image_shapes=([1, None, None, None, 1], [1, None, None, None, 2])):
        self.keys = keys
        self.image_shapes = image_shapes

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, shape in zip(self.keys, self.image_shapes):
            parsed_features[key] = image_features[key].set_shape(shape)
        return parsed_features


class DefineShape(ImageProcessor):
    def __init__(self, keys=('image_array', 'mask_array'), image_shapes=([None, None, None, 1], [None, None, None, 2])):
        self.keys = keys
        self.image_shapes = image_shapes

    def parse(self, image_features):
        _check_keys_(image_features, self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, shape in zip(self.keys, self.image_shapes):
            image = image_features[key]
            image.set_shape(shape)
            parsed_features[key] = image
        return parsed_features


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


class ToCategorical(ImageProcessor):
    def __init__(self, annotation_keys=('annotation',), number_of_classes=(2,), indexes=(2,),
                 tuple_length=2):
        self.annotation_keys = annotation_keys
        self.number_of_classes = number_of_classes
        self.indexes = indexes
        self.tuple_length = tuple_length

    def parse(self, image_features, *args, **kwargs):
        if isinstance(image_features, dict):
            _check_keys_(image_features, keys=self.annotation_keys)
            parsed_features = {key: value for key, value in image_features.items()}
            for key, num_classes in zip(self.annotation_keys, self.number_of_classes):
                y = image_features[key]
                y = tf.squeeze(y)
                parsed_features[key] = tf.cast(tf.one_hot(tf.cast(y, tf.int32), num_classes), dtype=y.dtype)
            return parsed_features
        else:
            outputs = tuple(tf.cast(tf.one_hot(tf.cast(image_features[i], tf.int32), self.number_of_classes[self.indexes.index(i)]), 'int32')
                            if i in self.indexes else image_features[i]
                            for i in range(self.tuple_length)
                            )
            return outputs


class Squeeze(ImageProcessor):
    def __init__(self, image_keys=('image',), indexes=(1,), tuple_length=2):
        """
        Designed to squeeze tf arrays
        :param image_keys:
        """
        self.image_keys = image_keys
        self.indexes = indexes
        self.tuple_length = tuple_length

    def parse(self, image_features, *args, **kwargs):
        if isinstance(image_features, dict):
            _check_keys_(image_features, self.image_keys)
            parsed_features = {key: value for key, value in image_features.items()}
            for key in self.image_keys:
                parsed_features[key] = tf.squeeze(image_features[key])
            return parsed_features
        else:
            outputs = tuple(tf.squeeze(image_features[i])
                if i in self.indexes else image_features[i]
                for i in range(self.tuple_length)
                )
            return outputs


class ExpandDimension(ImageProcessor):
    def __init__(self, axis=-1, image_keys=('image', 'annotation')):
        self.axis = axis
        self.image_keys = image_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key in self.image_keys:
            parsed_features[key] = tf.expand_dims(image_features[key], axis=self.axis)
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
        for key in self.input_keys:
            parsed_features[key] = tf.repeat(image_features[key], axis=self.axis, repeats=self.repeats)
        return parsed_features



class TakeAxis(ImageProcessor):
    def __init__(self, keys=('mask_array',), wanted_axis=(1,)):
        self.keys = keys
        self.wanted_axis = wanted_axis

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, axis in zip(self.keys, self.wanted_axis):
            image = image_features[key]
            parsed_features[key] = image[..., axis]
        return parsed_features


class FlipImages(ImageProcessor):
    def __init__(self, keys=('image', 'mask'), seed=None,
                 flip_up_down=False, flip_left_right=False, on_global_3D=True, image_shape=(32, 320, 320, 3)):
        self.og_shape = image_shape
        self.flip_up_down = flip_up_down
        self.flip_left_right = flip_left_right
        """
        I know that this has height_factor equal to 0, do not change it! We reshape things later
        """
        self.random_flip_up_down = None
        self.random_flip_left_right = None
        if flip_up_down:
            self.random_flip_up_down = tf.keras.layers.RandomFlip(mode='vertical', seed=seed)
        if flip_left_right:
            self.random_flip_left_right = tf.keras.layers.RandomFlip(mode='horizontal', seed=seed)
        self.keys = keys
        self.global_3D = on_global_3D

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        combine_images = [image_features[i] for i in self.keys]
        flip_image = tf.concat(combine_images, axis=-1)
        og_shape = self.og_shape
        if self.global_3D:
            flip_image = tf.reshape(flip_image, [og_shape[0] * og_shape[1]] + [i for i in og_shape[2:]])
        if self.flip_up_down:
            flip_image = self.random_flip_up_down(flip_image)
        if self.flip_left_right:
            flip_image = self.random_flip_left_right(flip_image)
        if self.global_3D:
            flip_image = tf.reshape(flip_image, og_shape)
        flipped_image = flip_image
        start_dim = 0
        for key in self.keys:
            dimension = image_features[key].shape[-1]
            end_dim = start_dim + dimension
            new_image = flipped_image[..., start_dim:end_dim]
            new_image.set_shape(self.og_shape[:-1] + (end_dim - start_dim,))
            start_dim += dimension
            parsed_features[key] = new_image
        return parsed_features


class RandomCrop(ImageProcessor):
    def __init__(self, keys_to_crop=('image_array', 'annotation_array'), crop_dimensions=(32, 32, 32, 1),
                 min_start_stop=None):
        if min_start_stop is None:
            min_start_stop = tuple([(None, None) for _ in range(len(crop_dimensions))])
        self.keys_to_crop = keys_to_crop
        self.crop_dimensions = crop_dimensions
        self.min_start_stop = min_start_stop

    def parse(self, image_features, *args, **kwargs):
        # Ensure all required keys are in the dictionary
        _check_keys_(image_features, keys=self.keys_to_crop)
        parsed_features = {key: value for key, value in image_features.items()}

        # Stack the features along the channel dimension
        images = [image_features[i] for i in self.keys_to_crop]
        image = tf.concat(images, axis=-1)

        # Determine random start indices for cropping
        crop_dimensions = self.crop_dimensions
        start_indices = []
        for i in range(len(crop_dimensions)):
            start_stop = self.min_start_stop[i]
            start_val = start_stop[0]
            if start_val is None:
                start_val = 0
            stop_val = start_stop[1]
            if stop_val is None:
                stop_val = image.shape[i]
            max_start = min((image.shape[i] - crop_dimensions[i], stop_val))
            if max_start > 0:
                start_index = tf.random.uniform([], start_val, max_start, dtype=tf.int32)
            else:
                start_index = 0
            start_indices.append(start_index)
        slices = [slice(start, start + size) for start, size in zip(start_indices, crop_dimensions)]
        cropped_image = image[slices]

        # Assign the cropped image parts to the respective keys
        start_dim = 0
        for key in self.keys_to_crop:
            dimension = image_features[key].shape[-1]
            end_dim = start_dim + dimension
            new_image = cropped_image[..., start_dim:end_dim]
            new_image.set_shape(self.crop_dimensions[:-1] + (end_dim - start_dim,))
            start_dim += dimension
            parsed_features[key] = new_image
        return parsed_features


class ShiftImages(ImageProcessor):
    def __init__(self, keys=('image', 'mask'), fill_value=None, fill_mode="reflect", interpolation="bilinear",
                 seed=None, height_factor=0.0, width_factor=0.0, vert_factor=0.0, on_global_3D=True,
                 image_shape=(32, 320, 320, 3)):
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
        self.og_shape = image_shape
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        """
        I know that this has height_factor equal to 0, do not change it! We reshape things later
        """
        self.random_translation_height = None
        self.random_translation_width = None
        self.random_translation_vert = None
        if height_factor != 0.0:
            self.random_translation_height = tf.keras.layers.RandomTranslation(height_factor=0.0,
                                                                               width_factor=height_factor,
                                                                               interpolation=interpolation,
                                                                               fill_value=fill_value, seed=seed,
                                                                               fill_mode=fill_mode)
        if width_factor != 0.0:
            self.random_translation_width = tf.keras.layers.RandomTranslation(height_factor=0.0,
                                                                              width_factor=width_factor,
                                                                              interpolation=interpolation,
                                                                              fill_value=fill_value, seed=seed,
                                                                              fill_mode=fill_mode)
        if vert_factor != 0.0 and on_global_3D:
            self.random_translation_vert = tf.keras.layers.RandomTranslation(height_factor=vert_factor,
                                                                             width_factor=0.0,
                                                                             interpolation=interpolation,
                                                                             fill_value=fill_value, seed=seed,
                                                                             fill_mode=fill_mode)
        self.keys = keys
        self.global_3D = on_global_3D

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        combine_images = [image_features[i] for i in self.keys]
        shift_image = tf.concat(combine_images, axis=-1)
        og_shape = self.og_shape
        if self.random_translation_height:
            if self.global_3D:
                shift_image = tf.reshape(shift_image, [og_shape[0] * og_shape[1]] + [i for i in og_shape[2:]])
            shift_image = self.random_translation_height(shift_image)
            if self.global_3D:
                shift_image = tf.reshape(shift_image, og_shape)
        if self.random_translation_width:
            if self.global_3D:
                shift_image = tf.reshape(tf.transpose(shift_image, [0, 2, 1, 3]), [og_shape[0] * og_shape[1], og_shape[2]] + [i for i in og_shape[3:]])
            shift_image = self.random_translation_width(shift_image)
            if self.global_3D:
                shift_image = tf.reshape(shift_image, og_shape)
                shift_image = tf.transpose(shift_image, [0, 2, 1, 3])
        if self.random_translation_vert and self.global_3D:
            shift_image = tf.reshape(shift_image, [og_shape[0], og_shape[1] * og_shape[2]] + [i for i in og_shape[3:]])
            shift_image = self.random_translation_vert(shift_image)
            shift_image = tf.reshape(shift_image, og_shape)
        shifted_image = shift_image
        start_dim = 0
        for key in self.keys:
            dimension = image_features[key].shape[-1]
            end_dim = start_dim + dimension
            new_image = shifted_image[..., start_dim:end_dim]
            start_dim += dimension
            parsed_features[key] = new_image
        return parsed_features


class MultiplyImagesByConstant(ImageProcessor):
    def __init__(self, keys=('image',), values=(1,), indexes=(0,), tuple_length=2):
        """
        :param keys: tuple of keys for addition
        :param values: tuple of values for addition
        """
        self.keys = keys
        self.values = values
        self.indexes = indexes
        self.tuple_length = tuple_length

    def parse(self, image_features, *args, **kwargs):
        if isinstance(image_features, dict):
            _check_keys_(input_features=image_features, keys=self.keys)
            parsed_features = {key: value for key, value in image_features.items()}
            for key, value in zip(self.keys, self.values):
                parsed_features[key] = tf.multiply(image_features[key], tf.cast(value, image_features[key].dtype))
            return parsed_features
        else:
            values = self.values
            indexes = self.indexes
            outputs = tuple(tf.multiply(image_features[i], tf.cast(values[indexes.index(i)], image_features[i].dtype))
                            if i in indexes else image_features[i]
                            for i in range(self.tuple_length)
                            )
            return outputs


class TakeExpOfKey(ImageProcessor):
    def __init__(self, input_keys=('pdos_array',)):
        self.input_keys = input_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for input_key in self.input_keys:
            parsed_features[input_key] = tf.exp(image_features[input_key])
        return parsed_features


class CreateNewKey(ImageProcessor):
    def __init__(self, input_keys=('pdos_array',), output_keys=('new_pdos_array',)):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            parsed_features[output_key] = tf.add(image_features[input_key], tf.cast(0, image_features[input_key].dtype))
        return parsed_features


class AddImagesTogether(ImageProcessor):
    def __init__(self, keys=('pdos_array', 'drr_array'), out_key='pdos_drr_combined'):
        """
        :param keys:
        """
        self.keys = keys
        self.out_key = out_key

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        parsed_features[self.out_key] = tf.add(image_features[self.keys[0]], image_features[self.keys[1]])
        return parsed_features


class MultiplyImagesTogether(ImageProcessor):
    def __init__(self, keys=('pdos_array', 'drr_array'), out_key='pdos_drr_combined'):
        """
        :param keys:
        """
        self.keys = keys
        self.out_key = out_key

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        parsed_features[self.out_key] = tf.multiply(image_features[self.keys[0]], image_features[self.keys[1]])
        return parsed_features


class Add_Constant(ImageProcessor):
    def __init__(self, keys=('image',), values=(0,), indexes=(0,), tuple_lenth=2):
        """
        :param keys: tuple of keys for addition
        :param values: tuple of values for addition
        """
        self.keys = keys
        self.values = values
        self.indexes = indexes
        self.tuple_length = tuple_lenth

    def parse(self, image_features, *args, **kwargs):
        if isinstance(image_features, dict):
            _check_keys_(input_features=image_features, keys=self.keys)
            parsed_features = {key: value for key, value in image_features.items()}
            for key, value in zip(self.keys, self.values):
                parsed_features[key] = tf.add(image_features[key], tf.cast(value, image_features[key].dtype))
            return parsed_features
        else:
            outputs = []
            for i in range(self.tuple_length):
                if i in self.indexes:
                    # Add the value to the tensor, preserving the batch dimension
                    value_to_add = tf.cast(self.values[self.indexes.index(i)], image_features[i].dtype)
                    new_tensor = tf.add(image_features[i], value_to_add)
                    outputs.append(new_tensor)
                else:
                    outputs.append(image_features[i])
            return tuple(outputs)

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
        parsed_features = {key: value for key, value in image_features.items()}
        parsed_features['image'] = tf.add(image_features['image'], self.means)
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
        for key, mean_val, std_val in zip(self.keys, self.mean_values, self.std_values):
            mean_val = tf.constant(mean_val, dtype=image_features[key].dtype)
            std_val = tf.constant(std_val, dtype=image_features[key].dtype)
            parsed_features[key] = (image_features[key] - mean_val) / std_val
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
        for key in self.annotation_keys:
            parsed_features[key] = tf.argmax(image_features[key], axis=self.axis)
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
        for key, from_value, to_value in zip(self.key_list, self.from_list, self.to_list):
            from_tensor = tf.constant(from_value, dtype=image_features[key].dtype)
            to_tensor = tf.constant(to_value, dtype=image_features[key].dtype)
            parsed_features[key] = tf.where(image_features[key] == from_tensor, to_tensor, image_features[key])
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
        for key, channel, new_key in zip(self.image_keys, self.channels, self.out_keys):
            parsed_features[new_key] = image_features[key][..., channel]
        return parsed_features


class MaskKeys(CombineAnnotations):
    def __init__(self, key_tuple=('annotation',), from_values_tuple=(2,), to_values_tuple=(1,)):
        super().__init__(key_tuple=key_tuple, from_values_tuple=from_values_tuple, to_values_tuple=to_values_tuple)


class Combined_Annotations(ImageProcessor):
    def __init__(self, values=[tf.constant(1, dtype='int8'), tf.constant(2, dtype='int8')]):
        self.values = values

    def parse(self, image_features, *args, **kwargs):
        parsed_features = {key: value for key, value in image_features.items()}
        for value in self.values:
            value = tf.constant(value, dtype=image_features['annotation'].dtype)
            parsed_features['annotation'] = tf.where(image_features['annotation'] == value,
                                                    tf.constant(1, dtype=image_features['annotation'].dtype),
                                                    image_features['annotation'])
        return parsed_features


class CreateDiseaseKey(ImageProcessor):
    def parse(self, image_features, *args, **kwargs):
        parsed_features = {key: value for key, value in image_features.items()}
        value = tf.constant(1, dtype=image_features['primary_liver'].dtype)
        parsed_features['disease'] = tf.where(image_features['primary_liver'] > value,
                                             value,
                                             tf.constant(0, dtype=image_features['primary_liver'].dtype))
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
        for guiding_key, new_key in zip(self.guiding_keys, self.new_keys):
            annotation = image_features[guiding_key]
            mask = tf.expand_dims(tf.reduce_sum(annotation[..., 1:], axis=-1), axis=-1)
            parsed_features[new_key] = mask
        return parsed_features


class Cast_Data(ImageProcessor):
    def __init__(self, keys=('image', 'annotation',), dtypes=('float16', 'float16'),
                 indexes=(1,), tuple_length=2):
        """
        :param keys: tuple of keys
        :param dtypes: tuple of datatypes
        """
        self.keys = keys
        self.dtypes = dtypes
        self.indexes = indexes
        self.tuple_length = tuple_length

    def parse(self, image_features, *args, **kwargs):
        if isinstance(image_features, dict):
            _check_keys_(input_features=image_features, keys=self.keys)
            parsed_features = {key: value for key, value in image_features.items()}
            for key, dtype in zip(self.keys, self.dtypes):
                parsed_features[key] = tf.cast(image_features[key], dtype=dtype)
            return parsed_features
        else:
            outputs = tuple(tf.cast(image_features[i], self.dtypes[self.indexes.index(i)])
                            if i in self.indexes else image_features[i]
                            for i in range(self.tuple_length)
                            )
            return outputs


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


class Threshold_Images(ImageProcessor):
    def __init__(self, keys=('image',), lower_bounds=(-np.inf,), upper_bounds=(np.inf,), divides=(True,),
                 indexes=(0,), tuple_length=2):
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
        self.indexes = indexes
        self.tuple_length = tuple_length

    def parse(self, image_features, *args, **kwargs):
        if isinstance(image_features, dict):
            _check_keys_(image_features, self.keys)
            parsed_features = {key: value for key, value in image_features.items()}
            for key, lower_bound, upper_bound, divide in zip(self.keys, self.lower_bounds, self.upper_bounds, self.divides):
                parsed_features[key] = tf.where(parsed_features[key] > tf.cast(upper_bound, dtype=parsed_features[key].dtype),
                                            tf.cast(upper_bound, dtype=parsed_features[key].dtype), parsed_features[key])
                parsed_features[key] = tf.where(parsed_features[key] < tf.cast(lower_bound, dtype=parsed_features[key].dtype),
                                            tf.cast(lower_bound, dtype=parsed_features[key].dtype), parsed_features[key])
                if divide:
                    parsed_features[key] = tf.divide(parsed_features[key], tf.cast(tf.subtract(upper_bound, lower_bound),
                                                                                dtype=parsed_features[key].dtype))
            return parsed_features
        else:
            outputs = tuple(
            tf.divide(
                tf.where(image_features[i] > tf.cast(self.upper_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                         tf.cast(self.upper_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                         tf.where(image_features[i] < tf.cast(self.lower_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                                  tf.cast(self.lower_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                                  image_features[i])),
                tf.cast(tf.subtract(self.upper_bounds[self.indexes.index(i)], self.lower_bounds[self.indexes.index(i)]), dtype=image_features[i].dtype)
            )
            if i in self.indexes and self.divides[self.indexes.index(i)] else
            tf.where(image_features[i] > tf.cast(self.upper_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                     tf.cast(self.upper_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                     tf.where(image_features[i] < tf.cast(self.lower_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                              tf.cast(self.lower_bounds[self.indexes.index(i)], dtype=image_features[i].dtype),
                              image_features[i]))
            if i in self.indexes else image_features[i]
            for i in range(self.tuple_length)
        )
            return outputs


class PadImages(ImageProcessor):
    def __init__(self, keys=('image', 'annotation'), pad_left=(100, 100), pad_top=(100, 100), out_size=(256, 256)):
        self.keys = keys
        self.pad_left = pad_left
        self.pad_top = pad_top
        self.out_size = out_size

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, pad_left, pad_top, out_size in zip(self.keys, self.pad_left, self.pad_top, self.out_size):
            image = image_features[key]
            new_image = tf.image.pad_to_bounding_box(image, pad_left, pad_top, out_size, out_size)
            parsed_features[key] = new_image
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
        for key, image_rows, image_cols, is_mask, out_key in zip(self.keys, self.image_rows, self.image_cols,
                                                                 self.is_mask, self.out_keys):
            image_rows = tf.constant(image_rows)
            image_cols = tf.constant(image_cols)
            parsed_features[key] = tf.image.resize_with_crop_or_pad(image_features[key],
                                                                   target_width=image_cols,
                                                                   target_height=image_rows)
            if is_mask and image_features[key].shape[-1] != 1:
                array = image_features[key]
                array = array[..., 1:]  # remove background
                background = tf.expand_dims(1 - tf.reduce_sum(array, axis=-1), axis=-1)
                array = tf.concat([background, array], axis=-1)
                parsed_features[out_key] = array
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
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
        parsed_features['image'] = image_cube
        parsed_features['annotation'] = annotation_cube
        return parsed_features


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
        parsed_features = {key: value for key, value in image_features.items()}
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
            parsed_features['image'] = image_features['image'][z_start:z_stop, r_start:r_stop, c_start:c_stop, ...]
            parsed_features['annotation'] = image_features['annotation'][z_start:z_stop, r_start:r_stop, c_start:c_stop,
                                           ...]
        return parsed_features


if __name__ == '__main__':
    pass
