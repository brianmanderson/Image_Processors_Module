__author__ = 'Brian M Anderson'
# Created on 4/8/2020
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt


class Image_Processor(object):
    def parse(self, *args, **kwargs):
        return args, kwargs


class Decode_Images_Annotations(Image_Processor):
    def __init__(self, d_type_dict=None):
        self.d_type_dict = d_type_dict

    def parse(self, image_features, *args, **kwargs):
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


class Random_Noise(Image_Processor):
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
                data = tf.cast(data,'float32')
                data += tf.random.uniform(shape=[], minval=0.0, maxval=self.max_noise,
                                          dtype='float32') * tf.random.normal(tf.shape(image_features['image']),
                                                                              mean=0.0, stddev=1.0, dtype='float32')
                data = tf.cast(data, dtype)
                image_features[key] = data
        return image_features


class Combine_image_RT_Dose(Image_Processor):
    def parse(self, input_features, *args, **kwargs):
        image = input_features['image']
        rt = input_features['annotation']
        dose = input_features['dose']
        output = tf.concat([image,rt,dose],axis=-1)
        input_features['combined'] = output
        return input_features

from tensorflow.keras.utils import to_categorical
class Fuzzy_Segment_Liver_Lobes(Image_Processor):
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
        annotation = to_categorical(annotation,9)
        image_features['annotation'] = annotation
        return image_features
        size = tf.random.uniform([2],minval=self.min_val, maxval=self.max_val)
        filter_shape = tuple(tf.cast(tf.divide(size,image_features['spacing'][:2]), dtype=tf.dtypes.float32))

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
        kernel = tf.ones(shape=filter_shape, dtype=annotation.dtype)

        annotation = tf.nn.depthwise_conv2d(annotation, kernel, strides=(1, 1, 1, 1), padding="VALID")
        annotation = tf.divide(annotation, area)
        annotation = tf.divide(annotation, tf.expand_dims(tf.reduce_sum(annotation, axis=-1),axis=-1))
        image_features['annotation'] = annotation
        return image_features


class Return_Outputs(Image_Processor):
    '''
    No image processors should occur after this, this will turn your dictionary into a set of tensors, usually
    image, annotation
    '''
    def __init__(self, wanted_keys_dict={'inputs':['image'],'outputs':['annotation']}):
        assert type(wanted_keys_dict) is dict, 'You need to pass a dictionary to Return_Outputs in the form of ' \
                                               '{"inputs":["image"],"outputs":["annotation"]}, etc.'
        self.wanted_keys_dict = wanted_keys_dict

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


class Pad_Z_Images_w_Reflections(Image_Processor):
    '''
    This will not work for parallelized.. because the z dimension is None unknown to start
    '''
    def __init__(self, z_images=32):
        self.z_images = tf.constant(z_images)

    def parse(self, image_features, *args, **kwargs):
        dif = tf.subtract(self.z_images,image_features['image'].shape[0])
        image_features['image'] = tf.concat([image_features['image'],tf.reverse(image_features['image'], axis=[0])[:dif]],axis=0)
        image_features['annotation'] = tf.concat([image_features['annotation'],tf.reverse(image_features['annotation'], axis=[0])[:dif]],axis=0)
        return image_features


class Ensure_Image_Proportions(Image_Processor):
    def __init__(self, image_rows=512, image_cols=512):
        self.image_rows = tf.constant(image_rows)
        self.image_cols = tf.constant(image_cols)

    def parse(self, image_features, *args, **kwargs):
        assert len(image_features['image'].shape) > 2, 'You should do an expand_dimensions before this!'
        image_features['image'] = tf.image.resize_with_crop_or_pad(image_features['image'], target_width=self.image_rows,
                                                                   target_height=self.image_cols)
        image_features['annotation'] = tf.image.resize_with_crop_or_pad(image_features['annotation'], target_width=self.image_rows,
                                                                        target_height=self.image_cols)
        return image_features


class Return_Add_Mult_Disease(Image_Processor):
    def __init__(self, on_disease=True):
        self.on_disease = on_disease

    def parse(self, image_features, *args, **kwargs):
        annotation = image_features['annotation']
        mask = tf.where(annotation > 0, 1, 0)
        if self.on_disease:
            annotation = tf.where(annotation == 2, 1, 0)
            image_features['annotation'] = annotation
        image_features['mask'] = mask
        return image_features



class Expand_Dimensions(Image_Processor):
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


class Repeat_Channel(Image_Processor):
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


class Return_Lung(Image_Processor):
    def __init__(self, dual_output=False):
        self.dual_output = dual_output

    def parse(self, image_features, *args, **kwargs):
        if self.dual_output:
            image_features['lung'] = tf.cast(image_features['annotation'] > 0,dtype='float32')
        return image_features


class Normalize_Images(Image_Processor):
    def __init__(self, mean_val=0, std_val=1):
        '''
        :param mean_val: Mean value to normalize to
        :param std_val: Standard deviation value to normalize to
        '''
        self.mean_val, self.std_val = tf.constant(mean_val, dtype='float32'), tf.constant(std_val, dtype='float32')

    def parse(self, image_features, *args, **kwargs):
        image_features['image'] = (image_features['image'] - self.mean_val)/self.std_val
        return image_features


class Combined_Annotations(Image_Processor):
    def __init__(self, values=[tf.constant(1, dtype='int8'),tf.constant(2, dtype='int8')]):
        self.values = values

    def parse(self, image_features, *args, **kwargs):
        for value in self.values:
            value = tf.constant(value, dtype=image_features['annotation'].dtype)
            image_features['annotation'] = tf.where(image_features['annotation'] == value,
                                                    tf.constant(1, dtype=image_features['annotation'].dtype),
                                                    image_features['annotation'])
        return image_features


class Cast_Data(Image_Processor):
    def __init__(self, key_type_dict=None):
        '''
        :param key_type_dict: A dictionary of keys and datatypes wanted {'image':'float32'}
        '''
        assert key_type_dict is not None and type(key_type_dict) is dict, 'Need to provide a key_type_dict, something' \
                                                                          ' like {"image":"float32"}'
        self.key_type_dict = key_type_dict

    def parse(self, image_features, *args, **kwargs):
        for key in self.key_type_dict:
            if key in image_features:
                image_features[key] = tf.cast(image_features[key], dtype=self.key_type_dict[key])
        return image_features


class Flip_Images(Image_Processor):
    def __init__(self, flip_lr=True, flip_up_down=True):
        self.flip_lr = flip_lr
        self.flip_up_down = flip_up_down

    def parse(self, image_features, *args, **kwargs):
        if self.flip_lr:
            if tf.random.uniform(shape=[], minval=0, maxval=2, dtype='int32') == tf.constant(1,dtype='int32'):
                print('flipped lr')
                image_features['image'] = tf.image.flip_left_right(image_features['image'])
                image_features['annotation'] = tf.image.flip_left_right(image_features['annotation'])
        if self.flip_up_down:
            if tf.random.uniform(shape=[], minval=0, maxval=2, dtype='int32') == tf.constant(1, dtype='int32'):
                print('flipped ud')
                image_features['image'] = tf.image.flip_up_down(image_features['image'])
                image_features['annotation'] = tf.image.flip_up_down(image_features['annotation'])
        return image_features


class Threshold_Images(Image_Processor):
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf):
        '''
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        '''
        self.lower = tf.constant(lower_bound, dtype='float32')
        self.upper = tf.constant(upper_bound, dtype='float32')

    def parse(self, image_features, *args, **kwargs):
        image_features['image'] = tf.where(image_features['image'] > self.upper, self.upper, image_features['image'])
        image_features['image'] = tf.where(image_features['image'] < self.lower, self.lower, image_features['image'])
        return image_features


class Clip_Images(Image_Processor):
    def __init__(self, annotations_index=None, bounding_box_expansion=(10, 10, 10), power_val_z=1, power_val_r=1,
                 power_val_c=1, min_images=0, min_rows=0, min_cols=0):
        self.annotations_index = annotations_index
        self.bounding_box_expansion = tf.convert_to_tensor(bounding_box_expansion)
        self.power_val_z, self.power_val_r, self.power_val_c = tf.constant(power_val_z), tf.constant(power_val_r), tf.constant(power_val_c)
        self.min_images, self.min_rows, self.min_cols = tf.constant(min_images), tf.constant(min_rows), tf.constant(min_cols)

    def parse(self, image_features, *args, **kwargs):
        zero = tf.constant(0)
        image = image_features['image']
        annotation = image_features['annotation']
        img_shape = tf.shape(image)
        if self.annotations_index:
            bounding_box = image_features['bounding_boxes_{}'.format(self.annotations_index)][0] # Assuming one bounding box
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

        remainder_z = tf.math.floormod(self.power_val_z - z_total,self.power_val_z) if tf.math.floormod(z_total,
                                                                                                        self.power_val_z) != zero else zero
        remainder_r = tf.math.floormod(self.power_val_r - r_total, self.power_val_r) if tf.math.floormod(r_total,
                                                                                                         self.power_val_r) != zero else zero
        remainder_c = tf.math.floormod(self.power_val_c - r_total, self.power_val_c) if tf.math.floormod(r_total,
                                                                                                         self.power_val_c) != zero else zero
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        min_images = tf.maximum(self.min_images,min_images)
        min_rows = tf.maximum(self.min_rows, min_rows)
        min_cols = tf.maximum(self.min_cols, min_cols)
        output_dims = tf.convert_to_tensor([min_images, min_rows, min_cols])
        image_cube = image[z_start:z_start + min_images, r_start:r_start + min_rows, c_start:c_start + min_cols,...]
        annotation_cube = annotation[z_start:z_start + min_images, r_start:r_start + min_rows, c_start:c_start + min_cols]
        img_shape = tf.shape(image_cube)
        size_dif = output_dims - img_shape[:3]
        if tf.reduce_max(size_dif) > 0:
            paddings = tf.convert_to_tensor([[size_dif[0], zero], [size_dif[1], zero], [size_dif[2], zero], [zero, zero]])
            image_cube = tf.pad(image_cube,paddings=paddings,constant_values=tf.reduce_min(image))
            annotation_cube = tf.pad(annotation_cube, paddings=paddings)
        image_features['image'] = image_cube
        image_features['annotation'] = annotation_cube
        return image_features


class Pull_Subset(Image_Processor):
    def __init__(self, max_batch=32):
        self.max_batch = max_batch

    def parse(self, images, annotations, *args, **kwargs):
        num_images = images.shape[0]
        if num_images > self.max_batch:
            random_index = tf.random.uniform(shape=[], minval=0, maxval=num_images - self.max_batch, dtype='int32')
            images = images[random_index:random_index+self.max_batch,...]
            annotations = annotations[random_index:random_index+self.max_batch,...]
        return images, annotations


class Pull_Bounding_Box(Image_Processor):
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
        self.max_volume, self.max_voxels = tf.constant(max_volume * 1000, dtype='float'), tf.constant(max_voxels, dtype='float')

    def parse(self, image_features, *args, **kwargs):
        if self.annotation_index is not None:
            keys = ['bounding_boxes_{}_{}_{}'.format(i,j,self.annotation_index) for i in ['r','z','c'] for j in ['start','stop']]
            for key in keys:
                if key not in image_features:
                    return image_features
            z_start = image_features['bounding_boxes_z_start_{}'.format(self.annotation_index)]
            r_start = image_features['bounding_boxes_r_start_{}'.format(self.annotation_index)]
            c_start = image_features['bounding_boxes_c_start_{}'.format(self.annotation_index)]
            z_stop = image_features['bounding_boxes_z_stop_{}'.format(self.annotation_index)]
            r_stop = image_features['bounding_boxes_r_stop_{}'.format(self.annotation_index)]
            c_stop = image_features['bounding_boxes_c_stop_{}'.format(self.annotation_index)]
            image_features['image'] = image_features['image'][z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
            image_features['annotation'] = image_features['annotation'][z_start:z_stop,r_start:r_stop,c_start:c_stop,...]
        return image_features


if __name__ == '__main__':
    pass
