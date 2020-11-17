__author__ = 'Brian M Anderson'

# Created on 4/28/2020
import SimpleITK as sitk
import numpy as np
from _collections import OrderedDict
from .Resample_Class.src.NiftiResampler.ResampleTools import ImageResampler
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import os, pickle
from math import ceil, floor
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt


class ImageProcessor(object):
    def parse(self, input_features):
        return input_features


def save_obj(path, obj):  # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return None


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = OrderedDict()
        return out


def return_feature(data):
    if type(data) is int:
        return _int64_feature(tf.constant(data, dtype='int64'))
    elif type(data) is np.ndarray:
        return _bytes_feature(data.tostring())
    elif type(data) is str:
        return _bytes_feature(tf.constant(data))
    elif type(data) is np.float32:
        return _float_feature(tf.constant(data, dtype='float32'))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_features(values):
    return tf.train.Features(float_list=tf.train.FloatList(values=[values]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def return_example_proto(base_dictionary, image_dictionary_for_pickle={}, data_type_dictionary={}):
    feature = {}
    for key in base_dictionary:
        data = base_dictionary[key]
        if type(data) is int:
            feature[key] = _int64_feature(tf.constant(data, dtype='int64'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.int64)
        elif type(data) is np.ndarray:
            for index, shape_value in enumerate(data.shape):
                if '{}_size_{}'.format(key, index) not in base_dictionary:
                    feature['{}_size_{}'.format(key, index)] = _int64_feature(tf.constant(shape_value, dtype='int64'))
                    image_dictionary_for_pickle['{}_size_{}'.format(key, index)] = tf.io.FixedLenFeature([], tf.int64)
            feature[key] = _bytes_feature(data.tostring())
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.string)
                data_type_dictionary[key] = data.dtype
        elif type(data) is str:
            feature[key] = _bytes_feature(tf.constant(data))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.string)
        elif type(data) is np.float32:
            feature[key] = _float_feature(tf.constant(data, dtype='float32'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.float32)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


def serialize_example(input_features_dictionary, image_processors=None, record_writer=None, verbose=False):
    get_features(input_features_dictionary, image_processors=image_processors, record_writer=record_writer,
                 verbose=verbose)


class RecordWriter(ImageProcessor):
    def __init__(self, out_path=None, out_file=None):
        assert out_path is not None, "You need to pass a base file path..."
        self.out_path = out_path
        self.out_file = out_file
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    def write_records(self, input_features):
        keys = list(input_features.keys())
        filename = self.out_file
        if self.out_file is None:
            image_name = os.path.split(input_features[keys[0]]['image_path'])[-1].split('.nii')[0]
            filename = os.path.join(self.out_path, '{}.tfrecord'.format(image_name))
        features = {}
        d_type = {}
        writer = tf.io.TFRecordWriter(filename)
        examples = 0
        for key in input_features.keys():
            example_proto = return_example_proto(input_features[key], features, d_type)
            writer.write(example_proto.SerializeToString())
            examples += 1
        writer.close()
        fid = open(filename.replace('.tfrecord', '_Num_Examples.txt'), 'w+')
        fid.write(str(examples))
        fid.close()
        save_obj(filename.replace('.tfrecord', '_features.pkl'), features)
        save_obj(filename.replace('.tfrecord', '_dtype.pkl'), d_type)
        del input_features
        return {}


def get_features(features, image_processors=None, record_writer=None, verbose=0):
    if image_processors is not None:
        for image_processor in image_processors:
            features, _ = down_dictionary(features, OrderedDict(), 0)
            if verbose:
                print(image_processor)
            for key in features.keys():
                features[key] = image_processor.parse(features[key])
        features, _ = down_dictionary(features, OrderedDict(), 0)
    record_writer.write_records(features)


def down_dictionary(input_dictionary, out_dictionary=OrderedDict(), out_index=0):
    if 'out_path' in input_dictionary.keys():
        out_dictionary['Example_{}'.format(out_index)] = input_dictionary
        out_index += 1
        return out_dictionary, out_index
    else:
        for key in input_dictionary.keys():
            out_dictionary, out_index = down_dictionary(input_dictionary[key], out_dictionary, out_index)
    return out_dictionary, out_index


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    Taken from tf.keras.utils.to_categorical
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_start_stop(annotation, extension=np.inf, desired_val=1):
    if len(annotation.shape) > 3:
        annotation = np.argmax(annotation, axis=-1)
    non_zero_values = np.where(np.max(annotation, axis=(1, 2)) >= desired_val)[0]
    start, stop = -1, -1
    if non_zero_values.any():
        start = int(non_zero_values[0])
        stop = int(non_zero_values[-1])
        start = max([start - extension, 0])
        stop = min([stop + extension, annotation.shape[0]])
    return start, stop


def get_bounding_boxes(annotation_handle, value):
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    RelabelComponent = sitk.RelabelComponentImageFilter()
    RelabelComponent.SortByObjectSizeOn()
    stats = sitk.LabelShapeStatisticsImageFilter()
    thresholded_image = sitk.BinaryThreshold(annotation_handle, lowerThreshold=value, upperThreshold=value + 1)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    connected_image = RelabelComponent.Execute(connected_image)
    stats.Execute(connected_image)
    bounding_boxes = [stats.GetBoundingBox(l) for l in stats.GetLabels()]
    num_voxels = np.asarray([stats.GetNumberOfPixels(l) for l in stats.GetLabels()]).astype('float32')
    return bounding_boxes, num_voxels


class Remove_Smallest_Structures(object):
    def __init__(self):
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.RelabelComponent.SortByObjectSizeOn()

    def remove_smallest_component(self, annotation_handle):
        label_image = self.Connected_Component_Filter.Execute(
            sitk.BinaryThreshold(sitk.Cast(annotation_handle, sitk.sitkFloat32), lowerThreshold=0.01,
                                 upperThreshold=np.inf))
        label_image = self.RelabelComponent.Execute(label_image)
        output = sitk.BinaryThreshold(sitk.Cast(label_image, sitk.sitkFloat32), lowerThreshold=0.1, upperThreshold=1.0)
        return output


class Remove_Lowest_Probabilty_Structure(object):
    def __init__(self):
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.RelabelComponent.SortByObjectSizeOn()

    def remove_lowest_probability(self, image_slice):
        stats = sitk.LabelShapeStatisticsImageFilter()
        thresholded_image = sitk.GetImageFromArray(image_slice) > 0
        connected_image = self.Connected_Component_Filter.Execute(thresholded_image)
        stats.Execute(connected_image)
        if self.Connected_Component_Filter.GetObjectCount() < 2:
            return image_slice
        current = 0
        for value in range(1, self.Connected_Component_Filter.GetObjectCount() + 1):
            mask = sitk.GetArrayFromImage(connected_image == value)
            prob = np.max(image_slice[mask == 1])
            if prob > current:
                current = prob
                out_mask = mask
        image_slice[out_mask == 0] = 0
        return image_slice


class Gaussian_Uncertainty(ImageProcessor):
    def __init__(self, sigma=None):
        '''
        :param sigma: Desired sigma, in mm, in x, y, z direction
        '''
        self.sigma = sigma

    def parse(self, input_features):
        remove_lowest_probability = Remove_Lowest_Probabilty_Structure()
        remove_smallest = Remove_Smallest_Structures()
        annotations = input_features['annotation']
        spacing = input_features['spacing']
        filtered = np.zeros(annotations.shape)
        filtered[..., 0] = annotations[..., 0]
        if len(annotations.shape) == 3:
            num_classes = np.max(annotations)
        else:
            num_classes = annotations.shape[-1]
        for i in range(1, num_classes):
            sigma = self.sigma[i - 1]
            if type(sigma) is not list:
                sigma = [sigma / spacing[i] for i in range(3)]
            else:
                sigma = [sigma[i] / spacing[i] for i in range(3)]
            annotation = annotations[..., i]
            if sigma[-1] != np.min(sigma):
                print('Make sure you put this in as x, y, z. Not z, x, y!')
            sigma = [sigma[-1], sigma[0], sigma[1]]  # now make it match image shape [z, x, y]
            filtered[..., i] = gaussian_filter(annotation, sigma=sigma, mode='constant')
        filtered[annotations[..., 0] == 1] = 0
        filtered[..., 0] = annotations[..., 0]
        # Now we've normed, but still have the problem that unconnected structures can still be there..
        for i in range(1, num_classes):
            annotation = filtered[..., i]
            annotation[annotation < 0.05] = 0
            slices = np.where(np.max(annotation, axis=(1, 2)) > 0)
            for slice in slices[0]:
                annotation[slice] = remove_lowest_probability.remove_lowest_probability(annotation[slice])
            mask_handle = remove_smallest.remove_smallest_component(sitk.GetImageFromArray(annotation) > 0)
            mask = sitk.GetArrayFromImage(mask_handle)
            masked_filter = filtered[..., i] * mask
            filtered[..., i] = masked_filter
        norm = np.sum(filtered, axis=-1)
        filtered[..., 0] += (norm == 0).astype('int')
        norm[norm == 0] = 1
        filtered /= norm[..., None]
        input_features['annotation'] = filtered
        return input_features


class Combine_Annotations(ImageProcessor):
    def __init__(self, annotation_input=[5, 6, 7, 8], to_annotation=5):
        self.annotation_input = annotation_input
        self.to_annotation = to_annotation

    def parse(self, input_features):
        annotation = input_features['annotation']
        assert len(annotation.shape) == 3 or len(
            annotation.shape) == 4, 'To combine annotations the size has to be 3 or 4'
        if len(annotation.shape) == 3:
            for val in self.annotation_input:
                annotation[annotation == val] = self.to_annotation
        elif len(annotation.shape) == 4:
            annotation[..., self.to_annotation] += annotation[..., self.annotation_input]
            del annotation[..., self.annotation_input]
        input_features['annotation'] = annotation
        return input_features


class To_Categorical(ImageProcessor):
    def __init__(self, num_classes=None, annotation_key='annotation'):
        self.num_classes = num_classes
        self.annotation_key = annotation_key

    def parse(self, input_features):
        _check_keys_(input_features=input_features, keys=self.annotation_key)
        input_features[self.annotation_key] = to_categorical(input_features[self.annotation_key], self.num_classes)
        input_features['num_classes_{}'.format(self.annotation_key)] = self.num_classes
        return input_features


class Resample_LiTs(ImageProcessor):
    def __init__(self, desired_output_spacing=(None, None, None)):
        self.desired_output_spacing = desired_output_spacing

    def parse(self, input_features):
        input_spacing = tuple([float(i) for i in input_features['spacing']])
        image_handle = sitk.GetImageFromArray(input_features['image'])
        image_handle.SetSpacing(input_spacing)
        annotation_handle = sitk.GetImageFromArray(input_features['annotation'])
        annotation_handle.SetSpacing(input_spacing)
        output_spacing = []
        for index in range(3):
            if self.desired_output_spacing[index] is None:
                spacing = input_spacing[index]
                output_spacing.append(spacing)
            else:
                output_spacing.append(self.desired_output_spacing[index])
        output_spacing = tuple(output_spacing)
        if output_spacing != input_spacing:
            resampler = ImageResampler()
            print('Resampling {} to {}'.format(input_spacing, output_spacing))
            image_handle = resampler.resample_image(input_image_handle=image_handle, output_spacing=output_spacing)
            annotation_handle = resampler.resample_image(input_image_handle=annotation_handle,
                                                         output_spacing=output_spacing)
            input_features['image'] = sitk.GetArrayFromImage(image_handle)
            input_features['annotation'] = sitk.GetArrayFromImage(annotation_handle)
            input_features['spacing'] = np.asarray(annotation_handle.GetSpacing(), dtype='float32')
        return input_features


class Resampler(ImageProcessor):
    def __init__(self, resample_keys=('image', 'annotation'), resample_interpolators=('Linear', 'Nearest'),
                 desired_output_spacing=(None, None, None), make_512=False):
        """
        :param resample_keys: tuple of keys in input_features to resample
        :param resample_interpolators: tuple of SimpleITK interpolators, 'Linear' or 'Nearest'
        :param desired_output_spacing: desired output spacing, (row, col, z)
        :param make_512: binary, make the image be 512x512?
        """
        self.desired_output_spacing = desired_output_spacing
        self.resample_keys = resample_keys
        self.resample_interpolators = resample_interpolators
        self.make_512 = make_512

    def parse(self, input_features):
        resampler = ImageResampler()
        _check_keys_(input_features=input_features, keys=self.resample_keys)
        for key, interpolator in zip(self.resample_keys, self.resample_interpolators):
            image_handle = input_features[key]
            input_spacing = None
            if 'spacing' in input_features.keys():
                input_spacing = tuple([float(i) for i in input_features['spacing']])
            elif '{}_spacing'.format(key) in input_features.keys():
                input_spacing = tuple([float(i) for i in input_features['{}_spacing'.format(key)]])
            assert type(image_handle) is sitk.Image or input_spacing is not None, 'Either need to pass a SimpleITK ' \
                                                                                  'Image or "spacing" key'
            if input_spacing is None:
                input_spacing = image_handle.GetSpacing()
            image_array = None
            if type(image_handle) is np.ndarray:
                image_array = image_handle
                image_handle = sitk.GetImageFromArray(image_handle)
                image_handle.SetSpacing(input_spacing)

            output_spacing = []
            for index in range(3):
                if self.desired_output_spacing[index] is None:
                    if input_spacing[index] < 0.5 and self.make_512:
                        spacing = input_spacing[index] * 2
                    else:
                        spacing = input_spacing[index]
                    output_spacing.append(spacing)
                else:
                    output_spacing.append(self.desired_output_spacing[index])
            output_spacing = tuple(output_spacing)
            if output_spacing != input_spacing:
                print('Resampling {} to {}'.format(input_spacing, output_spacing))
                if image_array is None:
                    image_array = sitk.GetArrayFromImage(image_handle)
                if len(image_array.shape) == 3:
                    image_handle = resampler.resample_image(input_image_handle=image_handle,
                                                            output_spacing=output_spacing,
                                                            interpolator=interpolator)
                    input_features[key] = sitk.GetArrayFromImage(image_handle)
                    input_features['{}_spacing'.format(key)] = np.asarray(self.desired_output_spacing, dtype='float32')
                else:
                    output = []
                    for i in range(image_array.shape[-1]):
                        reduced_handle = sitk.GetImageFromArray(image_array[..., i])
                        reduced_handle.SetSpacing(input_spacing)
                        resampled_handle = resampler.resample_image(input_image_handle=reduced_handle,
                                                                    output_spacing=output_spacing,
                                                                    interpolator=interpolator)
                        output.append(sitk.GetArrayFromImage(resampled_handle)[..., None])
                    stacked = np.concatenate(output, axis=-1)
                    stacked[..., 0] = 1 - np.sum(stacked[..., 1:], axis=-1)
                    input_features[key] = stacked
                    input_features['{}_spacing'.format(key)] = np.asarray(self.desired_output_spacing, dtype='float32')
        input_features['spacing'] = np.asarray(self.desired_output_spacing, dtype='float32')
        return input_features


class Cast_Data(ImageProcessor):
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
                image_features[key] = image_features[key].astype(self.key_type_dict[key])
        return image_features


class Add_Images_And_Annotations(ImageProcessor):
    def __init__(self, nifti_path_keys=('image_path', 'annotation_path'), out_keys=('image', 'annotation'),
                 dtypes=('float32', 'int8')):
        self.nifti_path_keys, self.out_keys, self.dtypes = nifti_path_keys, out_keys, dtypes

    def parse(self, input_features):
        _check_keys_(input_features=input_features, keys=self.nifti_path_keys)
        for nifti_path_key, out_key, dtype in zip(self.nifti_path_keys, self.out_keys, self.dtypes):
            image_handle = sitk.ReadImage(input_features[nifti_path_key])
            image_array = sitk.GetArrayFromImage(image_handle)
            input_features[out_key] = image_array.astype(dtype=dtype)
            input_features['{}_spacing'.format(out_key)] = np.asarray(image_handle.GetSpacing(), dtype='float32')
        return input_features


class Add_Dose(ImageProcessor):
    def parse(self, input_features):
        image_path = input_features['image_path']
        dose_path = image_path.replace('Data', 'Dose')
        dose_handle = sitk.ReadImage(dose_path)
        dose = sitk.GetArrayFromImage(dose_handle).astype('float32')
        spacing = dose_handle.GetSpacing()
        input_features['dose'] = dose
        input_features['dose_images'] = dose.shape[0]
        input_features['dose_rows'] = dose.shape[1]
        input_features['dose_cols'] = dose.shape[2]
        input_features['dose_spacing_images'] = spacing[0]
        input_features['dose_spacing_rows'] = spacing[1]
        input_features['dose_spacing_cols'] = spacing[2]
        return input_features


class Clip_Images_By_Extension(ImageProcessor):
    def __init__(self, extension=np.inf):
        self.extension = extension

    def parse(self, input_features):
        image = input_features['image']
        annotation = input_features['annotation']
        start, stop = get_start_stop(annotation, self.extension)
        if start != -1 and stop != -1:
            image, annotation = image[start:stop, ...], annotation[start:stop, ...]
        input_features['image'] = image
        input_features['annotation'] = annotation.astype('int8')
        return input_features


class Normalize_MRI(ImageProcessor):
    def parse(self, input_features):
        image_handle = sitk.GetImageFromArray(input_features['image'])
        image = input_features['image']

        normalizationFilter = sitk.IntensityWindowingImageFilter()
        upperPerc = np.percentile(image, 99)
        lowerPerc = np.percentile(image, 1)

        normalizationFilter.SetOutputMaximum(255.0)
        normalizationFilter.SetOutputMinimum(0.0)
        normalizationFilter.SetWindowMaximum(upperPerc)
        normalizationFilter.SetWindowMinimum(lowerPerc)

        normalizedImage = normalizationFilter.Execute(image_handle)

        image = sitk.GetArrayFromImage(normalizedImage)
        input_features['image'] = image
        return input_features


class N4BiasCorrection(ImageProcessor):
    def parse(self, input_features):
        image_handle = sitk.GetImageFromArray(input_features['image'])
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([int(2) * 2])
        try:
            N4_normalized_image = corrector.Execute(image_handle)
        except RuntimeError:
            N4_normalized_image = corrector.Execute(image_handle)
        input_features['image'] = sitk.GetArrayFromImage(N4_normalized_image)
        return input_features


class Split_Disease_Into_Cubes(ImageProcessor):
    def __init__(self, disease_annotation=None, cube_size=(16, 120, 120), min_voxel_volume=0, max_voxels=np.inf):
        '''
        :param disease_annotation: integer for disease annotation
        '''
        self.disease_annotation = disease_annotation
        self.cube_size = np.asarray(cube_size)
        self.min_voxel_volume = min_voxel_volume
        self.max_voxels = max_voxels
        assert disease_annotation is not None, 'Provide an integer for what is disease'

    def parse(self, input_features):
        if 'bounding_boxes_{}'.format(self.disease_annotation) not in input_features:
            Add_Bounding_Box = Add_Bounding_Box_Indexes([self.disease_annotation], add_to_dictionary=False)
            input_features = Add_Bounding_Box.parse(input_features)
        if 'bounding_boxes_{}'.format(self.disease_annotation) in input_features:
            bounding_boxes = input_features['bounding_boxes_{}'.format(self.disease_annotation)]
            voxel_volumes = input_features['voxel_volumes_{}'.format(self.disease_annotation)]
            del input_features['voxel_volumes_{}'.format(self.disease_annotation)]
            del input_features['bounding_boxes_{}'.format(self.disease_annotation)]
            image_base = input_features['image']
            annotation_base = input_features['annotation']
            out_features = OrderedDict()
            for cube_index, [box, voxels] in enumerate(zip(bounding_boxes, voxel_volumes)):
                if voxels < self.min_voxel_volume or voxels > self.max_voxels:
                    continue
                z_start, z_stop, r_start, r_stop, c_start, c_stop = add_bounding_box_to_dict(box, return_indexes=True)
                box_size = [z_stop - z_start, r_stop - r_start, c_stop - c_start]
                remainders = np.asarray([self.cube_size[i] - box_size[i] % self.cube_size[i]
                                         if box_size[i] % self.cube_size[i] != 0 else 0 for i in range(3)])
                z_start, z_stop, r_start, r_stop, c_start, c_stop = expand_box_indexes(z_start, z_stop, r_start, r_stop,
                                                                                       c_start, c_stop,
                                                                                       annotation_shape=
                                                                                       annotation_base.shape,
                                                                                       bounding_box_expansion=
                                                                                       remainders)
                image = image_base[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                annotation = annotation_base[z_start:z_stop, r_start:r_stop, c_start:c_stop]

                stack_image, stack_annotation = [image[None, ...]], [annotation[None, ...]]
                for axis in range(3):
                    output_images = []
                    output_annotations = []
                    for i in stack_image:
                        split = i.shape[axis + 1] // self.cube_size[axis]
                        if split > 1:
                            output_images += np.array_split(i, split, axis=axis + 1)
                        else:
                            output_images += [i]
                    for i in stack_annotation:
                        split = i.shape[axis + 1] // self.cube_size[axis]
                        if split > 1:
                            output_annotations += np.array_split(i, split, axis=axis + 1)
                        else:
                            output_annotations += [i]
                    stack_image = output_images
                    stack_annotation = output_annotations
                for box_index, [image_cube, annotation_cube] in enumerate(zip(stack_image, stack_annotation)):
                    temp_feature = OrderedDict()
                    image_cube, annotation_cube = image_cube[0], annotation_cube[0]
                    temp_feature['image'] = image_cube[:self.cube_size[0]]
                    temp_feature['annotation'] = annotation_cube[:self.cube_size[0]]
                    for key in input_features:  # Bring along anything else we care about
                        if key not in temp_feature.keys():
                            temp_feature[key] = input_features[key]
                    out_features['Disease_Box_{}_{}'.format(cube_index, box_index)] = temp_feature
            input_features = out_features
            return input_features
        return input_features


class Distribute_into_3D(ImageProcessor):
    def __init__(self, min_z=0, max_z=np.inf, max_rows=np.inf, max_cols=np.inf, mirror_small_bits=True,
                 chop_ends=False, desired_val=1):
        self.max_z = max_z
        self.min_z = min_z
        self.max_rows, self.max_cols = max_rows, max_cols
        self.mirror_small_bits = mirror_small_bits
        self.chop_ends = chop_ends
        self.desired_val = desired_val

    def parse(self, input_features):
        out_features = OrderedDict()
        start_chop = 0
        image_base = input_features['image']
        annotation_base = input_features['annotation']
        image_path = input_features['image_path']
        spacing = input_features['spacing']
        z_images_base, rows, cols = image_base.shape
        if self.max_rows != np.inf:
            rows = min([rows, self.max_rows])
        if self.max_cols != np.inf:
            cols = min([cols, self.max_cols])
        image_base, annotation_base = image_base[:, :rows, :cols], annotation_base[:, :rows, :cols]
        step = min([self.max_z, z_images_base])
        for index in range(z_images_base // step + 1):
            image_features = OrderedDict()
            if start_chop >= z_images_base:
                continue
            image = image_base[start_chop:start_chop + step, ...]
            annotation = annotation_base[start_chop:start_chop + step, ...]
            start_chop += step
            if image.shape[0] < max([step, self.min_z]):
                if self.mirror_small_bits:
                    while image.shape[0] < max([step, self.min_z]):
                        mirror_image = np.flip(image, axis=0)
                        mirror_annotation = np.flip(annotation, axis=0)
                        image = np.concatenate([image, mirror_image], axis=0)
                        annotation = np.concatenate([annotation, mirror_annotation], axis=0)
                    image = image[:max([step, self.min_z])]
                    annotation = annotation[:max([step, self.min_z])]
                elif self.chop_ends:
                    continue
            start, stop = get_start_stop(annotation, extension=0, desired_val=self.desired_val)
            if start == -1 or stop == -1:
                continue  # no annotation here
            image_features['image_path'] = image_path
            image_features['image'] = image
            image_features['annotation'] = annotation
            image_features['start'] = start
            image_features['stop'] = stop
            image_features['spacing'] = spacing
            for key in input_features.keys():
                if key not in image_features.keys():
                    image_features[key] = input_features[key]  # Pass along all other keys.. be careful
            out_features['Image_{}'.format(index)] = image_features
        input_features = out_features
        return input_features


class Distribute_into_2D(ImageProcessor):

    def parse(self, input_features):
        out_features = OrderedDict()
        image = input_features['image']
        annotation = input_features['annotation']
        image_path = input_features['image_path']
        spacing = input_features['spacing']
        z_images_base, rows, cols = annotation.shape[:3]
        if len(annotation.shape) > 3:
            input_features['num_classes'] = annotation.shape[-1]
        for index in range(z_images_base):
            image_features = OrderedDict()
            image_features['image_path'] = image_path
            image_features['image'] = image[index]
            image_features['annotation'] = annotation[index]
            image_features['spacing'] = spacing[:-1]
            for key in input_features.keys():
                if key not in image_features.keys():
                    image_features[key] = input_features[key]  # Pass along all other keys.. be careful
            out_features['Image_{}'.format(index)] = image_features
        input_features = out_features
        return input_features


class NormalizeParotidMR(ImageProcessor):
    def parse(self, input_features):
        images = input_features['image']
        data = images.flatten()
        counts, bins = np.histogram(data, bins=1000)
        count_index = 0
        count_value = 0
        while count_value / np.sum(counts) < .3:  # Throw out the bottom 30 percent of data, as that is usually just 0s
            count_value += counts[count_index]
            count_index += 1
        min_bin = bins[count_index]
        data = data[data > min_bin]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val) / std_val
        input_features['image'] = images
        return input_features


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


class AddByValues(ImageProcessor):
    def __init__(self, image_keys=('image',), values=(1.,)):
        """
        :param image_keys: tuple of keys to divide by the value
        :param values: values by which to add by
        """
        self.image_keys = image_keys
        self.values = values

    def parse(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key, value in zip(self.image_keys, self.values):
            image_array = input_features[key]
            image_array += value
            input_features[key] = image_array
        return input_features


class DistributeIntoRecurrenceCubes(ImageProcessor):
    def __init__(self, rows=128, cols=128, images=32):
        self.rows, self.cols, self.images = rows, cols, images
    """
    Highly specialized for the task of model prediction, likely won't be useful for others
    """
    def parse(self, input_features):
        out_features = OrderedDict()
        primary_array = input_features['primary_image']
        image_size = primary_array.shape
        secondary_array = input_features['secondary_image']
        primary_mask = input_features['primary_mask']
        '''
        Now, find centroids in the cases
        '''
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        stats = sitk.LabelShapeStatisticsImageFilter()

        no_recurred_image = sitk.GetImageFromArray((primary_mask == 1).astype('int'))
        connected_image = Connected_Component_Filter.Execute(no_recurred_image)
        stats.Execute(connected_image)
        no_recurrence_centroids = [no_recurred_image.TransformPhysicalPointToIndex(stats.GetCentroid(l))
                                   for l in stats.GetLabels()]

        recurred_image = sitk.GetImageFromArray((primary_mask == 2).astype('int'))
        connected_image = Connected_Component_Filter.Execute(recurred_image)
        stats.Execute(connected_image)
        recurrence_centroids = [recurred_image.TransformPhysicalPointToIndex(stats.GetCentroid(l))
                                for l in stats.GetLabels()]
        for value, cube_name, centroids in zip([0, 1], ['Non_Recurrence_Cube_{}', 'Recurrence_Cube_{}'],
                                               [no_recurrence_centroids, recurrence_centroids]):
            for index, centroid in enumerate(centroids):
                temp_feature = OrderedDict()
                col_center, row_center, z_center = centroid
                z_start = max([0, z_center - self.images // 2])
                z_stop = min([image_size[0], z_center + self.images // 2])
                row_start = max([0, row_center - self.rows // 2])
                row_stop = min([image_size[1], row_center + self.rows // 2])
                col_start = max([0, col_center - self.cols // 2])
                col_stop = min([image_size[2], col_center + self.cols // 2])
                primary_cube = primary_array[z_start:z_stop, row_start:row_stop, col_start:col_stop]
                secondary_cube = secondary_array[z_start:z_stop, row_start:row_stop, col_start:col_stop]
                out_cube = np.stack([primary_cube, secondary_cube], axis=-1)
                img_shape = out_cube.shape
                pads = [self.images - img_shape[0], self.rows - img_shape[1], self.cols - img_shape[2], 0]
                pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
                out_cube = np.pad(out_cube, pads, constant_values=np.min(out_cube))
                temp_feature['image'] = out_cube
                temp_feature['annotation'] = to_categorical(value, 2)
                wanted_keys = ('primary_image_path', 'out_path', 'out_file', 'spacing')
                for key in wanted_keys:  # Bring along anything else we care about
                    if key not in temp_feature.keys():
                        temp_feature[key] = input_features[key]
                out_features[cube_name.format(index)] = temp_feature
        return out_features


class DivideByValues(ImageProcessor):
    def __init__(self, image_keys=('image',), values=(1.,)):
        """
        :param image_keys: tuple of keys to divide by the value
        :param values: values by which to divide by
        """
        self.image_keys = image_keys
        self.values = values

    def parse(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key, value in zip(self.image_keys, self.values):
            image_array = input_features[key]
            image_array /= value
            input_features[key] = image_array
        return input_features


class Threshold_Images(ImageProcessor):
    def __init__(self, image_key='image', lower_bound=-np.inf, upper_bound=np.inf, divide=True):
        """
        :param image_key: key for images in the image_features dictionary
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        """
        self.lower = lower_bound
        self.upper = upper_bound
        self.image_key = image_key
        self.divide = divide

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.image_key)
        image = image_features[self.image_key]
        image[image < self.lower] = self.lower
        image[image > self.upper] = self.upper
        if self.divide:
            image = image / (self.upper - self.lower)
        image_features[self.image_key] = image
        return image_features


class Normalize_to_annotation(ImageProcessor):
    def __init__(self, image_key='image', annotation_key='annotation', annotation_value_list=None, mirror_max=False,
                 lower_percentile=None, upper_percentile=None):
        """
        :param image_key: key which corresponds to an image to be normalized
        :param annotation_key: key which corresponds to an annotation image used for normalization
        :param annotation_value_list: a list of values that you want to be normalized across
        :param mirror_max:
        :param lower_percentile:
        :param upper_percentile:
        """
        assert annotation_value_list is not None, 'Need to provide a list of values'
        self.annotation_value_list = annotation_value_list
        self.mirror_max = mirror_max
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.image_key = image_key
        self.annotation_key = annotation_key

    def parse(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.image_key, self.annotation_key))
        images = input_features[self.image_key]
        annotation = input_features[self.annotation_key]
        if len(annotation.shape) == 3:
            mask = np.zeros(annotation.shape)
            for value in self.annotation_value_list:
                mask += annotation == value
        else:
            mask = np.zeros(annotation.shape[:-1])
            for value in self.annotation_value_list:
                mask += annotation[..., value]
        data = images[mask > 0].flatten()
        if self.lower_percentile is not None and self.upper_percentile is not None:
            lower_bound = np.percentile(data, 25)
            upper_bound = np.percentile(data, 75)
            data = data[np.where((data >= lower_bound) & (data <= upper_bound))]
            mean_val, std_val = np.mean(data), np.std(data)
            images = (images - mean_val) / std_val
            input_features[self.image_key] = images
            return input_features
        counts, bins = np.histogram(data, bins=100)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        peak = bins[count_index]
        data_reduced = data[np.where((data > peak - 150) & (data < peak + 150))]
        counts, bins = np.histogram(data_reduced, bins=1000)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        half_counts = counts - np.max(counts) // 2
        half_upper = np.abs(half_counts[count_index + 1:])
        max_50 = np.where(half_upper == np.min(half_upper))[0][0]

        half_lower = np.abs(half_counts[:count_index - 1][-1::-1])
        min_50 = np.where(half_lower == np.min(half_lower))[0][0]

        min_values = bins[count_index - min_50]
        if self.mirror_max:
            min_values = bins[count_index - max_50]  # Good for non-normal distributions, just mirror the other FWHM
        max_values = bins[count_index + max_50]
        data = data[np.where((data >= min_values) & (data <= max_values))]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val) / std_val
        input_features[self.image_key] = images
        return input_features


def expand_box_indexes(z_start, z_stop, r_start, r_stop, c_start, c_stop, annotation_shape, bounding_box_expansion):
    z_start = max([0, z_start - floor(bounding_box_expansion[0] / 2)])
    z_stop = min([annotation_shape[0], z_stop + ceil(bounding_box_expansion[0] / 2)])
    r_start = max([0, r_start - floor(bounding_box_expansion[1] / 2)])
    r_stop = min([annotation_shape[1], r_stop + ceil(bounding_box_expansion[1] / 2)])
    c_start = max([0, c_start - floor(bounding_box_expansion[2] / 2)])
    c_stop = min([annotation_shape[2], c_stop + ceil(bounding_box_expansion[2] / 2)])
    return z_start, z_stop, r_start, r_stop, c_start, c_stop


class Box_Images(ImageProcessor):
    def __init__(self, image_key='image', annotation_key='annotation', wanted_vals_for_bbox=None,
                 bounding_box_expansion=(5, 10, 10), power_val_z=1, power_val_r=1,
                 power_val_c=1, min_images=None, min_rows=None, min_cols=None):
        """
        :param image_key: key which corresponds to an image to be normalized
        :param annotation_key: key which corresponds to an annotation image used for normalization
        :param wanted_vals_for_bbox:
        :param bounding_box_expansion:
        :param power_val_z:
        :param power_val_r:
        :param power_val_c:
        :param min_images:
        :param min_rows:
        :param min_cols:
        """
        assert type(wanted_vals_for_bbox) is list, 'Provide a list for bboxes'
        self.wanted_vals_for_bbox = wanted_vals_for_bbox
        self.bounding_box_expansion = bounding_box_expansion
        self.power_val_z, self.power_val_r, self.power_val_c = power_val_z, power_val_r, power_val_c
        self.min_images, self.min_rows, self.min_cols = min_images, min_rows, min_cols
        self.image_key, self.annotation_key = image_key, annotation_key

    def parse(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.image_key, self.annotation_key))
        annotation = input_features[self.annotation_key]
        image = input_features[self.image_key]
        if len(annotation.shape) > 3:
            mask = np.zeros(annotation.shape[:-1])
            argmax_annotation = np.argmax(annotation, axis=-1)
            for val in self.wanted_vals_for_bbox:
                mask[argmax_annotation == val] = 1
        else:
            mask = np.zeros(annotation.shape)
            for val in self.wanted_vals_for_bbox:
                mask[annotation == val] = 1
        for val in [1]:
            add_indexes = Add_Bounding_Box_Indexes([val], label_name='mask')
            input_features['mask'] = mask
            add_indexes.parse(input_features)
            del input_features['mask']
            z_start, z_stop, r_start, r_stop, c_start, c_stop = add_bounding_box_to_dict(
                input_features['bounding_boxes_{}'.format(val)][0], return_indexes=True)

            z_start, z_stop, r_start, r_stop, c_start, c_stop = expand_box_indexes(z_start, z_stop, r_start, r_stop,
                                                                                   c_start, c_stop,
                                                                                   annotation_shape=annotation.shape,
                                                                                   bounding_box_expansion=
                                                                                   self.bounding_box_expansion)

            z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
            remainder_z, remainder_r, remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                    self.power_val_r - r_total % self.power_val_r if r_total % self.power_val_r != 0 else 0, \
                                                    self.power_val_c - c_total % self.power_val_c if c_total % self.power_val_c != 0 else 0
            remainders = np.asarray([remainder_z, remainder_r, remainder_c])
            z_start, z_stop, r_start, r_stop, c_start, c_stop = expand_box_indexes(z_start, z_stop, r_start, r_stop,
                                                                                   c_start, c_stop,
                                                                                   annotation_shape=
                                                                                   annotation.shape,
                                                                                   bounding_box_expansion=
                                                                                   remainders)
            min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
            remainders = [0, 0, 0]
            if self.min_images is not None:
                remainders[0] = max([0, self.min_images - min_images])
                min_images = max([min_images, self.min_images])
            if self.min_rows is not None:
                remainders[1] = max([0, self.min_rows - min_rows])
                min_rows = max([min_rows, self.min_rows])
            if self.min_cols is not None:
                remainders[2] = max([0, self.min_cols - min_cols])
                min_cols = max([min_cols, self.min_cols])
            remainders = np.asarray(remainders)
            z_start, z_stop, r_start, r_stop, c_start, c_stop = expand_box_indexes(z_start, z_stop, r_start, r_stop,
                                                                                   c_start, c_stop,
                                                                                   annotation_shape=
                                                                                   annotation.shape,
                                                                                   bounding_box_expansion=
                                                                                   remainders)
            image_cube = image[z_start:z_stop, r_start:r_stop, c_start:c_stop]
            annotation_cube = annotation[z_start:z_stop, r_start:r_stop, c_start:c_stop]
            img_shape = image_cube.shape
            pads = [min_images - img_shape[0], min_rows - img_shape[1], min_cols - img_shape[2]]
            pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
            image_cube = np.pad(image_cube, pads, constant_values=np.min(image_cube))
            if len(annotation.shape) > 3:
                pads += [[0, 0]]
            annotation_cube = np.pad(annotation_cube, pads)
            if len(annotation.shape) > 3:
                annotation_cube[..., 0] = 1 - np.sum(annotation_cube[..., 1:], axis=-1)
            input_features[self.annotation_key] = annotation_cube
            input_features[self.image_key] = image_cube
        return input_features


class Add_Bounding_Box_Indexes(ImageProcessor):
    def __init__(self, wanted_vals_for_bbox=None, add_to_dictionary=False, label_name='annotation'):
        '''
        :param wanted_vals_for_bbox: a list of values in integer form for bboxes
        '''
        assert type(wanted_vals_for_bbox) is list, 'Provide a list for bboxes'
        self.wanted_vals_for_bbox = wanted_vals_for_bbox
        self.add_to_dictionary = add_to_dictionary
        self.label_name = label_name

    def parse(self, input_features):
        _check_keys_(input_features=input_features, keys=self.label_name)
        annotation_base = input_features[self.label_name]
        for val in self.wanted_vals_for_bbox:
            temp_val = val
            if len(annotation_base.shape) > 3:
                annotation = (annotation_base[..., val] > 0).astype('int')
                temp_val = 1
            else:
                annotation = annotation_base
            slices = np.where(annotation == temp_val)
            if slices:
                bounding_boxes, voxel_volumes = get_bounding_boxes(sitk.GetImageFromArray(annotation), temp_val)
                input_features['voxel_volumes_{}'.format(val)] = voxel_volumes
                input_features['bounding_boxes_{}'.format(val)] = bounding_boxes
                input_features = add_bounding_box_to_dict(input_features=input_features, bounding_box=bounding_boxes[0],
                                                          val=val, return_indexes=False,
                                                          add_to_dictionary=self.add_to_dictionary)
        return input_features


def add_bounding_box_to_dict(bounding_box, input_features=None, val=None, return_indexes=False,
                             add_to_dictionary=False):
    c_start, r_start, z_start, c_stop, r_stop, z_stop = bounding_box
    z_stop, r_stop, c_stop = z_start + z_stop, r_start + r_stop, c_start + c_stop
    if return_indexes:
        return z_start, z_stop, r_start, r_stop, c_start, c_stop
    if add_to_dictionary:
        input_features['bounding_boxes_z_start_{}'.format(val)] = z_start
        input_features['bounding_boxes_r_start_{}'.format(val)] = r_start
        input_features['bounding_boxes_c_start_{}'.format(val)] = c_start
        input_features['bounding_boxes_z_stop_{}'.format(val)] = z_stop
        input_features['bounding_boxes_r_stop_{}'.format(val)] = r_stop
        input_features['bounding_boxes_c_stop_{}'.format(val)] = c_stop
    return input_features


if __name__ == '__main__':
    pass
