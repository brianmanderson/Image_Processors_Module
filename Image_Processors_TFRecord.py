__author__ = 'Brian M Anderson'
# Created on 4/28/2020
import SimpleITK as sitk
import numpy as np
from _collections import OrderedDict
from .Resample_Class.Resample_Class import Resample_Class_Object
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import os, pickle
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, plt



class Image_Processor(object):
    def parse(self, input_features):
        return input_features


def save_obj(path, obj): # Save almost anything.. dictionary, list, etc.
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
    feature = OrderedDict()
    for key in base_dictionary:
        data = base_dictionary[key]
        if type(data) is int:
            feature[key] = _int64_feature(tf.constant(data, dtype='int64'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.int64)
        elif type(data) is np.ndarray:
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


def serialize_example(image_path, annotation_path, image_processors=None, record_writer=None, verbose=False):
    get_features(image_path,annotation_path, image_processors=image_processors, record_writer=record_writer,
                 verbose=verbose)


class Record_Writer(Image_Processor):
    def __init__(self, file_path=None):
        assert file_path is not None, "You need to pass a base file path..."
        self.file_path = file_path
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def parse(self, input_features):
        keys = list(input_features.keys())
        image_name = os.path.split(input_features[keys[0]]['image_path'])[-1].split('.nii')[0]
        filename = os.path.join(self.file_path,'{}.tfrecord'.format(image_name))
        features = OrderedDict()
        d_type = OrderedDict()
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


def get_features(image_path, annotation_path, image_processors=None, record_writer=None, verbose=0):
    features = OrderedDict()
    features['image_path'] = image_path
    features['annotation_path'] = annotation_path
    if image_processors is not None:
        for image_processor in image_processors:
            features, _ = down_dictionary(features, OrderedDict(), 0)
            if verbose:
                print(image_processor)
            for key in features.keys():
                features[key] = image_processor.parse(features[key])
        features, _ = down_dictionary(features, OrderedDict(), 0)
    record_writer.parse(features)


def down_dictionary(input_dictionary, out_dictionary=OrderedDict(), out_index=0):
    if 'image_path' in input_dictionary.keys():
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
        annotation = np.argmax(annotation,axis=-1)
    non_zero_values = np.where(np.max(annotation,axis=(1,2)) >= desired_val)[0]
    start, stop = -1, -1
    if non_zero_values.any():
        start = int(non_zero_values[0])
        stop = int(non_zero_values[-1])
        start = max([start - extension, 0])
        stop = min([stop + extension, annotation.shape[0]])
    return start, stop


def get_bounding_boxes(annotation_handle,value):
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    stats = sitk.LabelShapeStatisticsImageFilter()
    thresholded_image = sitk.BinaryThreshold(annotation_handle,lowerThreshold=value,upperThreshold=value+1)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
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
            sitk.BinaryThreshold(sitk.Cast(annotation_handle,sitk.sitkFloat32), lowerThreshold=0.01,
                                 upperThreshold=np.inf))
        label_image = self.RelabelComponent.Execute(label_image)
        output = sitk.BinaryThreshold(sitk.Cast(label_image,sitk.sitkFloat32), lowerThreshold=0.1,upperThreshold=1.0)
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
        for value in range(1, self.Connected_Component_Filter.GetObjectCount()+1):
            mask = sitk.GetArrayFromImage(connected_image == value)
            prob = np.max(image_slice[mask==1])
            if prob > current:
                current = prob
                out_mask = mask
        image_slice[out_mask==0] = 0
        return image_slice


class Gaussian_Uncertainty(Image_Processor):
    def __init__(self, sigma=None):
        '''
        :param sigma: Desired sigma, in mm, in z, x, y direction
        '''
        self.sigma = sigma

    def parse(self, input_features):
        remove_lowest_probability = Remove_Lowest_Probabilty_Structure()
        remove_smallest = Remove_Smallest_Structures()
        annotations = input_features['annotation']
        spacing = input_features['spacing']
        filtered = np.zeros(annotations.shape)
        filtered[...,0] = annotations[...,0]
        if len(annotations.shape) == 3:
            num_classes = np.max(annotations)
        else:
            num_classes = annotations.shape[-1]
        for i in range(1,num_classes):
            sigma = self.sigma[i-1]
            sigma = [sigma/spacing[0], sigma/spacing[1], sigma/spacing[2]]
            annotation = annotations[...,i]
            filtered[...,i] = gaussian_filter(annotation,sigma=sigma,mode='constant')
        filtered[annotations[...,0] == 1] = 0
        filtered[...,0] = annotations[...,0]
        # Now we've normed, but still have the problem that unconnected structures can still be there..
        for i in range(1,num_classes):
            annotation = filtered[...,i]
            annotation[annotation < 0.05] = 0
            slices = np.where(np.max(annotation,axis=(1,2))>0)
            for slice in slices[0]:
                annotation[slice] = remove_lowest_probability.remove_lowest_probability(annotation[slice])
            mask_handle = remove_smallest.remove_smallest_component(sitk.GetImageFromArray(annotation)>0)
            mask = sitk.GetArrayFromImage(mask_handle)
            masked_filter = filtered[...,i]*mask
            filtered[..., i] = masked_filter
        norm = np.sum(filtered, axis=-1)
        filtered[...,0] += (norm == 0).astype('int')
        norm[norm == 0] = 1
        filtered /= norm[...,None]
        input_features['annotation'] = filtered
        return input_features


class Combine_Annotations(Image_Processor):
    def __init__(self, annotation_input=[5,6,7,8], to_annotation=5):
        self.annotation_input = annotation_input
        self.to_annotation = to_annotation

    def parse(self, input_features):
        annotation = input_features['annotation']
        assert len(annotation.shape) == 3 or len(annotation.shape) == 4, 'To combine annotations the size has to be 3 or 4'
        if len(annotation.shape) == 3:
            for val in self.annotation_input:
                annotation[annotation == val] = self.to_annotation
        elif len(annotation.shape) == 4:
            annotation[..., self.to_annotation] += annotation[..., self.annotation_input]
            del annotation[..., self.annotation_input]
        input_features['annotation'] = annotation
        return input_features


class To_Categorical(Image_Processor):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def parse(self, input_features):
        annotation = input_features['annotation']
        input_features['annotation'] = to_categorical(annotation,self.num_classes)
        input_features['num_classes'] = self.num_classes
        return input_features


class Resample_LiTs(Image_Processor):
    def __init__(self, desired_output_spacing=(None,None,None)):
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
            resampler = Resample_Class_Object()
            print('Resampling {} to {}'.format(input_spacing,output_spacing))
            image_handle = resampler.resample_image(input_image=image_handle,input_spacing=input_spacing,
                                                    output_spacing=output_spacing,is_annotation=False)
            annotation_handle = resampler.resample_image(input_image=annotation_handle,input_spacing=input_spacing,
                                                         output_spacing=output_spacing,is_annotation=False)
            input_features['image'] = sitk.GetArrayFromImage(image_handle)
            input_features['annotation'] = sitk.GetArrayFromImage(annotation_handle)
            input_features['spacing'] = np.asarray(annotation_handle.GetSpacing(), dtype='float32')
        return input_features


class Resampler(Image_Processor):
    def __init__(self, desired_output_spacing=(None,None,None), make_512=False, binary_annotation=True):
        self.desired_output_spacing = desired_output_spacing
        self.binary_annotation = binary_annotation
        self.make_512 = make_512

    def parse(self, input_features):
        input_spacing = tuple([float(i) for i in input_features['spacing']])
        image_handle = sitk.GetImageFromArray(input_features['image'])
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
            resampler = Resample_Class_Object()
            print('Resampling {} to {}'.format(input_spacing,output_spacing))
            image_handle = resampler.resample_image(input_image=image_handle,input_spacing=input_spacing,
                                                    output_spacing=output_spacing,is_annotation=False)
            if len(input_features['annotation'].shape) == 3:
                annotation_handle = sitk.GetImageFromArray(input_features['annotation'])
                annotation_handle = resampler.resample_image(input_image=annotation_handle,input_spacing=input_spacing,
                                                             output_spacing=output_spacing,is_annotation=self.binary_annotation)
            else:
                annotation = input_features['annotation']
                output = []
                for i in range(annotation.shape[-1]):
                    output.append(resampler.resample_image(annotation[...,i], input_spacing=input_spacing,
                                                           output_spacing=output_spacing,
                                                           is_annotation=self.binary_annotation)[...,None])
                stacked = np.concatenate(output, axis=-1)
                stacked[...,0] = 1-np.sum(stacked[...,1:],axis=-1)
                annotation_handle = sitk.GetImageFromArray(stacked)
                annotation_handle.SetSpacing(image_handle.GetSpacing())
                annotation_handle.SetDirection(image_handle.GetDirection())
                annotation_handle.SetOrigin(image_handle.GetOrigin())
            input_features['image'] = sitk.GetArrayFromImage(image_handle)
            input_features['annotation'] = sitk.GetArrayFromImage(annotation_handle)
            input_features['spacing'] = np.asarray(annotation_handle.GetSpacing(), dtype='float32')
        return input_features


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
                image_features[key] = image_features[key].astype(self.key_type_dict[key])
        return image_features


class Add_Images_And_Annotations(Image_Processor):
    def parse(self, input_features):
        image_path = input_features['image_path']
        annotation_path = input_features['annotation_path']
        image_handle, annotation_handle = sitk.ReadImage(image_path), sitk.ReadImage(annotation_path)
        annotation = sitk.GetArrayFromImage(annotation_handle).astype('int8')
        image = sitk.GetArrayFromImage(image_handle).astype('float32')
        input_features['image'] = image
        input_features['annotation'] = annotation
        input_features['spacing'] = np.asarray(annotation_handle.GetSpacing(), dtype='float32')
        return input_features


class Add_Dose(Image_Processor):
    def parse(self, input_features):
        image_path = input_features['image_path']
        dose_path = image_path.replace('Data','Dose')
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


class Clip_Images_By_Extension(Image_Processor):
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


class Normalize_MRI(Image_Processor):
    def parse(self, input_features):
        image_handle = sitk.GetImageFromArray(input_features['image'])
        image = input_features['image']

        normalizationFilter = sitk.IntensityWindowingImageFilter()
        upperPerc = np.percentile(image, 99)
        lowerPerc = np.percentile(image,1)

        normalizationFilter.SetOutputMaximum(255.0)
        normalizationFilter.SetOutputMinimum(0.0)
        normalizationFilter.SetWindowMaximum(upperPerc)
        normalizationFilter.SetWindowMinimum(lowerPerc)

        normalizedImage = normalizationFilter.Execute(image_handle)

        image = sitk.GetArrayFromImage(normalizedImage)
        input_features['image'] = image
        return input_features


class N4BiasCorrection(Image_Processor):
    def parse(self, input_features):
        image_handle = sitk.GetImageFromArray(input_features['image'])
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([int(2)*2])
        try:
            N4_normalized_image = corrector.Execute(image_handle)
        except RuntimeError:
            N4_normalized_image = corrector.Execute(image_handle)
        input_features['image'] = sitk.GetArrayFromImage(N4_normalized_image)
        return input_features



class Split_Disease_Into_Cubes(Image_Processor):
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
                                                                                       remainders//2+1)
                image = image_base[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                annotation = annotation_base[z_start:z_stop, r_start:r_stop, c_start:c_stop]

                stack_image, stack_annotation = [image[None,...]], [annotation[None,...]]
                for axis in range(3):
                    output_images = []
                    output_annotations = []
                    for i in stack_image:
                        split = i.shape[axis+1] // self.cube_size[axis]
                        if split > 1:
                            output_images += np.array_split(i, split, axis=axis+1)
                        else:
                            output_images += [i]
                    for i in stack_annotation:
                        split = i.shape[axis+1] // self.cube_size[axis]
                        if split > 1:
                            output_annotations += np.array_split(i, split, axis=axis+1)
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


class Distribute_into_3D(Image_Processor):
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
            rows = min([rows,self.max_rows])
        if self.max_cols != np.inf:
            cols = min([cols,self.max_cols])
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
                continue # no annotation here
            image_features['image_path'] = image_path
            image_features['image'] = image
            image_features['annotation'] = annotation
            image_features['start'] = start
            image_features['stop'] = stop
            image_features['z_images'] = image.shape[0]
            image_features['rows'] = image.shape[1]
            image_features['cols'] = image.shape[2]
            image_features['spacing'] = spacing
            for key in input_features.keys():
                if key not in image_features.keys():
                    image_features[key] = input_features[key] # Pass along all other keys.. be careful
            out_features['Image_{}'.format(index)] = image_features
        input_features = out_features
        return input_features


class Distribute_into_2D(Image_Processor):

    def parse(self, input_features):
        out_features = OrderedDict()
        image = input_features['image']
        annotation = input_features['annotation']
        image_path = input_features['image_path']
        spacing = input_features['spacing']
        z_images_base, rows, cols = annotation.shape
        for index in range(z_images_base):
            image_features = OrderedDict()
            image_features['image_path'] = image_path
            image_features['image'] = image[index]
            image_features['annotation'] = annotation[index]
            image_features['rows'] = rows
            image_features['cols'] = cols
            image_features['spacing'] = spacing[:-1]
            out_features['Image_{}'.format(index)] = image_features
        input_features = out_features
        return input_features


class NormalizeParotidMR(Image_Processor):
    def parse(self, input_features):
        images = input_features['image']
        data = images.flatten()
        counts, bins = np.histogram(data, bins=1000)
        count_index = 0
        count_value = 0
        while count_value/np.sum(counts) < .3: # Throw out the bottom 30 percent of data, as that is usually just 0s
            count_value += counts[count_index]
            count_index += 1
        min_bin = bins[count_index]
        data = data[data>min_bin]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val)/std_val
        input_features['image'] = images
        return input_features


class Normalize_to_annotation(Image_Processor):
    def __init__(self, annotation_value_list=None, mirror_max=False, lower_percentile=None, upper_percentile=None):
        '''
        :param annotation_value: mask values to normalize over, [1]
        '''
        assert annotation_value_list is not None, 'Need to provide a list of values'
        self.annotation_value_list = annotation_value_list
        self.mirror_max = mirror_max
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def parse(self, input_features):
        images = input_features['image']
        annotation = input_features['annotation']
        mask = np.zeros(annotation.shape)
        for value in self.annotation_value_list:
            mask += annotation == value
        data = images[mask > 0].flatten()
        if self.lower_percentile is not None and self.upper_percentile is not None:
            lower_bound = np.percentile(data,25)
            upper_bound = np.percentile(data,75)
            data = data[np.where((data >= lower_bound) & (data <= upper_bound))]
            mean_val, std_val = np.mean(data), np.std(data)
            images = (images - mean_val) / std_val
            input_features['image'] = images
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
        input_features['image'] = images
        return input_features


def expand_box_indexes(z_start, z_stop, r_start, r_stop, c_start, c_stop, annotation_shape, bounding_box_expansion):
    z_start = max([0, z_start - bounding_box_expansion[0]])
    z_stop = min([annotation_shape[0], z_stop + bounding_box_expansion[0]])
    r_start = max([0, r_start - bounding_box_expansion[1]])
    r_stop = min([annotation_shape[1], r_stop + bounding_box_expansion[1]])
    c_start = max([0, c_start - bounding_box_expansion[2]])
    c_stop = min([annotation_shape[2], c_stop + bounding_box_expansion[2]])
    return z_start, z_stop, r_start, r_stop, c_start, c_stop


class Box_Images(Image_Processor):
    def __init__(self, wanted_vals_for_bbox=None,
                 bounding_box_expansion=(5,10,10), power_val_z=1, power_val_r=1,
                 power_val_c=1, min_images=None, min_rows=None, min_cols=None):
        '''
        :param wanted_vals_for_bbox: a list of values in integer form for bboxes
        :param box_imaages_and_annotations: True/False box up the images now?
        '''
        assert type(wanted_vals_for_bbox) is list, 'Provide a list for bboxes'
        self.wanted_vals_for_bbox=wanted_vals_for_bbox
        self.bounding_box_expansion = bounding_box_expansion
        self.power_val_z, self.power_val_r, self.power_val_c = power_val_z, power_val_r, power_val_c
        self.min_images, self.min_rows, self.min_cols = min_images, min_rows, min_cols

    def parse(self, input_features):
        annotation = input_features['annotation']
        image = input_features['image']
        if len(annotation.shape) > 3:
            mask = np.zeros(annotation.shape[:-1])
            argmax_annotation = np.argmax(annotation, axis=-1)
            for val in self.wanted_vals_for_bbox:
                mask[argmax_annotation==val] = 1
        else:
            mask = np.zeros(annotation.shape)
            for val in self.wanted_vals_for_bbox:
                mask[annotation == val] = 1
        for val in [1]:
            add_indexes = Add_Bounding_Box_Indexes([val],label_name='mask')
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
                                                                                   remainders // 2 + 1)
            min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
            if self.min_images is not None:
                min_images = max([min_images, self.min_images])
            if self.min_rows is not None:
                min_rows = max([min_rows, self.min_rows])
            if self.min_cols is not None:
                min_cols = max([min_cols, self.min_cols])
            out_images = np.ones([min_images, min_rows, min_cols], dtype=image.dtype) * np.min(image)
            if len(annotation.shape) > 3:
                out_annotations = np.zeros([min_images, min_rows, min_cols, annotation.shape[-1]], dtype=annotation.dtype)
                out_annotations[..., 0] = 1
            else:
                out_annotations = np.zeros([min_images, min_rows, min_cols], dtype=annotation.dtype)
            image_cube = image[z_start:z_start + min_images, r_start:r_start + min_rows, c_start:c_start + min_cols]
            annotation_cube = annotation[z_start:z_start + min_images, r_start:r_start + min_rows,
                              c_start:c_start + min_cols]
            img_shape = image_cube.shape
            out_images[:img_shape[0], :img_shape[1], :img_shape[2], ...] = image_cube
            out_annotations[:img_shape[0], :img_shape[1], :img_shape[2], ...] = annotation_cube
            input_features['annotation'] = out_annotations
            input_features['image'] = out_images
        return input_features


class Add_Bounding_Box_Indexes(Image_Processor):
    def __init__(self, wanted_vals_for_bbox=None, add_to_dictionary=False, label_name='annotation'):
        '''
        :param wanted_vals_for_bbox: a list of values in integer form for bboxes
        '''
        assert type(wanted_vals_for_bbox) is list, 'Provide a list for bboxes'
        self.wanted_vals_for_bbox=wanted_vals_for_bbox
        self.add_to_dictionary = add_to_dictionary
        self.label_name = label_name

    def parse(self, input_features):
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
                                                          val=val, return_indexes=False, add_to_dictionary=self.add_to_dictionary)
        return input_features


def add_bounding_box_to_dict(bounding_box, input_features=None, val=None, return_indexes=False, add_to_dictionary=False):
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
