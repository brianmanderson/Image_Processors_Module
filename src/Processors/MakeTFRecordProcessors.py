__author__ = 'Brian M Anderson'
# Created on 3/5/2021

import sys
import os.path

from tensorflow.python.data.ops.optional_ops import Optional

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import SimpleITK as sitk
import numpy as np
from collections import OrderedDict
from NiftiResampler.ResampleTools import ImageResampler
from scipy.ndimage.filters import gaussian_filter
import copy
from math import ceil, floor
import cv2
from skimage import morphology
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image, plt
from typing import List, Dict, Optional, Tuple


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


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


def get_bounding_boxes(annotation_handle, lower_threshold, upper_threshold=None):
    if upper_threshold is None:
        upper_threshold = lower_threshold + 1
    annotation_handle: sitk.Image
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    RelabelComponent = sitk.RelabelComponentImageFilter()
    RelabelComponent.SortByObjectSizeOn()
    stats = sitk.LabelShapeStatisticsImageFilter()
    thresholded_image = sitk.BinaryThreshold(annotation_handle, lowerThreshold=lower_threshold, upperThreshold=upper_threshold)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    connected_image = RelabelComponent.Execute(connected_image)
    stats.Execute(connected_image)
    bounding_boxes = [stats.GetBoundingBox(l) for l in stats.GetLabels()]
    num_voxels = np.asarray([stats.GetNumberOfPixels(l) for l in stats.GetLabels()]).astype('float32')
    return bounding_boxes, num_voxels


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if height is not None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    if width is not None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        if dim is None:
            dim = (width, int(h * r))
        else:
            dim = min([dim, (width, int(h * r))])

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def variable_remove_non_liver(annotations, threshold=0.5, is_liver=False):
    image_size_1 = annotations.shape[1]
    image_size_2 = annotations.shape[2]
    compare = copy.deepcopy(annotations)
    if is_liver:
        images_filt = gaussian_filter(copy.deepcopy(annotations), [0, .75, .75])
    else:
        images_filt = gaussian_filter(copy.deepcopy(annotations), [0, 1.5, 1.5])
    compare[compare < .01] = 0
    compare[compare > 0] = 1
    compare = compare.astype('int')
    for i in range(annotations.shape[0]):
        image = annotations[i, :, :]
        out_image = np.zeros([image_size_1,image_size_2])

        labels = morphology.label(compare[i, :, :],connectivity=1)
        for xxx in range(1,labels.max() + 1):
            overlap = image[labels == xxx]
            pred = sum(overlap)/overlap.shape[0]
            cutoff = threshold
            if pred < 0.75:
                cutoff = 0.15
            if cutoff != 0.95 and overlap.shape[0] < 500 and is_liver:
                k = copy.deepcopy(compare[i, :, :])
                k[k > cutoff] = 1
                out_image[labels == xxx] = k[labels == xxx]
            elif not is_liver:
                image_filt = images_filt[i, :, :]
                image_filt[image_filt < threshold] = 0
                image_filt[image_filt > 0] = 1
                image_filt = image_filt.astype('int')
                out_image[labels == xxx] = image_filt[labels == xxx]
            else:
                image_filt = images_filt[i, :, :]
                image_filt[image_filt < cutoff] = 0
                image_filt[image_filt > 0] = 1
                image_filt = image_filt.astype('int')
                out_image[labels == xxx] = image_filt[labels == xxx]
        annotations[i, :, :] = out_image
    return annotations


def remove_non_liver(annotations, threshold=0.5, max_volume=9999999.0, min_volume=0.0, max_area=99999.0, min_area=0.0,
                     do_3D = True, do_2D=False, spacing=None):
    '''
    :param annotations: An annotation of shape [Z_images, rows, columns]
    :param threshold: Threshold of probability from 0.0 to 1.0
    :param max_volume: Max volume of structure allowed
    :param min_volume: Minimum volume of structure allowed, in ccs
    :param max_area: Max volume of structure allowed
    :param min_area: Minimum volume of structure allowed
    :param do_3D: Do a 3D removal of structures, only take largest connected structure
    :param do_2D: Do a 2D removal of structures, only take largest connected structure
    :param spacing: Spacing of elements, in form of [z_spacing, row_spacing, column_spacing]
    :return: Masked annotation
    '''
    min_volume = min_volume * (10 * 10 * 10)  # cm to mm3
    annotations = copy.deepcopy(annotations)
    annotations = np.squeeze(annotations)
    if not annotations.dtype == 'int':
        annotations[annotations < threshold] = 0
        annotations[annotations > 0] = 1
        annotations = annotations.astype('int')
    if do_3D:
        labels = morphology.label(annotations, connectivity=1)
        if np.max(labels) > 1:
            area = []
            max_val = 0
            for i in range(1,labels.max()+1):
                new_area = labels[labels == i].shape[0]
                if spacing is not None:
                    volume = np.prod(spacing) * new_area
                    if volume > max_volume:
                        continue
                    elif volume < min_volume:
                        continue
                area.append(new_area)
                if new_area == max(area):
                    max_val = i
            labels[labels != max_val] = 0
            labels[labels > 0] = 1
            annotations = labels
    if do_2D:
        slice_indexes = np.where(np.sum(annotations,axis=(1,2))>0)
        if slice_indexes:
            for slice_index in slice_indexes[0]:
                labels = morphology.label(annotations[slice_index], connectivity=1)
                if np.max(labels) == 1:
                    continue
                area = []
                max_val = 0
                for i in range(1, labels.max() + 1):
                    new_area = labels[labels == i].shape[0]
                    if spacing is not None:
                        temp_area = np.prod(spacing[1:]) * new_area / 100
                        if temp_area > max_area:
                            continue
                        elif temp_area < min_area:
                            continue
                    area.append(new_area)
                    if new_area == max(area):
                        max_val = i
                labels[labels != max_val] = 0
                labels[labels > 0] = 1
                annotations[slice_index] = labels
    return annotations


class ImageProcessor(object):
    def pre_process(self, input_features):
        return input_features

    def post_process(self, input_features):
        return input_features


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

    def __repr__(self):
        return "Remove Smallest Structure"

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

    def __repr__(self):
        return "Remove Lowest Probability Structure"

class BinValuesConvertClass(object):
    def __init__(self, initial_min, initial_max, output_value) -> None:
        self.initial_min = initial_min
        self.initial_max = initial_max
        self.output_value = output_value


class Bin_Values(ImageProcessor):
    def __init__(self, change_keys, output_keys, bin_values: List[BinValuesConvertClass]):
        self.change_keys = change_keys
        self.output_keys = output_keys
        self.bin_values = bin_values

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.change_keys)
        for input_key, output_key in zip(self.change_keys, self.output_keys):
            value_to_change = input_features[input_key]
            for bin_class in self.bin_values:
                if value_to_change >= bin_class.initial_min and value_to_change <= bin_class.initial_max:
                    input_features[output_key] = bin_class.output_value
        return input_features


class Gaussian_Uncertainty(ImageProcessor):
    def __init__(self, sigma=None):
        '''
        :param sigma: Desired sigma, in mm, in x, y, z direction
        '''
        self.sigma = sigma

    def pre_process(self, input_features):
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


class AddConstant(ImageProcessor):
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
            image_features[key] = image_features[key] + value
        return image_features

    def __repr__(self):
        return f"Add constants {self.values} to {self.keys}"

class Combine_Annotations(ImageProcessor):
    def __init__(self, annotation_input=[5, 6, 7, 8], to_annotation=5):
        self.annotation_input = annotation_input
        self.to_annotation = to_annotation

    def pre_process(self, input_features):
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

    def __repr__(self):
        return f"Combine annotations from values {self.annotation_input} to {self.to_annotation}"

class To_Categorical(ImageProcessor):
    def __init__(self, num_classes=None, annotation_keys=('annotation',)):
        self.num_classes = num_classes
        self.annotation_keys = annotation_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.annotation_keys)
        for key in self.annotation_keys:
            annotation = input_features[key]
            input_features[key] = to_categorical(annotation, self.num_classes).astype(annotation.dtype)
            input_features['num_classes_{}'.format(key)] = self.num_classes
        return input_features

    def __repr__(self):
        return f"To categorical from {self.annotation_keys} to {self.num_classes} classes"

class ToCategorical(To_Categorical):
    def __init__(self, num_classes=None, annotation_keys=('annotation',)):
        super(ToCategorical, self).__init__(num_classes=num_classes, annotation_keys=annotation_keys)


class SmoothingPredictionRecursiveGaussian(ImageProcessor):
    def __init__(self, sigma=(0.1, 0.1, 0.0001), pred_axis=[1], prediction_key='prediction'):
        self.sigma = sigma
        self.pred_axis = pred_axis
        self.prediction_key = prediction_key

    def smooth(self, handle):
        return sitk.BinaryThreshold(sitk.SmoothingRecursiveGaussian(handle), lowerThreshold=.01, upperThreshold=np.inf)

    def post_process(self, input_features):
        pred = input_features[self.prediction_key]
        for axis in self.pred_axis:
            k = sitk.GetImageFromArray(pred[..., axis])
            k.SetSpacing(self.dicom_handle.GetSpacing())
            k = self.smooth(k)
            pred[..., axis] = sitk.GetArrayFromImage(k)
        input_features[self.prediction_key] = pred
        return input_features


class Iterate_Overlap(ImageProcessor):
    def __init__(self, on_liver_lobes=True, max_iterations=10, prediction_key='prediction', ground_truth_key='annotations',
                 dicom_handle_key='primary_handle'):
        self.max_iterations = max_iterations
        self.on_liver_lobes = on_liver_lobes
        MauererDistanceMap = sitk.SignedMaurerDistanceMapImageFilter()
        MauererDistanceMap.SetInsideIsPositive(True)
        MauererDistanceMap.UseImageSpacingOn()
        MauererDistanceMap.SquaredDistanceOff()
        self.MauererDistanceMap = MauererDistanceMap
        self.Remove_Smallest_Structure = Remove_Smallest_Structures()
        self.Smooth_Annotation = SmoothingPredictionRecursiveGaussian()
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key
        self.dicom_handle_key = dicom_handle_key

    def remove_56_78(self, annotations):
        amounts = np.sum(annotations, axis=(1, 2))
        indexes = np.where((np.max(amounts[:, (5, 6)], axis=-1) > 0) & (np.max(amounts[:, (7, 8)], axis=-1) > 0))
        if indexes:
            indexes = indexes[0]
            for i in indexes:
                if amounts[i, 5] < amounts[i, 8]:
                    annotations[i, ..., 8] += annotations[i, ..., 5]
                    annotations[i, ..., 5] = 0
                else:
                    annotations[i, ..., 5] += annotations[i, ..., 8]
                    annotations[i, ..., 8] = 0
                if amounts[i, 6] < amounts[i, 7]:
                    annotations[i, ..., 7] += annotations[i, ..., 6]
                    annotations[i, ..., 6] = 0
                else:
                    annotations[i, ..., 6] += annotations[i, ..., 7]
                    annotations[i, ..., 7] = 0
        return annotations

    def iterate_annotations(self, annotations_out, ground_truth_out, spacing, allowed_differences=50, z_mult=1):
        '''
        :param annotations:
        :param ground_truth:
        :param spacing:
        :param allowed_differences:
        :param max_iteration:
        :param z_mult: factor by which to ensure slices don't bleed into ones above and below
        :return:
        '''
        self.Remove_Smallest_Structure.spacing = self.dicom_handle.GetSpacing()
        self.Smooth_Annotation.spacing = self.dicom_handle.GetSpacing()
        annotations_out[ground_truth_out == 0] = 0
        min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(ground_truth_out)
        annotations = annotations_out[min_z:max_z, min_r:max_r, min_c:max_c, ...]
        ground_truth = ground_truth_out[min_z:max_z, min_r:max_r, min_c:max_c, ...]
        spacing[-1] *= z_mult
        differences = [np.inf]
        index = 0
        while differences[-1] > allowed_differences and index < self.max_iterations:
            index += 1
            print('Iterating {}'.format(index))
            # if self.on_liver_lobes:
            #     annotations = self.remove_56_78(annotations)
            previous_iteration = copy.deepcopy(np.argmax(annotations, axis=-1))
            for i in range(1, annotations.shape[-1]):
                annotation_handle = sitk.GetImageFromArray(annotations[..., i])
                annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
                pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(annotation_handle)
                annotations[..., i] = sitk.GetArrayFromImage(pruned_handle)
                slices = np.where(annotations[..., i] == 1)
                if slices:
                    slices = np.unique(slices[0])
                    for ii in range(len(slices)):
                        image_handle = sitk.GetImageFromArray(annotations[slices[ii], ..., i][None, ...])
                        pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(image_handle)
                        annotations[slices[ii], ..., i] = sitk.GetArrayFromImage(pruned_handle)

            annotations = self.make_distance_map(annotations, ground_truth, spacing=spacing)
            differences.append(np.abs(
                np.sum(previous_iteration[ground_truth == 1] - np.argmax(annotations, axis=-1)[ground_truth == 1])))
        annotations_out[min_z:max_z, min_r:max_r, min_c:max_c, ...] = annotations
        annotations_out[ground_truth_out == 0] = 0
        return annotations_out

    def run_distance_map(self, array, spacing):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        output = self.MauererDistanceMap.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

    def make_distance_map(self, pred, liver, reduce=True, spacing=(0.975, 0.975, 2.5)):
        '''
        :param pred: A mask of your predictions with N channels on the end, N=0 is background [# Images, rows, cols, N]
        :param liver: A mask of the desired region [# Images, rows, cols]
        :param reduce: Save time and only work on masked region
        :return:
        '''
        liver = np.squeeze(liver)
        pred = np.squeeze(pred)
        pred = np.round(pred).astype('int')
        min_z, min_r, min_c, max_z, max_r, max_c = 0, 0, 0, pred.shape[0], pred.shape[1], pred.shape[2]

        if reduce:
            min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(liver)
        reduced_pred = pred[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_liver = liver[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_output = np.zeros(reduced_pred.shape)
        for i in range(1, pred.shape[-1]):
            temp_reduce = reduced_pred[..., i]
            output = self.run_distance_map(temp_reduce, spacing)
            reduced_output[..., i] = output
        reduced_output[reduced_output > 0] = 0
        reduced_output = np.abs(reduced_output)
        reduced_output[..., 0] = np.inf
        output = np.zeros(reduced_output.shape, dtype='int')
        mask = reduced_liver == 1
        values = reduced_output[mask]
        output[mask, np.argmin(values, axis=-1)] = 1
        pred[min_z:max_z, min_r:max_r, min_c:max_c] = output
        return pred

    def pre_process(self, input_features):
        self.dicom_handle = input_features[self.dicom_handle_key]

    def post_process(self, input_features):
        self.dicom_handle = input_features[self.dicom_handle_key]
        pred = input_features[self.prediction_key]
        ground_truth = input_features[self.ground_truth_key]
        pred = self.iterate_annotations(pred, ground_truth, spacing=list(self.dicom_handle.GetSpacing()), z_mult=1)
        input_features[self.prediction_key] = pred
        return input_features


class Rename_Lung_Voxels_Ground_Glass(Iterate_Overlap):
    def pre_process(self, input_features):
        self.dicom_handle = input_features[self.dicom_handle_key]
        pred = input_features[self.prediction_key]
        mask = np.sum(pred[..., 1:], axis=-1)
        lungs = np.stack([mask, mask], axis=-1)
        lungs = self.iterate_annotations(lungs, mask, spacing=list(self.dicom_handle.GetSpacing()), z_mult=1)
        lungs = lungs[..., 1]
        pred[lungs == 0] = 0
        pred[..., 2] = lungs # Just put lungs in as entirety
        input_features[self.prediction_key] = pred
        return input_features


class Resample_LiTs(ImageProcessor):
    def __init__(self, desired_output_spacing=(None, None, None)):
        self.desired_output_spacing = desired_output_spacing

    def pre_process(self, input_features):
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


class DeepCopyKey(ImageProcessor):
    def __init__(self, from_keys=('annotation',), to_keys=('annotation_original',)):
        self.from_keys, self.to_keys = from_keys, to_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.from_keys)
        for from_key, to_key in zip(self.from_keys, self.to_keys):
            input_features[to_key] = copy.deepcopy(input_features[from_key])
        return input_features


class AddSpacing(ImageProcessor):
    def __init__(self, spacing_handle_key='primary_handle'):
        self.spacing_handle_key = spacing_handle_key

    def pre_process(self, input_features):
        input_features['spacing'] = input_features[self.spacing_handle_key].GetSpacing()
        return input_features


class DilateSitkImages:
    def __init__(self, kernel_radius: Tuple[float, float, float], dilate_keys=('annotation_handle',),
                 kernel_type=sitk.sitkBall):
        """
        :param kernel_radius: tuple of the kernel radius to dilate, in row, column, z
        :param dilate_keys: keys to dilate
        :param kernel_type: type of kernel to use
        """
        self.dilate_filter = sitk.BinaryDilateImageFilter()
        self.dilate_filter.SetKernelType(kernel_type)
        self.dilate_filter.SetKernelRadius(kernel_radius)
        self.dilate_keys = dilate_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.dilate_keys)
        for key in self.dilate_keys:
            mask_handle = input_features[key]
            assert type(mask_handle) is sitk.Image, 'Pass a SimpleITK Image'
            out_array = []
            mask_array = sitk.GetArrayFromImage(mask_handle)
            for z in range(mask_array.shape[-1]):
                temp_handle = sitk.GetImageFromArray(mask_array[..., z])
                _orient_handles_(temp_handle, mask_handle)
                temp_handle = self.dilate_filter.Execute(temp_handle)
                out_array.append(sitk.GetArrayFromImage(temp_handle)[..., None])
            out_array = np.concatenate(out_array, axis=-1)
            mask_handle = sitk.GetImageFromArray(out_array)
            input_features[key] = mask_handle
        return input_features


class DilateNiftiiHandles(DilateSitkImages):
    def __init__(self, kernel_radius: Tuple[float, float, float], dilate_keys=('annotation_handle',),
                 kernel_type=sitk.sitkBall):
        super().__init__(kernel_radius, dilate_keys, kernel_type)


class ResampleSITKHandlesToAnotherHandle(ImageProcessor):
    def __init__(self, resample_keys=('image_handle', 'annotation_handle'),
                 reference_handle_keys=('image_handle',), resample_interpolators=('Linear', 'Nearest'), verbose=True):
        """
        :param resample_keys: tuple of keys in input_features to resample
        :param reference_handle_keys: a tuple of keys to resample to
        :param verbose: binary, print when changing spacing
        """
        self.reference_handle_keys = reference_handle_keys
        self.resample_keys = resample_keys
        self.resample_interpolators = resample_interpolators
        self.verbose = verbose

    def pre_process(self, input_features):
        resampler = ImageResampler()
        _check_keys_(input_features=input_features, keys=self.resample_keys)
        _check_keys_(input_features=input_features, keys=self.reference_handle_keys)
        for key, reference_key, interpolator in zip(self.resample_keys, self.reference_handle_keys,
                                                    self.resample_interpolators):
            image_handle = input_features[key]
            reference_handle = input_features[reference_key]
            assert type(image_handle) is sitk.Image, 'Pass a SimpleITK Image'
            input_spacing = image_handle.GetSpacing()
            output_spacing = reference_handle.GetSpacing()
            input_features['{}_original_spacing'.format(key)] = np.asarray(input_spacing, dtype='float32')
            input_features['{}_output_spacing'.format(key)] = np.asarray(output_spacing, dtype='float32')
            input_features['{}_original_size'.format(key)] = np.asarray(image_handle.GetSize(), dtype='float32')
            input_features['output_spacing'] = np.asarray(output_spacing, dtype='float32')
            if output_spacing != input_spacing or image_handle.GetSize() != reference_handle.GetSize():
                if self.verbose:
                    print('Resampling {} to {}'.format(input_spacing, output_spacing))
                image_handle = resampler.resample_image(input_image_handle=image_handle,
                                                        ref_resampling_handle=reference_handle,
                                                        interpolator=interpolator)
                input_features[key] = image_handle
                input_features['{}_spacing'.format(key)] = np.asarray(output_spacing, dtype='float32')
        return input_features


class SetSITKOrigin(ImageProcessor):
    def __init__(self, keys=('image_handle', 'annotation_handle'),
                 desired_output_origin=(None, None, None), verbose=True):
        self.keys = keys
        self.verbose = verbose
        self.desired_output_origin = desired_output_origin

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.keys)
        for key in self.keys:
            image_handle = input_features[key]
            assert type(image_handle) is sitk.Image, 'Pass a SimpleITK Image'
            input_origin = image_handle.GetOrigin()
            output_origin = []
            for index in range(3):
                if self.desired_output_origin[index] is None:
                    output_origin.append(input_origin[index])
                else:
                    output_origin.append(self.desired_output_origin[index])
            image_handle.SetOrigin(output_origin)
            input_features[key] = image_handle
        return input_features


class ResampleSITKHandles(ImageProcessor):
    def __init__(self, resample_keys=('image_handle', 'annotation_handle'),
                 resample_interpolators=('Linear', 'Nearest'), desired_output_size=None,
                 desired_output_spacing=(None, None, None), verbose=True):
        """
        :param resample_keys: tuple of keys in input_features to resample
        :param resample_interpolators: tuple of SimpleITK interpolators, 'Linear' or 'Nearest'
        :param desired_output_spacing: desired output spacing, (row, col, z)
        :param verbose: binary, print when changing spacing
        """
        self.desired_output_spacing = desired_output_spacing
        self.resample_keys = resample_keys
        self.resample_interpolators = resample_interpolators
        self.verbose = verbose
        self.desired_output_size = desired_output_size

    def pre_process(self, input_features):
        resampler = ImageResampler()
        _check_keys_(input_features=input_features, keys=self.resample_keys)
        for key, interpolator in zip(self.resample_keys, self.resample_interpolators):
            image_handle = input_features[key]
            input_size = image_handle.GetSize()
            assert type(image_handle) is sitk.Image, 'Pass a SimpleITK Image'
            input_spacing = image_handle.GetSpacing()
            output_spacing = []
            output_size = None
            if self.desired_output_size is not None:
                output_size = []
            for index in range(3):
                if self.desired_output_spacing[index] is None:
                    if (self.desired_output_size[index] is None or
                            self.desired_output_size[index] == input_size[index]):
                        output_spacing.append(input_spacing[index])
                    else:
                        new_space = input_size[index]*input_spacing[index] / output_size[index]
                        output_spacing.append(new_space)
                else:
                    output_spacing.append(self.desired_output_spacing[index])
                if self.desired_output_size is not None:
                    if self.desired_output_size[index] is None:
                        output_size.append(input_size[index])
                    else:
                        output_size.append(self.desired_output_size[index])
            output_spacing = tuple(output_spacing)
            input_features['{}_original_spacing'.format(key)] = np.asarray(input_spacing, dtype='float32')
            input_features['{}_output_spacing'.format(key)] = np.asarray(output_spacing, dtype='float32')
            input_features['{}_original_size'.format(key)] = np.asarray(input_size, dtype='float32')
            input_features['output_spacing'] = np.asarray(output_spacing, dtype='float32')
            input_features['{}_spacing'.format(key)] = np.asarray(self.desired_output_spacing, dtype='float32')
            if output_spacing != input_spacing or input_size != tuple(output_size):
                if self.verbose:
                    print('Resampling {} to {}'.format(input_spacing, output_spacing))
                image_handle = resampler.resample_image(input_image_handle=image_handle,
                                                        output_spacing=output_spacing,
                                                        interpolator=interpolator,
                                                        output_size=output_size)
                input_features[key] = image_handle
        return input_features

    def __repr__(self):
        return (f"Resample {self.resample_keys} to {self.desired_output_spacing} or {self.desired_output_size}"
                f" with {self.resample_interpolators}")


class Resampler(ImageProcessor):
    def __init__(self, resample_keys=('image', 'annotation'), resample_interpolators=('Linear', 'Nearest'),
                 desired_output_spacing=(None, None, None), make_512=False, verbose=True,
                 post_process_resample_keys=None, post_process_original_spacing_keys=None,
                 post_process_interpolators=None):
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
        self.verbose = verbose
        self.post_process_resample_keys = post_process_resample_keys
        self.post_process_original_spacing_keys = post_process_original_spacing_keys
        self.post_process_interpolators = post_process_interpolators

    def pre_process(self, input_features):
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
            if type(image_handle) is np.ndarray:
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
            input_features['{}_original_spacing'.format(key)] = np.asarray(input_spacing, dtype='float32')
            input_features['{}_output_spacing'.format(key)] = np.asarray(output_spacing, dtype='float32')
            input_features['output_spacing'] = np.asarray(output_spacing, dtype='float32')
            if output_spacing != input_spacing:
                if self.verbose:
                    print('Resampling {} to {}'.format(input_spacing, output_spacing))
                image_handle = resampler.resample_image(input_image_handle=image_handle,
                                                        output_spacing=output_spacing,
                                                        interpolator=interpolator)
                input_features[key] = image_handle
                input_features['{}_spacing'.format(key)] = np.asarray(self.desired_output_spacing, dtype='float32')
        return input_features

    def post_process(self, input_features):
        if self.post_process_resample_keys is None:
            return input_features
        if self.post_process_original_spacing_keys is None:
            return input_features
        resampler = ImageResampler()
        _check_keys_(input_features=input_features, keys=self.post_process_resample_keys)
        for spacing_key, key, interpolator in zip(self.post_process_original_spacing_keys,
                                                  self.post_process_resample_keys,
                                                  self.post_process_interpolators):
            original_handle = input_features[spacing_key]
            if type(original_handle) is sitk.Image:
                output_spacing = original_handle.GetSpacing()
                ref_handle = original_handle
            else:
                output_spacing = tuple([float(i) for i in input_features['{}_original_spacing'.format(spacing_key)]])
                ref_handle = None
            image_handle = input_features[key]
            input_spacing = tuple([float(i) for i in input_features['output_spacing']])
            if output_spacing != input_spacing:
                image_array = None
                if type(image_handle) is np.ndarray:
                    image_array = image_handle
                    image_shape = image_handle.shape
                else:
                    image_shape = image_handle.GetSize()
                if self.verbose:
                    print('Resampling {} to {}'.format(input_spacing, output_spacing))
                if len(image_shape) == 3:
                    if type(image_handle) is np.ndarray:
                        image_handle = sitk.GetImageFromArray(image_handle)
                        image_handle.SetSpacing(input_spacing)
                    if ref_handle is not None:
                        image_handle.SetDirection(ref_handle.GetDirection())
                        image_handle.SetOrigin(ref_handle.GetOrigin())
                    image_handle = resampler.resample_image(input_image_handle=image_handle,
                                                            output_spacing=output_spacing,
                                                            interpolator=interpolator,
                                                            ref_resampling_handle=ref_handle)
                    input_features[key] = sitk.GetArrayFromImage(image_handle)
                else:
                    if image_array is None:
                        image_array = sitk.GetArrayFromImage(image_handle)
                    output = []
                    for i in range(image_array.shape[-1]):
                        reduced_handle = sitk.GetImageFromArray(image_array[..., i])
                        reduced_handle.SetSpacing(input_spacing)
                        if ref_handle is not None:
                            reduced_handle.SetDirection(ref_handle.GetDirection())
                            reduced_handle.SetOrigin(ref_handle.GetOrigin())
                        resampled_handle = resampler.resample_image(input_image_handle=reduced_handle,
                                                                    output_spacing=output_spacing,
                                                                    interpolator=interpolator,
                                                                    ref_resampling_handle=ref_handle)
                        output.append(sitk.GetArrayFromImage(resampled_handle)[..., None])
                    stacked = np.concatenate(output, axis=-1)
                    stacked[..., 0] = 1 - np.sum(stacked[..., 1:], axis=-1)
                    input_features[key] = stacked
        return input_features


class CreateTupleFromKeys(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation'), output_key='combined'):
        """
        :param image_keys: tuple of image keys
        :param output_key: key for output
        """
        self.image_keys = image_keys
        self.output_key = output_key

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        input_features[self.output_key] = tuple([input_features[i] for i in self.image_keys])
        return input_features


class CombineKeys(ImageProcessor):
    def __init__(self, image_keys=('primary_image', 'secondary_image'), output_key='combined', axis=-1):
        self.image_keys = image_keys
        self.output_key = output_key
        self.axis = axis

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        combine_images = [input_features[i] for i in self.image_keys]
        input_features[self.output_key] = np.concatenate(combine_images, axis=self.axis)
        return input_features


class ArgMax(ImageProcessor):
    def __init__(self, image_keys=('prediction',), axis=-1):
        self.image_keys = image_keys
        self.axis = axis

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key in self.image_keys:
            pred = input_features[key]
            pred = np.argmax(pred, axis=self.axis)
            input_features[key] = pred
        return input_features

    def __repr__(self):
        return f"Argmax {self.image_keys} with axis {self.axis}"


class Ensure_Image_Proportions(ImageProcessor):
    def __init__(self, image_rows=512, image_cols=512, image_keys=('image',), post_process_keys=('image', 'prediction')):
        self.wanted_rows = image_rows
        self.wanted_cols = image_cols
        self.image_keys = image_keys
        self.post_process_keys = post_process_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key in self.image_keys:
            images = input_features[key]
            og_image_size = np.squeeze(images.shape)
            if len(og_image_size) == 4:
                self.og_rows, self.og_cols = og_image_size[-3], og_image_size[-2]
            else:
                self.og_rows, self.og_cols = og_image_size[-2], og_image_size[-1]
            self.resize = False
            self.pads = None
            if self.og_rows != self.wanted_rows or self.og_cols != self.wanted_cols:
                self.resize = True
                if str(images.dtype).find('int') != -1:
                    out_dtype = 'int'
                else:
                    out_dtype = images.dtype
                images = [image_resize(i, self.wanted_rows, self.wanted_cols, inter=cv2.INTER_LINEAR)[None, ...] for i in
                          images.astype('float32')]
                images = np.concatenate(images, axis=0).astype(out_dtype)
                print('Resizing {} to {}'.format(self.og_rows, images.shape[1]))
                self.pre_pad_rows, self.pre_pad_cols = images.shape[1], images.shape[2]
                if self.wanted_rows != self.pre_pad_rows or self.wanted_cols != self.pre_pad_cols:
                    print('Padding {} to {}'.format(self.pre_pad_rows, self.wanted_rows))
                    self.pads = [0, self.wanted_rows - images.shape[1], self.wanted_cols - images.shape[2]]
                    if len(images.shape) == 4:
                        self.pads.append(0)
                    self.pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in self.pads]
                    images = np.pad(images, pad_width=self.pads, constant_values=np.min(images))
            input_features[key] = images
        return input_features

    def post_process(self, input_features):
        if self.pads is not None and not self.resize:
            return input_features
        _check_keys_(input_features=input_features, keys=self.post_process_keys)
        for key in self.post_process_keys:
            pred = input_features[key]
            if str(pred.dtype).find('int') != -1:
                out_dtype = 'int'
            else:
                out_dtype = pred.dtype
            pred = pred.astype('float32')
            if self.pads is not None:
                pred = pred[self.pads[0][0]:pred.shape[0] - self.pads[0][1],
                            self.pads[1][0]:pred.shape[1] - self.pads[1][1],
                            self.pads[2][0]:pred.shape[2] - self.pads[2][1]]
            if self.resize:
                pred = [image_resize(i, self.og_rows, self.og_cols, inter=cv2.INTER_LINEAR)[None, ...] for i in pred]
                pred = np.concatenate(pred, axis=0)
            input_features[key] = pred.astype(out_dtype)
        return input_features


class VGGNormalize(ImageProcessor):
    def __init__(self, image_keys=('image',)):
        """
        :param image_keys: tuple of image keys
        """
        self.image_keys = image_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key in self.image_keys:
            images = input_features[key]
            images[..., 0] -= 123.68
            images[..., 1] -= 116.78
            images[..., 2] -= 103.94
            input_features[key] = images
        return input_features


class Threshold_Prediction(ImageProcessor):
    def __init__(self, threshold=0.0, single_structure=True, is_liver=False, min_volume=0.0,
                 prediction_keys=('prediction')):
        '''
        :param threshold:
        :param single_structure:
        :param is_liver:
        :param min_volume: in ccs
        '''
        self.threshold = threshold
        self.is_liver = is_liver
        self.min_volume = min_volume
        self.single_structure = single_structure
        self.prediction_keys = prediction_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.prediction_keys)
        for key in self.prediction_keys:
            pred = input_features[key]
            if self.is_liver:
                pred[..., -1] = variable_remove_non_liver(pred[..., -1], threshold=0.2, is_liver=True)
            if self.threshold != 0.0:
                for i in range(1, pred.shape[-1]):
                    pred[..., i] = remove_non_liver(pred[..., i], threshold=self.threshold, do_3D=self.single_structure,
                                                    min_volume=self.min_volume)
            input_features[key] = pred
        return input_features


class GetSeedPoints(object):
    def __init__(self):
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.stats = sitk.LabelShapeStatisticsImageFilter()

    def get_seed_points(self, input_handle):
        connected_image = self.Connected_Component_Filter.Execute(input_handle)
        self.stats.Execute(connected_image)
        labels = [l for l in self.stats.GetLabels()]
        all_centroids = [input_handle.TransformPhysicalPointToIndex(self.stats.GetCentroid(l))
                         for l in labels]
        return all_centroids


class GrowFromSeedPoints(object):
    def __init__(self):
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.Connected_Threshold.SetUpper(2)
        self.Connected_Threshold.SetLower(0.5)
        self.stats = sitk.LabelShapeStatisticsImageFilter()
        self.Connected_Threshold.SetUpper(2)

    def grow_from_seed_points(self, seed_points, input_handle):
        self.Connected_Threshold.SetSeedList(seed_points)
        threshold_prediction = self.Connected_Threshold.Execute(sitk.Cast(input_handle, sitk.sitkFloat32))
        return threshold_prediction


class CombineLungLobes(ImageProcessor):
    def __init__(self, prediction_key='prediction', dicom_handle_key='primary_handle'):
        self.prediction_key = prediction_key
        self.dicom_handle_key = dicom_handle_key
        self.seed_finder = GetSeedPoints()
        self.seed_grower = GrowFromSeedPoints()

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.prediction_key,))
        pred = input_features[self.prediction_key]
        lungs = np.sum(pred[..., 1:], axis=-1)
        left_lung = copy.deepcopy(lungs)
        right_lung = copy.deepcopy(lungs)
        left_lung[:, :, :left_lung.shape[2]//2] = 0
        right_lung[:, :, right_lung.shape[2]//2:] = 0
        left_lung = remove_non_liver(left_lung, threshold=0.5, do_3D=True, min_volume=0, do_2D=False,
                                     max_volume=np.inf, spacing=input_features[self.dicom_handle_key].GetSpacing())
        right_lung = remove_non_liver(right_lung, threshold=0.5, do_3D=True, min_volume=0, do_2D=False,
                                      max_volume=np.inf, spacing=input_features[self.dicom_handle_key].GetSpacing())
        combined_lungs = left_lung + right_lung
        combined_lungs[combined_lungs > 0] = 1
        seeds = self.seed_finder.get_seed_points(sitk.GetImageFromArray(combined_lungs))
        grown_lungs = self.seed_grower.grow_from_seed_points(seed_points=seeds,
                                                             input_handle=sitk.GetImageFromArray(lungs))
        grown_lungs = sitk.GetArrayFromImage(grown_lungs)
        pred[grown_lungs == 0] = 0
        pred[..., -1] = grown_lungs
        input_features[self.prediction_key] = pred
        return input_features


class RemoveDisconnectedStructures(ImageProcessor):
    def __init__(self, single_structure=True, perform_2D=False, min_volume=0.0, image_keys=('prediction',),
                 indexes=None, max_volume=np.inf, dicom_handle_key='primary_handle'):
        self.single_structure = single_structure
        self.perform_2D = perform_2D
        self.min_volume, self.max_volume = min_volume, max_volume
        self.image_keys = image_keys
        self.dicom_handle_key = dicom_handle_key
        self.indexes = indexes

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys + (self.dicom_handle_key,))
        for key in self.image_keys:
            pred = input_features[key]
            if self.indexes is None:
                indexes = range(1, pred.shape[-1])
            else:
                indexes = self.indexes
            for i in indexes:
                pred[..., i] = remove_non_liver(pred[..., i], threshold=0.5, do_3D=self.single_structure,
                                                min_volume=self.min_volume, do_2D=self.perform_2D,
                                                max_volume=self.max_volume,
                                                spacing=input_features[self.dicom_handle_key].GetSpacing())
        return input_features


class SqueezeDimensions(ImageProcessor):
    def __init__(self, image_keys=None, post_prediction_keys=None):
        self.post_prediction_keys = post_prediction_keys
        self.image_keys = image_keys

    def pre_process(self, input_features):
        if self.image_keys:
            _check_keys_(input_features, self.image_keys)
            for key in self.image_keys:
                input_features[key] = np.squeeze(input_features[key])
        return input_features

    def post_process(self, input_features):
        if self.post_prediction_keys:
            _check_keys_(input_features, self.post_prediction_keys)
            for key in self.post_prediction_keys:
                input_features[key] = np.squeeze(input_features[key])
        return input_features

    def __repr__(self):
        return f"Squeeze {self.image_keys}"


class ExpandDimensions(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation'), axis=-1, post_process_keys=('image', 'prediction')):
        self.image_keys = image_keys
        self.axis = axis
        self.post_process_keys = post_process_keys

    def pre_process(self, input_features):
        _check_keys_(input_features, self.image_keys)
        for key in self.image_keys:
            input_features[key] = np.expand_dims(input_features[key], axis=self.axis)
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features, self.post_process_keys)
        for key in self.post_process_keys:
            i = input_features[key]
            i = np.squeeze(i, axis=self.axis)
            input_features[key] = i
        return input_features

    def __repr__(self):
        return f"Expand {self.image_keys} on {self.axis} axes"


class RepeatChannel(ImageProcessor):
    def __init__(self, num_repeats=3, axis=-1, image_keys=('image',)):
        self.num_repeats = num_repeats
        self.axis = axis
        self.image_keys = image_keys

    def pre_process(self, input_features):
        for key in self.image_keys:
            images = input_features[key]
            input_features[key] = np.repeat(images, self.num_repeats, axis=self.axis)
        return input_features

    def __repr__(self):
        return f"Repeat {self.image_keys} on {self.axis} axes {self.num_repeats} times"


class CastHandle(ImageProcessor):
    def __init__(self, image_handle_keys, d_type_keys):
        """
        :param image_handle_keys: tuple of image handle keys ('primary_handle', )
        :param d_type_keys: tuple of dtype keys ('float32', )
        """
        self.image_handle_keys = image_handle_keys
        self.d_type_keys = d_type_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_handle_keys)
        for key, dtype in zip(self.image_handle_keys, self.d_type_keys):
            if dtype.find('float') != -1:
                dtype = sitk.sitkFloat32
            elif dtype == 'int16':
                dtype = sitk.sitkInt16
            elif dtype == 'int32':
                dtype = sitk.sitkInt32
            else:
                dtype = None
            assert dtype is not None, 'Need to provide a dtype to cast of float, int16, or int32'
            input_features[key] = sitk.Cast(input_features[key], dtype)
        return input_features

    def __repr__(self):
        return f"Cast {self.image_handle_keys} to {self.d_type_keys}"


class CastData(ImageProcessor):
    def __init__(self, image_keys=('image',), dtypes=('float32',)):
        """
        :param image_keys: tuple of image keys in dictionary
        :param dtypes: tuple of string data types
        """
        self.image_keys = image_keys
        self.dtypes = dtypes

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key, dtype in zip(self.image_keys, self.dtypes):
            input_features[key] = input_features[key].astype(dtype)
        return input_features

    def __repr__(self):
        return f"Cast {self.image_keys} to {self.dtypes}"


class ConvertArrayToHandle(ImageProcessor):
    def __init__(self, array_keys=('gradients',), out_keys=('gradients_handle',)):
        """
        :param array_keys: tuple of array keys in dictionary
        :param out_keys: tuple of string names for arrays to be named
        """
        self.array_keys = array_keys
        self.out_keys = out_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.array_keys)
        for array_key, out_key in zip(self.array_keys, self.out_keys):
            input_features[out_key] = sitk.GetImageFromArray(input_features[array_key])
        return input_features

    def __repr__(self):
        return f"Convert NumPy Arrays {self.array_keys} to Simple ITK Images called {self.out_keys}"


def _orient_handles_(moving_handle, fixed_handle):
    moving_handle.SetSpacing(fixed_handle.GetSpacing())
    moving_handle.SetOrigin(fixed_handle.GetOrigin())
    moving_handle.SetDirection(fixed_handle.GetDirection())
    return moving_handle


class OrientHandleToAnother(ImageProcessor):
    def __init__(self, moving_handle_keys=('gradients_handle',), fixed_handle_keys=('primary_handle',)):
        """
        :param moving_handle_keys: tuple of moving handle keys
        :param fixed_handle_keys: tuple of fixed handle keys
        """
        self.moving_handle_keys = moving_handle_keys
        self.fixed_handle_keys = fixed_handle_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.moving_handle_keys + self.fixed_handle_keys)
        for moving_key, fixed_key in zip(self.moving_handle_keys, self.fixed_handle_keys):
            moving_handle = input_features[moving_key]
            fixed_handle = input_features[fixed_key]
            input_features[moving_key] = _orient_handles_(moving_handle=moving_handle, fixed_handle=fixed_handle)
        return input_features


class Add_Images_And_Annotations(ImageProcessor):
    def __init__(self, nifti_path_keys=('image_path', 'annotation_path'), out_keys=('image', 'annotation'),
                 dtypes=('float32', 'int8')):
        self.nifti_path_keys, self.out_keys, self.dtypes = nifti_path_keys, out_keys, dtypes

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.nifti_path_keys)
        for nifti_path_key, out_key, dtype in zip(self.nifti_path_keys, self.out_keys, self.dtypes):
            image_handle = sitk.ReadImage(input_features[nifti_path_key])
            image_array = sitk.GetArrayFromImage(image_handle)
            input_features[out_key] = image_array.astype(dtype=dtype)
            input_features['{}_spacing'.format(out_key)] = np.asarray(image_handle.GetSpacing(), dtype='float32')
        return input_features


class AddNifti(ImageProcessor):
    def __init__(self, nifti_path_keys=('image_path', 'annotation_path'), out_keys=('image', 'annotation')):
        self.nifti_path_keys, self.out_keys = nifti_path_keys, out_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.nifti_path_keys)
        for nifti_path_key, out_key in zip(self.nifti_path_keys, self.out_keys):
            image_handle = sitk.ReadImage(input_features[nifti_path_key])
            input_features[out_key] = image_handle
        return input_features

    def __repr__(self):
        return f"Load files at {self.nifti_path_keys} to Simple ITK Images called {self.out_keys}"

class LoadNifti(AddNifti):
    def __init__(self, nifti_path_keys=('image_path', 'annotation_path'), out_keys=('image', 'annotation')):
        super(LoadNifti, self).__init__(nifti_path_keys=nifti_path_keys, out_keys=out_keys)


class DeleteKeys(ImageProcessor):
    def __init__(self, keys_to_delete=('primary_image_nifti',)):
        self.keys_to_delete = keys_to_delete

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.keys_to_delete)
        for key in self.keys_to_delete:
            del input_features[key]
        return input_features

    def __repr__(self):
        return f"Delete keys {self.keys_to_delete}"


class ArrayToNiftii(ImageProcessor):
    def __init__(self, array_keys=('prediction',), out_keys=('prediction_handle',)):
        self.array_keys, self.out_keys = array_keys, out_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.array_keys)
        for array_key, out_key in zip(self.array_keys, self.out_keys):
            image_array = input_features[array_key]
            assert type(image_array) is np.ndarray, 'Only NumPy arrays should be passed here!'
            image_handle = sitk.GetImageFromArray(image_array)
            input_features[out_key] = image_handle
        return input_features

    def __repr__(self):
        return f"Convert NumPy Arrays {self.array_keys} to SimpleITK handles {self.out_keys}"


class NiftiToArray(ImageProcessor):
    def __init__(self, nifti_keys=('image_path', 'annotation_path'), out_keys=('image', 'annotation'),
                 dtypes=('float32', 'int8')):
        self.nifti_keys, self.out_keys, self.dtypes = nifti_keys, out_keys, dtypes

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.nifti_keys)
        for nifti_key, out_key, dtype in zip(self.nifti_keys, self.out_keys, self.dtypes):
            image_handle = input_features[nifti_key]
            assert type(image_handle) is sitk.Image, 'Only SimpleITK Images should be passed here!'
            image_array = sitk.GetArrayFromImage(image_handle)
            input_features[out_key] = image_array.astype(dtype=dtype)
            input_features['{}_spacing'.format(out_key)] = np.asarray(image_handle.GetSpacing(), dtype='float32')
        return input_features

    def __repr__(self):
        return f"Convert SimpleITK Images {self.nifti_keys} to NumPy Arrays {self.out_keys}"


class SimpleITKImageToArray(NiftiToArray):
    def __init__(self, nifti_keys=('image_path', 'annotation_path'), out_keys=('image', 'annotation'),
                 dtypes=('float32', 'int8')):
        super(SimpleITKImageToArray, self).__init__(nifti_keys=nifti_keys, out_keys=out_keys, dtypes=dtypes)


class SplitArray(ImageProcessor):
    def __init__(self, array_keys=('image_path', 'annotation_path'), out_keys=('image', 'annotation'),
                 axis_index=(0, 1)):
        self.array_keys, self.out_keys, self.axis_index = array_keys, out_keys, axis_index

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.array_keys)
        for array_key, out_key, axis_index in zip(self.array_keys, self.out_keys, self.axis_index):
            input_features[out_key] = input_features[array_key][..., axis_index]
        return input_features


class Clip_Images_By_Extension(ImageProcessor):
    def __init__(self, extension=np.inf, clipping_keys=('image_array', 'mask_array'),
                 guiding_keys=('mask_array', 'image_array')):
        self.extension = extension
        self.clipping_keys = clipping_keys
        self.guiding_keys = guiding_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.clipping_keys + self.guiding_keys)
        for guiding_key, clipping_key in zip(self.guiding_keys, self.clipping_keys):
            image = input_features[clipping_key]
            annotation = input_features[guiding_key]
            start, stop = get_start_stop(annotation, self.extension)
            if start != -1 and stop != -1:
                image = image[start:stop, ...]
            input_features[clipping_key] = image
        return input_features

    def __repr__(self):
        return (f"Clipping/Padding NumPy Arrays {self.guiding_keys} by {self.clipping_keys} and "
                f"extension of {self.extension}")


class Normalize_MRI(ImageProcessor):
    def pre_process(self, input_features):
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
    def pre_process(self, input_features):
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
    def __init__(self, disease_annotation=None, cube_size=(16, 120, 120), min_voxel_volume=0, max_voxels=np.inf,
                 image_key='image', annotation_key='annotation', minimum_percentage=0.):
        """
        :param disease_annotation:
        :param cube_size:
        :param min_voxel_volume:
        :param max_voxels:
        :param image_key:
        :param annotation_key:
        :param minimum_percentage: a minimum percentage of the image to be filled with a contour to be passed along
        """
        self.disease_annotation = disease_annotation
        self.cube_size = np.asarray(cube_size)
        self.min_voxel_volume = min_voxel_volume
        self.max_voxels = max_voxels
        self.image_key = image_key
        self.annotation_key = annotation_key
        self.minimum_percentage = minimum_percentage
        assert disease_annotation is not None, 'Provide an integer for what is disease'

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.image_key, self.annotation_key))
        if 'bounding_boxes_{}'.format(self.disease_annotation) not in input_features:
            Add_Bounding_Box = Add_Bounding_Box_Indexes([self.disease_annotation], add_to_dictionary=False,
                                                        label_name=self.annotation_key)
            input_features = Add_Bounding_Box.pre_process(input_features)
        if 'bounding_boxes_{}'.format(self.disease_annotation) in input_features:
            bounding_boxes = input_features['bounding_boxes_{}'.format(self.disease_annotation)]
            voxel_volumes = input_features['voxel_volumes_{}'.format(self.disease_annotation)]
            del input_features['voxel_volumes_{}'.format(self.disease_annotation)]
            del input_features['bounding_boxes_{}'.format(self.disease_annotation)]
            image_base = input_features[self.image_key]
            annotation_base = input_features[self.annotation_key]
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
                pads = np.asarray([self.cube_size[i] - image.shape[i] % self.cube_size[i]
                                   if image.shape[i] % self.cube_size[i] != 0 else 0 for i in range(3)])
                if np.max(pads) != 0:
                    if len(image.shape) > 3:
                        pads = np.append(pads, [0])
                    pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
                    image = np.pad(image, pads, constant_values=np.min(image))

                    pads = np.asarray([self.cube_size[i] - annotation.shape[i] % self.cube_size[i]
                                       if annotation.shape[i] % self.cube_size[i] != 0 else 0 for i in range(3)])
                    if len(annotation.shape) > 3:
                        pads = np.append(pads, [0])
                    pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
                    annotation = np.pad(annotation, pads)
                    if len(annotation.shape) > 3:
                        annotation[..., 0] = 1 - np.sum(annotation[..., 1:], axis=-1)
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
                    argmax_annotation = np.argmax(annotation_cube, axis=-1)
                    number_of_contours = np.sum(argmax_annotation > 0)
                    percentage = number_of_contours / np.prod(argmax_annotation.shape) * 100
                    if percentage < self.minimum_percentage:
                        continue
                    temp_feature[self.image_key] = image_cube[:self.cube_size[0]]
                    temp_feature[self.annotation_key] = annotation_cube[:self.cube_size[0]]
                    for key in input_features:  # Bring along anything else we care about
                        if key not in temp_feature.keys():
                            temp_feature[key] = input_features[key]
                    out_features['Box_{}_{}'.format(cube_index, box_index)] = temp_feature
            input_features = out_features
        return input_features


class Distribute_into_3DOld(ImageProcessor):
    def __init__(self, image_keys=('image_key', 'mask_key'), min_z=0, max_z=np.inf, max_rows=np.inf, max_cols=np.inf, mirror_small_bits=True,
                 chop_ends=False, desired_val=1):
        self.max_z = max_z
        self.min_z = min_z
        self.max_rows, self.max_cols = max_rows, max_cols
        self.mirror_small_bits = mirror_small_bits
        self.chop_ends = chop_ends
        self.desired_val = desired_val
        self.image_keys = image_keys

    def pre_process(self, input_features):
        out_features = OrderedDict()
        start_chop = 0
        image_base = input_features['image']
        annotation_base = input_features['annotation']
        image_path = input_features['image_path']
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
            for key in input_features.keys():
                if key not in image_features.keys():
                    image_features[key] = input_features[key]  # Pass along all other keys.. be careful
            out_features['Image_{}'.format(index)] = image_features
        input_features = out_features
        return input_features


class Distribute_into_3D(ImageProcessor):
    def __init__(self, image_keys=('image_key', 'mask_key')):
        self.image_keys = image_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        out_features = OrderedDict()
        image_features = OrderedDict()
        for key in self.image_keys:
            image = input_features[key]
            image_features[key] = image
        out_features[0] = image_features
        for index in out_features:
            for key in input_features:
                if key not in out_features[index]:
                    out_features[index][key] = input_features[key]
        return out_features


class Distribute_into_2D(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation')):
        self.image_keys = image_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        out_features = OrderedDict()
        for key in self.image_keys:
            image = input_features[key]
            z_images_base = image.shape[0]
            for index in range(z_images_base):
                if index in out_features:
                    image_features = out_features[index]
                else:
                    image_features = OrderedDict()
                image_features[key] = image[index]
                out_features[index] = image_features
        for index in out_features:
            for key in input_features:
                if key not in out_features[index]:
                    out_features[index][key] = input_features[key]
        return out_features


class DistributeInTo2DSlices(Distribute_into_2D):
    def __init__(self, image_keys=('image', 'annotation')):
        super(DistributeInTo2DSlices, self).__init__(image_keys=image_keys)


class NormalizeParotidMR(ImageProcessor):
    def __init__(self, image_keys=('image',)):
        self.image_keys = image_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key in self.image_keys:
            images = input_features[key]
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
            input_features[key] = images
        return input_features


class AddByValues(ImageProcessor):
    def __init__(self, image_keys=('image',), values=(1.,)):
        """
        :param image_keys: tuple of keys to divide by the value
        :param values: values by which to add by
        """
        self.image_keys = image_keys
        self.values = values

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key, value in zip(self.image_keys, self.values):
            image_array = input_features[key]
            image_array += value
            input_features[key] = image_array
        return input_features


class AddLiverKey(ImageProcessor):
    def __init__(self, annotation_keys=('primary_mask',), actions=('arg_max',), out_keys=('liver,',)):
        """
        :param annotation_keys: list of annotation keys
        :param actions: 'sum', or 'arg_max'
        :param out_keys: list of keys to add to dictionary
        """
        self.annotation_keys = annotation_keys
        self.actions = actions
        self.out_keys = out_keys

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.annotation_keys)
        for annotation_key, action, out_key in zip(self.annotation_keys, self.actions, self.out_keys):
            annotation = input_features[annotation_key]
            if action == 'sum':
                output = np.sum(annotation, axis=-1)[..., None]
                input_features[out_key] = output
            elif action == 'arg_max':
                output = np.argmax(annotation, axis=-1)[..., None]
                output[output > 0] = 1
                input_features[out_key] = output
        return input_features


class DistributeIntoCubes(ImageProcessor):
    def __init__(self, rows=128, cols=128, images=32, resize_keys_tuple=None, wanted_keys=('spacing', ),
                 image_keys=('primary_image', 'secondary_image_deformed'), mask_key='primary_mask',
                 mask_value=1, channel_value=None, out_mask_name='primary_liver'):
        """
        mask_value = int, value to reference as the mask
        channel_value = int, channel to reference as the mask, should be None if mask_value is not None
        """
        if channel_value is not None:
            assert mask_value is None, 'If you are using channel_value, mask_value should be None'
        self.rows, self.cols, self.images = rows, cols, images
        self.resize_keys_tuple = resize_keys_tuple
        self.wanted_keys = wanted_keys
        self.image_keys = image_keys
        self.mask_key = mask_key
        self.mask_value = mask_value
        self.channel_value = channel_value
        self.out_mask_name = out_mask_name
    """
    Highly specialized for the task of model prediction, likely won't be useful for others
    """
    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.mask_key,))
        out_features = OrderedDict()
        primary_mask = input_features[self.mask_key]
        image_size = primary_mask.shape
        '''
        Now, find centroids in the cases
        '''
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        Connected_Component_Filter.FullyConnectedOn()
        stats = sitk.LabelShapeStatisticsImageFilter()
        if self.channel_value is None:
            cube_image = sitk.GetImageFromArray((primary_mask == self.mask_value).astype('int'))
        else:
            cube_image = sitk.GetImageFromArray((primary_mask[..., self.channel_value]).astype('int'))
        connected_image = Connected_Component_Filter.Execute(cube_image)
        stats.Execute(connected_image)
        labels = [l for l in stats.GetLabels()]
        all_centroids = [cube_image.TransformPhysicalPointToIndex(stats.GetCentroid(l))
                         for l in labels]

        for value, cube_name, centroids, label, image in zip([0], ['Cube_{}'],
                                                             [all_centroids],
                                                             [labels],
                                                             [connected_image]):
            for index, centroid in enumerate(centroids):
                temp_feature = OrderedDict()
                col_center, row_center, z_center = centroid
                z_start_pad, z_stop_pad, r_start_pad, r_stop_pad, c_start_pad, c_stop_pad = 0, 0, 0, 0, 0, 0
                z_start = z_center - self.images // 2
                if z_start < 0:
                    z_start_pad = abs(z_start)
                    z_start = 0
                z_stop = z_center + self.images // 2
                if z_stop > image_size[0]:
                    z_stop_pad = z_stop - image_size[0]
                    z_stop = image_size[0]
                r_start = row_center - self.rows // 2
                if r_start < 0:
                    r_start_pad = abs(r_start)
                    r_start = 0
                r_stop = row_center + self.rows // 2
                if r_stop > image_size[1]:
                    r_stop_pad = r_stop - image_size[1]
                    r_stop = image_size[1]
                c_start = col_center - self.cols // 2
                if c_start < 0:
                    c_start_pad = abs(c_start)
                    c_start = 0
                c_stop = col_center + self.cols // 2
                if c_stop > image_size[2]:
                    c_stop_pad = c_stop - image_size[2]
                    c_stop = image_size[2]
                index_mask = sitk.GetArrayFromImage(image)
                index_mask[index_mask != label[index]] = 0
                index_mask[index_mask > 0] = 1
                index_mask = index_mask.astype('int')
                index_mask = index_mask[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                primary_liver_cube = primary_mask[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                pads = [[z_start_pad, z_stop_pad], [r_start_pad, r_stop_pad], [c_start_pad, c_stop_pad]]
                if np.max(pads) > 0:
                    primary_liver_cube = np.pad(primary_liver_cube, pads, constant_values=np.min(primary_liver_cube))
                    index_mask = np.pad(index_mask, pads, constant_values=np.min(index_mask))
                for image_key in self.image_keys:
                    primary_array = input_features[image_key]
                    primary_array_cube = primary_array[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                    if np.max(pads) > 0:
                        if len(primary_array_cube.shape) > len(pads):
                            primary_array_cube = np.pad(primary_array_cube, pads + [[0, 0]],
                                                        constant_values=np.min(primary_array_cube))
                        else:
                            primary_array_cube = np.pad(primary_array_cube, pads,
                                                        constant_values=np.min(primary_array_cube))

                    temp_feature[image_key] = primary_array_cube

                temp_feature['z_start'] = z_start
                temp_feature['image_size'] = image_size
                temp_feature['z_start_pad'] = z_start_pad
                temp_feature['r_start'] = r_start
                temp_feature['r_start_pad'] = r_start_pad
                temp_feature['c_start'] = c_start
                temp_feature['c_start_pad'] = c_start_pad
                # primary_liver_cube[primary_liver_cube > 0] = 1  # Make it so we have liver at 1, and disease as 2
                # primary_liver_cube[index_mask == 1] = 2
                primary_liver_cube = primary_liver_cube.astype('int8')
                temp_feature[self.out_mask_name] = primary_liver_cube
                if self.wanted_keys is not None:
                    for key in self.wanted_keys:  # Bring along anything else we care about
                        if key not in temp_feature.keys():
                            temp_feature[key] = input_features[key]
                else:
                    for key in input_features.keys():
                        if key not in temp_feature.keys():
                            temp_feature[key] = input_features[key]
                out_features[cube_name.format(index)] = temp_feature
        return out_features

    def post_process(self, input_features):
        if self.resize_keys_tuple is not None:
            _check_keys_(input_features=input_features, keys=self.resize_keys_tuple)
            og_image_size = input_features['image_size']
            z_start = input_features['z_start']
            r_start = input_features['r_start']
            c_start = input_features['c_start']
            for key in self.resize_keys_tuple:
                image = input_features[key]
                pads = [[z_start, 0], [r_start, 0], [c_start, 0]]
                image = np.pad(image, pads, constant_values=np.min(image))
                pads = [[0, og_image_size[0] - image.shape[0]], [0, og_image_size[1] - image.shape[1]],
                        [0, og_image_size[2] - image.shape[2]]]
                image = np.pad(image, pads, constant_values=np.min(image))
                input_features[key] = image
        return input_features


def get_bounding_box_indexes(annotation, bbox=(0,0,0)):
    '''
    :param annotation: A binary image of shape [# images, # rows, # cols, channels]
    :return: the min and max z, row, and column numbers bounding the image
    '''
    annotation = np.squeeze(annotation)
    if annotation.dtype != 'int':
        annotation[annotation>0.1] = 1
        annotation = annotation.astype('int')
    indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
    min_z_s, max_z_s = indexes[0], indexes[-1]
    min_z_s = max([0, min_z_s - bbox[0]])
    max_z_s = min([annotation.shape[0], max_z_s + bbox[0]])
    '''
    Get the row values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    min_r_s, max_r_s = indexes[0], indexes[-1]
    min_r_s = max([0, min_r_s - bbox[1]])
    max_r_s = min([annotation.shape[1], max_r_s + bbox[1]])
    '''
    Get the col values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    min_c_s, max_c_s = indexes[0], indexes[-1]
    min_c_s = max([0, min_c_s - bbox[2]])
    max_c_s = min([annotation.shape[2], max_c_s + bbox[2]])
    return min_z_s, max_z_s, min_r_s, max_r_s, min_c_s, max_c_s


class Iterate_Lobe_Annotations(object):
    def __init__(self):
        self.remove_smallest = Remove_Smallest_Structures()
        MauererDistanceMap = sitk.SignedMaurerDistanceMapImageFilter()
        MauererDistanceMap.SetInsideIsPositive(True)
        MauererDistanceMap.UseImageSpacingOn()
        MauererDistanceMap.SquaredDistanceOff()
        self.BinaryfillFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryfillFilter.SetFullyConnected(True)
        self.BinaryfillFilter = sitk.BinaryMorphologicalClosingImageFilter()
        self.BinaryfillFilter.SetKernelRadius((3, 3, 1))
        self.BinaryfillFilter.SetKernelType(sitk.sitkBall)
        self.MauererDistanceMap = MauererDistanceMap

    def remove_56_78(self, annotations):
        amounts = np.sum(annotations, axis=(1, 2))
        indexes = np.where((np.max(amounts[:, (5, 6)], axis=-1) > 0) & (np.max(amounts[:, (7, 8)], axis=-1) > 0))
        if indexes:
            indexes = indexes[0]
            for i in indexes:
                if amounts[i, 5] < amounts[i, 8]:
                    annotations[i, ..., 8] += annotations[i, ..., 5]
                    annotations[i, ..., 5] = 0
                else:
                    annotations[i, ..., 5] += annotations[i, ..., 8]
                    annotations[i, ..., 8] = 0
                if amounts[i, 6] < amounts[i, 7]:
                    annotations[i, ..., 7] += annotations[i, ..., 6]
                    annotations[i, ..., 6] = 0
                else:
                    annotations[i, ..., 6] += annotations[i, ..., 7]
                    annotations[i, ..., 7] = 0
        return annotations

    def iterate_annotations(self, annotations_base, ground_truth_base, spacing, allowed_differences=50,
                            max_iteration=15, reduce2D=True):
        '''
        :param annotations:
        :param ground_truth:
        :param spacing:
        :param allowed_differences:
        :param max_iteration:
        :param z_mult: factor by which to ensure slices don't bleed into ones above and below
        :return:
        '''
        differences = [np.inf]
        index = 0
        liver = np.squeeze(ground_truth_base)
        min_z_s, max_z_s, min_r_s, max_r_s, min_c_s, max_c_s = get_bounding_box_indexes(liver)
        annotations = annotations_base[min_z_s:max_z_s, min_r_s:max_r_s, min_c_s:max_c_s]
        ground_truth = ground_truth_base[min_z_s:max_z_s, min_r_s:max_r_s, min_c_s:max_c_s]
        while differences[-1] > allowed_differences and index < max_iteration:
            previous_iteration = copy.deepcopy(np.argmax(annotations, axis=-1))
            for i in range(1, annotations.shape[-1]):
                annotation = annotations[..., i]
                if reduce2D:
                    # start = time.time()
                    slices = np.where(np.max(annotation, axis=(1, 2)) > 0)
                    for slice in slices[0]:
                        annotation[slice] = sitk.GetArrayFromImage(self.remove_smallest.remove_smallest_component(
                            sitk.GetImageFromArray(annotation[slice].astype('float32')) > 0))
                    # print('Took {} seconds'.format(time.time()-start))
                # start = time.time()
                annotations[..., i] = sitk.GetArrayFromImage(self.remove_smallest.remove_smallest_component(
                    sitk.GetImageFromArray(annotation.astype('float32')) > 0))
                # print('Took {} seconds'.format(time.time() - start))
            # annotations = self.remove_56_78(annotations)
            summed = np.sum(annotations, axis=-1)
            annotations[summed > 1] = 0
            annotations[annotations > 0] = 1
            annotations[..., 0] = 1 - ground_truth
            # start = time.time()
            annotations = self.make_distance_map(annotations, ground_truth, spacing=spacing)
            differences.append(np.abs(
                np.sum(previous_iteration[ground_truth == 1] - np.argmax(annotations, axis=-1)[ground_truth == 1])))
            index += 1
        annotations_base[min_z_s:max_z_s, min_r_s:max_r_s, min_c_s:max_c_s] = annotations
        annotations_base[..., 0] = 1 - ground_truth_base
        return annotations_base

    def run_distance_map(self, array, spacing):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        output = self.MauererDistanceMap.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

    def make_distance_map(self, pred, liver, spacing=(0.975, 0.975, 2.5)):
        '''
        :param pred: A mask of your predictions with N channels on the end, N=0 is background [# Images, 512, 512, N]
        :param liver: A mask of the desired region [# Images, 512, 512]
        :param reduce: Save time and only work on masked region
        :return:
        '''
        liver = np.squeeze(liver)
        pred = np.squeeze(pred)
        pred = np.round(pred).astype('int')
        min_z, min_r, min_c = 0, 0, 0
        max_z, max_r, max_c = pred.shape[:3]
        reduced_pred = pred[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_liver = liver[min_z:max_z, min_r:max_r, min_c:max_c]
        reduced_output = np.zeros(reduced_pred.shape)
        for i in range(1, pred.shape[-1]):
            temp_reduce = reduced_pred[..., i]
            output = self.run_distance_map(temp_reduce, spacing)
            reduced_output[..., i] = output
        reduced_output[reduced_output > 0] = 0
        reduced_output = np.abs(reduced_output)
        reduced_output[..., 0] = np.inf
        output = np.zeros(reduced_output.shape, dtype='int')
        mask = reduced_liver == 1
        values = reduced_output[mask]
        output[mask, np.argmin(values, axis=-1)] = 1
        pred[min_z:max_z, min_r:max_r, min_c:max_c] = output
        return pred


def createthreshold(predictionimage, seeds, thresholdvalue):
    Connected_Threshold = sitk.ConnectedThresholdImageFilter()
    Connected_Threshold.SetUpper(2)
    Connected_Threshold.SetSeedList(seeds)
    Connected_Threshold.SetLower(thresholdvalue)
    threshold_prediction = Connected_Threshold.Execute(sitk.Cast(predictionimage, sitk.sitkFloat32))
    del Connected_Threshold, predictionimage, seeds, thresholdvalue
    return threshold_prediction


def createseeds(predictionimage, seed_value):
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    stats = sitk.LabelShapeStatisticsImageFilter()
    thresholded_image = sitk.BinaryThreshold(sitk.Cast(predictionimage, sitk.sitkFloat32), lowerThreshold=seed_value)
    connected_image = Connected_Component_Filter.Execute(thresholded_image)
    stats.Execute(connected_image)
    seeds = [stats.GetCentroid(l) for l in stats.GetLabels()]
    seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in seeds]
    del stats, Connected_Component_Filter, connected_image, predictionimage, seed_value
    return seeds


class Threshold_and_Expand_New(ImageProcessor):
    def __init__(self, seed_threshold_value=None, lower_threshold_value=None, prediction_key='prediction',
                 ground_truth_key='annotation', dicom_handle_key='primary_handle'):
        self.seed_threshold_value = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.stats = sitk.LabelShapeStatisticsImageFilter()
        self.lower_threshold_value = lower_threshold_value
        self.Connected_Threshold.SetUpper(2)
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key
        self.Iterate_Lobe_Annotations_Class = Iterate_Lobe_Annotations()
        self.dicom_handle_key = dicom_handle_key

    def pre_process(self, input_features):
        pred = input_features[self.prediction_key]
        ground_truth = input_features[self.ground_truth_key]
        out_prediction = np.zeros(pred.shape).astype('float32')
        for i in range(1, out_prediction.shape[-1]):
            out_prediction[..., i] = sitk.GetArrayFromImage(
                createthreshold(sitk.GetImageFromArray(pred[..., i].astype('float32')),
                                createseeds(sitk.GetImageFromArray(pred[..., i].astype('float32')),
                                            self.seed_threshold_value[i - 1]),
                                self.lower_threshold_value[i - 1]))
        summed_image = np.sum(out_prediction, axis=-1)
        # stop = time.time()
        out_prediction[summed_image > 1] = 0
        out_prediction = self.Iterate_Lobe_Annotations_Class.iterate_annotations(
            out_prediction, ground_truth > 0,
            spacing=input_features[self.dicom_handle_key].GetSpacing(),
            max_iteration=10, reduce2D=False)
        input_features[self.prediction_key] = out_prediction
        return input_features


class ThresholdToMask(ImageProcessor):
    def __init__(self, mask_keys=('annotation',), lower_bounds=(255/2,)):
        self.mask_keys = mask_keys
        self.lower_bounds = lower_bounds

    def pre_process(self, input_features):
        _check_keys_(input_features, self.mask_keys)
        for mask_key, lower_threshold in zip(self.mask_keys, self.lower_bounds):
            mask_array = input_features[mask_key]
            mask_array[mask_array < lower_threshold] = 0
            mask_array[mask_array > 0] = 1
            input_features[mask_key] = mask_array
        return input_features


class MaskOneBasedOnOther(ImageProcessor):
    def __init__(self, guiding_keys=('annotation',), changing_keys=('image',), guiding_values=(1,), mask_values=(-1,),
                 methods=('equal_to',)):
        """
        :param guiding_keys: keys which will guide the masking of another key
        :param changing_keys: keys which will be masked
        :param guiding_values: values which will define the mask
        :param mask_values: values which will be changed
        :param methods: method of masking, 'equal_to', 'less_than', 'greater_than'
        """
        self.guiding_keys, self.changing_keys = guiding_keys, changing_keys
        self.guiding_values, self.mask_values = guiding_values, mask_values
        for method in methods:
            assert method in ('equal_to', 'less_than', 'greater_than'), 'Only provide a method of equal_to, ' \
                                                                        'less_than, or greater_than'
        self.methods = methods

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.guiding_keys)
        _check_keys_(input_features=input_features, keys=self.changing_keys)
        for guiding_key, changing_key, guiding_value, mask_value, method in zip(self.guiding_keys, self.changing_keys,
                                                                                self.guiding_values, self.mask_values,
                                                                                self.methods):
            if method == 'equal_to':
                input_features[changing_key] = np.where(input_features[guiding_key] == guiding_value,
                                                        mask_value, input_features[changing_key])
            elif method == 'less_than':
                input_features[changing_key] = np.where(input_features[guiding_key] < guiding_value,
                                                        mask_value, input_features[changing_key])
            elif method == 'greater_than':
                input_features[changing_key] = np.where(input_features[guiding_key] > guiding_value,
                                                        mask_value, input_features[changing_key])
        return input_features


class MaskKeys(ImageProcessor):
    def __init__(self, key_tuple=('annotation',), from_values_tuple=(2,), to_values_tuple=(1,)):
        """
        :param key_tuple: tuple of key names that will be present in image_features
        :param from_values_tuple: tuple of values that we will change from
        :param to_values_tuple: tuple of values that we will change to
        """
        self.key_list = key_tuple
        self.from_list = from_values_tuple
        self.to_list = to_values_tuple

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.key_list)
        for key, from_value, to_value in zip(self.key_list, self.from_list, self.to_list):
            input_features[key] = np.where(input_features[key] == from_value, to_value, input_features[key])
        return input_features


class DistributeIntoRecurrenceCubes(ImageProcessor):
    def __init__(self, rows=128, cols=128, images=32):
        self.rows, self.cols, self.images = rows, cols, images
    """
    Highly specialized for the task of model prediction, likely won't be useful for others
    """
    def pre_process(self, input_features):
        out_features = OrderedDict()
        primary_array = input_features['primary_image']
        image_size = primary_array.shape
        secondary_array = input_features['secondary_image']
        secondary_deformed_array = input_features['secondary_image_deformed']
        primary_mask = input_features['primary_mask']
        secondary_mask = input_features['secondary_mask']
        '''
        Now, find centroids in the cases
        '''
        Connected_Component_Filter_no_recurred = sitk.ConnectedComponentImageFilter()
        Connected_Component_Filter_no_recurred.FullyConnectedOn()
        stats_no_recurred = sitk.LabelShapeStatisticsImageFilter()

        no_recurred_image = sitk.GetImageFromArray((primary_mask == 1).astype('int'))
        connected_image_no_recurred = Connected_Component_Filter_no_recurred.Execute(no_recurred_image)
        stats_no_recurred.Execute(connected_image_no_recurred)
        no_recurrence_labels = [l for l in stats_no_recurred.GetLabels()]
        no_recurrence_centroids = [no_recurred_image.TransformPhysicalPointToIndex(stats_no_recurred.GetCentroid(l))
                                   for l in no_recurrence_labels]


        Connected_Component_Filter_recurred = sitk.ConnectedComponentImageFilter()
        Connected_Component_Filter_recurred.FullyConnectedOn()
        stats_recurred = sitk.LabelShapeStatisticsImageFilter()

        recurred_image = sitk.GetImageFromArray((primary_mask == 2).astype('int'))
        connected_image_recurred = Connected_Component_Filter_recurred.Execute(recurred_image)
        stats_recurred.Execute(connected_image_recurred)
        recurrence_labels = [l for l in stats_recurred.GetLabels()]
        recurrence_centroids = [recurred_image.TransformPhysicalPointToIndex(stats_recurred.GetCentroid(l))
                                for l in recurrence_labels]
        for value, cube_name, centroids, label, image in zip([0, 1], ['Non_Recurrence_Cube_{}', 'Recurrence_Cube_{}'],
                                                             [no_recurrence_centroids, recurrence_centroids],
                                                             [no_recurrence_labels, recurrence_labels],
                                                             [connected_image_no_recurred, connected_image_recurred]):
            for index, centroid in enumerate(centroids):
                temp_feature = OrderedDict()
                col_center, row_center, z_center = centroid
                z_start_pad, z_stop_pad, r_start_pad, r_stop_pad, c_start_pad, c_stop_pad = 0, 0, 0, 0, 0, 0
                z_start = z_center - self.images // 2
                if z_start < 0:
                    z_start_pad = abs(z_start)
                    z_start = 0
                z_stop = z_center + self.images // 2
                if z_stop > image_size[0]:
                    z_stop_pad = z_stop - image_size[0]
                    z_stop = image_size[0]
                r_start = row_center - self.rows // 2
                if r_start < 0:
                    r_start_pad = abs(r_start)
                    r_start = 0
                r_stop = row_center + self.rows // 2
                if r_stop > image_size[1]:
                    r_stop_pad = r_stop - image_size[1]
                    r_stop = image_size[1]
                c_start = col_center - self.cols // 2
                if c_start < 0:
                    c_start_pad = abs(c_start)
                    c_start = 0
                c_stop = col_center + self.cols // 2
                if c_stop > image_size[2]:
                    c_stop_pad = c_stop - image_size[2]
                    c_stop = image_size[2]
                primary_cube = primary_array[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                secondary_cube = secondary_array[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                secondary_deformed_cube = secondary_deformed_array[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                index_mask = sitk.GetArrayFromImage(image)
                index_mask[index_mask != label[index]] = 0
                index_mask[index_mask > 0] = 1
                index_mask = index_mask.astype('int')
                index_mask = index_mask[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                primary_liver_cube = primary_mask[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                secondary_liver_cube = secondary_mask[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                pads = [[z_start_pad, z_stop_pad], [r_start_pad, r_stop_pad], [c_start_pad, c_stop_pad]]
                if np.max(pads) > 0:
                    primary_cube = np.pad(primary_cube, pads, constant_values=np.min(primary_cube))
                    secondary_cube = np.pad(secondary_cube, pads, constant_values=np.min(secondary_cube))
                    secondary_deformed_cube = np.pad(secondary_deformed_cube, pads,
                                                     constant_values=np.min(secondary_deformed_cube))
                    primary_liver_cube = np.pad(primary_liver_cube, pads, constant_values=np.min(primary_liver_cube))
                    secondary_liver_cube = np.pad(secondary_liver_cube, pads, constant_values=np.min(secondary_liver_cube))
                    index_mask = np.pad(index_mask, pads, constant_values=np.min(index_mask))
                temp_feature['primary_image'] = primary_cube
                temp_feature['secondary_image'] = secondary_cube
                temp_feature['secondary_image_deformed'] = secondary_deformed_cube
                primary_liver_cube[primary_liver_cube > 0] = 1  # Make it so we have liver at 1, and disease as 2
                primary_liver_cube[index_mask == 1] = 2
                primary_liver_cube = primary_liver_cube.astype('int8')
                secondary_liver_cube = secondary_liver_cube.astype('int8')
                temp_feature['primary_liver'] = primary_liver_cube
                temp_feature['secondary_liver'] = secondary_liver_cube
                temp_feature['annotation'] = to_categorical(value, 2)
                wanted_keys = ('primary_image_path', 'file_name', 'spacing')
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

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key, value in zip(self.image_keys, self.values):
            image_array = input_features[key]
            image_array /= value
            input_features[key] = image_array
        return input_features


class MultiplyByValues(ImageProcessor):
    def __init__(self, image_keys=('image',), values=(1.,)):
        """
        :param image_keys: tuple of keys to multiply by the value
        :param values: values by which to multiply by
        """
        self.image_keys = image_keys
        self.values = values

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key, value in zip(self.image_keys, self.values):
            image_array = input_features[key]
            image_array *= value
            input_features[key] = image_array
        return input_features


class ChangeArrayByArgInArray(ImageProcessor):
    def __init__(self, reference_keys=('image',), value_args=(np.max,),
                 target_keys=('image',), change_args=(np.multiply,)):
        """
        :param reference_key: a key for a reference array
        :param value_arg: a function to obtain some value from the reference array
        :param target_key: a key for the target array to have a change_arg applied
        :param change_arg: a function to apply some change
        """
        self.reference_keys = reference_keys
        self.value_args = value_args
        self.target_keys = target_keys
        self.change_args = change_args

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.reference_keys + self.target_keys)
        for ref_key, value_arg, target_key, change_arg in zip(self.reference_keys, self.value_args,
                                                              self.target_keys, self.change_args):
            ref_array = input_features[ref_key]
            target_array = input_features[target_key]
            value = value_arg(ref_array)
            input_features[target_key] = change_arg(target_array, value)
        return input_features


class Fill_Binary_Holes(ImageProcessor):
    def __init__(self, dicom_handle_key: str, prediction_key: Optional[str] = None,
                 prediction_keys: Optional[Tuple] = None):
        self.BinaryfillFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryfillFilter.SetFullyConnected(True)
        if prediction_keys is None and prediction_key is not None:
            prediction_keys = (prediction_key,)
        self.prediction_keys = prediction_keys
        self.dicom_handle_key = dicom_handle_key

    def pre_process(self, input_features):
        dicom_handle = input_features[self.dicom_handle_key]
        for pred_key in self.prediction_keys:
            pred = input_features[pred_key]
            for class_num in range(1, pred.shape[-1]):
                temp_pred = pred[..., class_num]
                k = sitk.GetImageFromArray(temp_pred.astype('int'))
                k.SetSpacing(dicom_handle.GetSpacing())
                output = self.BinaryfillFilter.Execute(k)
                output_array = sitk.GetArrayFromImage(output)
                pred[..., class_num] = output_array
            pred[..., 0] = 0
            input_features[pred_key] = pred.astype('int8')
        return input_features


class MinimumVolumeandAreaPrediction(ImageProcessor):
    '''
    This should come after prediction thresholding
    '''

    def __init__(self, min_volume=0.0, min_area=0.0, max_area=np.inf, pred_axis=(1,),
                 prediction_key: Optional[str] = 'prediction', dicom_handle_key='primary_handle', largest_only=False,
                 prediction_keys: Optional[Tuple] = None):
        '''
        :param min_volume: Minimum volume of structure allowed, in cm3
        :param min_area: Minimum area of structure allowed, in cm2
        :param max_area: Max area of structure allowed, in cm2
        :return: Masked annotation
        '''
        self.min_volume = min_volume * 1000  # cm3 to mm3
        self.min_area = min_area * 100
        self.max_area = max_area * 100
        self.pred_axis = pred_axis
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        if prediction_keys is None:
            prediction_keys = (prediction_key,)
        self.prediction_keys = prediction_keys
        self.dicom_handle_key = dicom_handle_key
        self.largest_only = largest_only

    def pre_process(self, input_features):

        dicom_handle = input_features[self.dicom_handle_key]
        for pred_key in self.prediction_keys:
            pred = input_features[pred_key]
            for axis in self.pred_axis:
                temp_pred = pred[..., axis]
                if self.min_volume != 0:
                    label_image = self.Connected_Component_Filter.Execute(sitk.GetImageFromArray(temp_pred) > 0)
                    self.RelabelComponent.SetMinimumObjectSize(
                        int(self.min_volume / np.prod(dicom_handle.GetSpacing())))
                    label_image = self.RelabelComponent.Execute(label_image)
                    temp_pred = sitk.GetArrayFromImage(label_image > 0)
                if self.min_area != 0 or self.max_area != np.inf:
                    slice_indexes = np.where(np.sum(temp_pred, axis=(1, 2)) > 0)
                    if slice_indexes:
                        slice_spacing = np.prod(dicom_handle.GetSpacing()[:-1])
                        for slice_index in slice_indexes[0]:
                            labels = morphology.label(temp_pred[slice_index], connectivity=1)
                            for i in range(1, labels.max() + 1):
                                new_area = labels[labels == i].shape[0]
                                temp_area = slice_spacing * new_area
                                if temp_area > self.max_area:
                                    labels[labels == i] = 0
                                    continue
                                elif temp_area < self.min_area:
                                    labels[labels == i] = 0
                                    continue
                            labels[labels > 0] = 1
                            temp_pred[slice_index] = labels
                if self.min_volume != 0:
                    label_image = self.Connected_Component_Filter.Execute(sitk.GetImageFromArray(temp_pred) > 0)
                    self.RelabelComponent.SetMinimumObjectSize(
                        int(self.min_volume / np.prod(dicom_handle.GetSpacing())))
                    label_image = self.RelabelComponent.Execute(label_image)
                    if self.largest_only:
                        temp_pred = sitk.GetArrayFromImage(label_image == 1)
                    else:
                        temp_pred = sitk.GetArrayFromImage(label_image > 0)
                pred[..., axis] = temp_pred
            input_features[pred_key] = pred
        return input_features


class Threshold_and_Expand(ImageProcessor):
    def __init__(self, seed_threshold_values=None, lower_threshold_values=None,
                 prediction_key: Optional[str] = 'prediction', prediction_keys: Optional[Tuple] = None):
        self.seed_threshold_values = seed_threshold_values
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.stats = sitk.LabelShapeStatisticsImageFilter()
        self.lower_threshold_values = lower_threshold_values
        self.Connected_Threshold.SetUpper(2)
        if prediction_keys is None:
            prediction_keys = (prediction_key,)
        self.prediction_keys = prediction_keys

    def pre_process(self, input_features):
        for pred_key, seed_threshold, lower_threshold in zip(self.prediction_keys, self.seed_threshold_values,
                                                         self.lower_threshold_values):
            pred = input_features[pred_key]
            for i in range(1, pred.shape[-1]):
                temp_pred = pred[..., i]
                output = np.zeros(temp_pred.shape)
                expanded = False
                if len(temp_pred.shape) == 4:
                    temp_pred = temp_pred[0]
                    expanded = True
                prediction = sitk.GetImageFromArray(temp_pred)
                overlap = temp_pred > seed_threshold
                if np.max(overlap) > 0:
                    seeds = np.transpose(np.asarray(np.where(overlap > 0)))[..., ::-1]
                    seeds = [[int(i) for i in j] for j in seeds]
                    self.Connected_Threshold.SetLower(lower_threshold)
                    self.Connected_Threshold.SetSeedList(seeds)
                    output = sitk.GetArrayFromImage(self.Connected_Threshold.Execute(prediction))
                    if expanded:
                        output = output[None, ...]
                pred[..., i] = output
            input_features[pred_key] = pred
        return input_features


class Threshold_Images(ImageProcessor):
    def __init__(self, image_keys=('image',), lower_bound=-np.inf, upper_bound=np.inf, divide=False):
        """
        :param image_keys: tuple key for images in the image_features dictionary
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        """
        self.lower = lower_bound
        self.upper = upper_bound
        self.image_keys = image_keys
        self.divide = divide

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key in self.image_keys:
            image = input_features[key]
            image[image < self.lower] = self.lower
            image[image > self.upper] = self.upper
            if self.divide:
                image = image / (self.upper - self.lower)
            input_features[key] = image
        return input_features


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

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.image_key, self.annotation_key))
        images = input_features[self.image_key]
        liver = input_features[self.annotation_key]
        data = images[liver > 0].flatten()
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


class NormalizeToAnnotation(Normalize_to_annotation):
    def __init__(self, image_key='image', annotation_key='annotation', annotation_value_list=None, mirror_max=False,
                 lower_percentile=None, upper_percentile=None):
        super(NormalizeToAnnotation, self).__init__(image_key=image_key, annotation_key=annotation_key,
                                                    annotation_value_list=annotation_value_list,
                                                    mirror_max=mirror_max, lower_percentile=lower_percentile,
                                                    upper_percentile=upper_percentile)


def expand_box_indexes(z_start, z_stop, r_start, r_stop, c_start, c_stop, annotation_shape, bounding_box_expansion):
    z_start = max([0, z_start - floor(bounding_box_expansion[0] / 2)])
    z_stop = min([annotation_shape[0], z_stop + ceil(bounding_box_expansion[0] / 2)])
    r_start = max([0, r_start - floor(bounding_box_expansion[1] / 2)])
    r_stop = min([annotation_shape[1], r_stop + ceil(bounding_box_expansion[1] / 2)])
    c_start = max([0, c_start - floor(bounding_box_expansion[2] / 2)])
    c_stop = min([annotation_shape[2], c_stop + ceil(bounding_box_expansion[2] / 2)])
    return z_start, z_stop, r_start, r_stop, c_start, c_stop


class Box_Images(ImageProcessor):
    def __init__(self, image_keys=('image',), annotation_key='annotation', wanted_vals_for_bbox=None,
                 bounding_box_expansion=(5, 10, 10), power_val_z=1, power_val_r=1,
                 power_val_c=1, min_images=None, min_rows=None, min_cols=None,
                 post_process_keys=('image', 'annotation', 'prediction'), pad_values=None, on_needles=False):
        """
        :param image_keys: keys which corresponds to an image to be normalized
        :param annotation_key: key which corresponds to an annotation image used for normalization
        :param wanted_vals_for_bbox:
        :param bounding_box_expansion:
        :param power_val_z:
        :param power_val_r:
        :param power_val_c:
        :param min_images:
        :param min_rows:
        :param min_cols:
        :param on_needles: arbitrary addition to allow me to only keep digitized parts of needles
        """
        assert type(wanted_vals_for_bbox) in [list, tuple], 'Provide a list for bboxes'
        self.wanted_vals_for_bbox = wanted_vals_for_bbox
        self.bounding_box_expansion = bounding_box_expansion
        self.power_val_z, self.power_val_r, self.power_val_c = power_val_z, power_val_r, power_val_c
        self.min_images, self.min_rows, self.min_cols = min_images, min_rows, min_cols
        self.image_keys, self.annotation_key = image_keys, annotation_key
        self.post_process_keys = post_process_keys
        if pad_values is None:
            pad_values = [None for _ in range(len(image_keys))]
        self.pad_values = pad_values
        self.on_needles = on_needles

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys + (self.annotation_key,))
        annotation = input_features[self.annotation_key]
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
            add_indexes.pre_process(input_features)
            del input_features['mask']
            z_start, z_stop, r_start, r_stop, c_start, c_stop = add_bounding_box_to_dict(
                input_features['bounding_boxes_{}'.format(val)], return_indexes=True, on_needles=self.on_needles)

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
            input_features['z_r_c_start'] = [z_start, r_start, c_start]
            for key, pad_value in zip(self.image_keys, self.pad_values):
                image = input_features[key]
                input_features['og_shape'] = image.shape
                input_features['og_shape_{}'.format(key)] = image.shape
                image_cube = image[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                img_shape = image_cube.shape
                pads = [min_images - img_shape[0], min_rows - img_shape[1], min_cols - img_shape[2]]
                pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
                if pad_value is None:
                    pad_value = np.min(image_cube)
                while len(image_cube.shape) > len(pads):
                    pads += [[0, 0]]
                image_cube = np.pad(image_cube, pads, constant_values=pad_value)
                input_features[key] = image_cube.astype(image.dtype)
                input_features['pads'] = [pads[i][0] for i in range(3)]
            annotation_cube = annotation[z_start:z_stop, r_start:r_stop, c_start:c_stop]
            pads = [min_images - annotation_cube.shape[0], min_rows - annotation_cube.shape[1],
                    min_cols - annotation_cube.shape[2]]
            if len(annotation.shape) > 3:
                pads = np.append(pads, [0])
            pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
            annotation_cube = np.pad(annotation_cube, pads)
            if len(annotation.shape) > 3:
                annotation_cube[..., 0] = 1 - np.sum(annotation_cube[..., 1:], axis=-1)
            input_features[self.annotation_key] = annotation_cube.astype(annotation.dtype)
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.post_process_keys)
        for key in self.post_process_keys:
            image = input_features[key]
            pads = input_features['pads']
            image = image[pads[0]:, pads[1]:, pads[2]:, ...]
            pads = [(i, 0) for i in input_features['z_r_c_start']]
            while len(image.shape) > len(pads):
                pads += [(0, 0)]
            image = np.pad(image, pads, constant_values=np.min(image))
            og_shape = input_features['og_shape']
            im_shape = image.shape
            if im_shape[0] > og_shape[0]:
                dif = og_shape[0] - im_shape[0]
                image = image[:dif]
            if im_shape[1] > og_shape[1]:
                dif = og_shape[1] - im_shape[1]
                image = image[:, :dif]
            if im_shape[2] > og_shape[2]:
                dif = og_shape[2] - im_shape[2]
                image = image[:, :, :dif]
            im_shape = image.shape
            pads = [(0, og_shape[0] - im_shape[0]), (0, og_shape[1] - im_shape[1]), (0, og_shape[2] - im_shape[2])]
            if len(image.shape) > 3:
                pads += [(0, 0)]
            image = np.pad(image, pads, constant_values=np.min(image))
            input_features[key] = image
        return input_features


def return_largest_bounding_box(bounding_boxes, number_of_voxels):
    num_voxel = 0
    out_box = bounding_boxes[0]
    for bbox, voxel_num in zip(bounding_boxes, number_of_voxels):
        if voxel_num > num_voxel:
            out_box = bbox
            num_voxel = voxel_num
    return out_box


class CropHandlesAboutValues(ImageProcessor):
    def __init__(self, input_keys=("image_handle", "dose_handle"), guiding_key="dose_handle", min_value=0.5,
                 upper_value=None, power_val_z=1, power_val_x=1, power_val_y=1):
        """
        We have power vals here for a reason. If we are cropping about a dose distribution and the 'cube' that comes
        out is 55x110x110, but we need our final 'cube' to be 64x128x128, why not bump it out a little?
        By default you will get the minimal cube, but the power_val can 'pad' with the natural image
        """
        self.input_keys = input_keys
        self.guiding_key = guiding_key
        self.min_value = min_value
        self.upper_value = upper_value
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y
    
    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.input_keys + (self.guiding_key,))
        image_handle: sitk.Image
        guide_handle: sitk.Image
        guide_handle = input_features[self.guiding_key]
        bounding_boxes, num_voxels = get_bounding_boxes(guide_handle, lower_threshold=self.min_value, upper_threshold=self.upper_value)
        if len(bounding_boxes) > 0:
            bounding_box = return_largest_bounding_box(bounding_boxes, num_voxels)  # Bounding box is row, col, z, rows, cols, zs
            row_size, col_size, z_size = guide_handle.GetSize()
            row_start, col_start, z_start = bounding_box[0], bounding_box[1], bounding_box[2]
            row_stop = row_start + bounding_box[3]
            col_stop = col_start + bounding_box[4]
            z_stop = z_start + bounding_box[5]
            r_total, c_total, z_total = bounding_box[3], bounding_box[4], bounding_box[5]
            remainder_z, remainder_r, remainder_c = (self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0,
                                                     self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0,
                                                     self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0)
            row_start = max([0, row_start - remainder_r//2])
            col_start = max([0, col_start - remainder_c//2])
            z_start = max([0, z_start - remainder_z//2])
            row_stop = min([row_size, row_stop + remainder_r//2])
            col_stop = min([col_size, col_stop + remainder_c // 2])
            z_stop = min([z_size, z_stop + remainder_z // 2])
            for image_key in self.input_keys:
                image_handle = input_features[image_key]
                image_handle = image_handle[row_start:row_stop, col_start:col_stop, z_start:z_stop]
                input_features[image_key] = image_handle
                input_features['crop_row_start'] = row_start
                input_features['crop_col_start'] = row_start
                input_features['crop_z_start'] = row_start
        return input_features


def largest_component_2D_slice(binary_image):
    # Get the size of the 3D image
    size = binary_image.GetSize()
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    # Initialize an empty list to store the processed slices
    processed_slices = []

    # Iterate over each 2D slice along the z-axis
    for z in range(size[2]):
        # Extract the 2D slice
        slice_2D = binary_image[:, :, z]

        # Label connected components in the 2D slice
        label_image = sitk.ConnectedComponent(slice_2D)

        # Analyze the connected components
        label_shape_filter.Execute(label_image)

        # Find the label of the largest component
        labels = label_shape_filter.GetLabels()
        largest_label = 0
        if len(labels) != 0:
            largest_label = max(label_shape_filter.GetLabels(),
                                key=lambda label: label_shape_filter.GetPhysicalSize(label))
        # Create a binary mask of the largest component
        largest_component = sitk.BinaryThreshold(label_image, lowerThreshold=largest_label,
                                                 upperThreshold=largest_label, insideValue=1, outsideValue=0)
        largest_component = sitk.BinaryFillhole(largest_component)
        # Convert the largest component back to numpy array and add to the list
        processed_slices.append(sitk.GetArrayFromImage(largest_component))

    # Stack the 2D slices back into a 3D numpy array
    processed_3D_array = np.stack(processed_slices, axis=0)

    # Convert the numpy array back to a SimpleITK image
    processed_3D_image = sitk.GetImageFromArray(processed_3D_array)
    processed_3D_image.CopyInformation(binary_image)  # Keep original spatial information
    return processed_3D_image


class MaintainOverlapBySlicesImages(ImageProcessor):
    def __init__(self, changing_handles=('body_handle',), guiding_handle='mask_handle'):
        self.changing_handles = changing_handles
        self.guiding_handle = guiding_handle

    def pre_process(self, input_features):
        guiding_handle: sitk.Image
        changing_handle: sitk.Image
        _check_keys_(input_features, self.changing_handles + (self.guiding_handle,))
        guiding_handle = input_features[self.guiding_handle]
        guiding_array = sitk.GetArrayFromImage(guiding_handle)
        has_numbers = np.sum(guiding_array, axis=(1, 2))
        indexes = np.where(has_numbers == 0)[0]
        if indexes.shape[0] > 0:
            for changing_key in self.changing_handles:
                changing_handle = input_features[changing_key]
                for i in indexes:
                    changing_handle[:, :, int(i)] = 0
                input_features[changing_key] = changing_handle
        return input_features


class IdentifyBodyContour(ImageProcessor):
    """
    Code for converting a SITK image of a patient into a binary SITK image, only taking the largest
    2D components
    """
    def __init__(self, image_key='image_handle', lower_threshold=-50, upper_threshold=1000,
                 out_label='body_handle', dilation_erosion_radius=3):
        self.image_key = image_key
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.out_label = out_label
        self.label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        self.dilation_erosion_radius = dilation_erosion_radius

    def pre_process(self, input_features):
        _check_keys_(input_features, (self.image_key,))
        image_handle = input_features[self.image_key]
        binary_image = sitk.BinaryThreshold(image_handle, lowerThreshold=self.lower_threshold,
                                            upperThreshold=self.upper_threshold, insideValue=1, outsideValue=0)
        # Step 1: Find largest connected component on a 2D slice basis
        binary_image = largest_component_2D_slice(binary_image)

        # Step 2: Remove outlier pieces, like table
        label_image = sitk.ConnectedComponent(binary_image)

        # Analyze the connected components
        self.label_shape_filter.Execute(label_image)

        # Find the label of the largest component
        largest_label = max(self.label_shape_filter.GetLabels(), key=lambda
            label: self.label_shape_filter.GetPhysicalSize(label))

        # Next, lets dilate a little to clean up
        binary_image = sitk.BinaryThreshold(label_image, lowerThreshold=largest_label,
                                            upperThreshold=largest_label, insideValue=1, outsideValue=0)
        # Perform binary dilation using the structuring element
        smeared_image = sitk.BinaryDilate(binary_image, kernelRadius=[self.dilation_erosion_radius]*3,
                                          kernelType=sitk.sitkBall)


        #And fill in any holes
        binary_image = sitk.BinaryFillhole(smeared_image)

        # And undo part of the dilation
        binary_image = sitk.BinaryErode(binary_image, kernelRadius=[self.dilation_erosion_radius] * 3,
                                        kernelType=sitk.sitkBall)
        input_features[self.out_label] = binary_image
        return input_features


class ConvertBodyContourToCentroidLine(ImageProcessor):
    def __init__(self, body_handle_key, out_key, extent_evaluated=1):
        self.body_handle_key = body_handle_key
        self.out_key = out_key
        self.label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        self.extent_evaluated = extent_evaluated

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.body_handle_key,))
        label_image = input_features[self.body_handle_key]
        label_image: sitk.Image
        if self.extent_evaluated != 1:
            image_x, image_y, image_z = label_image.GetSize()
            if self.extent_evaluated > 0:
                label_image = label_image[:, :, int(image_z*self.extent_evaluated):]
            else:
                label_image = label_image[:, :, :int(image_z * self.extent_evaluated)]
        mask_numpy = np.zeros(label_image.GetSize()[::-1]).astype('uint8')
        summed_image = np.sum(sitk.GetArrayFromImage(label_image), axis=(1, 2))
        for i in range(label_image.GetSize()[-1]):
            if summed_image[i] == 0:
                continue
            label_image_slice = label_image[..., i]
            self.label_shape_filter.Execute(label_image_slice)

            # Step 4: Calculate the centroid of the largest component
            centroid = self.label_shape_filter.GetCentroid(1)

            # Convert the centroid to index coordinates
            centroid_index = label_image_slice.TransformPhysicalPointToIndex(centroid)


            mask_numpy[i, centroid_index[1], centroid_index[0]] = 1

        # Create an empty binary image of the same size (all zeros)
        centroid_image = sitk.GetImageFromArray(mask_numpy)
        centroid_image.SetOrigin(label_image.GetOrigin())
        centroid_image.SetSpacing(label_image.GetSpacing())
        centroid_image.SetDirection(label_image.GetDirection())
        input_features[self.out_key] = centroid_image
        return input_features


class PadImages(ImageProcessor):
    def __init__(self, bounding_box_expansion=(10, 10, 10), power_val_z=1, power_val_x=1,
                 power_val_y=1, min_val=None, image_keys=('image', 'annotation'),
                 post_process_keys=('image', 'annotation', 'prediction'), mode='constant'):
        self.bounding_box_expansion = bounding_box_expansion
        self.min_val = min_val
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y
        self.image_keys = image_keys
        self.post_process_keys = post_process_keys
        self.mode = mode

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys)
        for key in self.image_keys:
            images = input_features[key]
            images_shape = images.shape
            self.og_shape = images_shape
            z_start, r_start, c_start = 0, 0, 0
            z_stop, r_stop, c_stop = images_shape[0], images_shape[1], images_shape[2]
            z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
            self.remainder_z, self.remainder_r, self.remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                                   self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0, \
                                                                   self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0
            pads = [self.remainder_z, self.remainder_r, self.remainder_c]
            self.pad = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
            if len(images_shape) > 3:
                self.pad = [[0, 0]] + self.pad
            if self.min_val is None:
                min_val = np.min(images)
            else:
                min_val = self.min_val
            if self.mode == 'constant':
                images = np.pad(images, self.pad, constant_values=min_val)
            elif self.mode == 'linear_ramp':
                images = np.pad(images, self.pad, mode=self.mode, end_values=min_val)
            else:
                images = np.pad(images, self.pad, mode=self.mode)
            input_features[key] = images
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.post_process_keys)
        if max([self.remainder_z, self.remainder_r, self.remainder_c]) == 0:
            return input_features
        for key in self.post_process_keys:
            pred = input_features[key]
            if len(pred.shape) == 3 or len(pred.shape) == 4:
                pred = pred[self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
                pred = pred[:self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]
            elif len(pred.shape) == 5:
                pred = pred[:, self.pad[0][0]:, self.pad[1][0]:, self.pad[2][0]:]
                pred = pred[:, :self.og_shape[0], :self.og_shape[1], :self.og_shape[2]]
            input_features[key] = pred
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

    def pre_process(self, input_features):
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
                input_features = add_bounding_box_to_dict(input_features=input_features, bounding_box=bounding_boxes,
                                                          val=val, return_indexes=False,
                                                          add_to_dictionary=self.add_to_dictionary)
        return input_features


def add_bounding_box_to_dict(bounding_box, input_features=None, val=None, return_indexes=False,
                             add_to_dictionary=False, on_needles=False):
    if type(bounding_box) is list:
        c_start_list, c_stop_list, r_start_list, r_stop_list, z_start_list, z_stop_list = [], [], [], [], [], []
        for bbox in bounding_box:
            c_start, r_start, z_start, c_stop, r_stop, z_stop = bbox
            z_stop, r_stop, c_stop = z_start + z_stop, r_start + r_stop, c_start + c_stop
            c_start_list.append(c_start)
            c_stop_list.append(c_stop)
            r_start_list.append(r_start)
            r_stop_list.append(r_stop)
            z_start_list.append(z_start)
            z_stop_list.append(z_stop)
        c_start, c_stop, r_start, r_stop, z_start, z_stop = min(c_start_list), max(c_stop_list), min(r_start_list), \
                                                            max(r_stop_list), min(z_start_list), max(z_stop_list)
        if on_needles:
            z_start = max(z_start_list)  # Only take the minimum digitization!

    else:
        c_start, r_start, z_start, c_stop, r_stop, z_stop = bounding_box
        z_stop, r_stop, c_stop = z_start + z_stop, r_start + r_stop, c_start + c_stop
    if add_to_dictionary:
        input_features['bounding_boxes_z_start_{}'.format(val)] = z_start
        input_features['bounding_boxes_r_start_{}'.format(val)] = r_start
        input_features['bounding_boxes_c_start_{}'.format(val)] = c_start
        input_features['bounding_boxes_z_stop_{}'.format(val)] = z_stop
        input_features['bounding_boxes_r_stop_{}'.format(val)] = r_stop
        input_features['bounding_boxes_c_stop_{}'.format(val)] = c_stop
    if return_indexes:
        return z_start, z_stop, r_start, r_stop, c_start, c_stop
    return input_features


if __name__ == '__main__':
    pass
