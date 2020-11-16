import SimpleITK as sitk
import numpy as np
import copy
from skimage import morphology


def get_bounding_box_indexes(annotation):
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
    '''
    Get the row values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    min_r_s, max_r_s = indexes[0], indexes[-1]
    '''
    Get the col values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    min_c_s, max_c_s = indexes[0], indexes[-1]
    return min_z_s, max_z_s, min_r_s, max_r_s, min_c_s, max_c_s


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


def remove_56_78(annotations):
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


class Iterate_Lobe_Annotations(object):
    def __init__(self, do_2D_pruning=True):
        MauererDistanceMap = sitk.SignedMaurerDistanceMapImageFilter()
        MauererDistanceMap.SetInsideIsPositive(True)
        MauererDistanceMap.UseImageSpacingOn()
        MauererDistanceMap.SquaredDistanceOff()
        self.MauererDistanceMap = MauererDistanceMap
        self.Remove_Smallest_Structure = Remove_Smallest_Structures()
        self.do_2D_pruning = do_2D_pruning

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

    def iterate_annotations(self, annotations_out, ground_truth_out, spacing, allowed_differences=50, max_iteration=15, z_mult=1):
        '''
        :param annotations:
        :param ground_truth:
        :param spacing:
        :param allowed_differences:
        :param max_iteration:
        :param z_mult: factor by which to ensure slices don't bleed into ones above and below
        :return:
        '''
        self.Remove_Smallest_Structure.spacing = spacing
        annotations_out[ground_truth_out == 0] = 0
        min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(ground_truth_out)
        annotations = annotations_out[min_z:max_z,min_r:max_r,min_c:max_c,...]
        ground_truth = ground_truth_out[min_z:max_z,min_r:max_r,min_c:max_c,...]
        spacing[-1] *= z_mult
        differences = [np.inf]
        index = 0
        while differences[-1] > allowed_differences and index < max_iteration:
            index += 1
            print('Iterating {}'.format(index))
            previous_iteration = copy.deepcopy(np.argmax(annotations,axis=-1))
            annotations = self.remove_56_78(annotations)
            for i in range(1, annotations.shape[-1]):
                annotation_handle = sitk.GetImageFromArray(annotations[...,i])
                annotation_handle.SetSpacing(spacing)
                pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(annotation_handle)
                annotations[..., i] = sitk.GetArrayFromImage(pruned_handle)
                if self.do_2D_pruning:
                    slices = np.where(annotations[...,i] == 1)
                    if slices:
                        slices = np.unique(slices[0])
                        for ii in range(len(slices)):
                            image_handle = sitk.GetImageFromArray(annotations[slices[ii],...,i][None,...])
                            pruned_handle = self.Remove_Smallest_Structure.remove_smallest_component(image_handle)
                            annotations[slices[ii], ..., i] = sitk.GetArrayFromImage(pruned_handle)

            annotations = self.make_distance_map(annotations, ground_truth,spacing=spacing)
            differences.append(np.abs(np.sum(previous_iteration[ground_truth==1]-np.argmax(annotations,axis=-1)[ground_truth==1])))
        annotations_out[min_z:max_z,min_r:max_r,min_c:max_c,...] = annotations
        annotations_out[ground_truth_out == 0] = 0
        return annotations_out

    def run_distance_map(self, array, spacing):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        output = self.MauererDistanceMap.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

    def make_distance_map(self, pred, liver, reduce=True, spacing=(0.975,0.975,2.5)):
        '''
        :param pred: A mask of your predictions with N channels on the end, N=0 is background [# Images, 512, 512, N]
        :param liver: A mask of the desired region [# Images, 512, 512]
        :param reduce: Save time and only work on masked region
        :return:
        '''
        liver = np.squeeze(liver)
        pred = np.squeeze(pred)
        pred = np.round(pred).astype('int')
        min_z, min_r, max_r, min_c, max_c = 0, 0, 512, 0, 512
        max_z = pred.shape[0]
        if reduce:
            min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(liver)
        reduced_pred = pred[min_z:max_z,min_r:max_r,min_c:max_c]
        reduced_liver = liver[min_z:max_z,min_r:max_r,min_c:max_c]
        reduced_output = np.zeros(reduced_pred.shape)
        for i in range(1,pred.shape[-1]):
            temp_reduce = reduced_pred[...,i]
            output = self.run_distance_map(temp_reduce, spacing)
            reduced_output[...,i] = output
        reduced_output[reduced_output>0] = 0
        reduced_output = np.abs(reduced_output)
        reduced_output[...,0] = np.inf
        output = np.zeros(reduced_output.shape,dtype='int')
        mask = reduced_liver == 1
        values = reduced_output[mask]
        output[mask,np.argmin(values,axis=-1)] = 1
        pred[min_z:max_z,min_r:max_r,min_c:max_c] = output
        return pred

    def post_process(self, images, pred, ground_truth, spacing):
        pred = self.iterate_annotations(pred, ground_truth, spacing=list(spacing), z_mult=1, max_iteration=10)
        return images, pred, ground_truth


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


class Fill_Missing_Segments(object):
    def __init__(self):
        MauererDistanceMap = sitk.SignedMaurerDistanceMapImageFilter()
        MauererDistanceMap.SetInsideIsPositive(True)
        MauererDistanceMap.UseImageSpacingOn()
        MauererDistanceMap.SquaredDistanceOff()
        self.MauererDistanceMap = MauererDistanceMap

    def iterate_annotations(self, annotations, ground_truth, spacing, allowed_differences=50, max_iteration=15, z_mult=1):
        '''
        :param annotations:
        :param ground_truth:
        :param spacing:
        :param allowed_differences:
        :param max_iteration:
        :param z_mult: factor by which to ensure slices don't bleed into ones above and below
        :return:
        '''
        annotations[ground_truth == 0] = 0
        re_organized_spacing = spacing[-1::-1]
        spacing[-1] *= z_mult
        differences = [np.inf]
        index = 0
        while differences[-1] > allowed_differences and index < max_iteration:
            index += 1
            print('Iterating {}'.format(index))
            previous_iteration = copy.deepcopy(np.argmax(annotations,axis=-1))
            annotations = remove_56_78(annotations)
            for i in range(1, annotations.shape[-1]):
                annotations[..., i] = remove_non_liver(annotations[..., i], do_3D=True, do_2D=True,min_area=.5,spacing=re_organized_spacing)
            annotations = self.make_distance_map(annotations, ground_truth,spacing=spacing)
            differences.append(np.abs(np.sum(previous_iteration[ground_truth==1]-np.argmax(annotations,axis=-1)[ground_truth==1])))
        return annotations

    def run_distance_map(self, array, spacing):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        output = self.MauererDistanceMap.Execute(image)
        output = sitk.GetArrayFromImage(output)
        return output

    def make_distance_map(self, pred, liver, reduce=True, spacing=(0.975,0.975,2.5)):
        '''
        :param pred: A mask of your predictions with N channels on the end, N=0 is background [# Images, 512, 512, N]
        :param liver: A mask of the desired region [# Images, 512, 512]
        :param reduce: Save time and only work on masked region
        :return:
        '''
        liver = np.squeeze(liver)
        pred = np.squeeze(pred)
        pred = np.round(pred).astype('int')
        min_z, min_r, max_r, min_c, max_c = 0, 0, 512, 0, 512
        max_z = pred.shape[0]
        if reduce:
            min_z, max_z, min_r, max_r, min_c, max_c = get_bounding_box_indexes(liver)
        reduced_pred = pred[min_z:max_z,min_r:max_r,min_c:max_c]
        reduced_liver = liver[min_z:max_z,min_r:max_r,min_c:max_c]
        reduced_output = np.zeros(reduced_pred.shape)
        for i in range(1,pred.shape[-1]):
            temp_reduce = reduced_pred[...,i]
            output = self.run_distance_map(temp_reduce, spacing)
            reduced_output[...,i] = output
        reduced_output[reduced_output>0] = 0
        reduced_output = np.abs(reduced_output)
        reduced_output[...,0] = np.inf
        output = np.zeros(reduced_output.shape,dtype='int')
        mask = reduced_liver == 1
        values = reduced_output[mask]
        output[mask,np.argmin(values,axis=-1)] = 1
        pred[min_z:max_z,min_r:max_r,min_c:max_c] = output
        return pred


if __name__ == '__main__':
    xxx = 1
    # Fill_Segments = Fill_Missing_Segments()
    # liver = np.load(os.path.join('.','liver.npy'))
    # pred = np.load(os.path.join('.','pred.npy'))
    # output = Fill_Segments.make_distance_map(pred,liver)