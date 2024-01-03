__author__ = 'Brian M Anderson'
# Created on 3/5/2021

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from scipy.ndimage import interpolation, filters
from tensorflow.python.keras.utils.np_utils import to_categorical
import cv2, math, copy, os, sys
from skimage.measure import block_reduce
from Fill_In_Segments_sitk import Fill_Missing_Segments
from NiftiResampler.ResampleTools import Resample_Class_Object, sitk
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image, plt
'''
Description of code
Annotations_To_Categorical(num_classes): classes for annotations to be turned into
Repeat_Channel(axis=-1, repeats=3, on_images=True, on_annotations=False)
Fuzzy_Segment_Liver_Lobes(variation, spacing): allow from some 'fuzzy' variation in annotations
Expand_Dimensions(axis): expand a dimension, normally they're pulled as [z_images, rows, columns]
Ensure_Image_Proportions(image_size-row, image_size_col): ensures images are proper proportions
Normalize_Images(mean_val, std_val): Normalize images
Threshold_Images(lower_bound, upper_bound): threshold, normally after normalizing, recommended -3.55 to +3.55
Add_Noise_To_Images(by_patient, variation): add noise to patient images
Random_Horizontal_Vertical_Flips(by_patient, h_flip, v_flip): randomly flip 2D images vertically/horizontally
Random_Scale_Processor(by_patient, variation): randomly scale images by certain amounts
Rotate_Images_2D_Processor(by_patient, image_size, variation): Randomly rotate images in 2D plane
Shift_Images_Processor(by_patient, variation, positive_to_negative): Shift images by a certain variation in 2D plane
Random_2D_Deformation_Processor(by_patient, image_size, variation): Randomly deform images in 2D plane
'''


class Image_Processor(object):
    def sitk_processes(self, image_handle, annotation_handle):
        '''
        This is for getting information on the sitk image for later
        :param image:
        :param annotation:
        :return:
        '''
        return image_handle, annotation_handle

    def preload_single_image_process(self, image, annotation):
        '''
        This is for image processes done loading on a slice by slice basis that only need to be done once, like
        normalizing a CT slice with a mean and std
        :param image: Some image of shape [n_row, m_col]
        :param annotation: Some image of shape [n_row, m_col]
        :return:
        '''
        return image, annotation

    def pre_load_whole_image_process(self, images, annotations):
        '''
        This is for image processes which will occur once on the entire 3D stack, only use if you're pulling a whole set
        :param images: Some image of shape [n_row, m_col]
        :param annotations: Some image of shape [n_row, m_col]
        :return:
        '''
        return images, annotations

    def post_load_process(self, images, annotations):
        '''
        This is for image processes which will vary between each load, for example, adding noise or perturbations
        :param images: Some image of shape [z_images, n_row, m_col]
        :param annotations: Some image of shape [z_images, n_row, m_col]
        :return:
        '''
        return images, annotations

    def post_load_all_patient_process(self, images, annotations, path_key=None, file_key=None):
        '''
        This is for image processes which go across the entire patient stack,
        :param images: [#patients, z_images, n_row, m_col, channel]
        :param annotations: [#patients, z_images, n_row, m_col, # classes]
        :return:
        '''
        return images, annotations

    def GetSpacing(self, path_key=None, file_key=None):
        if path_key in self.start_stop_dict:
            if file_key in self.start_stop_dict[path_key]:
                return self.start_stop_dict[path_key][file_key]['spacing']
        return None

    def set_start_stop_dict(self, start_stop_dict=None):
        if start_stop_dict is None:
            start_stop_dict = {}
        self.start_stop_dict = start_stop_dict
        return None


class Bring_Parotids_Together(Image_Processor):
    def preload_single_image_process(self, image, annotation):
        '''
        This is for image processes done loading on a slice by slice basis that only need to be done once, like
        normalizing a CT slice with a mean and std
        :param image: Some image of shape [n_row, m_col]
        :param annotation: Some image of shape [n_row, m_col]
        :return:
        '''
        annotation[annotation>1] = 1
        return image, annotation


class Pull_Cube_sitk(Image_Processor):
    def __init__(self, annotation_index=None, max_cubes=10, z_images=16, rows=100, cols=100, min_volume=0, min_voxels=0,
                 max_volume=np.inf, max_voxels=np.inf):
        self.annotation_index = annotation_index
        self.max_cubes = max_cubes
        self.z_images, self.rows, self.cols = z_images, rows, cols
        self.min_volume = min_volume * 1000
        self.min_voxels = min_voxels
        self.max_volume, self.max_voxels = max_volume * 1000, max_voxels
        if min_volume != 0 and min_voxels != 0:
            raise AssertionError('Cannot have both min_volume and min_voxels specified')
        if max_volume != np.inf and min_voxels != np.inf:
            raise AssertionError('Cannot have both max_volume and max_voxels specified')

    def post_load_all_patient_process(self, images, annotations, path_key=None, file_key=None):
        if self.annotation_index is not None:
            spacing = self.GetSpacing(path_key=path_key,file_key=file_key)
            Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
            stats = sitk.LabelShapeStatisticsImageFilter()
            images_shape = images.shape
            images = np.squeeze(images)
            annotations = np.squeeze(annotations)
            images_size, annotations_size = images.shape, annotations.shape
            seed_annotations = np.squeeze(annotations[...,self.annotation_index])
            thresholded_image = sitk.GetImageFromArray(seed_annotations.astype('int'))
            connected_image = Connected_Component_Filter.Execute(thresholded_image)
            RelabelComponentFilter = sitk.RelabelComponentImageFilter()
            if self.min_volume > 0 and spacing is not None:
                RelabelComponentFilter.SetMinimumObjectSize(int(self.min_volume/np.prod(spacing)))
                connected_image = RelabelComponentFilter.Execute(connected_image)
            elif self.min_voxels > 0:
                RelabelComponentFilter.SetMinimumObjectSize(self.min_voxels)
                connected_image = RelabelComponentFilter.Execute(connected_image)
            stats.Execute(connected_image)
            bounding_boxes = np.asarray([stats.GetBoundingBox(l) for l in stats.GetLabels()])
            if self.max_volume != np.inf and spacing is not None:
                volumes = np.asarray([stats.GetNumberOfPixels(l)*np.prod(spacing) for l in stats.GetLabels()])
                bounding_boxes = bounding_boxes[volumes<self.max_volume]
            elif self.max_voxels != np.inf:
                volumes = np.asarray([stats.GetNumberOfPixels(l) for l in stats.GetLabels()])
                bounding_boxes = bounding_boxes[volumes<self.max_voxels]
            perm = np.arange(len(bounding_boxes))
            np.random.shuffle(perm)
            bounding_boxes = bounding_boxes[perm]
            num_cubes = min([len(bounding_boxes),self.max_cubes])
            out_images = np.ones([num_cubes, self.z_images, self.rows, self.cols]) * np.min(images)
            out_annotations = np.zeros([num_cubes, self.z_images, self.rows, self.cols, annotations_size[-1]])
            max_bounds = seed_annotations.shape
            for index in range(num_cubes):
                bounding_box = bounding_boxes[index]
                c_start, r_start, z_start, _, _, _ = bounding_box
                c_stop, r_stop, z_stop = c_start + bounding_box[3], r_start + bounding_box[4], z_start + bounding_box[5]
                z_start -= (self.z_images // 2 - np.random.randint((z_stop-z_start)))
                c_start -= (self.cols // 2 - np.random.randint((c_stop - c_start)))
                r_start -= (self.rows // 2 - np.random.randint((r_stop - r_start)))
                z_start, r_start, c_start = [max([0,i]) for i in [z_start, r_start, c_start]]
                z_stop, r_stop, c_stop = z_start + self.z_images, r_start + self.rows, c_start + self.cols
                z_stop, r_stop, c_stop = [min([max_bounds[index], i]) for index, i in enumerate([z_stop, r_stop, c_stop])]
                image_cube = images[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                annotation_cube = annotations[z_start:z_stop, r_start:r_stop, c_start:c_stop, ...]
                img_shape = image_cube.shape
                out_images[index,:img_shape[0], :img_shape[1], :img_shape[2], ...] = image_cube
                out_annotations[index,:img_shape[0], :img_shape[1], :img_shape[2], ...] = annotation_cube
            out_images = out_images[...,None]
            if out_images.shape[-1] < images_shape[-1]:
                out_images = np.repeat(out_images,axis=-1,repeats=images_shape[-1])
            return out_images, out_annotations
        else:
            return images, annotations


class Mask_Pred_Within_Annotation(Image_Processor):
    def __init__(self, return_mask=False, liver_box=False, mask_image=False, remove_liver_layer_indexes=None,
                 threshold_value=0):
        self.return_mask = return_mask
        self.liver_box = liver_box
        self.mask_image = mask_image
        self.remove_liver_layer_indexes = remove_liver_layer_indexes
        self.threshold_value = threshold_value

    def post_load_all_patient_process(self, images, annotations, path_key=None, file_key=None):
        if self.mask_image:
            images[annotations[...,0] == 1] = self.threshold_value
        if self.return_mask:
            mask = np.sum(annotations[...,1:],axis=-1)[...,None]
            if self.remove_liver_layer_indexes is not None:  # In future predictions we do not want to predict liver, so toss it out
                annotations = annotations[..., self.remove_liver_layer_indexes]
                annotations[...,0] = 1-np.sum(annotations[...,1:],axis=-1)
            mask = np.repeat(mask,annotations.shape[-1],axis=-1)
            sum_vals = np.zeros(mask.shape)
            sum_vals[...,0] = 1 - mask[...,0]
            return [images,mask, sum_vals], annotations
        if self.remove_liver_layer_indexes is not None:
            annotations = annotations[..., self.remove_liver_layer_indexes]
        return images, annotations


class Pull_Cube_From_Image(Image_Processor):
    def __init__(self, desired_size, samples=1, random_z=True, annotation_index=-1):
        self.desired_size = desired_size
        self.samples = samples
        self.random_z = random_z
        self.annotation_index = annotation_index
        self.batch_range = np.arange(desired_size[0])

    def post_load_process(self, images, annotations):
        desired_size = self.desired_size
        samples = self.samples
        output_images = np.ones((samples,) + desired_size + (images.shape[-1],)) * np.min(images)
        output_annotations = np.zeros((samples,) + desired_size + (annotations.shape[-1],))
        z_locations, r_locations, c_locations = np.where(annotations[..., self.annotation_index] == 1)
        for i in range(samples):
            index = np.random.randint(len(z_locations))
            if self.random_z:
                np.random.shuffle(self.batch_range)
            z_start = max([0, z_locations[index] - self.batch_range[0]])
            z_stop = min([z_start + desired_size[0], images.shape[0]])
            r_start = max([0, r_locations[index] - desired_size[1] // 2])
            r_stop = min([r_start + desired_size[1], images.shape[1]])
            c_start = max([0, c_locations[index] - desired_size[2] // 2])
            c_stop = min([c_start + desired_size[2], images.shape[2]])
            image_cube = images[z_start:z_stop, r_start:r_stop, c_start:c_stop, ...]
            annotation_cube = annotations[z_start:z_stop, r_start:r_stop, c_start:c_stop, ...]
            output_images[i, :image_cube.shape[0], :image_cube.shape[1], :image_cube.shape[2], ...] = image_cube
            output_annotations[i, :image_cube.shape[0], :image_cube.shape[1], :image_cube.shape[2], ...] = annotation_cube
        return output_images, output_annotations


class Clip_Images(Image_Processor):
    def __init__(self, annotations_index=None, bounding_box_expansion=(10,10,10), power_val_z=1, power_val_x=1,
                 power_val_y=1, min_images=None, min_rows=None, min_cols=None):
        self.annotations_index = annotations_index
        self.bounding_box_expansion = bounding_box_expansion
        self.power_val_z, self.power_val_x, self.power_val_y = power_val_z, power_val_x, power_val_y
        self.min_images, self.min_rows, self.min_cols = min_images, min_rows, min_cols

    def post_load_process(self, images, annotations):
        if self.annotations_index:
            if len(self.annotations_index) > 1:
                liver = np.sum(annotations[...,self.annotations_index],axis=-1)
            else:
                liver = annotations[...,self.annotations_index]
            z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(liver)
            z_start = max([0,z_start-self.bounding_box_expansion[0]])
            z_stop = min([z_stop+self.bounding_box_expansion[0],images.shape[0]])
            r_start = max([0,r_start-self.bounding_box_expansion[1]])
            r_stop = min([images.shape[1],r_stop+self.bounding_box_expansion[1]])
            c_start = max([0,c_start-self.bounding_box_expansion[2]])
            c_stop = min([images.shape[2],c_stop+self.bounding_box_expansion[2]])
        else:
            z_stop, r_stop, c_stop, _ = images.shape
            z_start, r_start, c_start = 0, 0, 0
        z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
        remainder_z, remainder_r, remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                self.power_val_x - r_total % self.power_val_x if r_total % self.power_val_x != 0 else 0, \
                                                self.power_val_y - c_total % self.power_val_y if c_total % self.power_val_y != 0 else 0
        min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
        if self.min_images is not None:
            min_images = max([min_images,self.min_images])
        if self.min_rows is not None:
            min_rows = max([min_rows,self.min_rows])
        if self.min_cols is not None:
            min_cols = max([min_cols,self.min_cols])
        out_images = np.ones([min_images,min_rows,min_cols,images.shape[-1]])*np.min(images)
        out_annotations = np.zeros([min_images, min_rows, min_cols, annotations.shape[-1]])
        out_annotations[...,0] = 1
        image_cube = images[z_start:z_start + min_images,r_start:r_start + min_rows,c_start:c_start + min_cols,...]
        annotation_cube = annotations[z_start:z_start + min_images,r_start:r_start + min_rows,c_start:c_start + min_cols,...]
        img_shape = image_cube.shape
        out_images[:img_shape[0],:img_shape[1],:img_shape[2],...] = image_cube
        out_annotations[:img_shape[0],:img_shape[1],:img_shape[2],...] = annotation_cube
        return out_images, out_annotations


class Resample_Images(Image_Processor):
    def __init__(self, output_spacing=(None,None,2.5)):
        '''
        This is a little tricky... We only want to perform this task once, since it requires potentially large
        computation time, but it also requires that all individual image slices already be loaded
        '''
        self.output_spacing = output_spacing
        self.resampler = Resample_Class_Object()

    def sitk_processes(self, image_handle, annotation_handle):
        self.input_spacing = image_handle.GetSpacing()
        return image_handle, annotation_handle

    def pre_load_whole_image_process(self, images, annotations):
        output_spacing = []
        for index in range(3):
            if self.output_spacing[index] is None:
                output_spacing.append(self.input_spacing[index])
            else:
                output_spacing.append(self.output_spacing[index])
        output_spacing = tuple(output_spacing)
        if output_spacing != self.input_spacing:
            image_handle = sitk.GetImageFromArray(images)
            image_handle.SetSpacing(self.input_spacing)
            annotation_handle = sitk.GetImageFromArray(annotations)
            print('Resampling {} to {}'.format(self.input_spacing, output_spacing))
            image_handle = self.resampler.resample_image(input_image=image_handle, input_spacing=self.input_spacing,
                                                         output_spacing=output_spacing, is_annotation=False)
            annotation_handle = self.resampler.resample_image(input_image=annotation_handle, input_spacing=self.input_spacing,
                                                         output_spacing=output_spacing, is_annotation=True)
            images, annotations = sitk.GetArrayFromImage(image_handle), sitk.GetArrayFromImage(annotation_handle)
        return images, annotations


class Fuzzy_Segment_Liver_Lobes(Image_Processor):
    def __init__(self, variation=None, spacing=(1,1,5), run_as_preload=False):
        '''
        :param variation: margin to expand region, mm. np.arange(start=0, stop=1, step=1)
        :param spacing: Spacing of images, assumes this is constance
        '''
        self.variation = variation
        self.spacing = spacing
        self.run_as_preload = run_as_preload
        self.Fill_Missing_Segments_Class = Fill_Missing_Segments()
        self.patient_dictionary = {}

    def make_fuzzy_label(self, annotation, variation, patient_id):
        out_shape = annotation.shape
        annotation = np.squeeze(annotation)
        if patient_id not in self.patient_dictionary:
            distance_map = np.zeros(annotation.shape)
            for i in range(1, annotation.shape[-1]):
                temp_annotation = annotation[..., i].astype('int')
                distance_map[..., i] = self.Fill_Missing_Segments_Class.run_distance_map(temp_annotation,
                                                                                         spacing=self.spacing)
            distance_map[distance_map > 0] = 0
            distance_map = np.abs(distance_map)
            if not self.run_as_preload:
                self.patient_dictionary[patient_id] = copy.deepcopy(distance_map)
        if not self.run_as_preload:
            distance_map = copy.deepcopy(self.patient_dictionary[patient_id])
        distance_map[distance_map > variation] = variation  # Anything greater than 10 mm away set to 0
        distance_map = 1 - distance_map / variation
        distance_map[annotation[..., 0] == 1] = 0
        distance_map[..., 0] = annotation[..., 0]
        total = np.sum(distance_map, axis=-1)
        distance_map /= total[..., None]
        return np.reshape(distance_map, out_shape)

    def post_load_all_patient_process(self, images, annotations, path_key=None, file_key=None):
        '''
        :param images: Images set to values of 0 to max - min. This is done
        :param annotations:
        :return:
        '''
        if self.variation is not None and not self.run_as_preload:
            variation = self.variation[np.random.randint(len(self.variation))]
            patient_id = os.path.join(path_key,file_key)
            annotations = self.make_fuzzy_label(annotations, variation, patient_id)
        return images, annotations

    def pre_load_whole_image_process(self, images, annotations):
        '''
        :param images: Images set to values of 0 to max - min. This is done
        :param annotations:
        :return:
        '''
        if self.variation is not None and self.run_as_preload:
            variation = self.variation[np.random.randint(len(self.variation))]
            annotations = self.make_fuzzy_label(annotations, variation)
        return images, annotations


class Expand_Dimensions(Image_Processor):
    def __init__(self, axis=-1, on_patient=False, on_images=True, on_annotations=False):
        '''
        :param axis: axis to expand
        :param on_patient: expand on a 3D stack of images
        :param on_images: expand the axis on the images
        :param on_annotations: expand the axis on the annotations
        '''
        self.axis = axis
        self.on_patient = on_patient
        self.on_images = on_images
        self.on_annotations = on_annotations

    def preload_single_image_process(self, image, annotation):
        if self.on_images:
            image = np.expand_dims(image,axis=self.axis)
        if self.on_annotations:
            annotation = np.expand_dims(annotation,axis=self.axis)
        return image, annotation

    def post_load_process(self, images, annotations):
        if self.on_patient:
            if self.on_images:
                images = np.expand_dims(images,axis=self.axis)
            if self.on_annotations:
                annotations = np.expand_dims(annotations,axis=self.axis)
        return images, annotations


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

    def post_load_process(self, images, annotations):
        if self.on_images:
            images = np.repeat(images, self.repeats, axis=self.axis)
        if self.on_annotations:
            annotations = np.repeat(annotations, self.repeats, axis=self.axis)
        return images, annotations


class Annotations_To_Categorical(Image_Processor):
    def __init__(self, num_of_classes=2):
        '''
        :param num_of_classes: number of classes
        '''
        self.num_of_classes = num_of_classes
        if self.num_of_classes > 5:
            print('Warning, to_categorical drastically increases annotation size, can result in large RAM usage')

    def preload_single_image_process(self, images, annotations):
        '''
        :param images: Images set to values of 0 to max - min. This is done
        :param annotations:
        :return:
        '''
        annotations = to_categorical(annotations, self.num_of_classes).astype(annotations.dtype) # Sets it to float32, which can take up alot of memory, adding on post-load process
        return images, annotations

    def post_load_process(self, images, annotations):
        annotations = annotations.astype('float32')
        return images, annotations


class Ensure_Image_Proportions(Image_Processor):
    def __init__(self, image_size_row=512, image_size_col=512):
        self.image_size_row, self.image_size_col = image_size_row, image_size_col

    def convert_image_size(self, images, annotations):
        dif_1 = (self.image_size_row - images.shape[1])
        dif_2 = (self.image_size_col - images.shape[2])
        if dif_1 > 0 and dif_2 > 0:
            out_image = np.ones([1,self.image_size_row, self.image_size_col],dtype=images.dtype) * np.min(images)
            out_annotations = np.zeros([1, self.image_size_row, self.image_size_col],dtype=annotations.dtype)
            out_image[:, dif_1//2:dif_1//2 + images.shape[1], dif_2//2:dif_2//2 + images.shape[2],...] = images
            out_annotations[:, dif_1//2:dif_1//2 + images.shape[1], dif_2//2:dif_2//2 + images.shape[2],...] = annotations
            return out_image, out_annotations
        if dif_1 != 0:
            if dif_1 > 0:
                images = np.concatenate((images, images[:, :dif_1//2, ...]),axis=1)
                images = np.concatenate((images[:, -dif_1//2:, ...], images),axis=1)
                annotations = np.concatenate((annotations, annotations[:, :dif_1//2, ...]),axis=1)
                annotations = np.concatenate((annotations[:, -dif_1//2:, ...], annotations),axis=1)
            elif dif_1 < 0:
                images = images[:, abs(dif_1)//2:-abs(dif_1//2), ...]
                annotations = annotations[:, abs(dif_1)//2:-abs(dif_1//2), ...]
        if dif_2 != 0:
            if dif_2 > 0:
                images = np.concatenate((images, images[:, :, :dif_2//2, ...]),axis=2)
                images = np.concatenate((images[:, :, -dif_2//2:, ...], images),axis=2)
                annotations = np.concatenate((annotations, annotations[:, :, :dif_2//2, ...]),axis=2)
                annotations = np.concatenate((annotations[:, :, -dif_2//2:, ...], annotations),axis=2)
            elif dif_2 < 0:
                images = images[:, :, abs(dif_2)//2:-abs(dif_2//2), ...]
                annotations = annotations[:, :, abs(dif_2)//2:-abs(dif_2//2), ...]
        return images, annotations

    def preload_single_image_process(self, image, annotation):
        if image.shape[0] != 1:
            image = image[None, ...]
            annotation = annotation[None, ...]
        if image.shape[1] != self.image_size_row or image.shape[2] != self.image_size_col:
            image_dtype, annotation_dtype = image.dtype, annotation.dtype
            block = (image.shape[1]//self.image_size_row,image.shape[2]//self.image_size_col)
            block = np.max([block,(1,1)],axis=0)
            if np.max(block) > 1:
                block = tuple(block)
                image = block_reduce(image[0, ...], block, np.average).astype('float32')[None, ...]
                annotation = block_reduce(annotation[0, ...].astype('int'), block, np.max).astype('int')[None, ...]
            image, annotation = self.convert_image_size(image, annotation)
            image, annotation = image.astype(image_dtype), annotation.astype(annotation_dtype)
        return image, annotation


class Normalize_Images(Image_Processor):
    def __init__(self, mean_val=0, std_val=1):
        '''
        :param mean_val: Mean value to normalize to
        :param std_val: Standard deviation value to normalize to
        '''
        self.mean_val, self.std_val = mean_val, std_val

    def preload_single_image_process(self, images, annotations):
        images = (images - self.mean_val)/self.std_val
        return images, annotations


class Normalize_to_Liver(Image_Processor):
    def __init__(self):
        '''
        This is a little tricky... We only want to perform this task once, since it requires potentially large
        computation time, but it also requires that all individual image slices already be loaded
        '''
        # Now performing FWHM
        xxx = 1


    def pre_load_whole_image_process(self, images, annotations):
        liver = np.sum(annotations[..., 1:], axis=-1)
        data = images[liver == 1].flatten()
        counts, bins = np.histogram(data, bins=100)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        peak = bins[count_index]
        data_reduced = data[np.where((data>peak-150) & (data<peak+150))]
        counts, bins = np.histogram(data_reduced, bins=1000)
        bins = bins[:-1]
        count_index = np.where(counts == np.max(counts))[0][-1]
        half_counts = counts - np.max(counts) // 2
        half_upper = np.abs(half_counts[count_index+1:])
        max_50 = np.where(half_upper == np.min(half_upper))[0][0]

        half_lower = np.abs(half_counts[:count_index-1][-1::-1])
        min_50 = np.where(half_lower == np.min(half_lower))[0][0]

        min_values = bins[count_index - min_50]
        max_values = bins[count_index + max_50]
        data = data[np.where((data >= min_values) & (data <= max_values))]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val)/std_val
        return images, annotations


class Normalize_MR(Image_Processor):
    def __init__(self):
        '''
        This is a little tricky... We only want to perform this task once, since it requires potentially large
        computation time, but it also requires that all individual image slices already be loaded
        '''
        # Now performing FWHM
        xxx = 1


    def pre_load_whole_image_process(self, images, annotations):
        data = images[...]
        counts, bins = np.histogram(data, bins=100)
        count_index = 0
        count_value = 0
        while count_value/np.sum(counts) < .3: # Throw out the bottom 30 percent of data, as that is usually just 0s
            count_value += counts[count_index]
            count_index += 1
        min_bin = bins[count_index]
        data = data[data>min_bin]
        mean_val, std_val = np.mean(data), np.std(data)
        images = (images - mean_val)/std_val
        return images, annotations


class Threshold_Images(Image_Processor):
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf, inverse_image=False, post_load=True,
                 final_scale_value=None):
        '''
        :param lower_bound: Lower bound to threshold images, normally -3.55 if Normalize_Images is used previously
        :param upper_bound: Upper bound to threshold images, normally 3.55 if Normalize_Images is used previously
        :param inverse_image: Should the image be inversed after threshold?
        :param post_load: should this be done each iteration? If False, gets slotted under pre_load_process
        :param final_scale_value: Value to scale the entire image to (255 scales to 0-255), (1 scales to 0-1)
        '''
        self.lower = lower_bound
        self.upper = upper_bound
        self.inverse_image = inverse_image
        self.post_load = post_load
        self.final_scale_value = final_scale_value

    def post_load_process(self, images, annotations):
        if self.post_load:
            images[images<self.lower] = self.lower
            images[images>self.upper] = self.upper
            if self.final_scale_value is not None:
                images = (images - self.lower)/(self.upper - self.lower) * self.final_scale_value
            if self.inverse_image:
                if self.upper != np.inf and self.lower != -np.inf:
                    images = (self.upper + self.lower) - images
                else:
                    images = -1*images
        return images, annotations

    def preload_single_image_process(self, image, annotation):
        if not self.post_load:
            image[image<self.lower] = self.lower
            image[image>self.upper] = self.upper
            if self.final_scale_value is not None:
                image = (image - self.lower)/(self.upper - self.lower) * self.final_scale_value
            if self.inverse_image:
                if self.upper != np.inf and self.lower != -np.inf:
                    image = (self.upper + self.lower) - image
                else:
                    image = -1*image
        return image, annotation


class Add_Noise_To_Images(Image_Processor):
    def __init__(self, by_patient=False, variation=None):
        '''
        :param by_patient:
        :param variation: range of values np.round(np.arange(start=0, stop=1.0, step=0.1),2)
        '''
        self.by_patient = by_patient
        self.variation = variation

    def post_load_process(self, images, annotations):
        if self.variation is not None:
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                noisy_image = variation * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
                images += noisy_image
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    noisy_image = variation * np.random.normal(loc=0.0, scale=1.0, size=images[i].shape)
                    images[i] += noisy_image
        return images, annotations


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


class Random_Horizontal_Vertical_Flips(Image_Processor):
    def __init__(self, by_patient=False, h_flip=False, v_flip=False):
        '''
        :param by_patient: Process all images as single patient (flip together)
        :param h_flip: Perform horizontal flips?
        :param v_flip: Perform vertical flips?
        '''
        self.by_patient, self.h_flip, self.v_flip = by_patient, h_flip, v_flip

    def post_load_process(self, images, annotations):
        if self.h_flip or self.v_flip:
            if self.by_patient:
                if self.h_flip and np.random.randint(2) == 1:
                    images = images[..., :, ::-1]
                    annotations = annotations[..., :, ::-1]
                if self.v_flip and np.random.randint(2) == 1:
                    images = images[..., ::-1, :]
                    annotations = annotations[..., ::-1, :]
            else:
                for i in range(images.shape[0]):
                    if self.h_flip and np.random.randint(2) == 1:
                        images[i] = images[i, :, ::-1]
                        annotations[i] = annotations[i, :, ::-1]
                    if self.v_flip and np.random.randint(2) == 1:
                        images[i] = images[i, ::-1, :]
                        annotations[i] = annotations[i, ::-1, :]
        return images, annotations


class Random_Scale_Processor(Image_Processor):
    def __init__(self, by_patient=False, variation=None):
        '''
        :param by_patient: perform randomly on each image in stack, or on the entire collective
        :param variation: range of values by which to scale the image, np.round(np.arange(start=0, stop=2.6, step=0.5),2)
        '''
        self.by_patient = by_patient
        self.variation = variation

    def scale_image(self, im, variation=0, interpolator='linear'):
        if interpolator is 'linear':
            temp_scale = cv2.resize(im.astype('int16'), None, fx=1 + variation, fy=1 + variation,
                                    interpolation=cv2.INTER_LINEAR)
        elif interpolator is 'nearest':
            # QUICK FIX
            # changing interpolation of mask to linear because nearest remove the last annotation
            # due to some kind of border effect...
            # linear should still work as we return a 0-1 output as int16
            # temp_scale = cv2.resize(im.astype('int16'), None, fx=1 + variation, fy=1 + variation,
            #                         interpolation=cv2.INTER_NEAREST)
            temp_scale = cv2.resize(im.astype('int16'), None, fx=1 + variation, fy=1 + variation,
                                    interpolation=cv2.INTER_LINEAR)
        else:
            return im

        if len(temp_scale.shape) == 2:
            temp_scale = np.expand_dims(temp_scale, axis=-1)

        center = (temp_scale.shape[0] // 2, temp_scale.shape[1] // 2)
        if variation > 0:
            im = temp_scale[int(center[0] - 512 / 2):int(center[0] + 512 / 2),
                 int(center[1] - 512 / 2):int(center[1] + 512 / 2)]
        elif variation < 0:
            padx = (512 - temp_scale.shape[0]) / 2
            pady = (512 - temp_scale.shape[1]) / 2
            im = np.pad(temp_scale, [
                (math.floor(padx), math.ceil(padx)),
                (math.floor(pady), math.ceil(pady)), (0,0)], mode='constant',
                        constant_values=np.min(temp_scale))
        return im

    def post_load_process(self, images, annotations):
        if self.variation is not None or 0:
            min_val = np.min(images)
            images -= min_val
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                for i in range(images.shape[0]):
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            images += min_val
        return images, annotations

    def run_perturbation(self, images, annotations, variation):
        images = self.scale_image(images, variation, 'linear')
        output_annotation = np.zeros(annotations.shape, dtype=annotations.dtype)
        for val in range(1, int(annotations.max()) + 1):
            temp = copy.deepcopy(annotations).astype('int')
            temp[temp != val] = 0
            temp[temp > 0] = 1
            im = temp
            if np.max(im) != 0:
                im = self.scale_image(im, variation, 'nearest')
                im[im > 0.1] = val
                im[im < val] = 0
                output_annotation[im == val] = val
        annotations = output_annotation
        return images, annotations


class Rotate_Images_2D_Processor(Image_Processor):
    def __init__(self, by_patient=False, image_size=512, variation=None):
        '''
        :param image_size: size of image row/col
        :param by_patient: perform on all images in stack, or vary for each one
        :param variation: range of values np.round(np.arange(start=-5, stop=5, step=1))
        '''
        self.image_size = image_size
        self.variation = variation
        self.by_patient = by_patient
        self.M_image = {}

    def post_load_process(self, images, annotations):
        if self.variation is not None:
            min_val = np.min(images)
            images -= min_val
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                for i in range(images.shape[0]):
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            images += min_val
        return images, annotations

    def run_perturbation(self, image, annotation, variation):
        image_row, image_col = image.shape[-2:]
        if variation not in self.M_image.keys():
            M_image = cv2.getRotationMatrix2D((int(image_row) / 2, int(image_col) / 2), variation, 1)
            self.M_image[variation] = M_image
        else:
            M_image = self.M_image[variation]
        if variation != 0:
            # images = cv2.warpAffine(images,M_image, (int(shape_size_image), int(shape_size_image)))
            image = cv2.warpAffine(image, M_image, (int(image_row), int(image_col)),flags=cv2.INTER_LINEAR)
            output_annotation = np.zeros(annotation.shape, dtype=annotation.dtype)
            for val in range(1, int(annotation.max()) + 1):
                temp = copy.deepcopy(annotation).astype('int')
                temp[temp != val] = 0
                temp[temp > 0] = 1
                im = temp
                if np.max(im) != 0:
                    im = cv2.warpAffine(im, M_image,
                                        (int(image_row), int(image_col)),
                                        flags=cv2.INTER_NEAREST)
                    im[im > 0.1] = val
                    im[im < val] = 0
                    output_annotation[im == val] = val
                # output_annotation[annotations == val] = val
            annotation = output_annotation
        return image, annotation


class Shift_Images_Processor(Image_Processor):
    def __init__(self,by_patient=False, variation=0, positive_to_negative=False):
        '''
        :param by_patient: Perform the same scaling across all images in stack, or one by one
        :param variation: Range of shift variations in rows and columns, up to and including! So 5 is 0,1,2,3,4,5
        :param positive_to_negative: if True and variation is 30, will range from -30 to +30
        '''
        self.by_patient = by_patient
        if variation != 0:
            if positive_to_negative:
                self.variation_range = np.asarray(range(-variation,variation+1))
            else:
                self.variation_range = np.asarray(range(variation+1))
        else:
            self.variation_range = None

    def post_load_process(self, images, annotations):
        if self.variation_range is not None:
            min_val = np.min(images)
            images -= min_val
            if not self.by_patient:
                for i in range(len(images)):
                    variation_row = self.variation_range[np.random.randint(len(self.variation_range))]
                    variation_col = self.variation_range[np.random.randint(len(self.variation_range))]
                    images[i, :, :], annotations[i, :, :] = self.make_perturbation(images[i, :, :],annotations[i, :, :],
                                                                                   variation_row, variation_col)
            else:
                variation_row = self.variation_range[np.random.randint(len(self.variation_range))]
                variation_col = self.variation_range[np.random.randint(len(self.variation_range))]
                images, annotations = self.make_perturbation(images, annotations, variation_row, variation_col)
            images += min_val
        return images, annotations

    def make_perturbation(self, images, annotations, variation_row, variation_col):
        if len(images.shape) == 2:
            images = interpolation.shift(images, [variation_row, variation_col])
            annotations = interpolation.shift(annotations.astype('int'),
                                              [variation_row, variation_col])
        elif len(images.shape) == 3:
            images = interpolation.shift(images, [0, variation_row, variation_col])
            annotations = interpolation.shift(annotations.astype('int'),
                                              [0, variation_row, variation_col])
        return images, annotations


class Random_2D_Deformation_Processor(Image_Processor):
    def __init__(self, by_patient=False, image_size=512, variation=None):
        '''
        :param image_shape: shape of images row/col
        :param by_patient: perform randomly on each image in stack, or on the entire collective
        :param variation: range of values np.round(np.arange(start=0, stop=2.6, step=0.5),2)
        '''
        self.image_size = image_size
        self.by_patient = by_patient
        self.variation = variation

    def run_perturbation(self, images, annotations, variation):
        # generate random parameter --- will be the same for all slices of the same patients
        # for 3D use dz with same pattern than dx/dy
        random_state = np.random.RandomState(None)

        if len(images.shape) > 2:
            temp_img = images
        else:
            temp_img = images[:, :, None]

        shape = temp_img.shape
        sigma = self.image_size * 0.1
        alpha = self.image_size * variation
        dx = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx) #2d not used
        # dz = filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), 512*0.10, mode="constant", cval=0) * 512*variation

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        # indices_3d = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

        if len(images.shape) > 2:
            images = interpolation.map_coordinates(temp_img, indices, order=1, mode='constant',
                                                   cval=float(np.min(images))).reshape(shape)
        else:
            images = interpolation.map_coordinates(temp_img, indices, order=1, mode='constant',
                                                   cval=float(np.min(images))).reshape(shape)[:, :, 0]

        output_annotation = np.zeros(annotations.shape, dtype=annotations.dtype)

        for val in range(1, int(annotations.max()) + 1):
            temp = copy.deepcopy(annotations).astype('int')
            temp[temp != val] = 0
            temp[temp > 0] = 1

            if len(annotations.shape) > 2:
                im = interpolation.map_coordinates(temp, indices, order=0, mode='constant', cval=0).reshape(shape)
            else:
                im = interpolation.map_coordinates(temp[:, :, None], indices, order=0, mode='constant',
                                                   cval=0).reshape(
                    shape)[:, :, 0]

            im[im > 0.1] = val
            im[im < val] = 0
            output_annotation[im == val] = val
        annotations = output_annotation
        return images, annotations

    def post_load_process(self, images, annotations):
        if self.variation is not None:
            min_val = np.min(images)
            images -= min_val
            if self.by_patient:
                variation = self.variation[np.random.randint(len(self.variation))]
                for i in range(images.shape[0]):
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            else:
                for i in range(images.shape[0]):
                    variation = self.variation[np.random.randint(len(self.variation))]
                    images[i], annotations[i] = self.run_perturbation(images[i],annotations[i],variation)
            images += min_val
        return images, annotations


if __name__ == '__main__':
    pass
