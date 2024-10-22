__author__ = 'Brian M Anderson'
# Created on 3/5/2021
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
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
    def __call__(self, *args, **kwargs):
        return args, kwargs


class RandomNoise:
    def __init__(self, max_noise=2.5, wanted_keys=('image',)):
        self.max_noise = max_noise
        self.wanted_keys = wanted_keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.wanted_keys)
        if self.max_noise == 0.0:
            return image_features
        for key in self.wanted_keys:
            if key in image_features:
                data = image_features[key]
                noise = torch.empty_like(data).uniform_(0.0, self.max_noise) * torch.randn_like(data)
                image_features[key] = data + noise
        return image_features


class ResizeAndPad(ImageProcessor):
    def __init__(self, input_keys=('primary_image', 'secondary_image'),
                 output_keys=('primary_image_mag', 'secondary_image_mag'), resize_row_col=(133, 133),
                 output_size=(256, 256)):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.resize_row_col = resize_row_col
        self.output_size = output_size

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for image_key, resize_row_col, output_key, output_size in zip(self.input_keys, self.resize_row_col,
                                                                      self.output_keys, self.output_size):
            x = image_features[image_key]
            new_image = torch.nn.functional.interpolate(x.unsqueeze(0), size=resize_row_col, mode='bilinear', align_corners=False).squeeze(0)
            pad_top = (output_size - resize_row_col) // 2
            pad_bottom = output_size - resize_row_col - pad_top
            padded_image = torch.nn.functional.pad(new_image, (pad_top, pad_bottom, pad_top, pad_bottom), mode='constant', value=0)
            parsed_features[output_key] = padded_image
        return parsed_features


class CombineKeys:
    def __init__(self, image_keys=('primary_image', 'secondary_image'), output_key='combined', axis=-1):
        self.image_keys = image_keys
        self.output_key = output_key
        self.axis = axis

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.image_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        combine_images = [image_features[i] for i in self.image_keys]
        parsed_features[self.output_key] = torch.cat(combine_images, dim=self.axis)
        return parsed_features


class KeepOnlyKeys(ImageProcessor):
    """
    This should be your final image processor, this will turn your dictionary into a set of tensors
    """
    def __init__(self, keys_to_keep=('image',)):
        self.keys_to_keep = keys_to_keep

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys_to_keep)
        keys = list(image_features.keys())
        for key in keys:
            if key not in self.keys_to_keep:
                del image_features[key]
        return image_features


class ReturnOutputs(ImageProcessor):
    """
    This should be your final image processor, this will turn your dictionary into a set of tensors
    """
    def __init__(self, input_keys=('image',), output_keys=('annotation',), as_tuple=True):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.as_tuple = as_tuple

    def __call__(self, image_features, *args, **kwargs):
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
        return inputs, outputs


class RandomCrop(ImageProcessor):
    def __init__(self, keys_to_crop=('image_array', 'annotation_array'), crop_dimensions=(32, 32, 32, 1)):
        self.keys_to_crop = keys_to_crop
        self.crop_dimensions = crop_dimensions

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, keys=self.keys_to_crop)
        parsed_features = {key: value for key, value in image_features.items()}
        images = [image_features[i] for i in self.keys_to_crop]
        image = torch.cat(images, dim=-1)
        crop_dimensions = self.crop_dimensions
        start_indices = [torch.randint(0, image.shape[i] - crop_dimensions[i] + 1, (1,)).item() for i in range(len(crop_dimensions))]
        slices = tuple(slice(start, start + size) for start, size in zip(start_indices, crop_dimensions))
        cropped_image = image[slices]
        start_dim = 0
        for key in self.keys_to_crop:
            dimension = image_features[key].shape[-1]
            end_dim = start_dim + dimension
            new_image = cropped_image[..., start_dim:end_dim]
            start_dim += dimension
            parsed_features[key] = new_image
        return parsed_features


class ToCategorical(ImageProcessor):
    def __init__(self, annotation_keys=('annotation',), number_of_classes=(2,)):
        self.annotation_keys = annotation_keys
        self.number_of_classes = number_of_classes

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, keys=self.annotation_keys)
        for key, num_classes in zip(self.annotation_keys, self.number_of_classes):
            y = image_features[key].squeeze()
            one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes)
            image_features[key] = one_hot
        return image_features


class Squeeze(ImageProcessor):
    def __init__(self, image_keys=('image',)):
        """
        Designed to squeeze tf arrays
        :param image_keys:
        """
        self.image_keys = image_keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for key in self.image_keys:
            image_features[key] = image_features[key].squeeze()
        return image_features


class ToTensor(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation')):
        self.image_keys = image_keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for key in self.image_keys:
            image_features[key] = torch.from_numpy(image_features[key])
        return image_features


class ExpandDimension(ImageProcessor):
    def __init__(self, axis=-1, image_keys=('image', 'annotation')):
        self.axis = axis
        self.image_keys = image_keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for key in self.image_keys:
            image = image_features[key]
            image_features[key] = image.unsqueeze(self.axis)
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

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key in self.input_keys:
            image = torch.tensor(image_features[key])
            parsed_features[key] = image.repeat_interleave(self.repeats, dim=self.axis)
        return parsed_features


class TakeAxis(ImageProcessor):
    def __init__(self, keys=('mask_array',), wanted_axis=(1,)):
        self.keys = keys
        self.wanted_axis = wanted_axis

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, axis in zip(self.keys, self.wanted_axis):
            image = image_features[key]
            parsed_features[key] = image.select(-1, axis)
        return parsed_features


class FlipImages(ImageProcessor):
    def __init__(self, keys=('image', 'mask'), flip_up_down=False,
                 flip_left_right=False):
        self.flip_up_down = flip_up_down
        self.flip_left_right = flip_left_right
        self.keys = keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        combine_images = [image_features[i] for i in self.keys]
        flip_image = torch.cat(combine_images, dim=-1)
        if self.flip_up_down and torch.rand(1).item() > 0.5:
            flip_image = torch.flip(flip_image, dims=[1])  # Flip up-down
        if self.flip_left_right and torch.rand(1).item() > 0.5:
            flip_image = torch.flip(flip_image, dims=[2])  # Flip left-right
        flipped_image = flip_image
        start_dim = 0
        for key in self.keys:
            dimension = image_features[key].shape[-1]
            end_dim = start_dim + dimension
            new_image = flipped_image[..., start_dim:end_dim]
            start_dim += dimension
            image_features[key] = new_image
        return image_features


class ShiftImages(ImageProcessor):
    def __init__(self, keys=('image', 'mask'), fill_value=None, fill_mode="reflect", interpolation="bilinear",
                 height_factor=0.0, width_factor=0.0, vert_factor=0.0, on_global_3D=True,
                 image_shape=(32, 320, 320, 3)):
        self.og_shape = image_shape
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.vert_factor = vert_factor
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.keys = keys
        self.global_3D = on_global_3D

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        combine_images = [image_features[i] for i in self.keys]
        shift_image = np.concatenate(combine_images, axis=-1)
        og_shape = self.og_shape

        if self.height_factor != 0.0:
            shift_pixels = int(self.height_factor * shift_image.shape[1] * np.random.uniform(0, 1))
            shift_image = np.roll(shift_image, shift_pixels, axis=1)

        if self.width_factor != 0.0:
            shift_pixels = int(self.width_factor * shift_image.shape[2] * np.random.uniform(0, 1))
            shift_image = np.roll(shift_image, shift_pixels, axis=2)

        if self.vert_factor != 0.0:
            shift_pixels = int(self.vert_factor * shift_image.shape[0] * np.random.uniform(0, 1))
            shift_image = np.roll(shift_image, shift_pixels, axis=0)

        if self.global_3D and len(og_shape) == 4:
            shift_image = np.reshape(shift_image, og_shape)

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
    def __init__(self, keys=('image',), values=(1,)):
        """
        :param keys: tuple of keys for addition
        :param values: tuple of values for addition
        """
        self.keys = keys
        self.values = values

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, value in zip(self.keys, self.values):
            parsed_features[key] = image_features[key] * value
        return parsed_features


class TakeExpOfKey(ImageProcessor):
    def __init__(self, input_keys=('pdos_array',)):
        self.input_keys = input_keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for input_key in self.input_keys:
            parsed_features[input_key] = torch.exp(image_features[input_key])
        return parsed_features


class CreateNewKey(ImageProcessor):
    def __init__(self, input_keys=('pdos_array',), output_keys=('new_pdos_array',)):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            parsed_features[output_key] = image_features[input_key]
        return parsed_features


class AddImagesTogether(ImageProcessor):
    def __init__(self, keys=('pdos_array', 'drr_array'), out_key='pdos_drr_combined'):
        """
        :param keys:
        """
        self.keys = keys
        self.out_key = out_key

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        combined_image = torch.sum(torch.stack([image_features[key] for key in self.keys]), dim=0)
        parsed_features[self.out_key] = combined_image
        return parsed_features


class MultiplyImagesTogether(ImageProcessor):
    def __init__(self, keys=('pdos_array', 'drr_array'), out_key='pdos_drr_combined'):
        """
        :param keys:
        """
        self.keys = keys
        self.out_key = out_key

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        combined_image = torch.prod(torch.stack([image_features[key] for key in self.keys]), dim=0)
        parsed_features[self.out_key] = combined_image
        return parsed_features


class AddConstant(ImageProcessor):
    def __init__(self, keys=('image',), values=(0,)):
        """
        :param keys: tuple of keys for addition
        :param values: tuple of values for addition
        """
        self.keys = keys
        self.values = values

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, value in zip(self.keys, self.values):
            parsed_features[key] = image_features[key] + value
        return parsed_features


class NormalizeImages(ImageProcessor):
    def __init__(self, keys=('image',), mean_values=(0,), std_values=(1,),):
        """
        :param keys: tuple of image keys
        :param mean_values: tuple of mean values
        :param std_values: tuple of standard deviations
        """
        self.keys = keys
        self.mean_values = mean_values
        self.std_values = std_values

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, mean_val, std_val in zip(self.keys, self.mean_values, self.std_values):
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

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.annotation_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key in self.annotation_keys:
            parsed_features[key] = torch.argmax(image_features[key], dim=self.axis)
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

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.image_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, channel, new_key in zip(self.image_keys, self.channels, self.out_keys):
            parsed_features[new_key] = image_features[key][..., channel]
        return parsed_features


class CreateNewKeyFromArgSum(ImageProcessor):
    def __init__(self, guiding_keys=('annotation', ), new_keys=('mask',)):
        """
        :param guiding_keys: keys which will guide the masking of another key
        :param new_keys: keys which will be masked
        """
        self.guiding_keys = guiding_keys
        self.new_keys = new_keys

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.guiding_keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for guiding_key, new_key in zip(self.guiding_keys, self.new_keys):
            annotation = image_features[guiding_key]
            mask = torch.sum(annotation[..., 1:], dim=-1, keepdim=True)
            parsed_features[new_key] = mask
        return parsed_features


def get_torch_dtype(dtype_string):
    if dtype_string == 'float32':
        return torch.float32
    elif dtype_string == 'float':
        return torch.float
    elif dtype_string == 'int':
        return torch.int
    elif dtype_string == 'int64':
        return torch.int64
    elif dtype_string == 'int32':
        return torch.int32
    elif dtype_string == 'bool':
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_string}")


class PermuteAxis(ImageProcessor):
    def __init__(self, keys=('image', 'annotation',), axis=(3, 0, 1, 2)):
        """
        :param keys: tuple of keys
        :param dtypes: tuple of datatypes
        """
        self.keys = keys
        self.axis = axis

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key in self.keys:
            image_features[key] = image_features[key].permute(self.axis)
        return image_features


class CastData(ImageProcessor):
    def __init__(self, keys=('image', 'annotation',), dtypes=('float16', 'float16')):
        """
        :param keys: tuple of keys
        :param dtypes: tuple of datatypes
        """
        self.keys = keys
        self.dtypes = dtypes

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        for key, dtype_string in zip(self.keys, self.dtypes):
            dtype = get_torch_dtype(dtype_string)
            image_features[key] = image_features[key].to(dtype)
        return image_features


class ThresholdImages(ImageProcessor):
    def __init__(self, keys=('image',), lower_bounds=(-float('inf'),), upper_bounds=(float('inf'),), divides=(True,)):
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

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, lower_bound, upper_bound, divide in zip(self.keys, self.lower_bounds, self.upper_bounds, self.divides):
            parsed_features[key] = torch.clamp(image_features[key], min=lower_bound, max=upper_bound)
            if divide:
                parsed_features[key] = parsed_features[key] / (upper_bound - lower_bound)
        return parsed_features


class PadImages(ImageProcessor):
    def __init__(self, keys=('image', 'annotation'), pad_left=(100, 100), pad_top=(100, 100), out_size=(256, 256)):
        self.keys = keys
        self.pad_left = pad_left
        self.pad_top = pad_top
        self.out_size = out_size

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, pad_left, pad_top, out_size in zip(self.keys, self.pad_left, self.pad_top, self.out_size):
            image = image_features[key]
            padding = [
                (pad_top, out_size - image.shape[-2] - pad_top),
                (pad_left, out_size - image.shape[-1] - pad_left)
            ]
            if image.ndim == 3:
                padding = [(0, 0)] + padding
            new_image = torch.nn.functional.pad(image, pad=[p for pad_pair in reversed(padding) for p in pad_pair], mode='constant', value=0)
            parsed_features[key] = new_image
        return parsed_features


class ResizeWithCropPad(ImageProcessor):
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

    def __call__(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.keys)
        parsed_features = {key: value for key, value in image_features.items()}
        for key, image_rows, image_cols, is_mask, out_key in zip(self.keys, self.image_rows, self.image_cols,
                                                                 self.is_mask, self.out_keys):
            image = image_features[key]
            crop_height = min(image.shape[0], image_rows)
            crop_width = min(image.shape[1], image_cols)
            start_y = (image.shape[0] - crop_height) // 2
            start_x = (image.shape[1] - crop_width) // 2
            cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

            pad_height = max(0, image_rows - cropped_image.shape[0])
            pad_width = max(0, image_cols - cropped_image.shape[1])
            padded_image = np.pad(cropped_image, ((pad_height // 2, pad_height - pad_height // 2),
                                                  (pad_width // 2, pad_width - pad_width // 2), (0, 0)),
                                  mode='constant', constant_values=0)

            if is_mask and padded_image.shape[-1] != 1:
                array = padded_image[..., 1:]
                background = np.expand_dims(1 - np.sum(array, axis=-1), axis=-1)
                padded_image = np.concatenate([background, array], axis=-1)

            parsed_features[out_key] = padded_image
        return parsed_features


if __name__ == '__main__':
    pass
