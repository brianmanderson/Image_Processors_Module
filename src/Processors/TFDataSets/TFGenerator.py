import os
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image, plt
import glob
import pickle
import tensorflow as tf
import numpy as np


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


def return_parse_function(image_feature_description):

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)
    return _parse_image_function


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


class DataGeneratorClass(object):
    def __init__(self, record_paths=None, in_parallel=-1, delete_old_cache=False, shuffle=False, debug=False,
                 repeat=True):
        """
        :param record_paths: List of paths to a folder full of records files
        :param in_parallel: -1 is auto tune, None is None
        :param delete_old_cache: Boolean, delete the previous cache?
        :param shuffle: Boolean, shuffle the record names?
        :param debug: Boolean, debug process
        """
        self.delete_old_cache = delete_old_cache
        if in_parallel == -1:
            self.in_parallel = tf.data.AUTOTUNE
        else:
            self.in_parallel = in_parallel
        assert record_paths is not None, 'Need to pass a list of record names!'
        if not isinstance(record_paths, list):
            raise ValueError("Provide a list of record paths.")
        self.total_examples = 0
        data_set = None
        record_names = []
        for record_path in record_paths:
            assert os.path.isdir(record_path), 'Pass a directory, not a tfrecord\n{}'.format(record_path)
            record_names += [os.path.join(record_path, i) for i in os.listdir(record_path) if i.endswith('.tfrecord')]
        tfrecord_files = tf.data.Dataset.list_files(record_names)
        if repeat:
            tfrecord_files = tfrecord_files.repeat()
        if shuffle:
            tfrecord_files = tfrecord_files.shuffle(len(record_names))
        # raw_dataset = tfrecord_files.interleave(
        #     lambda filename: tf.data.TFRecordDataset(filename, num_parallel_reads=),
        #     cycle_length=2,  # Number of files to read concurrently
        #     num_parallel_calls=self.in_parallel
        # )
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=self.in_parallel)

        # raw_dataset = tf.data.TFRecordDataset(record_names, num_parallel_reads=self.in_parallel)
        features = None
        d_types = None
        for record_name in record_names:
            if features is None:
                features = load_obj(record_name.replace('.tfrecord', '_features.pkl'))
            if d_types is None:
                d_types = load_obj(record_name.replace('.tfrecord', '_dtype.pkl'))
            if os.path.exists(record_name.replace('.tfrecord', '_Num_Examples.txt')):
                fid = open(record_name.replace('.tfrecord', '_Num_Examples.txt'))
                examples = fid.readline()
                fid.close()
                self.total_examples += int(examples)
            else:
                self.total_examples += 1
        parsed_image_dataset = raw_dataset.map(tf.function(return_parse_function(features),),
                                               num_parallel_calls=self.in_parallel)
        Decode = DecodeImagesAnnotations(d_type_dict=d_types)
        if debug:
            data = next(iter(parsed_image_dataset))
            data = Decode.parse(image_features=data)
        self.data_set = parsed_image_dataset.map(tf.function(Decode.parse), num_parallel_calls=self.in_parallel)

    def compile_data_set(self, image_processors=None, debug=False):
        data = None
        if debug and data is None:
            data = next(iter(self.data_set))
        is_tuple = False
        if image_processors is not None:
            for image_processor in image_processors:
                print(image_processor)
                if type(image_processor) not in [dict, set]:
                    processor = image_processor.parse
                    if debug:
                        if data is None:
                            data = next(iter(self.data_set))
                        if data is not None:
                            data = image_processor.parse(data)
                    else:
                        processor = tf.function(image_processor.parse)
                    self.data_set = self.data_set.map(processor, num_parallel_calls=self.in_parallel)
                    # if True:
                    #     self.data_set = self.data_set.map(
                    #         lambda *features: processor(*features) if isinstance(features[0], dict) else
                    #         (processor(*features) if len(features) > 1 else processor(features[0])),
                    #         num_parallel_calls=self.in_parallel)
                    # elif not is_tuple:
                    #     self.data_set = self.data_set.map(lambda features: processor(features), num_parallel_calls=self.in_parallel)
                    # else:
                    #     self.data_set = self.data_set.map(lambda features: processor(*features), num_parallel_calls=self.in_parallel)
                elif type(image_processor) in [dict, set]:
                    data = None
                    value = None
                    if type(image_processor) is dict:
                        value = [image_processor[i] for i in image_processor][0]
                    if 'batch' in image_processor:
                        is_tuple = True
                        assert value is not None, "You need to provide a batch size with {'batch':batch_size}"
                        self.total_examples = self.total_examples//value
                        self.data_set = self.data_set.batch(value, drop_remainder=False)
                    elif 'cache' in image_processor:
                        if value is None:
                            self.data_set = self.data_set.cache()
                        else:
                            assert not os.path.isfile(value), 'Pass a path to {cache:path}, not a file!'
                            if not os.path.exists(value):
                                os.makedirs(value)
                            if self.delete_old_cache:
                                existing_files = glob.glob(os.path.join(value,'*cache.tfrecord*')) # Delete previous ones
                                for file in existing_files:
                                    os.remove(file)
                            path = os.path.join(value,'cache.tfrecord')
                            self.data_set = self.data_set.cache(path)
                    elif 'unbatch' in image_processor:
                        self.data_set = self.data_set.unbatch()
                    elif 'repeat' in image_processor:
                        self.data_set = self.data_set.repeat()
                    elif 'prefetch' in image_processor:
                        if value is not None:
                            self.data_set = self.data_set.prefetch(value)
                        else:
                            self.data_set = self.data_set.prefetch(self.in_parallel)
                else:
                    raise ModuleNotFoundError('Need to provide either a image processor, dict, or set!')

    def __len__(self):
        return self.total_examples