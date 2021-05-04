__author__ = 'Brian M Anderson'
# Created on 5/4/2021
import tensorflow as tf
import pickle
import os
from _collections import OrderedDict
import numpy as np


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


def return_example_proto(base_dictionary, image_dictionary_for_pickle=None, data_type_dictionary=None):
    if image_dictionary_for_pickle is None:
        image_dictionary_for_pickle = {}
    if data_type_dictionary is None:
        data_type_dictionary = {}
    feature = {}
    for key in base_dictionary:
        data = base_dictionary[key]
        if type(data) is int:
            feature[key] = _int64_feature(tf.constant(data, dtype='int64'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.int64)
        elif type(data) is float:
            feature[key] = _float_feature(tf.constant(data, dtype='float32'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.float32)
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


def serialize_example(input_features_dictionary, image_processors=None, verbose=False, record_writer=None):
    get_features(input_features_dictionary, image_processors=image_processors, verbose=verbose,
                 record_writer=record_writer)


def dictionary_to_tf_record(filename, input_features):
    """
    :param filename: .tfrecord filename
    :param input_features: dictionary of input_features, things like {'image': np.array}
    :return: empty dictionary
    """
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


class RecordWriter(object):
    def __init__(self, out_path, file_name_key='file_name', rewrite=False, **kwargs):
        self.file_name_key = file_name_key
        self.out_path = out_path
        self.rewrite = rewrite
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    def write_records(self, input_features):
        for example_key in input_features.keys():
            example = input_features[example_key]
            _check_keys_(example, self.file_name_key)
            image_name = example[self.file_name_key]
            filename = os.path.join(self.out_path, image_name)
            if not filename.endswith('.tfrecord'):
                filename += '.tfrecord'
            if not os.path.exists(filename) or self.rewrite:
                dictionary_to_tf_record(filename=filename, input_features=input_features)
            break


class RecordWriterRecurrence(RecordWriter):
    def write_records(self, input_features):
        non_recurred = -1
        recurred = -1
        for example_key in input_features.keys():
            example = input_features[example_key]
            _check_keys_(example, self.file_name_key)
            image_name = example[self.file_name_key]
            annotation = np.argmax(example['annotation'])
            if annotation == 0:
                non_recurred += 1
                out_path = os.path.join(self.out_path, 'No_Recurrence')
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                filename = os.path.join(out_path,
                                        image_name.replace('.tfrecord',
                                                           '_NoRecurrence_{}.tfrecord'.format(non_recurred)))
            else:
                recurred += 1
                out_path = os.path.join(self.out_path, 'Recurrence')
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                filename = os.path.join(out_path,
                                        image_name.replace('.tfrecord', '_Recurrence_{}.tfrecord'.format(recurred)))
            if not os.path.exists(filename) or self.rewrite:
                dictionary_to_tf_record(filename=filename, input_features={'out_example': example})


def get_features(features, image_processors=None, verbose=0, record_writer=None):
    if image_processors is not None:
        for image_processor in image_processors:
            features, _ = down_dictionary(features, OrderedDict(), 0)
            if verbose:
                print(image_processor)
            for key in features.keys():
                features[key] = image_processor.pre_process(features[key])
        features, _ = down_dictionary(features, OrderedDict(), 0)
    if record_writer is not None:
        record_writer.write_records(features)


def down_dictionary(input_dictionary, out_dictionary=None, out_index=0):
    if out_dictionary is None:
        out_dictionary = OrderedDict()
    for key in input_dictionary.keys():
        data = input_dictionary[key]
        if type(data) is dict or type(data) is OrderedDict:
            out_dictionary, out_index = down_dictionary(input_dictionary[key], out_dictionary, out_index)
        else:
            out_dictionary['Example_{}'.format(out_index)] = input_dictionary
            out_index += 1
            return out_dictionary, out_index
    return out_dictionary, out_index