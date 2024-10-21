__author__ = 'Brian M Anderson'
# Created on 5/4/2021
import sys
import os.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image, plt
try:
    import tensorflow as tf
except:
    print("No tensorflow imported")
import typing
import pickle
import os
from collections import OrderedDict
import numpy as np
from threading import Thread
import copy
from multiprocessing import cpu_count
from queue import *


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


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
        elif type(data) in [float, np.float32, np.float64]:
            feature[key] = _float_feature(tf.constant(float(data), dtype='float32'))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.float32)
        elif type(data) is np.ndarray:
            for index, shape_value in enumerate(data.shape):
                if '{}_size_{}'.format(key, index) not in base_dictionary:
                    feature['{}_size_{}'.format(key, index)] = _int64_feature(int(shape_value))
                    image_dictionary_for_pickle['{}_size_{}'.format(key, index)] = tf.io.FixedLenFeature([], tf.int64)
            feature[key] = _bytes_feature(data.tostring())
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.string)
                data_type_dictionary[key] = data.dtype
        elif type(data) is str:
            feature[key] = _bytes_feature(tf.constant(data))
            if key not in image_dictionary_for_pickle:
                image_dictionary_for_pickle[key] = tf.io.FixedLenFeature([], tf.string)
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
    example_proto = return_example_proto(input_features, features, d_type)
    writer.write(example_proto.SerializeToString())
    writer.close()
    save_obj(filename.replace('.tfrecord', '_features.pkl'), features)
    save_obj(filename.replace('.tfrecord', '_dtype.pkl'), d_type)
    del input_features
    return {}


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
            out_dictionary['{}'.format(out_index)] = input_dictionary
            out_index += 1
            return out_dictionary, out_index
    return out_dictionary, out_index


class RecordWriter(object):
    def __init__(self, out_path, file_name_key='file_name', rewrite=False):
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
            filename = filename.replace('.tfrecord', '_{}.tfrecord'.format(example_key))
            if not os.path.exists(filename) or self.rewrite:
                dictionary_to_tf_record(filename=filename, input_features=example)


class PickleRecordWriter(object):
    def __init__(self, out_path, file_name_key='file_name', rewrite=False,
                 wanted_keys=('ct_array', 'mask_array', 'ct_file')):
        self.file_name_key = file_name_key
        self.out_path = out_path
        self.rewrite = rewrite
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.wanted_keys = wanted_keys

    def write_records(self, input_features):
        for example_key in input_features.keys():
            full_example = input_features[example_key]
            _check_keys_(full_example, self.file_name_key)
            _check_keys_(full_example, self.wanted_keys)
            image_name = full_example[self.file_name_key]
            image_name = f"{image_name.replace('.tfrecord', '')}_{example_key}.pkl"
            filename = os.path.join(self.out_path, image_name)
            example = {key: value for key, value in full_example.items() if key in self.wanted_keys}
            save_obj(filename, example)


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
                dictionary_to_tf_record(filename=filename, input_features=example)


def worker_def(a):
    q = a[0]
    base_class = serialize_example
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                base_class(**item)
            except:
                print('Failed? {}'.format(item))
            q.task_done()


def return_data_dict(niftii_path):
    data_dict = {}
    image_files = [i for i in os.listdir(niftii_path) if i.find('Overall_Data') == 0]
    for file in image_files:
        iteration = file.split('_')[-1].split('.')[0]
        data_dict[iteration] = {'image_path': os.path.join(niftii_path, file),
                                'file_name': '{}.tfrecord'.format(file.split('.nii')[0])}

    annotation_files = [i for i in os.listdir(niftii_path) if i.find('Overall_mask') == 0]
    for file in annotation_files:
        iteration = file.split('_y')[-1].split('.')[0]
        data_dict[iteration]['annotation_path'] = os.path.join(niftii_path, file)
    return data_dict


def parallel_record_writer(dictionary_list=None, out_path=None, max_records=np.inf, image_processors=None,
                           recordwriter=None, thread_count=int(cpu_count() * .5), niftii_path=None, rewrite=False,
                           is_3D=True, extension=np.inf, special_actions=False, verbose=False, file_parser=None,
                           debug=False, **kwargs):
    """
    :param niftii_path: path to where Overall_Data and mask files are located
    :param out_path: path that we will write records to
    :param rewrite: Do you want to rewrite old records? True/False
    :param thread_count: specify 1 if debugging
    :param max_records: Can specify max number of records, for debugging purposes
    :param extension: extension above and below annotation, recommend np.inf for validation and test
    :param is_3D: Take the whole patient or break up into 2D images
    :param image_processors: a list of image processes that can take the image and annotation dictionary,
        see Image_Processors, TF_Record
    :param special_actions: if you're doing something special and don't want Add_Images_And_Annotations
    :param verbose: Binary, print processors as they go
    :param dictionary_list: a list of dictionaries, typically [{'image_path': path, 'annotation_path': path}]
    :return:
    """
    assert image_processors is not None, 'Please provide a list of image processors'
    if recordwriter is None:
        if out_path is None:
            out_path = niftii_path
        recordwriter = RecordWriter(out_path=out_path, file_name_key='file_name', rewrite=rewrite)
    threads = []
    q = None
    if not debug:
        q = Queue(maxsize=thread_count)
        a = [q, ]
        for worker in range(thread_count):
            t = Thread(target=worker_def, args=(a,))
            t.start()
            threads.append(t)
    if dictionary_list is None:
        if file_parser is None:
            data_dict = return_data_dict(niftii_path=niftii_path)
        else:
            data_dict = file_parser(**locals(), **kwargs)
    else:
        data_dict = dictionary_list
    counter = 0
    if type(data_dict) in (dict, OrderedDict):
        for iteration in data_dict.keys():
            item = copy.deepcopy(data_dict[iteration])
            input_item = OrderedDict()
            input_item['input_features_dictionary'] = item
            input_item['image_processors'] = image_processors
            input_item['record_writer'] = recordwriter
            input_item['verbose'] = verbose
            if not debug:
                q.put(input_item)
            else:
                serialize_example(**input_item)
            counter += 1
            if counter >= max_records:
                break
    else:
        data_dict = list(data_dict)
        while data_dict:
            item = data_dict.pop()
            input_item = OrderedDict()
            input_item['input_features_dictionary'] = copy.deepcopy(item)
            input_item['image_processors'] = image_processors
            input_item['record_writer'] = recordwriter
            input_item['verbose'] = verbose
            if not debug:
                q.put(input_item)
            else:
                serialize_example(**input_item)
            counter += 1
            if counter >= max_records:
                break
    if not debug:
        for i in range(thread_count):
            q.put(None)
        for t in threads:
            t.join()
    return None


if __name__ == '__main__':
    pass
