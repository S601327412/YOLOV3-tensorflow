import tensorflow as tf
import functools
from dataset_utils import dataset_utils
import os

from dataset_utils import batcher


def get_input(input_config):

    '''test
    record_path = "train.record"
    batch_size = 5'''

    record_path = input_config["record_path"]
    #batch_size = input_config["batch_size"]
    if not os.path.exists(record_path):
        raise FileNotFoundError("The record file is not found!")
    file_read_func = functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000)
    filename_dataset = tf.data.Dataset.from_tensor_slices(tf.unstack([record_path]))
    filename_dataset = filename_dataset.repeat()
    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            file_read_func,
            cycle_length=4,
            block_length=4,
            sloppy=False))
    records_dataset = records_dataset.shuffle(2048,reshuffle_each_iteration=False)
    tensor_dataset = records_dataset.map(
        dataset_utils.parser_tfrecord).repeat()
    dataset = tensor_dataset.prefetch(64)
    iterator = dataset_utils.make_initializable_iterator(dataset)

    return iterator

def read_and_transform_dataset(per_clone_batch_size,create_tensor_dict_fn,prefetch_queue_capacity=5,capacity=150,num_threads=8,data_augmentation_options=None):

    tensor_dict = create_tensor_dict_fn()

    #预处理
    #.........
    input_queue = batcher.BatchQueue(
        tensor_dict,
        batch_size=per_clone_batch_size,
        batch_queue_capacity=capacity,
        num_batch_queue_threads=num_threads,
        prefetch_queue_capacity=prefetch_queue_capacity)

    return input_queue

'''
def get_next(config):
    return get_input(config).get_next()

create_input_dict_fn = functools.partial(get_next,32)

data = create_input_dict_fn()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(tf.tables_initializer())
    dataset = data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(1):

        gtbox_w = dataset["bbox"][:,:,2:3]-dataset["bbox"][:,:,0:1]
        gtbox_h = dataset["bbox"][:,:,3:4]-dataset["bbox"][:,:,1:2]
        gtbox_wh = tf.concat([gtbox_w,gtbox_h],axis=2)
        print(sess.run([dataset["filename"],dataset["label"],gtbox_wh,dataset["bbox"]]))'''


