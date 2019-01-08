
import collections

import tensorflow as tf

from dataset_utils import prefetcher

rt_shape_str = '_runtime_shapes'


class BatchQueue(object):


  def __init__(self, tensor_dict, batch_size, batch_queue_capacity,
               num_batch_queue_threads, prefetch_queue_capacity):


    static_shapes = collections.OrderedDict(
        {key: tensor.get_shape() for key, tensor in tensor_dict.items()})

    runtime_shapes = collections.OrderedDict(
        {(key + rt_shape_str): tf.shape(tensor)
         for key, tensor in tensor_dict.items()})

    all_tensors = tensor_dict
    all_tensors.update(runtime_shapes)

    batched_tensors = tf.train.batch(
        all_tensors,
        capacity=batch_queue_capacity,
        batch_size=batch_size,
        dynamic_pad=True,
        num_threads=num_batch_queue_threads)

    self._queue = prefetcher.prefetch(batched_tensors,
                                      prefetch_queue_capacity)
    self._static_shapes = static_shapes
    self._batch_size = batch_size

  def dequeue(self):

    batched_tensors = self._queue.dequeue()

    tensors = {}
    shapes = {}
    for key, batched_tensor in batched_tensors.items():
      unbatched_tensor_list = tf.unstack(batched_tensor)
      for i, unbatched_tensor in enumerate(unbatched_tensor_list):
        if rt_shape_str in key:
          shapes[(key[:-len(rt_shape_str)], i)] = unbatched_tensor
        else:
          tensors[(key, i)] = unbatched_tensor


    tensor_dict_list = []
    batch_size = self._batch_size
    for batch_id in range(batch_size):
      tensor_dict = {}
      for key in self._static_shapes:
        tensor_dict[key] = tf.slice(tensors[(key, batch_id)],
                                    tf.zeros_like(shapes[(key, batch_id)]),
                                    shapes[(key, batch_id)])
        tensor_dict[key].set_shape(self._static_shapes[key])
      tensor_dict_list.append(tensor_dict)

    return tensor_dict_list

