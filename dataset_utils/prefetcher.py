
import tensorflow as tf


def prefetch(tensor_dict, capacity):

  names = list(tensor_dict.keys())
  dtypes = [t.dtype for t in tensor_dict.values()]
  shapes = [t.get_shape() for t in tensor_dict.values()]
  prefetch_queue = tf.PaddingFIFOQueue(capacity, dtypes=dtypes,
                                       shapes=shapes,
                                       names=names,
                                       name='prefetch_queue')
  enqueue_op = prefetch_queue.enqueue(tensor_dict)
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      prefetch_queue, [enqueue_op]))
  tf.summary.scalar('queue/%s/fraction_of_%d_full' % (prefetch_queue.name,
                                                      capacity),
                    tf.to_float(prefetch_queue.size()) * (1. / capacity))
  return prefetch_queue
