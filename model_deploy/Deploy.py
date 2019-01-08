#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import tensorflow.contrib.slim as slim

Clone = collections.namedtuple('Clone',
                               ['outputs',
                                'scope',
                                'device',
                               ])


def create_clones(config,model_fn,*args,**kwargs):
    clones = []
    args = args or []
    kwargs = kwargs or {}
    with slim.arg_scope([slim.model_variable, slim.variable],
                        device=config.variables_device()):
        # Create clones.
        for i in range(0, config.num_clones):
            with tf.name_scope(config.clone_scope(i)) as clone_scope:
                clone_device = config.clone_device(i)
                with tf.device(clone_device):
                    with tf.variable_scope(tf.get_variable_scope(),
                                           reuse=True if i > 0 else None):
                        outputs = model_fn(*args, **kwargs)
                    clones.append(Clone(outputs, clone_scope, clone_device))
    return clones


def optimize_clones(clones, optimizer,
                    regularization_losses=None,
                    **kwargs):

  grads_and_vars = []
  clones_losses = []
  num_clones = len(clones)
  if regularization_losses is None:
    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
  for clone in clones:
    with tf.name_scope(clone.scope):
      clone_loss, clone_grad = _optimize_clone(
          optimizer, clone, num_clones, regularization_losses, **kwargs)
      if clone_loss is not None:
        clones_losses.append(clone_loss)
        grads_and_vars.append(clone_grad)
      # Only use regularization_losses for the first clone
      regularization_losses = None
  # Compute the total_loss summing all the clones_losses.
  total_loss = tf.add_n(clones_losses, name='total_loss')
  # Sum the gradients across clones.
  grads_and_vars = _sum_clones_gradients(grads_and_vars)

  return total_loss, grads_and_vars

def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
                    **kwargs):

  sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
  clone_grad = None
  if sum_loss is not None:
    with tf.device(clone.device):
      clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
  return sum_loss, clone_grad


def _gather_clone_loss(clone, num_clones, regularization_losses):


  sum_loss = None

  clone_loss = None
  regularization_loss = None

  with tf.device(clone.device):
    all_losses = []
    clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
    if clone_losses:
      clone_loss = tf.add_n(clone_losses, name='clone_loss')
      if num_clones > 1:
        clone_loss = tf.div(clone_loss, 1.0,
                            name='scaled_clone_loss')
      all_losses.append(clone_loss)
    if regularization_losses:
      regularization_loss = tf.add_n(regularization_losses,
                                     name='regularization_loss')
      all_losses.append(regularization_loss)
    if all_losses:
      sum_loss = tf.add_n(all_losses)
  # Add the summaries out of the clone device block.
  if clone_loss is not None:
    tf.summary.scalar('/'.join(filter(None,
                                      ['Losses', clone.scope, 'clone_loss'])),
                      clone_loss)
  if regularization_loss is not None:
    tf.summary.scalar('Losses/regularization_loss', regularization_loss)
  return sum_loss


def _sum_clones_gradients(clone_grads):

  sum_grads = []
  for grad_and_vars in zip(*clone_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
    grads = []
    var = grad_and_vars[0][1]
    for g, v in grad_and_vars:
      assert v == var
      if g is not None:
        grads.append(g)
    if grads:
      if len(grads) > 1:
        sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        sum_grad = grads[0]
      #tf.summary.scalar("sum_grads",sum_grad)
      sum_grads.append((sum_grad, var))
  return sum_grads


class DeploymentConfig(object):

    def __init__(self,
                 num_clones=1,
                 clone_on_cpu=False,
                 replica_id=0,
                 num_replicas=1,
                 num_ps_tasks=0,
                 worker_job_name='worker',
                 ps_job_name='ps'
                 ):

        if num_replicas > 1:
            if num_ps_tasks < 1:
                raise ValueError('When using replicas num_ps_tasks must be positive')
        if num_replicas > 1 or num_ps_tasks > 0:
            if not worker_job_name:
                raise ValueError('Must specify worker_job_name when using replicas')
            if not ps_job_name:
                raise ValueError('Must specify ps_job_name when using parameter server')
        if replica_id >= num_replicas:
            raise ValueError('replica_id must be less than num_replicas')
        self._num_clones = num_clones
        self._clone_on_cpu = clone_on_cpu
        self._replica_id = replica_id
        self._num_replicas = num_replicas
        self._num_ps_tasks = num_ps_tasks
        self._ps_device = '/job:' + ps_job_name if num_ps_tasks > 0 else ''
        self._worker_device = '/job:' + worker_job_name if num_ps_tasks > 0 else ''

    @property
    def num_clones(self):
        return self._num_clones

    @property
    def clone_on_cpu(self):
        return self._clone_on_cpu

    @property
    def replica_id(self):
        return self._replica_id

    @property
    def num_replicas(self):
        return self._num_replicas

    @property
    def num_ps_tasks(self):
        return self._num_ps_tasks

    @property
    def ps_device(self):
        return self._ps_device

    @property
    def worker_device(self):
        return self._worker_device

    def caching_device(self):

        if self._num_ps_tasks > 0:
            return lambda op: op.device
        else:
            return None

    def clone_device(self, clone_index):

        if clone_index >= self._num_clones:
            raise ValueError('clone_index must be less than num_clones')
        device = ''
        if self._num_ps_tasks > 0:
            device += self._worker_device
        if self._clone_on_cpu:
            device += '/device:CPU:0'
        else:
            device += '/device:GPU:%d' % clone_index
        return device

    def clone_scope(self, clone_index):

        if clone_index >= self._num_clones:
            raise ValueError('clone_index must be less than num_clones')
        scope = ''
        if self._num_clones > 1:
            scope = 'clone_%d' % clone_index
        return scope

    def optimizer_device(self):

        if self._num_ps_tasks > 0 or self._num_clones > 0:
            return self._worker_device + '/device:CPU:0'
        else:
            return ''

    def inputs_device(self):
        device = ''
        if self._num_ps_tasks > 0:
            device += self._worker_device
        device += '/device:CPU:0'
        return device

    def variables_device(self):
        """Returns the device to use for variables created inside the clone.

        Returns:
          A value suitable for `tf.device()`.
        """
        device = ''
        if self._num_ps_tasks > 0:
            device += self._ps_device
        device += '/device:CPU:0'

        class _PSDeviceChooser(object):
            """Slim device chooser for variables when using PS."""

            def __init__(self, device, tasks):
                self._device = device
                self._tasks = tasks
                self._task = 0

            def choose(self, op):
                if op.device:
                    return op.device
                node_def = op if isinstance(op, tf.NodeDef) else op.node_def
                if node_def.op.startswith('Variable'):
                    t = self._task
                    self._task = (self._task + 1) % self._tasks
                    d = '%s/task:%d' % (self._device, t)
                    return d
                else:
                    return op.device

        if not self._num_ps_tasks:
            return device
        else:
            chooser = _PSDeviceChooser(device, self._num_ps_tasks)
            return chooser.choose