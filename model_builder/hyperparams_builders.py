import tensorflow as tf
import tensorflow.contrib.slim as slim



def build(hyperparams_config,is_training):
    batch_norm = None
    batch_normparams = None

    if "batch_norm" in hyperparams_config:
        batch_norm = slim.batch_norm
    batch_normparams = _build_batch_norm_params(hyperparams_config["batch_norm"],is_training=is_training)

    affect_ops = [slim.conv2d,slim.separable_conv2d]

    def scope_fn():
        with slim.arg_scope([slim.batch_norm],**batch_normparams):
            with slim.arg_scope(
                affect_ops,
                weights_regularizer=_build_regularizer(
                    hyperparams_config["regularizer"]),
                weights_initializer=_build_initializer(
                    hyperparams_config["initializer"]),
                activation_fn=_build_activation_fn(hyperparams_config["activation"]),
                normalizer_fn=batch_norm,
                biases_initializer=_build_bias_initializer(hyperparams_config["use_bias"]),
                padding="SAME",) as sc:
                return sc
    return scope_fn

def _build_batch_norm_params(batch_norm, is_training):

  batch_norm_params = {
      'decay': batch_norm["decay"],
      'center': batch_norm["center"],
      'scale': batch_norm["scale"],
      'epsilon': batch_norm["epsilon"],
      # Remove is_training parameter from here and deprecate it in the proto
      # once we refactor Faster RCNN models to set is_training through an outer
      # arg_scope in the meta architecture.
      'is_training': is_training and batch_norm["train"],
  }
  return batch_norm_params


def _build_activation_fn(activation_fn):
    if activation_fn["type"] == "Relu":
        return tf.nn.relu

    if activation_fn["type"] == "leaky_relu":
        return tf.nn.leaky_relu

    if activation_fn["type"] == 'Relu_6':
        return tf.nn.relu6

    if activation_fn["type"] == "Sigmoid":
        return tf.nn.sigmoid


def _build_initializer(initializer):

    if initializer["type"] == 'random_normal_initializer':
        return tf.random_normal_initializer(
            mean=initializer["mean"],
            stddev=initializer["stddev"])
    elif initializer["type"] == 'truncated_normal_initializer':
        return tf.truncated_normal_initializer(
            mean=initializer["mean"],
            stddev=initializer["stddev"])

def _build_regularizer(regularizer):

    if regularizer["type"] == 'l1_regularizer':
        return slim.l1_regularizer(scale=float(regularizer["weigth"]))

    if regularizer["type"] == 'l2_regularizer':
        return slim.l2_regularizer(scale=float(regularizer["weigth"]))

def _build_bias_initializer(biasinit):

    if biasinit == None:
        return None

    elif biasinit == 0.0:
        return tf.zeros_initializer

    elif biasinit == 1.0:
        return tf.ones_initializer

    else:
        raise ValueError('Unkown params {}'.format(biasinit))
