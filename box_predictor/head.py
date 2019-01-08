#-*-coding:utf-8-*-

import functools
import tensorflow as tf

slim = tf.contrib.slim



class ConvolutionalBoxHead(object):

    def __init__(self,
                 istrain,
                 use_depthwise,
                 box_code_size,
                 kernel_size):

        self._istrain=istrain
        self._use_depthwise=use_depthwise
        self._box_code_size=box_code_size
        self._kernel_size=kernel_size


    def predict(self,features,num_anchors):

        net = features

        if self._use_depthwise:

            box_codings = slim.separable_conv2d(
                net,None,[self._kernel_size,self._kernel_size],depth_multiplier=1,
                stride=1,padding='SAME',rate=1,scope="Box_predictor_depthwise")

            box_codings = slim.conv2d(box_codings,num_anchors * self._box_code_size,[1,1],
                                      activation_fn=None,normalizer_fn=None,normalizer_params=None,
                                      scope="Box_predictions")

        else:

            box_codings = slim.conv2d(net,num_anchors * self._box_code_size,[self._kernel_size,self._kernel_size],
                                      activation_fn=None,normalizer_fn=None,normalizer_params=None,
                                      scope="Box_predictions")
        batch_size = features.get_shape().as_list()[0]
        box_encodings = tf.reshape(box_codings,[batch_size,-1,self._box_code_size])

        return box_encodings


class ConvolutionalClassHead(object):

    def __init__(self,
                 istrain,
                 num_class,
                 use_dropout,
                 dropout_keep_prob,
                 kernel_size,
                 apply_sigmoid_to_scores=False,
                 bias=0.0,
                 use_depthwise=False):

        self._istrain = istrain
        self._num_class = num_class
        self._use_dropout = use_dropout
        self._dropout_keep_prob = dropout_keep_prob
        self._apply_sigmoid_to_scores = apply_sigmoid_to_scores
        self._kernel_size = kernel_size
        self._bias_init = bias
        self._use_depthwise = use_depthwise

    def predict(self,features,num_anchors):

        net = features

        if self._use_dropout:
            net = slim.dropout(net,keep_prob=self._dropout_keep_prob)

        if self._use_depthwise:

            class_predictions = slim.separable_conv2d(net,None,[self._kernel_size,self._kernel_size],depth_multiplier=1,
                                                      stride=1,padding="SAME",rate=1,scope="Class_predictions_depthwise")

            class_predictions = slim.conv2d(class_predictions,num_anchors * self._num_class,[1,1],activation_fn=None,
                                            normalizer_fn=None,normalizer_params=None,
                                            biases_initializer=tf.constant_initializer(self._bias_init),scope="Class_predictions")
        else:
            class_predictions = slim.conv2d(net,num_anchors * self._num_class,[self._kernel_size,self._kernel_size],
                                            activation_fn=None,normalizer_fn=None,normalizer_params=None,scope="Class_predictions")

        if self._apply_sigmoid_to_scores:
            class_predictions = tf.sigmoid(class_predictions)

        batch_size = features.get_shape().as_list()[0]

        Class_Predictions = tf.reshape(class_predictions,[batch_size,-1,self._num_class])

        return Class_Predictions

class ConvolutionalConfidenceHead(object):

    def __init__(self,
                 istrain,
                 kernel_size,
                 bias=0.0,
                 apply_sigmoid_to_scores=False,
                 use_depthwise=False):

        self._istrain = istrain
        self._kernel_size = kernel_size
        self._bias_init = bias
        self._apply_sigmoid_to_scores = apply_sigmoid_to_scores
        self._use_depthwise = use_depthwise


    def predict(self,features,num_anchors):
        net = features

        if self._use_depthwise:

            confidence_predictor = slim.separable_conv2d(net,None,[self._kernel_size,self._kernel_size],depth_multiplier=1,
                                                         stride=1,padding="SAME",rate=1,scope="Confidence_predict_depthwise")

            confidence_predictor = slim.conv2d(confidence_predictor,num_anchors,[1,1],activation_fn=None,
                                                normalizer_fn=None,normalizer_params=None,biases_initializer=tf.constant_initializer(
                                                self._bias_init),scope="Confidence_predictions")

        else:
            confidence_predictor = slim.conv2d(net,num_anchors,[self._kernel_size,self._kernel_size],activation_fn=None,
                                               normalizer_fn=None,normalizer_params=None,biases_initializer=tf.constant_initializer(
                                                self._bias_init),scope="Confidence_predictions")

        if self._apply_sigmoid_to_scores:
            confidence_predictor = tf.sigmoid(confidence_predictor)

        batch_size = features.get_shape().as_list()[0]

        confidence_predictions = tf.reshape(confidence_predictor,[batch_size,-1,1])

        return confidence_predictions