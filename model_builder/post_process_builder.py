#-*-coding:utf-8-*-


import tensorflow as tf
import functools
from preprocess import post_process

def build(config):
    non_max_suppressor_fn = _build_non_max_suppressor(config["NMS"])

    score_converter_fn = _build_score_converter(score_convert_type=config["score_convert"]["type"],
                                                logit_scale=config["score_convert"]["logit_scale"])


    return non_max_suppressor_fn,score_converter_fn


def _build_non_max_suppressor(config):

    if int(config["iou_threshold"]) < 0 or int(config["iou_threshold"]) > 1:

        raise ValueError("The iou_threshold must between [0,1]")

    non_max_suppressor_fn = functools.partial(
        post_process.batch_nms,
        iou_threshold=config["iou_threshold"],
        scores_threshold=config["scores_threshold"],
        max_total_detections=config["max_detection"],
        change_coordinate_frame=True
    )

    return non_max_suppressor_fn

def _build_score_converter(score_convert_type,logit_scale=1):

    if score_convert_type=="IDENTITY":
        return _score_converter_fn_with_logit_scale(tf.identity,logit_scale)

    elif score_convert_type=="SIGMOID":
        return _score_converter_fn_with_logit_scale(tf.sigmoid,logit_scale)

    elif score_convert_type=="SOFTMAX":

        return _score_converter_fn_with_logit_scale(tf.nn.softmax,logit_scale)

def _score_converter_fn_with_logit_scale(fn,logit_scale):

    if logit_scale < 0.0:
        raise ValueError("The logit_scale must be greater than 0.0!")

    logit_scale = tf.to_float(logit_scale)


    def score_converter_fn(logits):

        scaled_logits = tf.divide(logits,logit_scale)

        return fn(scaled_logits)

    return score_converter_fn