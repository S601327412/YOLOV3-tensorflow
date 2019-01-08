# -*-coding:utf-8-*-
import tensorflow as tf
from net import nets
from anchor import anchor_build_debug
from loss import losses_builder
from box_utils import boxes_utils
from model_builder import hyperparams_builders
from model_builder import box_predict_builder
from net import YOLOV_debug
def build(model_config,is_training):

    if model_config["model_type"]=="yoloV3":
        return _build_yoloV3_model(model_config,is_training)



def _build_yoloV3_model(model_config,is_training,reuse_weights=tf.AUTO_REUSE):
    sess = tf.InteractiveSession()
    num_class = model_config["num_class"]
    #构建超参数

    conv_hyperparams = hyperparams_builders.build(model_config["hyperparams"],is_training=is_training)

    #特征提取层
    feature_extractor = nets.YoloV3_FeatureExtractor(istrain=is_training,
                                                     hyperparams_fn=conv_hyperparams,
                                                     reuse_weights=reuse_weights,
                                                     batch_size=model_config["batch_size"])
    #YOLO 检测层
    if model_config["model_type"]=="mobile_net":
        BoxPredictor = box_predict_builder.box_predictor_build(istrain=is_training,num_class=num_class,
                                                           use_dropout=model_config["box_predictor"]["use_dropout"],
                                                           kernel_size=model_config["box_predictor"]["kernel_size"],
                                                           box_coder_size=model_config["box_predictor"]["coder_size"],
                                                           dropout_keep_prob=model_config["box_predictor"]["dropout_keep_prob"],
                                                           conv_hyperparams_fn=conv_hyperparams, \
                                                           apply_sigmoid_to_scores=model_config["box_predictor"]["apply_sigmoid_to_scores"],
                                                           class_prediction_bias_init=model_config["box_predictor"]["class_prediction_bias_init"],
                                                           confidence_prediction_bias_init=model_config["box_predictor"]["confidence_prediction_bias_init"],
                                                           use_depthwise=model_config["box_predictor"]["use_depthwise"])
    else:
        BoxPredictor = None

    #对网络输出进行解码匹配
    match = boxes_utils.Matcher(matched_threshold=model_config["Matcher"]["matched_threshold"],
                                unmatched_threshold=model_config["Matcher"]["unmatched_threshold"],)

    #生成anchors
    anchor_generator = anchor_build_debug.Anchor_Generator(model_config["Anchors"])

    #NMS验证模型时再写

    #构建loss函数
    classification_loss, localization_loss,confidence_loss,hard_example_mined= losses_builder.build(model_config)

    return YOLOV_debug.YOLOV3Meta(istrain=is_training,anchor_generator=anchor_generator,
                             matcher=match,feature_extractor=feature_extractor,
                             box_predictor=BoxPredictor,
                             non_max_suppression_fn=None,
                             classfication_loss=classification_loss,localization_loss=localization_loss,
                             confidence_loss=confidence_loss,hard_example_mined=hard_example_mined,num_class=num_class,iou_threshold=model_config["iou_threshold"],
                             cls_weigth=model_config["cls_weigth"],conf_weigth=model_config["confidence_weigth"],loc_weigth=model_config["loc_weigth"],
                             resize=model_config["resize"],batch_size=model_config["batch_size"],sess=sess)