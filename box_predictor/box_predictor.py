import tensorflow as tf
from box_predictor import head
import collections
BOX_PREDICTOR = "pre_box"
CLASS_PREDICTOR = "pre_class"
CONFIDENCE_PREDICTOR = "pre_confidence"
slim = tf.contrib.slim

class Box_prediction(object):

    def __init__(self,
                 istrain,
                 hyperparamsfn,
                 class_head,
                 box_head,
                 confidence_head):
        self._hyperparams_fn = hyperparamsfn
        self._is_train = istrain
        self._class_head = class_head
        self._box_head = box_head
        self._confidence_head = confidence_head


    def _predict(self,features,per_grid_num_anchor):

        if len(features)!= len(per_grid_num_anchor):
            raise ValueError("The features num must be equal anchors list length")

        predictions_list = []
        if len(features)>1:
            scopes = [tf.variable_scope("Box_Predictor_{}".format(i)) for i in range(len(features))]
        else:
            scopes = [tf.variable_scope("Box_Predictor")]

        sort_keys = [BOX_PREDICTOR,CLASS_PREDICTOR,CONFIDENCE_PREDICTOR]
        for indx,(image_features,scope,num_anchors) in enumerate(zip(features,scopes,per_grid_num_anchor)):

            net = image_features

            with scope:
                with slim.arg_scope(self._hyperparams_fn()):
                    with slim.arg_scope([slim.dropout],is_training=self._is_train):
                        predictions_dict = collections.OrderedDict()
                        for keys in sort_keys:

                            if keys==BOX_PREDICTOR:
                                head_obj = self._box_head
                            elif keys==CLASS_PREDICTOR:
                                head_obj = self._class_head
                            elif keys==CONFIDENCE_PREDICTOR:
                                head_obj = self._confidence_head

                            predictions = head_obj.predict(net,num_anchors)

                            predictions_dict[keys+"_{}".format(indx)] = predictions
                        predictions_list.append(predictions_dict)

        return predictions_list







