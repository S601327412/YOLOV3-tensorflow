import tensorflow as tf
from loss import losses

class loss(object):

  def __init__(self,
               gtbox,
               anchor,
               pre_box,
               pre_class,
               classlabel,
               pre_confidence,
               pre_negativite_confidence,
               weights_coordinate,
               gridcell,
               size,
               confidence_weights,
               class_weights=1.0,
               weights_noobj=0.5,
               numclass=7,):

    self._gtboxes = gtbox
    self._anchors = anchor
    self._positivite_box = pre_box
    self._positivite_confidence = pre_confidence
    self._negativite_confidence = pre_negativite_confidence
    self._weights_noobj = weights_noobj
    self._weights_coordinate = weights_coordinate
    self._numclass = numclass
    self._pre_class =pre_class
    self._gird_cell = gridcell
    self._truelabel = classlabel
    self._class_weights = class_weights
    self._size =size
    self._confidence_weights = confidence_weights

  def classfiction_loss(self):
    '''target_tensor shape is [batch_size,size,num_class]
       logits shape is [batch_size,gtbox_num,num_class]'''
    with tf.variable_scope("classfiction_loss",reuse=None):
      target_tensor = self._truelabel
      logits = self._pre_class
      per_entry_cross_ent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor,logits=logits))

      return per_entry_cross_ent*tf.constant(self._class_weights,tf.float32)

  def localfiction_loss(self,scale_size=None,resize=None):
    '''gtbox shape is [batch_size,num_box,4]
       pre_box shape is [batch_size,size,4]
       anchor shape is [batch_size,size,2]
       gird_cell shape is [size,2]'''
    with tf.variable_scope("locallization_loss",reuse=None):
      xmin_1,ymin_1,xmax_1,ymax_1 = tf.split(self._positivite_box,num_or_size_splits=4,axis=2)
      xmin_2,ymin_2,xmax_2,ymax_2 = tf.split(self._gtboxes,num_or_size_splits=4,axis=2)
      gt_w = (xmax_2 - xmin_2) * (resize / scale_size)
      gt_h = (ymax_2 - ymin_2) * (resize / scale_size)
      gt_x1 = (ymax_2 - ymin_2)/2.0 *(resize / scale_size)
      gt_y1 = (xmax_2 - xmin_2)/2.0 *(resize / scale_size)
      #gt_x2 = (ymax_2 - ymin_2)/2.0 * (resize // scale_size) + gt_w/2.0
      #gt_y2 = (xmax_2 - xmin_2)/2.0 * (resize // scale_size) + gt_h/2.0
      cx,cy = tf.split(self._gird_cell,num_or_size_splits=2,axis=2)
      pw,ph= tf.split((self._anchors[:,:,2:4]-self._anchors[:,:,0:2])*(resize/scale_size),num_or_size_splits=2,axis=2)
      pre_tx = (xmax_1-xmin_1)/2.0
      pre_ty = (ymax_1-ymin_1)/2.0
      pre_tw = (xmax_1-xmin_1) / scale_size
      pre_th = (ymax_1-ymin_1) / scale_size
      bx = ((tf.sigmoid(pre_tx)-cx) / scale_size)
      by = ((tf.sigmoid(pre_ty)-cy) / scale_size)
      bw = pw*tf.exp(pre_tw)
      bh = ph*tf.exp(pre_th)
      prediction_tensor = tf.concat([bx,by,bw,bh],axis=2)
      target_tensor = tf.concat([gt_x1,gt_y1,gt_w,gt_h],axis=2)

      return tf.reduce_mean(tf.losses.huber_loss(
        target_tensor,
        prediction_tensor,
        delta=1.0,
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
      ))*tf.constant(self._weights_coordinate,tf.float32)

  def confidence_loss(self,batch_size=None):
    with tf.variable_scope("confidence_loss"):
      size = self._positivite_confidence.shape.as_list()[1]
      obj_label = tf.tile(tf.expand_dims(tf.ones([batch_size,1],dtype=tf.float32),axis=1),[1,size,1])
      noobj_label = tf.tile(tf.expand_dims(tf.zeros([batch_size,1],dtype=tf.float32),axis=1),[1,size,1])
      obj_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_label,logits=self._positivite_confidence))
      noobj_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=noobj_label,logits=self._negativite_confidence))

      confidences_loss = obj_loss*tf.constant(self._confidence_weights,tf.float32)+noobj_loss*tf.constant(self._weights_noobj,tf.float32)

    return confidences_loss

def build(loss_config):
  hard_example = None

  if "hard_example" in loss_config:
    hard_example = _build_hard_example(loss_config["hard_example"])

  classfication_loss = _build_classfiction_loss(loss_config["classfication_loss"])
  localfication_loss = _build_localfication_loss(loss_config["localfication_loss"])
  confidence_loss = _build_confidence_loss(loss_config["confidence_loss"])


  return (classfication_loss,localfication_loss,confidence_loss,hard_example)

def _build_classfiction_loss(loss_config):

  if loss_config["loss_type"]=='weighted_sigmoid':

    return losses.WeightedSigmoidClassificationLoss()

  if loss_config["loss_type"]=='weighted_softmax':

    return losses.WeightedSoftmaxClassificationLoss(logit_scale=1.0)

def _build_localfication_loss(loss_config):


  loss_type = loss_config["loss_type"]

  if loss_type == 'weighted_l2':
    return losses.WeightedL2LocalizationLoss()

  if loss_type == 'weighted_smooth_l1':
    return losses.WeightedSmoothL1LocalizationLoss(
        delta=loss_config["delta"])


def _build_confidence_loss(loss_config):

  loss_type = loss_config["loss_type"]

  if loss_type=="Cross_Entropy_Loss":

    return losses.Cross_Entropy_Loss(weigths=loss_config["no_obj_weigth"])

def _build_hard_example(loss_config):
  max_negatives_per_positive = None
  num_hard_examples = None
  min_negatives_per_image = None

  if "max_negatives_per_positive" in loss_config:
    if loss_config["max_negatives_per_positive"]<0:
      max_negatives_per_positive = 0
    else:
      max_negatives_per_positive = loss_config["max_negatives_per_positive"]

  if "min_negatives_per_image" in loss_config:
    if loss_config["min_negatives_per_image"]<0:
      min_negatives_per_image = 0
    else:
      min_negatives_per_image = loss_config["min_negatives_per_image"]

  if "num_hard_examples" in loss_config:
    num_hard_examples = loss_config["num_hard_examples"]

  hard_example_miner = losses.Hard_example(num_example=num_hard_examples,iou_threshold=loss_config["iou_threshold"],
                                           loss_type=loss_config["loss_type"],loc_weigth=loss_config["loc_weigth"],
                                           cls_weigth=loss_config["cls_weigth"],obj_weigth=loss_config["obj_weigth"],max_positive=max_negatives_per_positive,
                                           max_negative=min_negatives_per_image)

  return hard_example_miner