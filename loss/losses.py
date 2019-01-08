import tensorflow as tf


class WeightedSigmoidClassificationLoss(object):

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    mask=None):
    if mask.dtype!=tf.float32:
        mask = tf.expand_dims(tf.cast(mask,tf.float32),axis=2)
    #weights = tf.expand_dims(weights, 2)
    prediction_tensor = tf.where(tf.is_nan(prediction_tensor),target_tensor,prediction_tensor)
    per_entry_cross_ent = mask*tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor)
    return per_entry_cross_ent


class WeightedSoftmaxClassificationLoss(object):

  def __init__(self, logit_scale=1.0):

    self._logit_scale = logit_scale

  def _compute_loss(self, prediction_tensor, target_tensor, weights,mask=None):
    if mask.dtype!=tf.float32:
        mask = tf.expand_dims(tf.cast(mask,tf.float32),axis=2)
    num_classes = prediction_tensor.get_shape().as_list()[-1]
    prediction_tensor = tf.divide(
        prediction_tensor, self._logit_scale, name='scale_logit')
    per_row_cross_ent = (mask*tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))

    return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights*mask

class WeightedL2LocalizationLoss(object):

  def _compute_loss(self, prediction_tensor, target_tensor, weights,mask=None):
    if mask.dtype!=tf.float32:
        mask = tf.expand_dims(tf.cast(mask,tf.float32),axis=2)
    weighted_diff = (prediction_tensor - target_tensor) * tf.expand_dims(
        weights, 2)
    square_diff = mask*0.5 * tf.square(weighted_diff)
    return tf.reduce_mean(square_diff, 2)

class WeightedSmoothL1LocalizationLoss(object):


  def __init__(self, delta=1.0):

    self._delta = delta

  def _compute_loss(self, prediction_tensor, target_tensor,grid_cell,weights,anchors,input_size,scale,mask=None):
    if mask.dtype!=tf.float32:
        mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=2)
    #scaling_stride = input_size / scale
    grid_cell = tf.expand_dims(tf.reshape(tf.cast(grid_cell, dtype=tf.float32), (-1,2)), axis=0)
    anchors = tf.expand_dims(anchors,axis=0)
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:4] - anchors[:, :, 1:2]
    anchors_wh = tf.concat([anchors_w, anchors_h], axis=2)
    gtbox_x = (target_tensor[:, :, 2:3] + target_tensor[:, :, 0:1]) / 2.0
    gtbox_y = (target_tensor[:, :, 3:4] + target_tensor[:, :, 1:2]) / 2.0
    gtbox_xy = tf.concat([gtbox_x, gtbox_y],axis=2)
    gtbox_w = target_tensor[:, :, 2:3] - target_tensor[:, :, 0:1]
    gtbox_h = target_tensor[:, :, 3:4] - target_tensor[:, :, 1:2]
    gtbox_wh = tf.concat([gtbox_w, gtbox_h],axis=2)
    true_xy = gtbox_xy * scale - grid_cell
    true_wh = tf.log(gtbox_wh*input_size / anchors_wh)
    true_wh = tf.where(mask,true_wh,tf.zeros_like(true_wh))
    box_loss_scale = 2 - true_wh[...,0]*true_wh[...,1]
    xy_loss = mask*box_loss_scale*tf.nn.sigmoid_cross_entropy_with_logits(labels=true_xy,logits=prediction_tensor[...,0:2])
    wh_loss = mask*box_loss_scale*tf.losses.huber_loss(labels=true_wh,predictions=prediction_tensor[...,2:4],delta=self._delta,
                                                       weights=weights,loss_collection=None,reduction=tf.losses.Reduction.NONE)

    return tf.reduce_sum(xy_loss,axis=2) + tf.reduce_sum(wh_loss,axis=2)

class Cross_Entropy_Loss(object):

    def __init__(self,weigths):

        self._weigths = weigths

    def _compute_loss(self,prediction_tensor,obj_mask,ignore_mask):
        if obj_mask.dtype != tf.float32:
            obj_mask = tf.expand_dims(tf.cast(obj_mask, tf.float32), axis=2)
        if ignore_mask.dtype!= tf.float32:
            ignore_mask = tf.expand_dims(tf.cast(ignore_mask,tf.float32),axis=2)

        loss = obj_mask*tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_tensor,labels=obj_mask) \
               + (1.0-obj_mask)*ignore_mask*self._weigths*tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_tensor,labels=obj_mask)
        return loss

class Hard_example(object):

    def __init__(self,
                 num_example,
                 iou_threshold=0.7,
                 loss_type='cls',
                 loc_weigth=0.5,
                 cls_weigth=0.5,
                 obj_weigth=0.5,
                 max_positive=3,
                 max_negative=3):
        self.num_example=num_example
        self.iou_threshold=iou_threshold
        self.loss_type=loss_type
        self.loc_weigth=loc_weigth
        self.cls_weigth=cls_weigth
        self.obj_weigth=obj_weigth
        self.max_positive=max_positive
        self.max_negative=max_negative
        self.num_negatives_list=None
        self.num_positives_list=None

    def __call__(self,
                 location_loss,
                 cls_loss,
                 obj_loss,
                 decode_box,
                 match_list):
        mined_loction_loss = []
        mined_cls_loss = []
        mined_confidence_loss =[]
        per_location_loss = tf.unstack(location_loss)
        per_cls_loss = tf.unstack(cls_loss)
        decode_box = tf.unstack(decode_box)
        match_list = tf.unstack(match_list)
        cls_loss = tf.reduce_sum(cls_loss,axis=2)
        obj_loss = tf.reduce_sum(obj_loss,axis=2)
        if not len(per_cls_loss) == len(per_location_loss) == len(decode_box):
            raise ValueError("location_losses:{}, cls_losses:{} and decoded_boxlist_list:{}"
                            "do not have compatible shapes.!" \
                            .format(location_loss.shape,cls_loss.shape,decode_box.shape))

        if not len(match_list) == len(decode_box):

            raise ValueError('match_list length must be equal decode_box length!')

        num_positive_list=[]
        num_negativie_list=[]
        for indx,boxes in enumerate(decode_box):
            match = match_list[indx]
            image_loss = cls_loss[indx]
            if self.loss_type=='cls':
                image_loss = cls_loss[indx]
            elif self.loss_type=='loc':
                image_loss = location_loss[indx]
            elif self.loss_type=='obj':
                image_loss=obj_loss[indx]
            else:
                image_loss*=self.cls_weigth
                image_loss+=location_loss[indx]*self.loc_weigth+obj_loss[indx]*self.obj_weigth
            if self.num_example is not None:
                num_hard_example = self.num_example
            else:
                num_hard_example = tf.shape(boxes)[0]
            #Select top-K box based on the value of the loss
            select_indices = tf.image.non_max_suppression(boxes,image_loss,num_hard_example,
                                                          iou_threshold=self.iou_threshold)
            if self.max_positive is not None:
                #Balance positive and negative sample ratio
                (selected_indices,num_positive,num_negativie) = self.select_balance_positice_negative_box(select_indices,match,
                                                                                                          self.max_positive,self.max_negative)
            num_positive_list.append(num_positive)
            num_negativie_list.append(num_negativie)
            mined_loction_loss.append(
                tf.reduce_sum(tf.gather(location_loss[indx], selected_indices)))
            mined_cls_loss.append(
                tf.reduce_sum(tf.gather(cls_loss[indx], selected_indices)))
            mined_confidence_loss.append(tf.reduce_sum(tf.gather(obj_loss[indx],select_indices)))
        location_loss = tf.reduce_sum(tf.stack(mined_loction_loss)) / float(len(decode_box))
        cls_loss = tf.reduce_sum(tf.stack(mined_cls_loss)) / float(len(decode_box))
        confidence_loss = tf.reduce_sum(tf.stack(mined_confidence_loss)) / float(len(decode_box))
        if  self.max_positive:
            self._num_positives_list = num_positive_list
            self._num_negatives_list = num_negativie_list
        return (location_loss, cls_loss,confidence_loss)


    def select_balance_positice_negative_box(self,indices,match,max_positive_per_negative,min_positive_per_negative):

        match = tf.cast(match,dtype=tf.float32)
        positives_indicator = tf.gather(tf.greater_equal(match,1),indices)
        negatives_indicator = tf.gather(tf.equal(match,0),indices)
        num_positives = tf.reduce_sum(tf.to_int32(positives_indicator))
        max_negatives = tf.maximum(min_positive_per_negative,
                                   tf.to_int32(max_positive_per_negative *
                                               tf.to_float(num_positives)))
        topk_negatives_indicator = tf.less_equal(
            tf.cumsum(tf.to_int32(negatives_indicator)), max_negatives)
        subsampled_selection_indices = tf.where(
            tf.logical_or(positives_indicator, topk_negatives_indicator))
        num_negatives = tf.size(subsampled_selection_indices) - num_positives
        return (tf.reshape(tf.gather(indices, subsampled_selection_indices), [-1]),
                num_positives, num_negatives)