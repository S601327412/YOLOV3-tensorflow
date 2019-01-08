#-*-coding:utf-8-*-


import tensorflow as tf

from box_utils import boxes_utils


def batch_nms(boxes,
              confidence,
              probability,
              iou_threshold,
              scores_threshold,
              max_total_detections,
              change_coordinate_frame=True):
    #sess = tf.InteractiveSession()
    #sess.run(tf.initialize_all_variables())
    #sess.run(tf.tables_initializer())
    num_class = probability.get_shape()[1]

    scores = probability * confidence
    num_scores = tf.shape(scores)[0]

    if change_coordinate_frame:
        if len(boxes.shape)!= 2:
            boxes = tf.reshape(boxes,(-1,4))
        boxes = boxes_utils.change_coordinate_frame(boxes)
        boxes = tf.expand_dims(boxes,axis=1)


    per_box_num_class = tf.unstack(boxes,axis=1)
    boxes_indx = range(num_class) if len(per_box_num_class) > 1 else [0] * num_class.value
    selected_boxes_list = []
    selected_scores_list = []
    selected_class_list = []
    for class_indx,box_indx in zip(range(num_class),boxes_indx):
        per_class_box = tf.unstack(boxes,axis=1)[box_indx]

        class_scores = tf.reshape(tf.slice(scores,[0,class_indx],tf.stack([num_scores,1])),[-1])

        positive_indx = tf.cast(tf.where(tf.greater(class_scores,scores_threshold)),tf.int32)
        gather_scores = tf.gather(class_scores,tf.squeeze(positive_indx))
        match_box = tf.gather(per_class_box,tf.squeeze(positive_indx,axis=1))

        select_indices = tf.image.non_max_suppression(match_box,gather_scores,num_scores,
                                                      iou_threshold=iou_threshold)

        selected_boxes = tf.gather(match_box,select_indices)
        selected_scores = tf.gather(class_scores,select_indices)
        selected_class = tf.zeros_like(selected_scores,dtype=tf.int32) + class_indx
        selected_class_list.append(selected_class)
        selected_boxes_list.append(selected_boxes)
        selected_scores_list.append(selected_scores)
    select_boxes = tf.concat(selected_boxes_list,axis=0)
    select_scores = tf.concat(selected_scores_list,axis=0)
    select_class = tf.concat(selected_class_list,axis=0)
    selected_num_scores = tf.shape(select_scores)[0]
    _, top_k_scores = tf.math.top_k(select_scores, selected_num_scores,
                                    sorted=True)
    sorted_boxes = tf.gather(tf.gather(select_boxes,top_k_scores),
                             tf.range(max_total_detections))

    sorted_scores = tf.gather(tf.gather(select_scores,top_k_scores),
                              tf.range(max_total_detections))

    sorted_class = tf.gather(tf.gather(select_class,top_k_scores),
                             tf.range(max_total_detections))


    return [sorted_boxes,sorted_scores,sorted_class]












