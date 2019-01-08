#-*-coding:utf-8-*-

import tensorflow as tf
class Detection_model(object):
    #主要用来读数据用

    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._groundtruth_dict = {}

    @property
    def num_classes(self):
        return self._num_classes

    def provide_groundtruth(self,groundtruth_box,groundtruth_class,filename,width,height):
        self._groundtruth_dict["gtbox"] = groundtruth_box
        self._groundtruth_dict["gtclass"] = groundtruth_class
        self._groundtruth_dict["filename"] = filename
        self._groundtruth_dict["width"] = width
        self._groundtruth_dict["height"] = height

    def groundtruth_list(self):
        return self._groundtruth_dict

    def process_gtbox(self,box,input_size,width,height):

        img_w = tf.cast(width,dtype=tf.float32)
        img_h = tf.cast(height,dtype=tf.float32)
        scale_size_h = input_size[1] / img_h
        scale_size_w = input_size[0] / img_w
        scale_size = tf.minimum(scale_size_h,scale_size_w)
        dy = (input_size[1]-img_h*scale_size) // 2.
        dx = (input_size[0]-img_w*scale_size) // 2.
        new_box_xmin = (box[:,0:1] *scale_size + dx) / input_size[0]
        new_box_xmax = (box[:,2:3] *scale_size + dx) / input_size[0]
        new_box_ymin = (box[:,1:2] *scale_size + dy) / input_size[1]
        new_box_ymax = (box[:,3:4] *scale_size + dy) / input_size[1]

        new_box = tf.concat([new_box_xmin,new_box_ymin,new_box_xmax,new_box_ymax],axis=1)

        return new_box

