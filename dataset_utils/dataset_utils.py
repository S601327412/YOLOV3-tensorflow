#coding:utf-8


import os
import tensorflow as tf
import yaml


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def recursive_parse_xml_to_dict(xml):

  if not len(xml):
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def parser_yaml(configfile):
  if not os.path.exists(configfile):
    raise IOError("The json file is not found!")
  f = open(configfile,'r',encoding='utf-8')
  context = f.read()
  config = yaml.load(context)
  return config

def parser_dense_tensor(tensor):
  return tf.sparse_tensor_to_dense(tensor)

def parser_tfrecord(tfrecord):
    '''
    filename_queue = tf.train.string_input_producer([tfrecord])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)'''
    features = tf.parse_single_example(
        tfrecord,
        features={
                  'image/height':tf.FixedLenFeature([],tf.int64),
                  'image/width':tf.FixedLenFeature([],tf.int64),
                  'image/filename':tf.FixedLenFeature([],tf.string),
                  'image/source_id':tf.FixedLenFeature([],tf.string),
                  'image/encoded':tf.FixedLenFeature([],tf.string),
                  'image/format':tf.FixedLenFeature([],tf.string),
                  'image/object/bbox/xmin':tf.VarLenFeature(tf.float32),
                  'image/object/bbox/ymin':tf.VarLenFeature(tf.float32),
                  'image/object/bbox/xmax':tf.VarLenFeature(tf.float32),
                  'image/object/bbox/ymax':tf.VarLenFeature(tf.float32),
                  'image/object/class/text':tf.VarLenFeature(tf.string),
                  'image/object/class/label':tf.VarLenFeature(tf.int64),
                  'image/keypoint':tf.FixedLenFeature([],tf.int64)
              })

    image = tf.cast(tf.image.decode_jpeg(features['image/encoded'],channels=3,dct_method="INTEGER_ACCURATE"),dtype=tf.uint8)
    #image = tf.reshape(image,[640,480,3])
    height = tf.cast(features['image/height'],dtype=tf.int64)
    width = tf.cast(features['image/width'],dtype=tf.int64)
    xmin = parser_dense_tensor(tf.cast(features['image/object/bbox/xmin'],dtype=tf.float32))
    ymin = parser_dense_tensor(tf.cast(features['image/object/bbox/ymin'],dtype=tf.float32))
    xmax = parser_dense_tensor(tf.cast(features['image/object/bbox/xmax'],dtype=tf.float32))
    ymax = parser_dense_tensor(tf.cast(features['image/object/bbox/ymax'],dtype=tf.float32))
    #label_text = tf.cast(features['image/object/class/text'],dtype=tf.string)
    label = parser_dense_tensor(tf.cast(features['image/object/class/label'],dtype=tf.int64))
    keypoint = tf.cast(features['image/keypoint'],dtype=tf.int64)
    filename = tf.reshape(tf.cast(features['image/filename'],dtype=tf.string),[1])
    bbox = tf.cast(tf.stack([xmin,ymin,xmax,ymax],axis=1),dtype=tf.float32)

    tensor_dict = {"image":image,
                   "bbox":bbox,
                   "label":label,
                   "filename":filename,
                   "width":width,
                   "height":height,
                   "keypoint":keypoint}
    return tensor_dict

def make_initializable_iterator(dataset):

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator

