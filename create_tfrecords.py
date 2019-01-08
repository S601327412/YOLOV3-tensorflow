import tensorflow as tf
import os
import tqdm
import glob
from PIL import Image
import argparse
from dataset_utils import dataset_utils
import io
from lxml import etree
import random
parser = argparse.ArgumentParser()
parser.add_argument('--imagepath',default='',type=str,help='The dataset_image path')
parser.add_argument('--xmlpath',default='',type=str,help='The dataset_image path')
parser.add_argument('--output',default='',type=str,help='The tfrecord ouputs_dir')
parser.add_argument('--labelfile',default='',type=str,help='The label file')

args = parser.parse_args()


def dict_to_example(example_name, annotation, image, label_dict,conut):
  encoded_jpg = image
  encoded_jpg_io = io.BytesIO(image)
  image = Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  width = int(image.width)
  height = int(image.height)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  if 'object' in annotation:
    for obj in annotation['object']:
      if obj['name'] in label_dict:
        xmin.append(float(obj['bndbox']['xmin']))
        ymin.append(float(obj['bndbox']['ymin']))
        xmax.append(float(obj['bndbox']['xmax']))
        ymax.append(float(obj['bndbox']['ymax']))
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_dict[obj['name']])

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_utils.int64_feature(height),
    'image/width': dataset_utils.int64_feature(width),
    'image/filename': dataset_utils.bytes_feature(
      example_name.encode('utf8')),
    'image/source_id': dataset_utils.bytes_feature(
      example_name.encode('utf8')),
    'image/encoded': dataset_utils.bytes_feature(encoded_jpg),
    'image/format': dataset_utils.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_utils.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_utils.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_utils.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_utils.float_list_feature(ymax),
    'image/object/class/text': dataset_utils.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_utils.int64_list_feature(classes),
    'image/keypoint':dataset_utils.int64_list_feature([conut])
  }))
  return example

sets = ['train','valid']

def main(args):
  for set in sets:
    if not os.path.exists(args.output):
        os.mkdir('dataset')
    if not os.path.exists(args.imagepath):
      raise IOError("the path is not exists!")
    images = glob.glob(args.imagepath + '/*.jpg')
    random.shuffle(images)
    traincounts = int(0.9*len(images))
    label_dict = dataset_utils.parser_yaml(args.labelfile)
    if set =="train":
      print("\n Create train set!")
      writer = tf.python_io.TFRecordWriter(os.path.join(args.output, set + '.record'))
      t_count = 0
      for image in tqdm.tqdm(images[:traincounts]):
        try:
          with tf.gfile.GFile(image,'rb') as fid:
            enconded_jpg = fid.read()
        except:
          raise IOError("image can not open!")

        filename = os.path.splitext(os.path.split(image)[1])[0]+'.xml'
        xmlfile = os.path.join(args.xmlpath,filename)
        if not os.path.exists(xmlfile):
          continue
        with tf.gfile.GFile(xmlfile, 'r') as fid:
          xml = fid.read()
        xml_str = etree.fromstring(xml)
        labeldict = dataset_utils.recursive_parse_xml_to_dict(xml_str)['annotation']
        tf_example = dict_to_example(image,labeldict,enconded_jpg,label_dict,t_count)
        t_count+=1
        writer.write(tf_example.SerializeToString())
      writer.close()
    else:
      print("Create valid sets!")
      writer = tf.python_io.TFRecordWriter(os.path.join(args.output, set + '.record'))
      v_count = 0
      for image in tqdm.tqdm(images[traincounts:]):
        try:
          with tf.gfile.GFile(image,'rb') as fid:
            enconded_jpg = fid.read()
        except:
          raise IOError("image can not open!")
        filename = os.path.splitext(os.path.split(image)[1])[0]+'.xml'
        xmlfile = os.path.join(args.xmlpath,filename)
        if not os.path.exists(xmlfile):
          continue
        with tf.gfile.GFile(xmlfile,'r') as fid:
          xml = fid.read()
        xml_str = etree.fromstring(xml)
        labeldict = dataset_utils.recursive_parse_xml_to_dict(xml_str)['annotation']
        tf_example = dict_to_example(image,labeldict,enconded_jpg,label_dict,v_count)
        v_count+=1
        writer.write(tf_example.SerializeToString())
      writer.close()

if __name__ =='__main__':
    main(args)