from sklearn.cluster import KMeans
import tensorflow as tf
from dataset_utils import dataset_utils
import glob
import sys
from lxml import etree


def k_means(data=None,numclass=6,label_dict=None,input_size=416.0):
  boxes = []
  for file in data:
    xmin =[]
    ymin =[]
    xmax = []
    ymax = []
    try:
      with open(file,"r") as fid:
        xml = fid.read()
    except:
      raise IOError("can't open the xmlfile")

    xml_str = etree.fromstring(xml)
    labeldict = dataset_utils.recursive_parse_xml_to_dict(xml_str)["annotation"]
    scale_size = 1.0
    if 'size' in labeldict:
      for size in labeldict['size']:
        if size=="width":
          scale_x = input_size / float(labeldict['size'][size])
        elif size=="height":
          scale_y = input_size / float(labeldict['size'][size])
        else:
          continue
      scale_size = min(scale_x,scale_y)

    if 'object' in labeldict:
      for obj in labeldict['object']:
        if obj['name'] in label_dict:
          xmin.append(float(obj['bndbox']['xmin']) * scale_size)
          ymin.append(float(obj['bndbox']['ymin']) * scale_size)
          xmax.append(float(obj['bndbox']['xmax']) * scale_size)
          ymax.append(float(obj['bndbox']['ymax']) * scale_size)
      for xmins,ymins,xmaxs,ymaxs in zip(xmin,ymin,xmax,ymax):
        boxes.append([xmins,ymins,xmaxs,ymaxs])
  with tf.device("/cpu:0"):
    kmeans = KMeans(n_clusters=numclass,n_init=200).fit(boxes)
    for indx,(x1,y1,x2,y2) in enumerate(kmeans.cluster_centers_):
      with open("keams_anchor.txt","a") as f:
        f.write("anchor_{}_corner_is_{}".format(indx,[x1,y1,x2,y2])+"\n" )
        f.write("pre_anchors_w_is{}_h_is{}:".format(x2-x1,y2-y1)+"\n")
  with tf.device('/gpu:0'):
    anchor = tf.convert_to_tensor(kmeans.cluster_centers_,dtype=tf.float32)
  print(anchor)
  return anchor


data = glob.glob("/home/ubuntu/下载/VOCdevkit/VOC2012/Annotations"+"/*.xml")
label_dict = dataset_utils.parser_yaml("../VOC_label.yaml")
k_means(data=data,label_dict=label_dict)
sys.exit()
