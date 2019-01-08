import tensorflow as tf
import numpy as np

def data_argument(Inputs,methods,gtboxs=None,resize=None,batch_size=None):
  if Inputs.shape[3]!=3:
    raise ValueError('The inputs channles must be 3')
  prebatch_images = []
  input_width = Inputs.shape[0]
  input_height = Inputs.shape[1]
  Inputs.set_shape([batch_size,None,None,3])
  inputs = tf.unstack(Inputs,axis=0)
  with tf.variable_scope('preprocess',reuse=None) as sc:
    if gtboxs == None:
      gtbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[4])

    #batch_image = []
    '''if inputs.dtype!=tf.float32:
      #image = tf.image.convert_image_dtype(inputs,dtype=tf.float32)
      #bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(inputs,bounding_boxes=gtbox,
                                                                       min_object_covered=0.5)

      #distort_image = tf.slice(inputs,bbox_begin,bbox_size)
      #image = tf.image.resize_images(distort_image,[input_height,input_width],method=np.random.randint(4))'''
    for input in inputs:
      input_s = tf.expand_dims(input,axis=0)
      if methods==0:
        image = tf.image.random_brightness(input_s, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif methods==1:
        image = tf.image.adjust_brightness(input,-0.2)
        image = tf.image.adjust_contrast(image,0.1)
      image = tf.image.convert_image_dtype(image *(1.0 / 255.0),dtype=tf.float32)
      image = tf.image.resize_images(image, [resize, resize])
      prebatch_images.append(image)

    return tf.concat(prebatch_images,axis=0)
