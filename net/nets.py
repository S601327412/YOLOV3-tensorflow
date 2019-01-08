import tensorflow as tf

import tensorflow.contrib.slim as slim
import functools
import os
import re
class YoloV3_FeatureExtractor(object):

  def __init__(self,
               istrain,
               hyperparams_fn,
               reuse_weights,
               batch_size,
               use_depthwise=True,
               depth_multiplier=2):
    self._istrain = istrain
    self._batchnorm_fn = hyperparams_fn
    self._reuse=reuse_weights
    self._batch_size = batch_size
    self._use_depthwise = use_depthwise
    self.depth_multiplier=depth_multiplier

  '''
  Yolo_53 net
  layer    filters     size             input        output
  0_conv     32       (3,3,stride=1)  416X416X3    416X416X32
  1_conv     64       (3,3,stride=2)  416X416X32   208X208X64
  2_conv     32       (1,1,stride=1)  208X208X64   208X208X32
  3_conv     64       (3,3,stride=1)  208X208X32   208X208X64
  4_Shortcut Layer 1+3 -->next_layer's input  
  5_conv     128      (3,3,stride=2)  208X208X64   104X104X128
  6_conv     64       (1,1,stride=1)  104X104X128  104X104X64
  7_conv     128      (3,3,stride=1)  104X104X64   104X104X128
  8_Shortcut Layer 5+7 -->next_layer's input 
  9_conv     64       (1,1,stride=1)  104X104X128   104X104X64
  10_conv    128      (3,3,stride=1)  104X104X64    104X104X128
  11_Shortcut Layer 8+10 -->next_layer's input
  12_conv    256      (3,3,stride=2)  104X104X128   52X52X256
  13_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  14_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  15_Shortcut Layer 12+14 -->next_layer's input
  16_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  17_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  18_Shortcut Layer 15+17 -->next_layer's input
  19_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  20_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  21_Shortcut Layer 18+20 -->next_layer's input
  22_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  23_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  24_Shortcut Layer 21+23 -->next_layer's input
  25_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  26_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  27_Shortcut Layer 24+26 -->next_layer's input
  28_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  29_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  30_Shortcut Layer 27+29 -->next_layer's input
  31_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  32_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  33_Shortcut Layer 30+32 -->next_layer's input
  34_conv    128      (1,1,stride=1)  52X52X256     52X52X128
  35_conv    256      (3,3,stride=1)  52X52X128     52X52X256
  36_Shortcut Layer 33+35 -->next_layer's input
  37_conv    512      (3,3,stride=2)  52X52X256     26X26X512
  38_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  39_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  40_Shortcut Layer 37+39 -->next_layer's input
  41_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  42_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  43_Shortcut Layer 40+42 -->next_layer's input
  44_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  45_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  46_Shortcut Layer 43+45 -->next_layer's input
  47_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  48_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  49_Shortcut Layer 46+48 -->next_layer's input
  50_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  51_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  52_Shortcut Layer 49+51 -->next_layer's input
  53_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  54_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  55_Shortcut Layer 52+54 -->next_layer's input
  56_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  57_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  58_Shortcut Layer 55+57 -->next_layer's input
  59_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  60_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  61_Shortcut Layer 58+60 -->next_layer's input
  62_conv    1024     (3,3,stride=2)  26X26X512     13X13X1024
  63_conv    512      (1,1,stride=1)  13X13X1024    13X13X512
  64_conv    1024     (3,3,stride=1)  13X13X512     13X13X1024
  65_Shortcut Layer 62+64 -->next_layer's input
  66_conv    512      (1,1,stride=1)  13X13X1024    13X13X512
  67_conv    1024     (3,3,stride=1)  13X13X512     13X13X1024
  68_Shortcut Layer  65+67 -->next_layer's input
  69_conv    512      (1,1,stride=1)  13X13X1024    13X13X512
  70_conv    1024     (3,3,stride=1)  13X13X512     13X13X1024
  71_Shortcut Layer  68+70 -->next_layer's input
  72_conv    512      (1,1,stride=1)  13X13X1024    13X13X512
  73_conv    1024     (3,3,stride=1)  13X13X512     13X13X1024
  74_Shortcut Layer  71+73 -->next_layer's input
  75_conv    512      (1,1,stride=1)  13x13x1024    13x13x512
  76_conv    1024     (3,3,stride=1)  13X13X512     13X13X1024
  77_conv    512      (1,1,stride=1)  13x13x1024    13x13x512
  78_conv    1024     (3,3,stride=1)  13X13X512     13X13X1024
  79_conv    512      (1,1,stride=1)  13x13x1024    13x13x512
  80_conv    1024     (3,3,stride=1)  13X13X512     13X13X1024
  81_conv    num_anchor*(4+1+num_class) (1,1,stride=1)  13X13X1024  13X13Xnum_anchor*(4+1+num_class)
  82 return 81_conv (13,13,3*(4+1+num_class))
  83 route 79_conv -->next_layer input
  84_conv    256      (1,1,stride=1)  13X13X512     13X13X256
  85_upsampleX2 -->(26,26,256) tf.image.resize_images or tf.nn.conv2d_transpose or use 1X1 conv
  86_route 85 61(concat) -->26X26X768 -->next_layer input
  87_conv    256      (1,1,stride=1)  26X26X768     26X26X256
  88_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  89_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  90_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  91_conv    256      (1,1,stride=1)  26X26X512     26X26X256
  92_conv    512      (3,3,stride=1)  26X26X256     26X26X512
  93_conv    num_anchor*(4+1+num_class)      (1,1,stride=1)  26X26X512     26X26Xnum_anchor*(4+1+num_class)
  94 return 93_conv (26,26,3*(4+1+num_class))
  95 route 91_conv -->next_layer input
  96_conv    128      (1,1,stride=1)  26X26X256     26X26X128
  97_upsampleX2 -->(52,52,128) tf.image.resize_images or tf.nn.conv2d_transpose or use 1X1 conv
  98 route 97 36(concat)-->52X52X384 -->next_layer input
  99_conv    128      (1,1,stride=1)  52X52X384     52X52X128
  100_conv   256      (3,3,stride=1)  52X52X128     52X52X256
  101_conv   128      (1,1,stride=1)  52X52X256     52X52X128
  102_conv   256      (3,3,stride=1)  52X52X128     52X52X256
  103_conv   128      (1,1,stride=1)  52X52X256     52X52X128
  104_conv   256      (3,3,stride=1)  52X52X128     52X52X256
  105_conv   num_anchor*(4+1+num_class)      (1,1,stride=1)  52X52X256     52X52Xnum_anchor*(4+1+num_class)
  106 return 105_conv (52,52,3*(4+1+num_class))
  '''
  def op(self,func, *args, **kwargs):

      return functools.partial(func, *args, **kwargs)

  def expand_conv(self,input_tensor, kernel_size, depth_multiplier,stride,scope,insert_1x1_conv=True):
      with tf.variable_scope(name_or_scope=scope) as s, \
              tf.name_scope(s.original_name_scope):
          depth = input_tensor.get_shape().as_list()[3]
          net = input_tensor
          padding = 'SAME'

          if insert_1x1_conv:
              net = slim.conv2d(net, depth_multiplier * depth, [1, 1], scope="1x1_conv")

          if self._use_depthwise:
              net = slim.separable_conv2d(net,None,kernel_size=kernel_size,stride=stride,
                                          padding=padding,rate=1,scope="depthwise")

              net = slim.conv2d(net,depth,[1,1],scope="project")
          else:
              net = slim.conv2d(net,depth,kernel_size=kernel_size,stride=stride,scope="project")

      return net

  #Using tiny-yolo scale is (13,13),(26,26)
  def extractor_feature(self,preprocess_input,num_anchors,num_class,debug,sess=None):
    if not debug:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.tables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners()
    featrue_list = []
    layer_list_1 = [self.op(slim.conv2d, kernel_size=[3,3], stride=1, num_outputs=16, scope="conv_1_3x3_16"),
                    self.op(slim.max_pool2d,kernel_size=[2,2],stride=2,scope="max_pool_1"),
                    self.op(slim.conv2d,kernel_size=[3,3],stride=1,num_outputs=32,scope="conv_2_3x3_32"),
                    self.op(slim.max_pool2d,kernel_size=[2,2],stride=2,scope="max_pool_2"),
                    self.op(slim.conv2d,kernel_size=[3,3],stride=1,num_outputs=64,scope="conv_3_3x3_64"),
                    self.op(slim.max_pool2d,kernel_size=[2,2],stride=2,scope="max_pool_3"),
                    self.op(slim.conv2d,kernel_size=[3,3],stride=1,num_outputs=128,scope="conv_4_3x3_128"),
                    self.op(slim.max_pool2d,kernel_size=[2,2],stride=2,scope="max_pool_4"),
                    self.op(slim.conv2d,kernel_size=[3,3],stride=1,num_outputs=256,scope="conv_5_3x3_256"),
                    self.op(slim.max_pool2d,kernel_size=[2,2],stride=2,scope="max_pool_5"),
                    self.op(slim.conv2d,kernel_size=[3,3],stride=1,num_outputs=512,scope="conv_6_3x3_512"),
                    self.op(slim.max_pool2d,kernel_size=[1,1],stride=1,scope="max_pool_s1_6"),
                    self.op(slim.conv2d,kernel_size=[3,3],stride=1,num_outputs=1024,scope="conv_7_3x3_1024"),]

    layer_list_2 = [self.op(slim.conv2d,kernel_size=[1,1],stride=1,num_outputs=256,scope="conv_8_1x1_256"),
                    self.op(slim.conv2d,kernel_size=[1,1],stride=1,num_outputs=512,scope="conv_9_1x1_512"),
                    self.op(slim.conv2d,kernel_size=[1,1],stride=1,num_outputs=num_anchors*(4+1+num_class),activation_fn=None,scope="scale_1")]

    layer_list_3 = [self.op(slim.conv2d,kernel_size=[1,1],stride=1,num_outputs=128,scope="conv_10_1x1_128"),
                    self.op(tf.image.resize_bilinear,size=[2,2],align_corners=True,name="Upsampel2D")]

    layer_list_4 = [self.op(slim.conv2d,kernel_size=[3,3],stride=1,num_outputs=256,scope="conv_11_3x3_256"),
                    self.op(slim.conv2d,kernel_size=[1,1],stride=1,num_outputs=num_anchors*(4+1+num_class),activation_fn=None,scope="scale_2")]

    with tf.variable_scope("tiny-yolo",reuse=self._reuse) as scope:
      with slim.arg_scope(self._batchnorm_fn()):

        input_tensor = tf.identity(preprocess_input,name="input_tensor")

        featrue_block1 = self._block(input_tensor,layer_list_1)
        key_in_block1 = [key for key in featrue_block1.keys() if re.search("conv_7_3x3_1024",key)][0]
        featrue_block2 = self._block(featrue_block1[key_in_block1],layer_list_2)
        conv_5_3x3_256 = [value for key,value in featrue_block1.items() if re.search("conv_5_3x3_256",key)][0]
        scale_1 = [value for key,value in featrue_block2.items() if re.search("scale_1",key)][0]
        #tf.summary.image(scale_1.op.name,scale_1[:,:,:,0:3],max_outputs=5)
        featrue_list.append(scale_1)
        featrue_block3 = self._block(featrue_block1[key_in_block1],layer_list_3)
        key_in_block3 = [key for key in featrue_block3.keys() if re.search("Upsampel2D",key)][0]
        featrue_block4 = self._block(tf.concat([featrue_block3[key_in_block3],conv_5_3x3_256],axis=3),layer_list_4)
        scale_2 = [value for key,value in featrue_block4.items() if re.search("scale_2",key)][0]
        #tf.summary.image(scale_2.op.name, scale_2[:, :, :, 0:3], max_outputs=5)
        featrue_list.append(scale_2)

        #tf.summary.histogram("feature_map_2", featrue_dict["scale_2"])
    return  featrue_list

  def _block(self,input,layer_list):
    featrue_map_dict = {}
    for block in layer_list:
      if block.func.__name__=='resize_bilinear':
        net = block(input,size=[2*(input.shape[1]),2*(input.shape[1])])

      else:
          net = block(input)
      featrue_map_dict[net.op.name] = net
      tf.summary.image(net.op.name, net[:, :, :, 2:3], max_outputs=5)
      input = net

    return featrue_map_dict



  def test_neural_net(self):
      Multiplier_depth = self.depth_multiplier
      featrue_list = []
      layer_list_1 = [self.op(slim.conv2d, kernel_size=[3, 3], stride=1, num_outputs=16, scope="conv_1_3x3_16"),
                      self.op(self.expand_conv, stride=2, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_16"),
                      self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=32, scope="conv_2_1x1_32"),
                      self.op(self.expand_conv, stride=2, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_32"),
                      self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=64, scope="conv_3_1x1_64"),
                      self.op(self.expand_conv, stride=2, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_64"),
                      self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=128, scope="conv_4_1x1_128"),
                      self.op(self.expand_conv, stride=2, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_128"),
                      self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=256, scope="conv_5_1x1_256",
                              activation_fn=None)]

      layer_list_2 = [self.op(tf.nn.leaky_relu, alpha=0.1, name="leaky_relu"),
                      self.op(self.expand_conv, stride=2, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_s2_256"),
                      self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=512, scope="conv_6_1x1_512"),
                      self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=768, scope="conv_7_1x1_768"),
                      self.op(self.expand_conv, stride=1, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_s1_768"),
                      self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=1024, scope="conv_8_1x1_1024")]

      layer_list_3 = [self.op(self.expand_conv, stride=1, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_s1_1024")]

      layer_list_4 = [
          self.op(slim.conv2d, kernel_size=[1, 1], stride=1, num_outputs=256, scope="conv_11_1x1_256",
                  activation_fn=None),
          self.op(tf.image.resize_bilinear, size=[2, 2], align_corners=True, name="Upsample2D")]

      layer_list_5 = [self.op(self.expand_conv, stride=1, kernel_size=[3, 3], depth_multiplier=Multiplier_depth,
                              scope="expand_outputs_s1_256")]