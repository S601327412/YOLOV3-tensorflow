#-*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.python.platform import flags
import os
import exporter

input_type = ["image_tensor","inputs"]
types = ["uint","float"]
flags.DEFINE_string('input_type','image_tensor','Type of input_node type,you can choice from '
                                                '[image_tensor,inputs]')
flags.DEFINE_string('data_type','uint','you can choice from [uint,float]')
flags.DEFINE_string('input_shape',None,"The inpnut_node's shape,if dimensions's values is -1,it's shape is unkown,"
                                       "but also you can specify for an input_node'shapes,default shape is [None,None,None,3]")
flags.DEFINE_string('pipline_config_path',None,"It's your model's config file's path")
flags.DEFINE_string('trained_checkpoint_prefix',None,"The path your */.ckpt")
flags.DEFINE_string('output_dict',None,"pb models output path")
flags.DEFINE_string('preprocess',None,"Wheather to add preprocess op into the final graph")
flags.DEFINE_string('postprocess',None,"Wheather to add a new output_nodes into the final graph")
flags.mark_flag_as_required('pipline_config_path')
flags.mark_flag_as_required('trained_checkpoint_prefix')
flags.mark_flag_as_required('output_dict')
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.input_type not in input_type:
        raise ValueError("The input_type must be one from 'image_tensor' or 'inputs':{}".format(FLAGS.input_type))
    if FLAGS.data_type not in types:
        raise ValueError("The data_type must be one from 'uint' or float:{}".format(FLAGS.data_type))
    
    if not os.path.exists(FLAGS.pipline_config_path):
        raise FileNotFoundError("The file {} is not found".format(FLAGS.pipline_config_path))
    
    if not os.path.exists(FLAGS.output_dict):
        os.mkdir(FLAGS.output_dict)
        print("Create model's save output path {}".format(FLAGS.output_dict))
    
    
    exporter.export_inference_graph(input_type=FLAGS.input_type,config=FLAGS.pipline_config_path,
                                    trained_checkpoint_prefix=FLAGS.trained_checkpoint_prefix,output_directory=FLAGS.output_dict)
    
if __name__ == '__main__':
    tf.app.run()