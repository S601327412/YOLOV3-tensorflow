#-*-coding:utf-8-*-
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from model_builder import model
from tensorflow.python.framework import graph_util
import tempfile
from dataset_utils import dataset_utils
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.client import session

Detection_scores = "detection_scores"
Detection_boxes = "detection_boxes"
Detection_class = "detection_class"
Featrue_class_scores = "class_scores"
Featrue_class_boxes = "class_boxes"
freeze_graph_with_def_protos = freeze_graph.freeze_graph_with_def_protos

def get_output_from_input_tensor(input_tensor,detect_model,preprocess=True,postpreprocess=True,
                                 output_collection_name=""):
    
    if  preprocess:
        preprocess_input,true_image_shapes = detect_model.preprocess(input_tensor)
    else:
        true_image_shapes = [1,input_tensor.shape.ndims[1],input_tensor.shape.ndims[2],3]
        preprocess_input = input_tensor
    output_tensor = detect_model._predictor(preprocess_input)
    if postpreprocess:
        postpreprocess_tensors = detect_model.post_process(output_tensor,true_image_shapes)
    else:
        postpreprocess_tensors = output_tensor
        return postpreprocess_tensors
    return add_output_tensor_nodes(postpreprocess_tensors,
                                   output_collection_name,
                                   postpreprocess)
    
    
def add_output_tensor_nodes(postpreprocess_tensors,output_collection_name,postpreprocess):
    
    if not output_collection_name:
        output_collection_name = 'inference_op'
    
    if postpreprocess:
        boxes = postpreprocess_tensors.get(Detection_boxes)

        scores = postpreprocess_tensors.get(Detection_scores)

        classes = postpreprocess_tensors.get(Detection_class)
        output_dict = {}
        
        output_dict[Detection_boxes] = tf.identity(boxes,name=Detection_boxes)

        output_dict[Detection_scores] = tf.identity(scores,name=Detection_scores)
        output_dict[Detection_class] = tf.identity(classes,name=Detection_class)

        for output_keys in output_dict.keys():
            tf.add_to_collection(output_collection_name,output_dict[output_keys])
    else:
        class_scores = tf.identity(postpreprocess_tensors.get(Featrue_class_scores),name=Featrue_class_scores)
        class_boxes = tf.identity(postpreprocess_tensors.get(Featrue_class_boxes),name=Featrue_class_boxes)
        output_dict = {Featrue_class_boxes:class_boxes,Featrue_class_scores:class_scores}
        
    return output_dict

def _image_tensor_input_placeholder(input_shape=None):

  if input_shape is None:
    input_shape = (None, None, None, 3)
  input_tensor = tf.placeholder(
      dtype=tf.uint8, shape=input_shape, name='image_tensor')
  return input_tensor, input_tensor

def _build_detection_graph(detect_model,input_shape,output_collection_name,input_type="image_tensor",
                           graph_hook_fn=None):

    if input_shape == None and input_type=="image_tensor":

        placeholder,input_tensor = _image_tensor_input_placeholder(input_shape)


    outputs = get_output_from_input_tensor(input_tensor=input_tensor,detect_model=detect_model,output_collection_name=output_collection_name)

    slim.get_or_create_global_step()

    return outputs,placeholder


def export_inference_graph(input_type,
                           config,
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None,
                           write_inference_graph=False):
  config = dataset_utils.parser_yaml(config)
  detection_model = model.build(config,
                                is_training=False)

  _export_inference_graph(
      input_type,
      detection_model,
      config["use_moving_averages"],
      trained_checkpoint_prefix,
      output_directory,
      input_shape,
      write_inference_graph
)

def _export_inference_graph(input_type,
      detection_model,
      use_moving_averages,
      trained_checkpoint_prefix,
      output_directory,
      input_shape,
      write_inference_graph):

    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                     'frozen_inference_graph.pb')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'model.ckpt')

    outputs, placeholder_tensor = _build_detection_graph(
        input_type=input_type,
        detect_model=detection_model,
        input_shape=input_shape,
        output_collection_name="")

    saver_kwargs = {}
    if use_moving_averages:
        if os.path.isfile(trained_checkpoint_prefix):
            saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
            temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
        else:
            temp_checkpoint_prefix = tempfile.mkdtemp()
        replace_variable_values_with_moving_averages(
            tf.get_default_graph(), trained_checkpoint_prefix,
            temp_checkpoint_prefix)
        checkpoint_to_use = temp_checkpoint_prefix
    else:
        checkpoint_to_use = trained_checkpoint_prefix

    saver = tf.train.Saver(**saver_kwargs)
    input_saver_def = saver.as_saver_def()

    write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=checkpoint_to_use)
    if write_inference_graph:
        inference_graph_def = tf.get_default_graph().as_graph_def()
        inference_graph_path = os.path.join(output_directory,
                                            'inference_graph.pbtxt')
        for node in inference_graph_def.node:
            node.device = ''
        with gfile.GFile(inference_graph_path, 'wb') as f:
            f.write(str(inference_graph_def))


    output_node_names = ','.join(outputs.keys())


    frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=checkpoint_to_use,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=frozen_graph_path,
        clear_devices=True,
        initializer_nodes='')

    write_saved_model(saved_model_path, frozen_graph_def,
                      placeholder_tensor, outputs)

def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
  for node in inference_graph_def.node:
    node.device = ''
  with tf.Graph().as_default():
    tf.import_graph_def(inference_graph_def, name='')
    with session.Session() as sess:
      saver = saver_lib.Saver(saver_def=input_saver_def,
                              save_relative_paths=True)
      saver.restore(sess, trained_checkpoint_prefix)
      saver.save(sess, model_path)

def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):

  with tf.Graph().as_default():
    with session.Session() as sess:

      tf.import_graph_def(frozen_graph_def, name='')

      builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

      tensor_info_inputs = {
          'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
      tensor_info_outputs = {}
      for k, v in outputs.items():
        tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

      detection_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs=tensor_info_inputs,
              outputs=tensor_info_outputs,
              method_name=signature_constants.PREDICT_METHOD_NAME))

      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  detection_signature,
          },
      )
      builder.save()