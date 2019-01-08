#coding:utf-8

import os
import argparse
from get_inputs import get_input
import functools
from dataset_utils import dataset_utils
from model_builder import model_debug
import trainer1

parser = argparse.ArgumentParser()
parser.add_argument("--train_config",default='',type=str,help="The training config path")
parser.add_argument("--log",default='',type=str,help="The log and checkpoint will be saved path")
parser.add_argument("--num_clones",default='1',type=int,help="The num_clones to use")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='2'
def main(args):
  if not os.path.exists(args.train_config):
    raise IOError("The train config file is not exists!")
  train_config = dataset_utils.parser_yaml(args.train_config)

  #if not os.path.exists(train_config["xml_path"]):
    #raise IOError("The xml_path is not found!")
  #xml_path = glob.glob(train_config["xml_path"] + "/*.xml")
  if not os.path.exists(train_config["label_file"]):
    raise IOError("The json_label is not exists!")
  if not os.path.exists(args.log):
    os.mkdir(args.log)
  label_dict = dataset_utils.parser_yaml(train_config["label_file"])


  model_fn = functools.partial(
    model_debug.build,
    model_config = train_config,
    is_training = True
  )
  input_config = train_config["input_config"]

  def get_next(config):
    return get_input(config).get_next()

  create_input_dict_fn = functools.partial(get_next,input_config)

  task = 0
  worker_replicas=1
  ps_tasks = 0
  worker_job_name = 'lonely_worker'
  is_chief = True

  trainer1.train(
    create_input_dict_fn,
    create_model_fn=model_fn,
    train_config=train_config,
    train_dir=args.log,
    task=task,
    num_clones=args.num_clones,
    worker_replicas=worker_replicas,
    ps_tasks=ps_tasks,
    worker_job_name=worker_job_name,
    is_chief = is_chief,
    clone_on_cpu=False)

if __name__ == "__main__":
  main(args)



