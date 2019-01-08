import tensorflow as tf



def build(config):
    optimizer = None
    summary_vars = []
    if  config["type"] =="rms_prop_optimizer":
        optimizer_config = config["rms_prop_optimizer"]
        learning_rate = create_learning_rate(optimizer_config["learning_rate"])
        summary_vars.append(learning_rate)
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=config["decay"],
            momentum=config["momentum_optimizer_value"],
            epsilon=config["epsilon"])
    if config["type"] == 'adam_optimizer':
        #config = optimizer_config.adam_optimizer
        learning_rate = create_learning_rate(config["learning_rate"])
        summary_vars.append(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
    if config["use_moving_averages"]:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer,average_decay=config["average_decay"])

    return optimizer,summary_vars


def create_learning_rate(config):
  learning_rate = None
  learning_rate_type = config["type"]

  if learning_rate_type == 'exponential_decay_learning_rate':
    learning_rate = tf.train.exponential_decay(
        config["initial_learning_rate"],
        tf.train.get_or_create_global_step(),
        config["decay_steps"],
        config["decay_factor"],
        staircase=config["staircase"], name='learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    lr = config["initial_learning_rate"]
    learning_rate = tf.constant(lr, dtype=tf.float32,
                                name='learning_rate')

  return learning_rate