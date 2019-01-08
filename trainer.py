import functools
import tensorflow as tf
from loss import optimizer_builder
from model_deploy import Deploy
import tensorflow.contrib.slim as slim
import get_inputs


def _create_losses(input_queue,model_fn,train_config):

    dectect_model = model_fn()
    batch_size = train_config["batch_size"]
    input_dict = input_queue.dequeue()

    width = []
    height = []
    images = []
    filename = []
    groundtruth_box = []
    groundtruth_class = []
    preprocess_images = []
    preprocess_box = []

    #widths, heights = dectect_model.get_width_height(input_dict[0]["image"])

    for i in range(batch_size):
        width.append(input_dict[i]["width"])
        height.append(input_dict[i]["height"])
        images.append(input_dict[i]["image"])
        groundtruth_box.append(input_dict[i]["bbox"])
        groundtruth_class.append(input_dict[i]["label"])
        filename.append(input_dict[i]["filename"])

    for image in images:
        resize_images = dectect_model.preprocess(image)
        preprocess_images.append(resize_images)

    images = tf.concat(preprocess_images,0)
    for indx,box in enumerate(groundtruth_box):
        resize_box = dectect_model.process_gtbox(box,train_config["resize"],width[indx],height[indx])
        preprocess_box.append(resize_box)

    dectect_model.provide_groundtruth(preprocess_box,groundtruth_class,filename,width,height)

    prediction_dict,true_image_shape = dectect_model._predictor(images)

    '''
    for indx in range(batch_size):
        scale_size,tensor = dectect_model.process_gtbox([416.0,416.0],indx)'''
    loss_dict = dectect_model._loss(prediction_dict)

    for loss_tensor in loss_dict.values():
        tf.losses.add_loss(loss_tensor)

def train(create_input_dict_fn,
          create_model_fn,
          train_config,
          train_dir,
          task,
          num_clones,
          worker_replicas,
          clone_on_cpu,
          ps_tasks,
          worker_job_name,
          is_chief,):

    #模型实例先留着,后面预训练模型时再用
    detection_model = create_model_fn()

    with tf.Graph().as_default():
        #配置类
        deploy_config = Deploy.DeploymentConfig(
            num_clones=num_clones,clone_on_cpu=clone_on_cpu,
            replica_id=task,num_replicas=worker_replicas,
            num_ps_tasks=ps_tasks,
            worker_job_name=worker_job_name)

        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        batch_size = train_config["batch_size"] // num_clones

        with tf.device(deploy_config.inputs_device()):
            #从tfrecord读数据，组成batch,生成样本队列
            input_queue =get_inputs.read_and_transform_dataset(per_clone_batch_size=batch_size,
                                                               create_tensor_dict_fn=create_input_dict_fn)
        #前向计算 forward
        model_fn = functools.partial(_create_losses,
                                     model_fn=create_model_fn,
                                     train_config=train_config)

        clones = Deploy.create_clones(deploy_config,model_fn,input_queue)

        first_clone_scope = clones[0].scope

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,first_clone_scope)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_summaries = set([])


        with tf.device(deploy_config.optimizer_device()):

            #构建优化器,返回优化器实例和参数
            training_optimizer, optimizer_summary_vars = optimizer_builder.build(
                train_config["optimizer"])
            for var in optimizer_summary_vars:
                global_summaries.add(tf.summary.scalar(var.op.name, var, family='LearningRate'))

        with tf.device(deploy_config.optimizer_device()):
            regularization_losses = None if train_config["add_regularization_loss"] else []
            #梯度Gradient
            total_loss, grads_and_vars = Deploy.optimize_clones(
                clones, training_optimizer,
                regularization_losses=regularization_losses)
            #检查Loss为NaN的情况
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')
            #反向传播 Back propagation
            grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                          global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops, name='update_barrier')
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')

        for model_var in slim.get_model_variables():
            global_summaries.add(tf.summary.histogram('ModelVars/' +
                                                      model_var.op.name, model_var))
        for loss_tensor in tf.losses.get_losses():
            global_summaries.add(tf.summary.scalar('Losses/' + loss_tensor.op.name,
                                                   loss_tensor))
        global_summaries.add(
            tf.summary.scalar('Losses/TotalLoss', tf.losses.get_total_loss()))


        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))
        summaries |= global_summaries

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)

        keep_checkpoint_every_n_hours = train_config["keep_checkpoint_every_n_hours"]
        saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        if train_config["num_steps"]:
            step = train_config["num_steps"]
        else:
            step = None
        #开始训练start
        slim.learning.train(
            train_tensor,
            logdir=train_dir,
            is_chief=is_chief,
            session_config=session_config,
            startup_delay_steps=0,
            summary_op=summary_op,
            number_of_steps=step,
            save_summaries_secs=120,
            saver=saver)