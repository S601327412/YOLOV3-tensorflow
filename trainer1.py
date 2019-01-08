
import tensorflow as tf

from model_deploy import Deploy
import tensorflow.contrib.slim as slim
import get_inputs

def _create_losses(input_queue,train_config,model_fn=None):
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

    images = tf.concat(preprocess_images, 0)

    for indx,box in enumerate(groundtruth_box):
        resize_box = dectect_model.process_gtbox(box,train_config["resize"],width[indx],height[indx])
        preprocess_box.append(resize_box)



    #prediction_dict = dectect_model._predictor(images)


    dectect_model.provide_groundtruth(preprocess_box, groundtruth_class, filename, width, height)

    prediction_dict = dectect_model._predictor(images)

    loss = dectect_model._loss(prediction_dict)

    for loss_tensor in loss.values():
        tf.losses.add_loss(loss_tensor)

    return loss


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
        _create_losses(input_queue,train_config,model_fn=create_model_fn)
'''
def match(gtbox, gtclass, anchors):

        batch_cls_target = []
        batch_loc_target = []
        match_list = []
        unmatch_list = []
        batch_iou_max = []

        iou_matrix = Iou_interface(anchors, gtbox)
        for i, sclae_iou in enumerate(iou_matrix):
            sclae_match = []
            sclae_unmatch = []
            sclae_cls_target = []
            sclae_loc_target = []
            iou_max = []
            for j, iou in enumerate(sclae_iou):
                tf.summary.histogram("scale_iou", iou)

                column_max_indx = tf.argmax(iou, axis=0)  # take the per column max value index,index value between[0~M]

                tf.summary.histogram("indx", column_max_indx)
                column_max_values = tf.reduce_max(iou, axis=0)  # take the per column max value

                tf.summary.histogram("values", column_max_values)
                iou_max.append(tf.expand_dims(column_max_values, axis=0))
                # Less than or equal to than threshold position is false,more than the threshold position is true,
                # The false position is no target anchors,true position is targeted anchors
                object_mask = tf.less_equal(0.5, column_max_values)
                tf.summary.histogram("obj_mask", tf.cast(object_mask, dtype=tf.float32))
                sclae_match.append(tf.expand_dims(object_mask, axis=0))
                noobj_mask = 1.0 - tf.cast(object_mask, tf.float32)
                sclae_unmatch.append(tf.expand_dims(noobj_mask, axis=0))
                # gtbox [M,4] column_max_indx's length is [N],broadcast gtbox to [N,4]
                new_box = tf.gather(gtbox[j], column_max_indx)
                sclae_loc_target.append(tf.expand_dims(new_box, axis=0))
                new_class = tf.gather(gtclass[j], column_max_indx)
                sclae_cls_target.append(tf.expand_dims(new_class, axis=0))
            batch_loc_target.append(tf.concat(sclae_loc_target, axis=0))
            batch_cls_target.append(tf.concat(sclae_cls_target, axis=0))
            match_list.append(tf.concat(sclae_match, axis=0))
            unmatch_list.append(tf.concat(sclae_unmatch, axis=0))
            batch_iou_max.append(tf.concat(iou_max, axis=0))
        return batch_cls_target,batch_loc_target,match_list,unmatch_list,batch_iou_max

def Iou(gtbox,bbox):

    area1 = area(bbox)
    area2 = area(gtbox)
    xmin1, ymin1, xmax1, ymax1 = tf.split(gtbox, num_or_size_splits=4, axis=1)
    xmin2, ymin2, xmax2, ymax2 = tf.split(bbox, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2, [1, 0]))
    all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2, [1, 0]))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2, [1, 0]))
    all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2, [1, 0]))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    innersize = tf.multiply(intersect_heights, intersect_widths)
    uion = tf.transpose(area1, [1, 0]) + area2 - innersize
    iou = tf.where(tf.equal(innersize, 0), tf.zeros_like(innersize),
                       tf.truediv(innersize, uion), name="iou")

    return iou

def Iou_interface(bbox,gtbox,name="Iou"):
      #gtbox (N,4)
      # bbox (13,13,3,4),(26,26,3,4),(52,52,3,4) if not tiny_yolo else (13,13,3,4),(26,26,3,4)
      # reshape (507,4) (2028,4)
    all_iou_matrix = []
    with tf.name_scope(name):
        if name=="Iou":
            for i,t_bbox in enumerate(bbox):
                iou_matrix = []
                t_bbox = tf.reshape(t_bbox,(-1,4))
                for indx,g_box in enumerate(gtbox):
                    iou = Iou(g_box,t_bbox)
                    iou_matrix.append(iou)
                all_iou_matrix.append(iou_matrix)

    return all_iou_matrix

def area(box_1):
    if box_1.shape.as_list()[1]==2:
        width,heigth = tf.split(box_1, num_or_size_splits=2, axis=1)
        area = tf.multiply(width,heigth)
    else:
        xmin,ymin,xmax,ymax = tf.split(box_1,num_or_size_splits=4,axis=1)
        area = (xmax-xmin) * (ymax - ymin)
    return area'''