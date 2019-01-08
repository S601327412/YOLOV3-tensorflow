import tensorflow as tf
#from box_utils import boxes_utils
import tensorflow.contrib.slim as slim
from model_builder import detection_model
from box_utils import boxes_utils

class YOLOV3Meta(detection_model.Detection_model):
    # BOX_PREDICTOR = "pre_box"
    # CLASS_PREDICTOR = "pre_class"
    # CONFIDENCE_PREDICTOR = "pre_confidence"
    def __init__(self,
                 istrain,
                 anchor_generator,
                 matcher,
                 feature_extractor,
                 non_max_suppression_fn,
                 score_convert_fn,
                 box_predictor,
                 classfication_loss,
                 localization_loss,
                 confidence_loss,
                 hard_example_mined,
                 iou_threshold,
                 num_class,
                 cls_weigth,
                 conf_weigth,
                 loc_weigth,
                 resize,
                 batch_size,):
        super(YOLOV3Meta,self).__init__(num_classes=num_class)
        self._istrain = istrain
        self._iou_threshold = iou_threshold
        self._anchor_generator = anchor_generator
        self._matcher = matcher
        self._feature_extractor = feature_extractor
        self._box_predictor = box_predictor
        self._nom_max_suppression_fn =non_max_suppression_fn
        self._score_convert = score_convert_fn
        self._classfication_loss = classfication_loss
        self._loclization_loss = localization_loss
        self._confidence_loss = confidence_loss
        self._hard_example_mined = hard_example_mined
        self._cls_weigth = cls_weigth
        self._conf_weigth = conf_weigth
        self._loc_weigth = loc_weigth
        self._resize = resize
        self._batch_size = batch_size
        self._scope = "Feature_extractor"
        self._anchors = None
        self._grid_cell = None
        self._add_summarize = True
        self._scale_size = None
        self.nodebug = True

    def _predictor(self,preprocess_input):
        feature_map_dict={}
        batchnorm_updates_collections = tf.GraphKeys.UPDATE_OPS

        with slim.arg_scope([slim.batch_norm],is_training=self._istrain,
                            updates_collections=batchnorm_updates_collections):
            with tf.variable_scope(self._scope,values=[preprocess_input]):
                feature_map = self._feature_extractor.extractor_feature(preprocess_input,num_anchors=3,num_class=self.num_classes,debug=self.nodebug)

        _anchors,self._grid_cell,self._scale_size,self.per_grid_num_anchor= self._anchor_generator._generator(
            feature_map,self._resize)

        self._anchors = boxes_utils.change_coordinate_frame(_anchors)
        #feature_map_list = self._box_predictor._predict(feature_map,per_grid_num_anchor)

        return feature_map



    def _loss(self,predictor_dict):
        with tf.name_scope("Loss",values=[predictor_dict,self._anchors]):
            with tf.name_scope("Match",values=[self.groundtruth_list()]):
                #Two-layer feature map's obj_mask,noobj_mask
                batch_cls_target,batch_loc_target,batch_match,batch_unmatch,batch_iou_max = self._target_assign(groundtruth_box=self.groundtruth_list()["gtbox"],
                                                         groundtruth_class=self.groundtruth_list()["gtclass"])

            if self._add_summarize:
                #Compute box average numbers,pos box ande neg box numbers at one batch
                self._summarize_target(self.groundtruth_list()["gtbox"],batch_match,batch_unmatch)

            num_anchors = len(self._anchors)
            num_featrue_map = len(predictor_dict)

            if num_anchors!=num_featrue_map:
                raise ValueError("The anchors numbers %d not equal featrue_map %d!"%(num_anchors,num_featrue_map))

            total_loc_loss = 0.0
            total_cls_loss = 0.0
            total_obj_loss = 0.0
            #Compute loss at two-layer feature map successively then sum all
            for i,matches in enumerate(batch_match):

                predictor_loc_target = tf.reshape(predictor_dict[i][:,:,:,0:12],(self._batch_size,-1,4))
                predictor_confiden_target = tf.reshape(predictor_dict[i][:,:,:,12:15],(self._batch_size,-1,1))
                predictor_cls_target = tf.reshape(predictor_dict[i][:,:,:,15:],(self._batch_size,-1,self.num_classes))

                pre_box_list = boxes_utils.decode(self._anchors[i],tf.unstack(predictor_loc_target),self._grid_cell[i],self._resize,scale=self._scale_size[i])
                pre_gtbox_iou = self._matcher.Iou_interface(pre_box_list,self.groundtruth_list()["gtbox"],name="ignore_iou")
                ignore_mask = []
                for iou in pre_gtbox_iou:
                    max_values = tf.reduce_max(iou,axis=0)
                    ignore = tf.greater_equal(self._iou_threshold,max_values)
                    ignore_mask.append(tf.expand_dims(ignore,axis=0))
                ignore_mask = tf.concat(ignore_mask,axis=0)

                #Cross entropy loss function
                cls_loss = self._classfication_loss._compute_loss(predictor_cls_target,
                                                    batch_cls_target[i],
                                                    mask=matches)

                #Enhanced squared error loss function d's defaults value is 1,y1 is gtclass，y2 prediction
                #Loss = 1 / 2 (y1-y2) if (y1-y2)<=d else d*(|y1-y2|- 1/2*d)
                loc_loss = self._loclization_loss._compute_loss(predictor_loc_target,
                                                  batch_loc_target[i],
                                                  grid_cell=self._grid_cell[i],
                                                  mask=matches,
                                                  weights=1.0,
                                                  input_size=self._resize[0],
                                                  scale=self._scale_size[i],
                                                  anchors = self._anchors[i])
                #Cross entropy loss function
                confiden_loss = self._confidence_loss._compute_loss(predictor_confiden_target,
                                                        obj_mask=matches,
                                                        ignore_mask=ignore_mask)
                if self._hard_example_mined:
                    localization_loss,classfication_loss,confidence_loss = self._hard_example_mined(loc_loss,cls_loss,confiden_loss,batch_loc_target[i],matches)
                    total_cls_loss+=classfication_loss
                    total_loc_loss+=localization_loss
                    total_obj_loss+=confidence_loss
                else:
                    localization_loss= tf.multiply(tf.reduce_sum(loc_loss) /float(self._batch_size),self._loc_weigth,name="localization_loss")
                    total_loc_loss+= localization_loss
                    classfication_loss= tf.multiply(tf.reduce_sum(cls_loss) / float(self._batch_size),self._cls_weigth,name="classfication_loss")
                    total_cls_loss+= classfication_loss
                    confidence_loss=tf.multiply(tf.reduce_sum(confiden_loss) / float(self._batch_size),self._conf_weigth,name="confidence_loss")
                    total_obj_loss+= confidence_loss
            tf.summary.scalar("class_loss",total_cls_loss)
            tf.summary.scalar("loc_loss",total_loc_loss)
            tf.summary.scalar("confidence_loss",total_obj_loss)
            loss_dict = {
                "class_loss":total_cls_loss,
                "loc_loss":total_loc_loss,
                "confidence_loss":total_obj_loss,
            }

        return loss_dict


    def _target_assign(self,groundtruth_box,groundtruth_class):
        groundtruth_box_list = groundtruth_box
        groundtruth_class_list = groundtruth_class
        groundtruth_class_with_one_hot = []
        numclass = self.num_classes

        for label in groundtruth_class_list:
            one_hot_label = tf.one_hot(label-1,depth=numclass,axis=1)
            groundtruth_class_with_one_hot.append(one_hot_label)

        return self._matcher.match(groundtruth_box_list,groundtruth_class_with_one_hot,self._anchors,size=self._resize[0])

    def _summarize_target(self, groundtruth_boxes_list, match_list,unmatch_list):
        pos_anchors_per_image = [self._num_matched_box(match) for match in match_list]
        neg_anchors_per_image = [self._num_matched_box(match) for match in unmatch_list]
        tf.summary.scalar('AvgNumGroundtruthBoxesPerImage',
                          tf.add_n([tf.shape(num_box)[0] for num_box in groundtruth_boxes_list])/len(groundtruth_boxes_list),
                          family='TargetAssignment')
        tf.summary.scalar('AvgNumPositiveAnchorsPerImage',
                          tf.reduce_mean(tf.to_float(pos_anchors_per_image)),
                          family='TargetAssignment')
        tf.summary.scalar('AvgNumNegativeAnchorsPerImage',
                          tf.reduce_mean(tf.to_float(neg_anchors_per_image)),
                          family='TargetAssignment')

    def preprocess(self,image):

        if image.dtype!=tf.float32:
            if tf.rank(image)==3:
                image = tf.expand_dims(tf.cast(image,dtype=tf.float32),axis=0)
        true_image_shape = image.shape
        return tf.image.resize_images(image,self._resize,method=0,align_corners=True) * (1.0 / 255.0),true_image_shape

    def _num_matched_box(self,match):

        return tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.cast(match,dtype=tf.int32),axis=1),[-1]))

    def get_width_height(self,image):

        rank = image.shape.ndims
        if rank==4:
            width = tf.cast(tf.shape(image)[1],dtype=tf.float32)
            height = tf.cast(tf.shape(image)[2],dtype=tf.float32)
        elif rank==3:
            width = tf.cast(tf.shape(image)[0], dtype=tf.float32)
            height = tf.cast(tf.shape(image)[1], dtype=tf.float32)

        return width,height

    def post_process(self,prediction_dict, true_image_shapes):

        with tf.name_scope("Post_Process"):
            detection_dict = {}
            detection_box_list = []
            detection_scores_list = []
            detection_class_list = []
            for i,prediction in enumerate(prediction_dict):
                num_anchor = self.per_grid_num_anchor[i] * prediction.shape[1].value * prediction.shape[2].value
                batch_size = true_image_shapes[0].value
                if batch_size is None:
                    batch_size = tf.shape(prediction)[0]
                box_predictor = tf.reshape(prediction[:,:,:,0:12],(batch_size,num_anchor,4))
                confidence_predictor = tf.reshape(prediction[:,:,:,12:15],(batch_size,num_anchor,1))
                class_probability = tf.reshape(prediction[:,:,:,15:],(batch_size,num_anchor,self.num_classes))

                arg = [box_predictor,confidence_predictor,class_probability]

                batch_detection = tf.map_fn(self.sigle_nms,arg,dtype=[tf.float32,tf.float32,tf.int32])
                detection_box_list.append(batch_detection[0])
                detection_scores_list.append(batch_detection[1])
                detection_class_list.append(batch_detection[2])

            detection_dict["detection_boxes"] = tf.concat(detection_box_list,axis=0)
            detection_dict["detection_scores"] = tf.concat(detection_scores_list,axis=0)
            detection_dict["detection_class"] = tf.concat(detection_class_list,axis=0)

            return detection_dict

    def sigle_nms(self,args):


        box_predictor = args[0]
        confidence_predictor = args[1]
        class_probability = args[2]
        boxes = boxes_utils.decode(anchors=self._anchors[self.i], bbox=box_predictor, grid=self._grid_cell[self.i],
                                   input_size=self._resize[self.i],
                                   scale=self._scale_size[self.i])

        class_probability = self._score_convert(class_probability)
        confidence_predictor = self._score_convert(confidence_predictor)
        batch_detection = self._nom_max_suppression_fn(boxes, confidence_predictor, class_probability)

        return batch_detection
