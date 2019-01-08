import tensorflow as tf


class Matcher(object):

    def __init__(self,
                 matched_threshold,
                 unmatched_threshold,):

        self._matched_threshold = matched_threshold
        self._unmatched_threshold = unmatched_threshold
        self._batch_iou_max = None
        self._match_list = None
        self._unmatched_list = None
        self._batch_cls_target = None
        self._batch_loc_target = None

    def Iou_interface(self,bbox,gtbox,name="Iou",size=None):
      #gtbox (N,4)
      # bbox (13,13,3,4),(26,26,3,4),(52,52,3,4) if not tiny_yolo else (13,13,3,4),(26,26,3,4)
      # reshape (507,4) (2028,4)
      all_iou_matrix = []
      with tf.name_scope(name):
          if name=="Iou":
              for i,t_bbox in enumerate(bbox):
                  iou_matrix = []
                  t_bbox = tf.reshape(t_bbox,(-1,4)) / float(size)
                  for indx,g_box in enumerate(gtbox):
                    iou = self.Iou(g_box,t_bbox)
                    iou_matrix.append(iou)
                  all_iou_matrix.append(iou_matrix)
          else:
              for i,t_bbox in enumerate(bbox):
                  t_bbox = box_change(t_bbox)
                  iou = self.Iou(gtbox[i],t_bbox)
                  all_iou_matrix.append(iou)

      return all_iou_matrix

    def Iou(self,bbox,gtbox):

        area1 = self.area(gtbox)
        area2 = self.area(bbox)
        xmin1, ymin1, xmax1, ymax1 = tf.split(bbox, num_or_size_splits=4, axis=1)
        xmin2, ymin2, xmax2, ymax2 = tf.split(gtbox, num_or_size_splits=4, axis=1)
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

    def area(self,box_1):
        if box_1.shape.as_list()[1]==2:
            width,heigth = tf.split(box_1, num_or_size_splits=2, axis=1)
            area = tf.multiply(width,heigth)
        else:
            xmin,ymin,xmax,ymax = tf.split(box_1,num_or_size_splits=4,axis=1)
            area = (xmax-xmin) * (ymax - ymin)
        return area

    def match(self,gtbox,gtclass,anchors,size):

        batch_cls_target = []
        batch_loc_target = []
        match_list = []
        unmatch_list = []
        batch_iou_max = []
        '''
        gtbox:Num M groundtruth box shape is [M,4]
        anchors:Num N anchors shape is [N,4]
        iou_matrix: M gtbox with N anchors iou_matrix,[M,N]
        '''

        iou_matrix = self.Iou_interface(anchors,gtbox,size=size)
        for i,sclae_iou in enumerate(iou_matrix):
            sclae_match = []
            sclae_unmatch = []
            sclae_cls_target = []
            sclae_loc_target = []
            iou_max = []
            for j,iou in enumerate(sclae_iou):
                tf.summary.histogram("scale_iou",iou)
                # take the per column max value index,index value between[0~M]
                column_max_indx = tf.argmax(iou,axis=0)
                tf.summary.histogram("indx",column_max_indx)
                # take the per column max value
                column_max_values = tf.reduce_max(iou,axis=0)
                tf.summary.histogram("values",column_max_values)
                iou_max.append(tf.expand_dims(column_max_values,axis=0))
                #Less than or equal to than threshold position is false,more than the threshold position is true,
                #The false position is no target anchors,true position is targeted anchors
                object_mask = tf.greater_equal(column_max_values,self._matched_threshold)
                tf.summary.histogram("obj_mask",tf.cast(object_mask,dtype=tf.float32))
                sclae_match.append(tf.expand_dims(object_mask,axis=0))
                noobj_mask = 1.0 - tf.cast(object_mask,tf.float32)
                sclae_unmatch.append(tf.expand_dims(noobj_mask,axis=0))
                #gtbox [M,4] column_max_indx's length is [N],broadcast gtbox to [N,4]
                new_box = tf.gather(gtbox[j],column_max_indx)
                sclae_loc_target.append(tf.expand_dims(new_box,axis=0))
                new_class = tf.gather(gtclass[j],column_max_indx)
                sclae_cls_target.append(tf.expand_dims(new_class,axis=0))
            batch_loc_target.append(tf.concat(sclae_loc_target,axis=0))
            batch_cls_target.append(tf.concat(sclae_cls_target,axis=0))
            match_list.append(tf.concat(sclae_match,axis=0))
            unmatch_list.append(tf.concat(sclae_unmatch,axis=0))
            batch_iou_max.append(tf.concat(iou_max,axis=0))
        self._match_list = match_list
        self._unmatched_list = unmatch_list
        self._batch_iou_max = batch_iou_max

        return batch_cls_target,batch_loc_target,self._match_list,self._unmatched_list,self._batch_iou_max

    def _num_match_box(self):

        num = tf.reshape(tf.where(tf.equal(self._match_list,1)),[-1])

        return tf.size(tf.cast(num,tf.int32))

    def _num_unmatch_box(self):

        num = tf.reshape(tf.where(tf.equal(self._unmatched_list,0),[-1]))

        return tf.size(tf.cast(num,tf.int32))

def decode(anchors,bbox,grid,input_size,scale,sess=None):
    pre_box_list = []
    if sess!=None:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.tables_initializer())
    if grid.shape.ndims!=2:
        grid = tf.reshape(grid,(-1,2))

    if anchors.shape.ndims!=2:

        anchors = tf.reshape(anchors,(-1,4))

    if isinstance(bbox,list):
        for box in bbox:
            pre_box = box_utils(box,anchors,grid,input_size,scale)
            pre_box_list.append(pre_box)

    elif isinstance(bbox,tf.Tensor):
        if bbox.shape.ndims==2:
            pre_box = box_utils(bbox, anchors, grid, input_size,scale)
            return pre_box

        elif bbox.shape.ndims==3:
            bbox = tf.unstack(bbox)
            for box in bbox:
                pre_box = box_utils(box, anchors, grid, input_size,scale)
                pre_box_list.append(tf.expand_dims(pre_box,axis=0))
                return tf.concat(pre_box_list,axis=0)
    else:
        raise ValueError("The bbox type must be tf.Tensor or tensor list,if tf.Tensor,the rank must be 2 or 3")

    return pre_box_list

def box_utils(bbox,anchors,grid,input_size,scale):

    ''' bx = sigmoid(tx)
    by = sigmoid(ty)
    bh = exp(tw) *anchor_w
    bh = exp(th) *anchor_h
    tx,ty,tw,th is net's outputs value
    '''

    t_x = bbox[:, 0:1]
    t_y = bbox[:, 1:2]
    t_w = bbox[:, 2:3]
    t_h = bbox[:, 3:4]
    anchors_w = anchors[:, 2:3] - anchors[:, 0:1]
    anchors_h = anchors[:, 3:4] - anchors[:, 1:2]
    anchors_wh = tf.concat([anchors_w, anchors_h], axis=1)
    box_xy = tf.concat([t_x, t_y], axis=1)
    box_wh = tf.concat([t_w, t_h], axis=1)
    pre_box_xy = (tf.sigmoid(box_xy) + grid) / scale
    pre_box_wh = tf.exp(box_wh) * anchors_wh / input_size
    pre_box = tf.concat([pre_box_xy, pre_box_wh], axis=1)

    return pre_box

def box_change(boxes):

    new_box = []
    if isinstance(boxes,list):
        for box in boxes:
            n_box = change_coordinate_frame(box)
            new_box.append(n_box)
    elif isinstance(boxes,tf.Tensor):
        if boxes.shape.ndims==3:
            box_list = tf.unstack(boxes)
            for box in box_list:
                n_box = change_coordinate_frame(box)
                new_box.append(tf.expand_dims(n_box,axis=0))
                return tf.concat(new_box,axis=0)
        elif boxes.shape.ndims==2:
            n_box = change_coordinate_frame(boxes)
            return n_box
    else:
        raise ValueError("The boxes must be Tensor of list or tf.Tensor")

    return new_box


def change_coordinate_frame(box):
    '''x_center = box[:,0:1], y_center = box[:,1:2] w = box[:,2:3] h = box[:,3:4]
    change to xmin,ymin,xmax,ymax style
    '''
    if isinstance(box,tf.Tensor):
        x, y, w, h = tf.split(box, num_or_size_splits=4, axis=1)
        xmin = tf.maximum(x - w / 2.0, 0.0)
        xmax = tf.maximum(x + w / 2.0, 0.0)
        ymin = tf.maximum(y - h / 2.0, 0.0)
        ymax = tf.maximum(y + h / 2.0, 0.0)
        n_box = tf.concat([xmin, ymin, xmax, ymax], axis=1)

        return n_box

    elif isinstance(box,list):
        n_boxes = []
        for b in box:
            x, y, w, h = tf.split(b, num_or_size_splits=4, axis=1)
            xmin = tf.maximum(x - w // 2.0,0.0)
            xmax = tf.maximum(x + w // 2.0,0.0)
            ymin = tf.maximum(y - h // 2.0,0.0)
            ymax = tf.maximum(y + h // 2.0,0.0)
            n_box = tf.concat([xmin, ymin, xmax, ymax], axis=1)
            n_boxes.append(n_box)

        return n_boxes



'''
all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2,[0,2,1]))
all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2,[0,2,1]))
intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2,[0,2,1]))
all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2,[0,2,1]))
intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
innersize = tf.multiply(intersect_heights,intersect_widths)
uion = tf.expand_dims(area1,axis=1)+tf.expand_dims(area2,axis=2)-tf.transpose(innersize,[0,2,1])
iou = tf.where(tf.equal(innersize,0),tf.zeros_like(innersize),tf.truediv(innersize,tf.transpose(uion,[0,2,1])))
return tf.cast(iou,dtype=tf.float32)'''







