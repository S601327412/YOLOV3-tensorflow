import tensorflow as tf


class Anchor_Generator(object):

    def __init__(self,
                 anchor_config,):

        self._config = anchor_config

    def _generator(self,scale_list,resize):
        #Scale up
        # scale_size_w = float(resize[0]) / image_width
        # scale_size_h = float(resize[1]) / image_height
        # scale_size = tf.minimum(scale_size_w,scale_size_h)
        anchor_list = []
        grid_cell_list = []
        num_anchor = len(self._config)

        scale = []
        ratio = []
        for tensor in scale_list:
            scale.append(tensor.get_shape().as_list()[1])
        for size in scale:
            ratio.append(resize[0] / size)
        if num_anchor!=len(scale_list):
            raise ValueError("The num_scale must be equal num_anchor !")
        per_grid_num_anchor=[]
        if num_anchor == 2:
            for i,j in enumerate(self._config):
                per_grid_num_anchor.append(len(j))
                anchor = tf.convert_to_tensor(j,dtype=tf.float32)

                anchors = tf.tile(tf.reshape(anchor,(1,1,len(j),-1)),(scale[i],scale[i],1,1))

                grid_cell_x = tf.tile(tf.reshape(tf.range(scale[i]),(1,-1,1,1)),[scale[i],1,3,1])
                grid_cell_y = tf.tile(tf.reshape(tf.range(scale[i]),(-1,1,1,1)),[1,scale[i],3,1])
                grid_cell = tf.cast(tf.concat([grid_cell_x,grid_cell_y],axis=3),dtype=tf.float32)
                grid_cell_list.append(grid_cell)

                anchor_n = tf.reshape(tf.concat([(grid_cell + 1 / 2)*ratio[i],anchors],axis=3),(-1,4))
                anchor_list.append(anchor_n)

        return anchor_list,grid_cell_list,scale,per_grid_num_anchor

